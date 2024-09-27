import gc
from contextlib import nullcontext
from pathlib import Path
import statistics
import shutil
import os
import math
import pyviz3d.visualizer as vis
from torch_scatter import scatter_mean
import matplotlib
from benchmark.evaluate_semantic_instance import evaluate
from collections import defaultdict
from sklearn.cluster import DBSCAN
from utils.votenet_utils.eval_det import eval_det
from datasets.scannet200.scannet200_splits import (
    HEAD_CATS_SCANNET_200,
    TAIL_CATS_SCANNET_200,
    COMMON_CATS_SCANNET_200,
    VALID_CLASS_IDS_200_VALIDATION,
)

import hydra
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
from models.metrics import IoU
import random
import colorsys
from typing import List, Tuple
import functools

torch.set_float32_matmul_precision('medium')

@functools.lru_cache(20)
def get_evenly_distributed_colors(
    count: int,
) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    # lru cache caches color tuples
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(
        map(
            lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(
                np.uint8
            ),
            HSV_tuples,
        )
    )


class RegularCheckpointing(pl.Callback):
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        general = pl_module.config.general
        trainer.save_checkpoint(f"{general.save_dir}/last-epoch.ckpt")
        print("Checkpoint created")


class InstanceSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.decoder_id = config.general.decoder_id

        if config.model.train_on_segments:
            self.mask_type = "segment_mask"
        else:
            self.mask_type = "masks"

        self.eval_on_segments = config.general.eval_on_segments

        self.config = config
        self.save_hyperparameters()
        # model
        self.model = hydra.utils.instantiate(config.model)
        self.optional_freeze = nullcontext
        if config.general.freeze_backbone:
            self.optional_freeze = torch.no_grad
        # loss
        self.ignore_label = config.data.ignore_label

        matcher = hydra.utils.instantiate(config.matcher)
        weight_dict = {
            "loss_ce": matcher.cost_class,
            "loss_mask": matcher.cost_mask,
            "loss_dice": matcher.cost_dice,
        }

        aux_weight_dict = {}
        for i in range(self.model.num_levels * self.model.num_decoders):
            if i not in self.config.general.ignore_mask_idx:
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict.items()}
                )
            else:
                aux_weight_dict.update(
                    {k + f"_{i}": 0.0 for k, v in weight_dict.items()}
                )
        weight_dict.update(aux_weight_dict)

        self.preds = dict()
        self.bbox_preds = dict()
        self.bbox_gt = dict()

        self.criterion = hydra.utils.instantiate(
            config.loss, matcher=matcher, weight_dict=weight_dict
        )

        # metrics
        self.confusion = hydra.utils.instantiate(config.metrics)
        self.iou = IoU()
        # misc
        self.labels_info = dict()
        self.losses = defaultdict(list)
        
        self.prepare_data()
        
    def forward(
        self, x, point2segment=None, raw_coordinates=None, is_eval=False
    ):
        with self.optional_freeze():
            x = self.model(
                x,
                point2segment,
                raw_coordinates=raw_coordinates,
                is_eval=is_eval,
            )
        return x
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if len(batch[0]) == 0:
            return []
        prediction = self.inference(batch[0])
        return prediction
    
    def inference(self, batch):
        # check if self.inference_c_fn exists
        if not hasattr(self, "inference_c_fn"):
            print("Instantiating inference_c_fn")
            self.inference_c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return self._inference(self.inference_c_fn(batch))

    def _inference(self, batch):
        data, target, file_names = batch
        inverse_maps = data.inverse_maps
        target_full = data.target_full
        original_colors = data.original_colors
        data_idx = data.idx
        original_normals = data.original_normals
        original_coordinates = data.original_coordinates

        # if len(target) == 0 or len(target_full) == 0:
        #    print("no targets")
        #    return None

        if len(data.coordinates) == 0:
            return 0.0

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]

        if raw_coordinates.shape[0] == 0:
            return 0.0

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )

        try:
            output = self.forward(
                data,
                point2segment=[
                    target[i]["point2segment"] for i in range(len(target))
                ],
                raw_coordinates=raw_coordinates,
                is_eval=True,
            )
        except RuntimeError as run_err:
            print(run_err)
            if (
                "only a single point gives nans in cross-attention"
                == run_err.args[0]
            ):
                return None
            else:
                raise run_err

        if self.config.data.test_mode != "test":
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(False)

            try:
                losses = self.criterion(
                    output, target, mask_type=self.mask_type
                )
            except ValueError as val_err:
                print(f"ValueError: {val_err}")
                print(f"data shape: {data.shape}")
                print(f"data feat shape:  {data.features.shape}")
                print(f"data feat nans:   {data.features.isnan().sum()}")
                print(f"output: {output}")
                print(f"target: {target}")
                print(f"filenames: {file_names}")
                raise val_err

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(True)

        if self.config.general.save_visualizations:
            backbone_features = (
                output["backbone_features"].F.detach().cpu().numpy()
            )
            from sklearn import decomposition

            pca = decomposition.PCA(n_components=3)
            pca.fit(backbone_features)
            pca_features = pca.transform(backbone_features)
            rescaled_pca = (
                255
                * (pca_features - pca_features.min())
                / (pca_features.max() - pca_features.min())
            )

        preds = self.infere_instance_step(
            output,
            target,
            target_full,
            inverse_maps,
            file_names,
            original_coordinates,
            original_colors,
            original_normals,
            raw_coordinates,
            data_idx,
            backbone_features=rescaled_pca
            if self.config.general.save_visualizations
            else None,
        )

        return preds

    def training_step(self, batch, batch_idx):
        data, target, file_names = batch

        if data.features.shape[0] > self.config.general.max_batch_size:
            print("data exceeds threshold")
            raise RuntimeError("BATCH TOO BIG")

        if len(target) == 0:
            print("no targets")
            return None

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )

        try:
            output = self.forward(
                data,
                point2segment=[target[i]["point2segment"] for i in range(len(target))],
                raw_coordinates=raw_coordinates,
            )
        except RuntimeError as run_err:
            print(run_err)
            if "only a single point gives nans in cross-attention" == run_err.args[0]:
                return None
            else:
                raise run_err

        try:
            losses = self.criterion(output, target, mask_type=self.mask_type)
        except ValueError as val_err:
            print(f"ValueError: {val_err}")
            print(f"data shape: {data.shape}")
            print(f"data feat shape:  {data.features.shape}")
            print(f"data feat nans:   {data.features.isnan().sum()}")
            print(f"output: {output}")
            print(f"target: {target}")
            print(f"filenames: {file_names}")
            raise val_err

        raw_logged_losses = {
            f"train_raw_{k}": v.detach().cpu().item() for k, v in losses.items()
        }

        scaled_logged_losses = {}
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                scaled_logged_losses[f"train_{k}"] = (
                    losses[k] * self.criterion.weight_dict[k]
                ).detach().cpu().item()
            else:
                losses.pop(k)

        # Log raw and scaled losses
        self.log_dict({**raw_logged_losses, **scaled_logged_losses}, prog_bar=True, sync_dist=True)

        # Return the sum of scaled losses
        return sum(losses.values())


    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def export(self, pred_masks, scores, pred_classes, file_names, decoder_id):
        root_path = f"eval_output"
        base_path = f"{root_path}/instance_evaluation_{self.config.general.experiment_name}_{self.current_epoch}/decoder_{decoder_id}"
        pred_mask_path = f"{base_path}/pred_mask"

        Path(pred_mask_path).mkdir(parents=True, exist_ok=True)

        file_name = file_names
        with open(f"{base_path}/{file_name}.txt", "w") as fout:
            real_id = -1
            for instance_id in range(len(pred_classes)):
                real_id += 1
                pred_class = pred_classes[instance_id]
                score = scores[instance_id]
                mask = pred_masks[:, instance_id].astype("uint8")

                if score > self.config.general.export_threshold:
                    # reduce the export size a bit. I guess no performance difference
                    np.savetxt(
                        f"{pred_mask_path}/{file_name}_{real_id}.txt",
                        mask,
                        fmt="%d",
                    )
                    fout.write(
                        f"pred_mask/{file_name}_{real_id}.txt {pred_class} {score}\n"
                    )

    def training_epoch_end(self, outputs):
        train_loss = sum([out["loss"].cpu().item() for out in outputs]) / len(
            outputs
        )
        results = {"train_loss_mean": train_loss}
        self.log_dict(results, prog_bar=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        # Ensure that outputs are available
        if outputs:
            all_logged_losses = defaultdict(list)
            total_val_loss = 0.0

            # Collect all relevant loss values from outputs
            for out in outputs:
                if "loss" in out:
                    total_val_loss += out["loss"].cpu().item()
                for key in out.keys():
                    all_logged_losses[key].append(out[key])

            # Calculate the mean of the total validation loss
            val_loss = total_val_loss / len(outputs)

            # Calculate and log the mean of each specific loss
            for loss_name, loss_values in all_logged_losses.items():
                mean_loss = sum(loss_values) / len(loss_values)
                self.log(loss_name, mean_loss, prog_bar=True, sync_dist=True)

            # Log the overall validation loss
            self.log("val_loss_mean", val_loss, prog_bar=True, sync_dist=True)

    def save_visualizations(
        self,
        target_full,
        full_res_coords,
        sorted_masks,
        sort_classes,
        file_name,
        original_colors,
        original_normals,
        sort_scores_values,
        point_size=20,
        sorted_heatmaps=None,
        query_pos=None,
        backbone_features=None,
    ):

        full_res_coords -= full_res_coords.mean(axis=0)

        gt_pcd_pos = []
        gt_pcd_normals = []
        gt_pcd_color = []
        gt_inst_pcd_color = []
        gt_boxes = []

        if "labels" in target_full:
            instances_colors = torch.from_numpy(
                np.vstack(
                    get_evenly_distributed_colors(
                        target_full["labels"].shape[0]
                    )
                )
            )
            for instance_counter, (label, mask) in enumerate(
                zip(target_full["labels"], target_full["masks"])
            ):
                if label == 255:
                    continue

                mask_tmp = mask.detach().cpu().numpy()
                mask_coords = full_res_coords[mask_tmp.astype(bool), :]

                if len(mask_coords) == 0:
                    continue

                gt_pcd_pos.append(mask_coords)
                mask_coords_min = full_res_coords[
                    mask_tmp.astype(bool), :
                ].min(axis=0)
                mask_coords_max = full_res_coords[
                    mask_tmp.astype(bool), :
                ].max(axis=0)
                size = mask_coords_max - mask_coords_min
                mask_coords_middle = mask_coords_min + size / 2

                gt_boxes.append(
                    {
                        "position": mask_coords_middle,
                        "size": size,
                        "color": self.validation_dataset.map2color([label])[0],
                    }
                )

                gt_pcd_color.append(
                    self.validation_dataset.map2color([label]).repeat(
                        gt_pcd_pos[-1].shape[0], 1
                    )
                )
                gt_inst_pcd_color.append(
                    instances_colors[instance_counter % len(instances_colors)]
                    .unsqueeze(0)
                    .repeat(gt_pcd_pos[-1].shape[0], 1)
                )

                gt_pcd_normals.append(
                    original_normals[mask_tmp.astype(bool), :]
                )

            gt_pcd_pos = np.concatenate(gt_pcd_pos)
            gt_pcd_normals = np.concatenate(gt_pcd_normals)
            gt_pcd_color = np.concatenate(gt_pcd_color)
            gt_inst_pcd_color = np.concatenate(gt_inst_pcd_color)

        v = vis.Visualizer()

        v.add_points(
            "RGB Input",
            full_res_coords,
            colors=original_colors,
            normals=original_normals,
            visible=True,
            point_size=point_size,
        )

        if backbone_features is not None:
            v.add_points(
                "PCA",
                full_res_coords,
                colors=backbone_features,
                normals=original_normals,
                visible=False,
                point_size=point_size,
            )

        if "labels" in target_full:
            v.add_points(
                "Semantics (GT)",
                gt_pcd_pos,
                colors=gt_pcd_color,
                normals=gt_pcd_normals,
                alpha=0.8,
                visible=False,
                point_size=point_size,
            )
            v.add_points(
                "Instances (GT)",
                gt_pcd_pos,
                colors=gt_inst_pcd_color,
                normals=gt_pcd_normals,
                alpha=0.8,
                visible=False,
                point_size=point_size,
            )

        pred_coords = []
        pred_normals = []
        pred_sem_color = []
        pred_inst_color = []

        for did in range(len(sorted_masks)):
            instances_colors = torch.from_numpy(
                np.vstack(
                    get_evenly_distributed_colors(
                        max(1, sorted_masks[did].shape[1])
                    )
                )
            )

            for i in reversed(range(sorted_masks[did].shape[1])):
                coords = full_res_coords[
                    sorted_masks[did][:, i].astype(bool), :
                ]

                mask_coords = full_res_coords[
                    sorted_masks[did][:, i].astype(bool), :
                ]
                mask_normals = original_normals[
                    sorted_masks[did][:, i].astype(bool), :
                ]

                label = sort_classes[did][i]

                if len(mask_coords) == 0:
                    continue

                pred_coords.append(mask_coords)
                pred_normals.append(mask_normals)

                pred_sem_color.append(
                    self.validation_dataset.map2color([label]).repeat(
                        mask_coords.shape[0], 1
                    )
                )

                pred_inst_color.append(
                    instances_colors[i % len(instances_colors)]
                    .unsqueeze(0)
                    .repeat(mask_coords.shape[0], 1)
                )

            if len(pred_coords) > 0:
                pred_coords = np.concatenate(pred_coords)
                pred_normals = np.concatenate(pred_normals)
                pred_sem_color = np.concatenate(pred_sem_color)
                pred_inst_color = np.concatenate(pred_inst_color)

                v.add_points(
                    "Semantics (Mask3D)",
                    pred_coords,
                    colors=pred_sem_color,
                    normals=pred_normals,
                    visible=False,
                    alpha=0.8,
                    point_size=point_size,
                )
                v.add_points(
                    "Instances (Mask3D)",
                    pred_coords,
                    colors=pred_inst_color,
                    normals=pred_normals,
                    visible=False,
                    alpha=0.8,
                    point_size=point_size,
                )

        v.save(
            f"{self.config['general']['save_dir']}/visualizations/{file_name}"
        )

    
    def eval_step(self, batch, batch_idx):
        data, target, file_names = batch

        if len(data.coordinates) == 0:
            return 0.0

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]

        if raw_coordinates.shape[0] == 0:
            return 0.0

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )

        try:
            output = self.forward(
                data,
                point2segment=[target[i]["point2segment"] for i in range(len(target))],
                raw_coordinates=raw_coordinates,
                is_eval=True,
            )
        except RuntimeError as run_err:
            print(run_err)
            if "only a single point gives nans in cross-attention" in run_err.args[0]:
                return None
            else:
                raise run_err

        if self.config.data.test_mode != "test":
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(False)

            try:
                losses = self.criterion(output, target, mask_type=self.mask_type)
            except ValueError as val_err:
                print(f"ValueError: {val_err}")
                print(f"data shape: {data.shape}")
                print(f"data feat shape:  {data.features.shape}")
                print(f"data feat nans:   {data.features.isnan().sum()}")
                print(f"output: {output}")
                print(f"target: {target}")
                print(f"filenames: {file_names}")
                raise val_err

            # Prepare dictionaries for raw and scaled losses
            raw_logged_losses = {}
            scaled_logged_losses = {}

            # Log raw (unscaled) losses
            for k, v in losses.items():
                raw_logged_losses[f"val_raw_{k}"] = v.detach().cpu().item()

            # Scale losses by their respective weights and log them
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    scaled_loss_value = losses[k] * self.criterion.weight_dict[k]
                    scaled_logged_losses[f"val_{k}"] = scaled_loss_value.detach().cpu().item()
                else:
                    losses.pop(k)

            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(True)

            # Return both raw and scaled logged losses
            return {**raw_logged_losses, **scaled_logged_losses}
        else:
            return 0.0





    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)


    def get_full_res_mask(
        self, mask, inverse_map, point2segment_full, is_heatmap=False
    ):
        mask = mask.detach().cpu()[inverse_map]  # full res

        if self.eval_on_segments and is_heatmap == False:
            mask = scatter_mean(
                mask, point2segment_full, dim=0
            )  # full res segments
            mask = (mask > 0.5).float()
            mask = mask.detach().cpu()[
                point2segment_full.cpu()
            ]  # full res points

        return mask

    def get_mask_and_scores(
        self, mask_cls, mask_pred, num_queries=100, num_classes=18, device=None
    ):
        if device is None:
            device = self.device
        labels = (
            torch.arange(num_classes, device=device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )

        if self.config.general.topk_per_image != -1:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                self.config.general.topk_per_image, sorted=True
            )
        else:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                num_queries, sorted=True
            )

        labels_per_query = labels[topk_indices]
        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[:, topk_indices]

        result_pred_mask = (mask_pred > 0).float()
        heatmap = mask_pred.float().sigmoid()

        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
            result_pred_mask.sum(0) + 1e-6
        )
        score = scores_per_query * mask_scores_per_image
        classes = labels_per_query

        return score, result_pred_mask, classes, heatmap

    def eval_instance_step(
        self,
        output,
        target_low_res,
        target_full_res,
        inverse_maps,
        file_names,
        full_res_coords,
        original_colors,
        original_normals,
        raw_coords,
        idx,
        first_full_res=False,
        backbone_features=None,
    ):
        label_offset = self.validation_dataset.label_offset
        prediction = output["aux_outputs"]
        prediction.append(
            {
                "pred_logits": output["pred_logits"],
                "pred_masks": output["pred_masks"],
            }
        )

        prediction[self.decoder_id][
            "pred_logits"
        ] = torch.functional.F.softmax(
            prediction[self.decoder_id]["pred_logits"], dim=-1
        )[
            ..., :-1
        ]

        all_pred_classes = list()
        all_pred_masks = list()
        all_pred_scores = list()
        all_heatmaps = list()
        all_query_pos = list()

        offset_coords_idx = 0
        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            if not first_full_res:
                if self.model.train_on_segments:
                    masks = (
                        prediction[self.decoder_id]["pred_masks"][bid]
                        .detach()
                        .cpu()[target_low_res[bid]["point2segment"].cpu()]
                    )
                else:
                    masks = (
                        prediction[self.decoder_id]["pred_masks"][bid]
                        .detach()
                        .cpu()
                    )

                if self.config.general.use_dbscan:
                    new_preds = {
                        "pred_masks": list(),
                        "pred_logits": list(),
                    }

                    curr_coords_idx = masks.shape[0]
                    curr_coords = raw_coords[
                        offset_coords_idx : curr_coords_idx + offset_coords_idx
                    ]
                    offset_coords_idx += curr_coords_idx

                    for curr_query in range(masks.shape[1]):
                        curr_masks = masks[:, curr_query] > 0

                        if curr_coords[curr_masks].shape[0] > 0:
                            clusters = (
                                DBSCAN(
                                    eps=self.config.general.dbscan_eps,
                                    min_samples=self.config.general.dbscan_min_points,
                                    n_jobs=-1,
                                )
                                .fit(curr_coords[curr_masks])
                                .labels_
                            )

                            new_mask = torch.zeros(curr_masks.shape, dtype=int)
                            new_mask[curr_masks] = (
                                torch.from_numpy(clusters) + 1
                            )

                            for cluster_id in np.unique(clusters):
                                original_pred_masks = masks[:, curr_query]
                                if cluster_id != -1:
                                    new_preds["pred_masks"].append(
                                        original_pred_masks
                                        * (new_mask == cluster_id + 1)
                                    )
                                    new_preds["pred_logits"].append(
                                        prediction[self.decoder_id][
                                            "pred_logits"
                                        ][bid, curr_query]
                                    )

                    scores, masks, classes, heatmap = self.get_mask_and_scores(
                        torch.stack(new_preds["pred_logits"]).cpu(),
                        torch.stack(new_preds["pred_masks"]).T,
                        len(new_preds["pred_logits"]),
                        self.model.num_classes - 1,
                    )
                else:
                    scores, masks, classes, heatmap = self.get_mask_and_scores(
                        prediction[self.decoder_id]["pred_logits"][bid]
                        .detach()
                        .cpu(),
                        masks,
                        prediction[self.decoder_id]["pred_logits"][bid].shape[
                            0
                        ],
                        self.model.num_classes - 1,
                    )

                masks = self.get_full_res_mask(
                    masks,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                )

                heatmap = self.get_full_res_mask(
                    heatmap,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                    is_heatmap=True,
                )

                if backbone_features is not None:
                    backbone_features = self.get_full_res_mask(
                        torch.from_numpy(backbone_features),
                        inverse_maps[bid],
                        target_full_res[bid]["point2segment"],
                        is_heatmap=True,
                    )
                    backbone_features = backbone_features.numpy()
            else:
                assert False, "not tested"
                masks = self.get_full_res_mask(
                    prediction[self.decoder_id]["pred_masks"][bid].cpu(),
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                )

                scores, masks, classes, heatmap = self.get_mask_and_scores(
                    prediction[self.decoder_id]["pred_logits"][bid].cpu(),
                    masks,
                    prediction[self.decoder_id]["pred_logits"][bid].shape[0],
                    self.model.num_classes - 1,
                    device="cpu",
                )

            masks = masks.numpy()
            heatmap = heatmap.numpy()

            sort_scores = scores.sort(descending=True)
            sort_scores_index = sort_scores.indices.cpu().numpy()
            sort_scores_values = sort_scores.values.cpu().numpy()
            sort_classes = classes[sort_scores_index]

            sorted_masks = masks[:, sort_scores_index]
            sorted_heatmap = heatmap[:, sort_scores_index]

            if self.config.general.filter_out_instances:
                keep_instances = set()
                pairwise_overlap = sorted_masks.T @ sorted_masks
                normalization = pairwise_overlap.max(axis=0)
                norm_overlaps = pairwise_overlap / normalization

                for instance_id in range(norm_overlaps.shape[0]):
                    # filter out unlikely masks and nearly empty masks
                    # if not(sort_scores_values[instance_id] < 0.3 or sorted_masks[:, instance_id].sum() < 500):
                    if not (
                        sort_scores_values[instance_id]
                        < self.config.general.scores_threshold
                    ):
                        # check if mask != empty
                        if not sorted_masks[:, instance_id].sum() == 0.0:
                            overlap_ids = set(
                                np.nonzero(
                                    norm_overlaps[instance_id, :]
                                    > self.config.general.iou_threshold
                                )[0]
                            )

                            if len(overlap_ids) == 0:
                                keep_instances.add(instance_id)
                            else:
                                if instance_id == min(overlap_ids):
                                    keep_instances.add(instance_id)

                keep_instances = sorted(list(keep_instances))
                all_pred_classes.append(sort_classes[keep_instances])
                all_pred_masks.append(sorted_masks[:, keep_instances])
                all_pred_scores.append(sort_scores_values[keep_instances])
                all_heatmaps.append(sorted_heatmap[:, keep_instances])
            else:
                all_pred_classes.append(sort_classes)
                all_pred_masks.append(sorted_masks)
                all_pred_scores.append(sort_scores_values)
                all_heatmaps.append(sorted_heatmap)

        if self.validation_dataset.dataset_name == "scannet200":
            all_pred_classes[bid][all_pred_classes[bid] == 0] = -1
            if self.config.data.test_mode != "test":
                target_full_res[bid]["labels"][
                    target_full_res[bid]["labels"] == 0
                ] = -1

        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            all_pred_classes[
                bid
            ] = self.validation_dataset._remap_model_output(
                all_pred_classes[bid].cpu() + label_offset
            )

            if (
                self.config.data.test_mode != "test"
                and len(target_full_res) != 0
            ):
                target_full_res[bid][
                    "labels"
                ] = self.validation_dataset._remap_model_output(
                    target_full_res[bid]["labels"].cpu() + label_offset
                )

                # PREDICTION BOX
                bbox_data = []
                for query_id in range(
                    all_pred_masks[bid].shape[1]
                ):  # self.model.num_queries
                    obj_coords = full_res_coords[bid][
                        all_pred_masks[bid][:, query_id].astype(bool), :
                    ]
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(
                            axis=0
                        ) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))

                        bbox_data.append(
                            (
                                all_pred_classes[bid][query_id].item(),
                                bbox,
                                all_pred_scores[bid][query_id],
                            )
                        )
                self.bbox_preds[file_names[bid]] = bbox_data

                # GT BOX
                bbox_data = []
                for obj_id in range(target_full_res[bid]["masks"].shape[0]):
                    if target_full_res[bid]["labels"][obj_id].item() == 255:
                        continue

                    obj_coords = full_res_coords[bid][
                        target_full_res[bid]["masks"][obj_id, :]
                        .cpu()
                        .detach()
                        .numpy()
                        .astype(bool),
                        :,
                    ]
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(
                            axis=0
                        ) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))
                        bbox_data.append(
                            (
                                target_full_res[bid]["labels"][obj_id].item(),
                                bbox,
                            )
                        )

                self.bbox_gt[file_names[bid]] = bbox_data

            if self.config.general.eval_inner_core == -1:
                self.preds[file_names[bid]] = {
                    "pred_masks": all_pred_masks[bid],
                    "pred_scores": all_pred_scores[bid],
                    "pred_classes": all_pred_classes[bid],
                }
            else:
                # prev val_dataset
                self.preds[file_names[bid]] = {
                    "pred_masks": all_pred_masks[bid][
                        self.test_dataset.data[idx[bid]]["cond_inner"]
                    ],
                    "pred_scores": all_pred_scores[bid],
                    "pred_classes": all_pred_classes[bid],
                }

            if self.config.general.save_visualizations:
                if "cond_inner" in self.test_dataset.data[idx[bid]]:
                    target_full_res[bid]["masks"] = target_full_res[bid][
                        "masks"
                    ][:, self.test_dataset.data[idx[bid]]["cond_inner"]]
                    self.save_visualizations(
                        target_full_res[bid],
                        full_res_coords[bid][
                            self.test_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        [self.preds[file_names[bid]]["pred_masks"]],
                        [self.preds[file_names[bid]]["pred_classes"]],
                        file_names[bid],
                        original_colors[bid][
                            self.test_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        original_normals[bid][
                            self.test_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        [self.preds[file_names[bid]]["pred_scores"]],
                        sorted_heatmaps=[
                            all_heatmaps[bid][
                                self.test_dataset.data[idx[bid]]["cond_inner"]
                            ]
                        ],
                        query_pos=all_query_pos[bid][
                            self.test_dataset.data[idx[bid]]["cond_inner"]
                        ]
                        if len(all_query_pos) > 0
                        else None,
                        backbone_features=backbone_features[
                            self.test_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        point_size=self.config.general.visualization_point_size,
                    )
                else:
                    self.save_visualizations(
                        target_full_res[bid],
                        full_res_coords[bid],
                        [self.preds[file_names[bid]]["pred_masks"]],
                        [self.preds[file_names[bid]]["pred_classes"]],
                        file_names[bid],
                        original_colors[bid],
                        original_normals[bid],
                        [self.preds[file_names[bid]]["pred_scores"]],
                        sorted_heatmaps=[all_heatmaps[bid]],
                        query_pos=all_query_pos[bid]
                        if len(all_query_pos) > 0
                        else None,
                        backbone_features=backbone_features,
                        point_size=self.config.general.visualization_point_size,
                    )

            if self.config.general.export:
                self.export(
                    self.preds[file_names[bid]]["pred_masks"],
                    self.preds[file_names[bid]]["pred_scores"],
                    self.preds[file_names[bid]]["pred_classes"],
                    file_names[bid],
                    self.decoder_id,
                )

    def infere_instance_step(
        self,
        output,
        target_low_res,
        target_full_res,
        inverse_maps,
        file_names,
        full_res_coords,
        original_colors,
        original_normals,
        raw_coords,
        idx,
        first_full_res=False,
        backbone_features=None,
    ):
        label_offset = self.validation_dataset.label_offset
        prediction = output["aux_outputs"]
        prediction.append(
            {
                "pred_logits": output["pred_logits"],
                "pred_masks": output["pred_masks"],
            }
        )

        prediction[self.decoder_id][
            "pred_logits"
        ] = torch.functional.F.softmax(
            prediction[self.decoder_id]["pred_logits"], dim=-1
        )[
            ..., :-1
        ]

        all_pred_classes = list()
        all_pred_masks = list()
        all_pred_scores = list()
        all_heatmaps = list()
        all_query_pos = list()

        offset_coords_idx = 0
        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            if not first_full_res:
                if self.model.train_on_segments:
                    masks = (
                        prediction[self.decoder_id]["pred_masks"][bid]
                        .detach()
                        .cpu()[target_low_res[bid]["point2segment"].cpu()]
                    )
                else:
                    masks = (
                        prediction[self.decoder_id]["pred_masks"][bid]
                        .detach()
                        .cpu()
                    )

                if self.config.general.use_dbscan:
                    new_preds = {
                        "pred_masks": list(),
                        "pred_logits": list(),
                    }

                    curr_coords_idx = masks.shape[0]
                    curr_coords = raw_coords[
                        offset_coords_idx : curr_coords_idx + offset_coords_idx
                    ]
                    offset_coords_idx += curr_coords_idx

                    for curr_query in range(masks.shape[1]):
                        curr_masks = masks[:, curr_query] > 0

                        if curr_coords[curr_masks].shape[0] > 0:
                            clusters = (
                                DBSCAN(
                                    eps=self.config.general.dbscan_eps,
                                    min_samples=self.config.general.dbscan_min_points,
                                    n_jobs=-1,
                                )
                                .fit(curr_coords[curr_masks])
                                .labels_
                            )

                            new_mask = torch.zeros(curr_masks.shape, dtype=int)
                            new_mask[curr_masks] = (
                                torch.from_numpy(clusters) + 1
                            )

                            for cluster_id in np.unique(clusters):
                                original_pred_masks = masks[:, curr_query]
                                if cluster_id != -1:
                                    new_preds["pred_masks"].append(
                                        original_pred_masks
                                        * (new_mask == cluster_id + 1)
                                    )
                                    new_preds["pred_logits"].append(
                                        prediction[self.decoder_id][
                                            "pred_logits"
                                        ][bid, curr_query]
                                    )

                    scores, masks, classes, heatmap = self.get_mask_and_scores(
                        torch.stack(new_preds["pred_logits"]).cpu(),
                        torch.stack(new_preds["pred_masks"]).T,
                        len(new_preds["pred_logits"]),
                        self.model.num_classes - 1,
                    )
                else:
                    scores, masks, classes, heatmap = self.get_mask_and_scores(
                        prediction[self.decoder_id]["pred_logits"][bid]
                        .detach()
                        .cpu(),
                        masks,
                        prediction[self.decoder_id]["pred_logits"][bid].shape[
                            0
                        ],
                        self.model.num_classes - 1,
                    )

                masks = self.get_full_res_mask(
                    masks,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                )

                heatmap = self.get_full_res_mask(
                    heatmap,
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                    is_heatmap=True,
                )

                if backbone_features is not None:
                    backbone_features = self.get_full_res_mask(
                        torch.from_numpy(backbone_features),
                        inverse_maps[bid],
                        target_full_res[bid]["point2segment"],
                        is_heatmap=True,
                    )
                    backbone_features = backbone_features.numpy()
            else:
                assert False, "not tested"
                masks = self.get_full_res_mask(
                    prediction[self.decoder_id]["pred_masks"][bid].cpu(),
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                )

                scores, masks, classes, heatmap = self.get_mask_and_scores(
                    prediction[self.decoder_id]["pred_logits"][bid].cpu(),
                    masks,
                    prediction[self.decoder_id]["pred_logits"][bid].shape[0],
                    self.model.num_classes - 1,
                    device="cpu",
                )

            masks = masks.numpy()
            heatmap = heatmap.numpy()

            sort_scores = scores.sort(descending=True)
            sort_scores_index = sort_scores.indices.cpu().numpy()
            sort_scores_values = sort_scores.values.cpu().numpy()
            sort_classes = classes[sort_scores_index]

            sorted_masks = masks[:, sort_scores_index]
            sorted_heatmap = heatmap[:, sort_scores_index]

            if self.config.general.filter_out_instances:
                keep_instances = set()
                pairwise_overlap = sorted_masks.T @ sorted_masks
                normalization = pairwise_overlap.max(axis=0)
                norm_overlaps = pairwise_overlap / normalization

                for instance_id in range(norm_overlaps.shape[0]):
                    # filter out unlikely masks and nearly empty masks
                    # if not(sort_scores_values[instance_id] < 0.3 or sorted_masks[:, instance_id].sum() < 500):
                    if not (
                        sort_scores_values[instance_id]
                        < self.config.general.scores_threshold
                    ):
                        # check if mask != empty
                        if not sorted_masks[:, instance_id].sum() == 0.0:
                            overlap_ids = set(
                                np.nonzero(
                                    norm_overlaps[instance_id, :]
                                    > self.config.general.iou_threshold
                                )[0]
                            )

                            if len(overlap_ids) == 0:
                                keep_instances.add(instance_id)
                            else:
                                if instance_id == min(overlap_ids):
                                    keep_instances.add(instance_id)

                keep_instances = sorted(list(keep_instances))
                all_pred_classes.append(sort_classes[keep_instances])
                all_pred_masks.append(sorted_masks[:, keep_instances])
                all_pred_scores.append(sort_scores_values[keep_instances])
                all_heatmaps.append(sorted_heatmap[:, keep_instances])
            else:
                all_pred_classes.append(sort_classes)
                all_pred_masks.append(sorted_masks)
                all_pred_scores.append(sort_scores_values)
                all_heatmaps.append(sorted_heatmap)

        preds = {}
        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            all_pred_classes[
                bid
            ] = self.validation_dataset._remap_model_output(
                all_pred_classes[bid].cpu() + label_offset
            )

            if (
                self.config.data.test_mode != "test"
                and len(target_full_res) != 0
            ):
                target_full_res[bid][
                    "labels"
                ] = self.validation_dataset._remap_model_output(
                    target_full_res[bid]["labels"].cpu() + label_offset
                )


            preds[bid] = {
                "pred_masks": all_pred_masks[bid],
                "pred_scores": all_pred_scores[bid],
                "pred_classes": all_pred_classes[bid],
            }

            # important stuff
            '''self.preds[file_names[bid]]["pred_masks"],
            self.preds[file_names[bid]]["pred_scores"],
            self.preds[file_names[bid]]["pred_classes"],'''
        return preds
        

    def eval_instance_epoch_end(self):
        log_prefix = "val"
        loss_results = {}

        # Aggregate and calculate mean losses
        for key, values in self.losses.items():
            loss_results[f"{log_prefix}_{key}_mean"] = statistics.mean(values)

        # Log the mean losses
        self.log_dict(loss_results)

        # Clear losses for the next epoch
        self.losses.clear()

        gc.collect()


    def test_epoch_end(self, outputs):
        if self.config.general.export:
            return
        
        # Ensure that outputs are available
        if outputs:
            all_logged_losses = defaultdict(list)

            # Collect all relevant loss values from outputs
            for out in outputs:
                for key in out.keys():
                    all_logged_losses[key].append(out[key])

            # Calculate and log the mean of each loss
            for loss_name, loss_values in all_logged_losses.items():
                mean_loss = sum(loss_values) / len(loss_values)
                self.log(loss_name, mean_loss, prog_bar=True, sync_dist=True)


    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=self.parameters()
        )
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(
                self.train_dataloader()
            )
        lr_scheduler = hydra.utils.instantiate(
            self.config.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def prepare_data(self):
        self.train_dataset = hydra.utils.instantiate(
            self.config.data.train_dataset
        )
        self.validation_dataset = hydra.utils.instantiate(
            self.config.data.validation_dataset
        )
        self.test_dataset = hydra.utils.instantiate(
            self.config.data.test_dataset
        )
        self.labels_info = self.train_dataset.label_info

    def on_load_checkpoint(self, checkpoint):
        # Get the list of optimizers
        optimizers = self.optimizers()
        
        # Ensure we correctly handle the case with multiple optimizers
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        
        # Set the learning rate to the value defined in the config for each optimizer
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.config.optimizer.lr
        print(f"Learning rate reset to {self.config.optimizer.lr}")

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        print("TEST COLLATION", self.config.data.test_collation)
        res = hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )
        return res