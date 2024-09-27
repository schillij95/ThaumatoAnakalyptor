#!/bin/bash
export OMP_NUM_THREADS=32

CURR_DBSCAN=14.0
CURR_TOPK=750
CURR_QUERY=160
CURR_SIZE=54

# Extend open file limit
ulimit -n 10240

# TEST
python main_instance_segmentation.py \
general.project_name="stpls3d_eval" \
data/datasets=stpls3d \
general.num_targets=2 \
data.num_labels=2 \
data.voxel_size=0.333 \
data.num_workers=32 \
data.cache_data=true \
data.cropping_v1=false \
general.reps_per_epoch=100 \
model.num_queries=${CURR_QUERY} \
general.on_crops=true \
model.config.backbone._target_=models.Res16UNet18B \
general.train_mode=false \
general.checkpoint="saved/train/last-epoch.ckpt" \
data.crop_length=${CURR_SIZE} \
general.eval_inner_core=50.0 \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN}
