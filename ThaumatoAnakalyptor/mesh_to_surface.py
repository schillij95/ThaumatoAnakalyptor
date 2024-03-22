### Julian Schilliger - ThaumatoAnakalyptor - 2023

import multiprocessing
import open3d as o3d
import argparse
import gc
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from PIL import Image
Image.MAX_IMAGE_PIXELS = None



class MyPredictionWriter(BasePredictionWriter):
    def __init__(self, save_path):
        super().__init__(write_interval="batch")  # or "epoch" for end of an epoch
        self.save_path = save_path
    
    def write_on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, prediction, batch_indices, batch, batch_idx: int, dataloader_idx: int) -> None:
        if prediction is None:
            return
        if len(prediction) == 0:
            return

        indexes_2d, values = prediction
        # TODO: Save the prediction to the save_path
        
        
class ObjDataset(Dataset):
    def __init__(self, path, save_path, grid_size=500):
        self.writer = MyPredictionWriter(save_path)
        self.path = path
        self.grid_size = grid_size
        self.grids_to_process = self.init_grids_to_process(path, grid_size)
        
        
    def init_grids_to_process(self, path, grid_size):
        grids_to_process = []

        # TODO
        
        grids_to_process = sorted(list(grids_to_process)) # Sort the blocks to process for deterministic behavior
        return grids_to_process
    
    def get_writer(self):
        return self.writer
    
    def __len__(self):
        return len(self.grids_to_process)

    def __getitem__(self, idx):
        grid_path = self.grids_to_process[idx]
        # TODO
        return None

class PPMAndTextureModel(pl.LightningModule):
    def __init__(self):
        print("instantiating model")
        super().__init__()

    def forward(self, x):
        # TODO
        return None
    
def ppm_and_texture(obj_path, grid_size=500, gpus=1, batch_size=1):
    dataset = ObjDataset(obj_path, obj_path, grid_size=grid_size)
    num_threads = multiprocessing.cpu_count() // int(1.5 * int(gpus))
    num_treads_for_gpus = 5
    num_workers = min(num_threads, num_treads_for_gpus)
    num_workers = max(num_workers, 1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=3)
    model = PPMAndTextureModel()
    
    writer = dataset.get_writer()
    trainer = pl.Trainer(callbacks=[writer], gpus=int(gpus), strategy="ddp")
    
    print("Start Rendering")
    # Run Rendering
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    print("Rendering done")
    
    return

def main(args):
    working_path = os.path.dirname(args.obj_path)
    base_name = os.path.splitext(os.path.basename(args.obj_path))[0]
    surface_path = os.path.join(working_path, f"{base_name}.memmap")

    # Construct the potential paths for the .tif and .png files
    tif_path = os.path.join(working_path, f"{base_name}.tif")
    png_path = os.path.join(working_path, f"{base_name}.png")

    # Check if the .tif version exists
    if os.path.exists(tif_path):
        image_path = tif_path
        print(f"Found TIF image at: {image_path}", end="\n")
    # If not, check if the .png version exists
    elif os.path.exists(png_path):
        image_path = png_path
        print(f"Found PNG image at: {image_path}", end="\n")
    # If neither exists, handle the case (e.g., error message)
    else:
        image_path = None
        print("No corresponding TIF or PNG image found.", end="\n")
    
    # Open the image file
    with Image.open(image_path) as img:
        # Get dimensions
        y_size, x_size = img.size
    print(f"Y-size: {y_size}, X-size: {x_size}", end="\n")

    mesh = o3d.io.read_triangle_mesh(args.obj_path)
    print(f"Loaded mesh from {args.obj_path}", end="\n")

    V = torch.tensor(np.asarray(mesh.vertices))
    N = torch.tensor(np.asarray(mesh.vertex_normals))
    UV = torch.tensor(np.asarray(mesh.triangle_uvs))  # Ensure your mesh has UV coordinates
    F = torch.tensor(np.asarray(mesh.triangles))

    # Adjust UV coordinates as per the requirement
    UV_scaled = UV * torch.tensor([y_size, x_size])

    del mesh, x, y, grid_x, grid_y, UV
    gc.collect()

    print(f"Computing coordinates...", end="\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('obj_path', type=str)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    main(args)