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
from .grid_to_pointcloud import load_grid


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
        print("Writing prediction to disk")
        # TODO: Save the prediction to the save_path
        
        
class MeshDataset(Dataset):
    """Dataset class for rendering a mesh."""
    def __init__(self, path, grid_cell_template, grid_size=500, r=32):
        """Initialize the dataset."""
        self.writer = MyPredictionWriter(os.join(path, "layers"))
        self.path = path
        self.grid_cell_template = grid_cell_template
        self.r = r
        self.load_mesh(path)

        self.grid_size = grid_size
        self.grids_to_process = self.init_grids_to_process(path, grid_size)
        
    def load_mesh(self, path):
        """Load the mesh from the given path and extract the vertices, normals, triangles, and UV coordinates."""
        # Get the working path and base name
        working_path = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]

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

        self.mesh = o3d.io.read_triangle_mesh(path)
        print(f"Loaded mesh from {path}", end="\n")

        self.vertices = np.asarray(self.mesh.vertices)
        self.normals = np.asarray(self.mesh.vertex_normals)
        self.triangles = np.asarray(self.mesh.triangles)
        uv = np.asarray(self.mesh.triangle_uvs)
        # scale numpy UV coordinates to the image size
        self.uv = uv * np.array([y_size, x_size])

        # vertices of triangles
        self.triangles_vertices = self.vertices[self.triangles]
        self.triangles_normals = self.normals[self.triangles]
        
    def init_grids_to_process(self):
        # Set up the vertices and triangles
        triangles_vertices_grid_index_raw = self.triangles_vertices / self.grid_size
        triangles_vertices_grid_index_rounded = np.round(triangles_vertices_grid_index_raw).astype(int)
        triangles_vertices_grid_index_mask_r = np.abs(triangles_vertices_grid_index_raw - triangles_vertices_grid_index_rounded) < self.r
        triangles_vertices_grid_index_r = self.triangles_vertices[triangles_vertices_grid_index_mask_r]

        # find all grids that have vertices in a r-padded bounding box
        triangles_vertices_grid_index_r_000 = np.floor(triangles_vertices_grid_index_raw).astype(int)

        triangles_vertices_grid_index_r_001 = np.floor((triangles_vertices_grid_index_r + np.array([0, 0, self.r])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_010 = np.floor((triangles_vertices_grid_index_r + np.array([0, self.r, 0])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_100 = np.floor((triangles_vertices_grid_index_r + np.array([self.r, 0, 0])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_011 = np.floor((triangles_vertices_grid_index_r + np.array([0, self.r, self.r])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_101 = np.floor((triangles_vertices_grid_index_r + np.array([self.r, 0, self.r])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_110 = np.floor((triangles_vertices_grid_index_r + np.array([self.r, self.r, 0])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_111 = np.floor((triangles_vertices_grid_index_r + np.array([self.r, self.r, self.r])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_002 = np.floor((triangles_vertices_grid_index_r + np.array([0, 0, -self.r])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_020 = np.floor((triangles_vertices_grid_index_r + np.array([0, -self.r, 0])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_200 = np.floor((triangles_vertices_grid_index_r + np.array([-self.r, 0, 0])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_022 = np.floor((triangles_vertices_grid_index_r + np.array([0, -self.r, -self.r])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_202 = np.floor((triangles_vertices_grid_index_r + np.array([-self.r, 0, -self.r])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_220 = np.floor((triangles_vertices_grid_index_r + np.array([-self.r, -self.r, 0])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_222 = np.floor((triangles_vertices_grid_index_r + np.array([-self.r, -self.r, -self.r])) / self.grid_size).astype(int)

        triangles_vertices_grid_index_r_012 = np.floor((triangles_vertices_grid_index_r + np.array([0, self.r, -self.r])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_021 = np.floor((triangles_vertices_grid_index_r + np.array([0, -self.r, self.r])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_102 = np.floor((triangles_vertices_grid_index_r + np.array([self.r, 0, -self.r])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_201 = np.floor((triangles_vertices_grid_index_r + np.array([-self.r, self.r, 0])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_120 = np.floor((triangles_vertices_grid_index_r + np.array([self.r, -self.r, 0])) / self.grid_size).astype(int)
        triangles_vertices_grid_index_r_210 = np.floor((triangles_vertices_grid_index_r + np.array([-self.r, 0, self.r])) / self.grid_size).astype(int)

        triangles_vertices_grid_index = np.concatenate([triangles_vertices_grid_index_r_000, triangles_vertices_grid_index_r_001, triangles_vertices_grid_index_r_010, triangles_vertices_grid_index_r_100, triangles_vertices_grid_index_r_011, triangles_vertices_grid_index_r_101, triangles_vertices_grid_index_r_110, triangles_vertices_grid_index_r_111, triangles_vertices_grid_index_r_002, triangles_vertices_grid_index_r_020, triangles_vertices_grid_index_r_200, triangles_vertices_grid_index_r_022, triangles_vertices_grid_index_r_202, triangles_vertices_grid_index_r_220, triangles_vertices_grid_index_r_222, triangles_vertices_grid_index_r_012, triangles_vertices_grid_index_r_021, triangles_vertices_grid_index_r_102, triangles_vertices_grid_index_r_201, triangles_vertices_grid_index_r_120, triangles_vertices_grid_index_r_210])

        # grids that contain at least one triangle
        grids_to_process = set(map(tuple, triangles_vertices_grid_index.reshape(-1, 3)))
        
        grids_to_process = sorted(list(grids_to_process)) # Sort the blocks to process for deterministic behavior
        return grids_to_process
    
    def get_writer(self):
        return self.writer
    
    def extract_triangles_mask(self, grid_index):
        # select all triangles that have at least one vertice in the grid with a r-padded bounding box
        selected_triangles_mask_padded = np.any(np.abs(self.triangles_vertices / self.grid_size - np.array(grid_index)) < self.r, axis=2)
        selected_triangles_mask_floored = np.any(np.floor(self.triangles_vertices / self.grid_size) == np.array(grid_index), axis=2)
        selected_triangles_mask = np.logical_or(selected_triangles_mask_padded, selected_triangles_mask_floored)
        return selected_triangles_mask
    
    def load_grid_cell(self, grid_index, uint8=False):
        # load grid cell from disk
        grid_cell = load_grid(self.grid_cell_template, grid_index, grid_block_size=self.grid_size, grid_block_size=self.grid_size, uint8=uint8)
        return grid_cell
    
    def __len__(self):
        return len(self.grids_to_process)

    def __getitem__(self, idx):
        grid_index = self.grids_to_process[idx]
        triangles_mask = self.extract_triangles_mask(grid_index)
        # Vertices and normals in triangles that have at least one vertice in the grid with a r-padded bounding box
        vertices = self.triangles_vertices[triangles_mask]
        normals = self.triangles_normals[triangles_mask]
        # UV of vertices in triangles that have at least one vertice in the grid with a r-padded bounding box
        uv = self.uv[triangles_mask]

        # load grid cell from disk
        grid_cell = self.load_grid_cell(grid_index)

        return grid_cell, vertices, normals, uv

class PPMAndTextureModel(pl.LightningModule):
    def __init__(self):
        print("instantiating model")
        super().__init__()

    def forward(self, x):
        # TODO
        return None
    
def ppm_and_texture(obj_path, grid_size=500, gpus=1, batch_size=1):
    dataset = MeshDataset(obj_path, obj_path, grid_size=grid_size)
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