### Julian Schilliger - ThaumatoAnakalyptor - 2023

import multiprocessing
import open3d as o3d
import argparse
import gc
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import tifffile


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
        self.writer = MyPredictionWriter(os.path.join(path, "layers"))
        self.path = path
        self.grid_cell_template = grid_cell_template
        self.r = r
        self.load_mesh(path)

        self.grid_size = grid_size
        self.grids_to_process = self.init_grids_to_process()

    def parse_mtl_for_texture_filenames(self, mtl_filepath):
        texture_filenames = []
        with open(mtl_filepath, 'r') as file:
            for line in file:
                if line.strip().startswith('map_Kd'):  # This line indicates a texture file in MTL
                    parts = line.split()
                    if len(parts) > 1:
                        texture_filenames.append(parts[1])  # The second part is the filename
        return texture_filenames
        
    def load_mesh(self, path):
        """Load the mesh from the given path and extract the vertices, normals, triangles, and UV coordinates."""
        # Get the working path and base name
        working_path = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]

        # Construct the potential paths for the .tif and .png files
        tif_path = os.path.join(working_path, f"{base_name}.tif")
        png_path = os.path.join(working_path, f"{base_name}.png")
        mtl_path = os.path.join(working_path, f"{base_name}.mtl")

        # Check if the .tif version exists
        if os.path.exists(tif_path):
            image_path = tif_path
            print(f"Found TIF image at: {image_path}", end="\n")
        # If not, check if the .png version exists
        elif os.path.exists(png_path):
            image_path = png_path
            print(f"Found PNG image at: {image_path}", end="\n")
        elif os.path.exists(mtl_path):
            texture_filenames = self.parse_mtl_for_texture_filenames(mtl_path)
            if len(texture_filenames) > 0:
                image_path = os.path.join(working_path, texture_filenames[0])
                print(f"Found material texture image at: {image_path}", end="\n")
            else:
                image_path = None
                print("No corresponding TIF, PNG, or MTL image found.", end="\n")
        # If neither exists, handle the case (e.g., error message)
        else:
            image_path = None
            print("No corresponding TIF or PNG image found.", end="\n")
        
        print("Texture Image Name:", image_path)
        if image_path:
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
        uv = np.asarray(self.mesh.triangle_uvs).reshape(-1, 3, 2)
        # scale numpy UV coordinates to the image size
        self.uv = uv * np.array([y_size, x_size])

        # vertices of triangles
        self.triangles_vertices = self.vertices[self.triangles]
        self.triangles_normals = self.normals[self.triangles]
        
    def init_grids_to_process(self):
        grids_to_process = set()
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    grids_to_process.update(map(tuple, np.floor((self.triangles_vertices + np.array([x*self.r, y*self.r, z*self.r])) / self.grid_size).astype(int).reshape(-1, 3)))
        
        grids_to_process = sorted(list(grids_to_process)) # Sort the blocks to process for deterministic behavior
        print(f"Number of grids to process: {len(grids_to_process)}")
        return grids_to_process
    
    def get_writer(self):
        return self.writer
    
    def extract_triangles_mask(self, grid_index):
        # select all triangles that have at least one vertice in the grid with a r-padded bounding box
        # grid_size * grid_index - r <= vertice <= grid_size * grid_index + r
        selected_triangles_mask = np.any(np.all(np.logical_and(self.triangles_vertices >= np.array(grid_index) * self.grid_size - self.r, self.triangles_vertices <= (np.array(grid_index) + 1) * self.grid_size + self.r), axis=2), axis=1)

        return selected_triangles_mask
    
    def load_grid_cell(self, grid_index, uint8=False):
        path = self.grid_cell_template.format(grid_index[0]+1, grid_index[1]+1, grid_index[2]+1)

        # Check if the file exists
        if not os.path.exists(path):
            # print(f"File {path} does not exist.")
            return None

        # Read the image
        with tifffile.TiffFile(path) as tif:
            grid_cell = tif.asarray()

        if uint8:
            grid_cell = np.uint8(grid_cell//256)
        return grid_cell
    
    def __len__(self):
        return len(self.grids_to_process)

    def __getitem__(self, idx):
        grid_index = self.grids_to_process[idx]
        triangles_mask = self.extract_triangles_mask(grid_index)
        # Vertices and normals in triangles that have at least one vertice in the grid with a r-padded bounding box
        print(f"Shape of triangles mask: {triangles_mask.shape}, shape of triangles vertices: {self.triangles_vertices.shape}, shape of UV: {self.uv.shape}")
        vertices = self.triangles_vertices[triangles_mask]
        normals = self.triangles_normals[triangles_mask]
        # UV of vertices in triangles that have at least one vertice in the grid with a r-padded bounding box
        uv = self.uv[triangles_mask]

        # load grid cell from disk
        grid_cell = self.load_grid_cell(grid_index)
        if grid_cell is None:
            return None, None, None, None
        grid_cell = grid_cell.astype(np.float32)

        # Convert NumPy arrays to PyTorch tensors
        vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
        normals_tensor = torch.tensor(normals, dtype=torch.float32)
        uv_tensor = torch.tensor(uv, dtype=torch.float32)
        grid_cell_tensor = torch.tensor(grid_cell, dtype=torch.float32)

        return grid_cell_tensor, vertices_tensor, normals_tensor, uv_tensor

class PPMAndTextureModel(pl.LightningModule):
    def __init__(self):
        print("instantiating model")
        super().__init__()

    def ppm(self, pts, tri):
        # pts B x 2
        # tri_pts 3 x 2
        # triangles 3
        v0 = tri[2, :] - tri[0, :]
        v1 = tri[1, :] - tri[0, :]
        v2 = pts - tri[0, :]

        dot00 = torch.dot(v0, v0)
        dot01 = torch.dot(v0, v1)
        dot11 = torch.dot(v1, v1)
        dot02 = torch.dot(v0, v2)
        dot12 = torch.dot(v1, v2)

        invDenom = 1 / (dot00 * dot11 - dot01.pow(2))
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        is_inside = (u >= 0) & (v >= 0) & ((u + v) <= 1 )

        u_vals = u[is_inside]
        v_vals = v[is_inside]
        w_vals = 1 - u_vals - v_vals

        bary_coords = torch.stack([u_vals, v_vals, w_vals], dim=1)
        bary_coords = normalize(bary_coords, p=1, dim=1)

        return bary_coords

    def points_in_triangles(self, pts, tri_pts):
        # pts B x 2
        # tri_pts tri x 3 x 2
        # triangles tri x 3
        v0 = tri_pts[:, 2, :] - tri_pts[:, 0, :]
        v1 = tri_pts[:, 1, :] - tri_pts[:, 0, :]
        v2 = pts - tri_pts[:, 0, :]

        dot00 = v0.pow(2).sum(dim=1)
        dot01 = (v0 * v1).sum(dim=1)
        dot11 = v1.pow(2).sum(dim=1)
        dot02 = (v2 * v0).sum(dim=1)
        dot12 = (v2 * v1).sum(dim=1)

        invDenom = 1 / (dot00 * dot11 - dot01.pow(2))
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        is_inside = (u >= 0) & (v >= 0) & ((u + v) <= 1 )

        bary_coords = torch.zeros((u.shape[0], 3), dtype=torch.float64, device=pts.device)
        triangle_indices = torch.where(is_inside, is_inside.float().argmax(dim=0), torch.tensor(-1, device=pts.device, dtype=torch.int32))
        inside_mask = triangle_indices != -1

        u_vals = torch.gather(u[inside_mask], 0, triangle_indices[inside_mask])
        v_vals = torch.gather(v[inside_mask], 0, triangle_indices[inside_mask])
        w_vals = 1 - u_vals - v_vals
        bary_coords[inside_mask] = torch.stack([u_vals, v_vals, w_vals], dim=1)
        bary_coords = normalize(bary_coords, p=1, dim=1)
        return triangle_indices, bary_coords
    
    def create_grid_points_tensor(self, starting_points, w, h):
        device = starting_points.device
        n = starting_points.shape[0]
        
        # Generate all combinations of offsets in the grid
        dx = torch.arange(w, device=device)  # Shape (w,)
        dy = torch.arange(h, device=device)  # Shape (h,)
        
        # Create a meshgrid from dx and dy
        mesh_dx, mesh_dy = torch.meshgrid(dx, dy, indexing='xy')  # Shapes (h, w)
        
        # Stack and reshape to create the complete offset grid
        offset_grid = torch.stack((mesh_dx, mesh_dy), dim=2).view(-1, 2)  # Shape (w*h, 2)
        
        # Expand starting points for broadcasting
        starting_points_expanded = starting_points.view(n, 1, 2)  # Shape (n, 1, 2)
        
        # Add starting points to offset grid (broadcasting in action)
        grid_points = starting_points_expanded + offset_grid  # Shape (n, w*h, 2)
        
        return grid_points

    def forward(self, x):
        new_order = [2,1,0]
        # grid_cell: B x W x W x W, vertices: B x T x 3 x 3, normals: B x T x 3 x 3, uv_coords_triangles: B x T x 3 x 2
        grid_cells, vertices, normals, uv_coords_triangles, grid_index = x
        
        # Handle the case where the grid cells are empty
        if grid_cells is None:
            return None
        
        # vertices = vertices[:, new_order, :]
        # normals = normals[:, new_order, :]
        print(f"Vertices: {vertices.shape}, Normals: {normals.shape}, UV: {uv_coords_triangles.shape}", end="\n")
        
        # Step 1: Compute AABBs for each triangle
        min_uv, _ = torch.min(uv_coords_triangles, dim=1)
        max_uv, _ = torch.max(uv_coords_triangles, dim=1)
        # Floor and ceil the UV coordinates
        min_uv = torch.floor(min_uv)
        max_uv = torch.ceil(max_uv)

        print(f"Scaled min UV: {min_uv.shape}, Scaled max UV: {max_uv.shape}", end="\n")

        # Find largest max-min difference for each 2d axis
        max_diff_uv, _ = torch.max(max_uv - min_uv, dim=0)
        # max_diff, _ = torch.max(max_diff_uv, dim=0)
        print(f"Max diff: {max_diff_uv}", end="\n")

        # starting_points = torch.flatten(min_uv, start_dim=0, end_dim=-2)
        # print(f"Starting points: {starting_points.shape}", end="\n")

        # Step 2: Generate Meshgrids for All Triangles
        # create grid points tensor
        grid_points = self.create_grid_points_tensor(min_uv, max_diff_uv[0], max_diff_uv[1])

        # Step 3: Copmute Barycentric Coordinates for All Triangles
        baryicentric_coords = self.ppm(grid_points, uv_coords_triangles)


        grid_indices = []  # List to collect the grid indices

        # # Step 2: Generate Meshgrids for All Triangles
        # for i in range(grid_cell.shape[0]):
        #     ppm_triangle = []  # List to collect all grid points
        #     for u in range(uv_coords_triangles[i].shape[0]):
        #         # Create a meshgrid of points within the AABB
        #         u_range = torch.arange(start=min_uv[i][u][0], end=max_uv[i][u][0]).int()
        #         v_range = torch.arange(start=min_uv[i][u][1], end=max_uv[i][u][1]).int()
        #         U, V = torch.meshgrid(u_range, v_range, indexing='xy')
        #         grid_points = torch.stack([U.flatten(), V.flatten()], dim=1)

        #         baryicentric_coords = self.ppm(grid_points, uv_coords_triangles[i])

        #         coords = torch.einsum('ij,i->i', vertices[i][u], baryicentric_coords)
        #         norms = normalize(torch.einsum('ij,i->i', normals[i][u], baryicentric_coords),dim=1)
                
        #         ppm_triangle.append(baryicentric_coords)
            
            


        # # Combine all points into a single tensor
        # combined_points = torch.cat(all_points, dim=0)

        # # Step 3: Remove Duplicate Points
        # unique_points, _ = torch.unique(combined_points, dim=0, return_inverse=True)

        # TODO
        return None
    
# Custom collation function
def custom_collate_fn(batch):
    # Initialize containers for the aggregated items
    grid_cells = []
    vertices = []
    normals = []
    uv_coords_triangles = []
    grid_index = []

    # Loop through each batch and aggregate its items
    for i, items in enumerate(batch):
        if items is None:
            continue
        grid_cell, vertic, normal, uv_coords_triangle = items
        if grid_cell is None:
            continue
        grid_cells.append(grid_cell)
        vertices.append(vertic)
        normals.append(normal)
        uv_coords_triangles.append(uv_coords_triangle)
        grid_index.append([i]*grid_cell.shape[0])
        
    if len(grid_cells) == 0:
        return None, None, None, None
        
    # Turn the lists into tensors
    grid_cells = torch.stack(grid_cells, dim=0)
    vertices = torch.concat(vertices, dim=0)
    normals = torch.concat(normals, dim=0)
    uv_coords_triangles = torch.concat(uv_coords_triangles, dim=0)
    grid_index = torch.tensor(grid_index, dtype=torch.int32)

    # Return a single batch containing all aggregated items
    return grid_cells, vertices, normals, uv_coords_triangles, grid_index
    
    
def ppm_and_texture(obj_path, grid_cell_path, grid_size=500, gpus=1, batch_size=1):
    grid_cell_template = os.path.join(grid_cell_path, "cell_yxz_{:03}_{:03}_{:03}.tif")
    dataset = MeshDataset(obj_path, grid_cell_template, grid_size=grid_size)
    num_threads = multiprocessing.cpu_count() // int(1.5 * int(gpus))
    num_treads_for_gpus = 5
    num_workers = min(num_threads, num_treads_for_gpus)
    num_workers = max(num_workers, 1)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, num_workers=num_workers, prefetch_factor=3)
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
    parser.add_argument('obj', type=str)
    parser.add_argument('grid_cell', type=str)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    ppm_and_texture(args.obj, gpus=args.gpus, grid_cell_path=args.grid_cell)