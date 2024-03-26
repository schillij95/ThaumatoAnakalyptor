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
from .rendering_utils.interpolate_image_3d import extract_from_image_4d
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import tifffile


class MyPredictionWriter(BasePredictionWriter):
    def __init__(self, save_path, image_size, r):
        super().__init__(write_interval="batch")  # or "epoch" for end of an epoch
        self.save_path = save_path
        self.image_size = image_size
        self.r = r
        # self.mmap_npz = np.memmap(os.path.join(os.path.dirname(save_path), "prediction.npz"), mode='w+', shape=(0, 3), dtype=np.float32)
        self.surface_volume_np = np.zeros((2*r+1, image_size[0], image_size[1]), dtype=np.float32)
    
    def write_on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, prediction, batch_indices, batch, batch_idx: int, dataloader_idx: int) -> None:
        if prediction is None:
            return
        if len(prediction) == 0:
            return

        values, indexes_3d = prediction
        indexes_3d = indexes_3d.cpu().numpy().astype(np.int32)
        values = values.cpu().numpy().astype(np.float32)
        if len(indexes_3d) == 0:
            return

        # print(f"Writing with shapes {indexes_3d.shape}, {values.shape}", end="\n")
        # print(f"Maximum Coordinates: {np.max(indexes_3d, axis=0)}, Surface volume size: {self.surface_volume_np.shape}", end="\n")
        # save into surface_volume_np
        # for i in range(indexes_3d.shape[0]):
        #     x, y, z = indexes_3d[i]
        #     self.surface_volume_np[z, x, y] = values[i]
        print("now vectorized")
        # vectorized version
        self.surface_volume_np[indexes_3d[:, 2], indexes_3d[:, 0], indexes_3d[:, 1]] = values
        print("Writing prediction to disk")
        # print(f"Sum of values: {np.sum(values)}, sum of surface volume: {np.sum(self.surface_volume_np)}", end="\n")
        # print(f"Sum of values: {np.sum(values)}", end="\n")
        
class MeshDataset(Dataset):
    """Dataset class for rendering a mesh."""
    def __init__(self, path, grid_cell_template, grid_size=500, r=32, max_side_triangle=20):
        """Initialize the dataset."""
        self.path = path
        self.grid_cell_template = grid_cell_template
        self.r = r
        self.max_side_triangle = max_side_triangle
        self.load_mesh(path)
        self.writer = MyPredictionWriter(os.path.join(path, "layers"), self.image_size, r)

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
        self.image_size = (y_size, x_size)

        # vertices of triangles
        self.triangles_vertices = self.vertices[self.triangles]
        self.triangles_normals = self.normals[self.triangles]

        self.adjust_triangle_sizes()

    def adjust_triangle_sizes(self):
        triangles_vertices = self.triangles_vertices
        triangles_normals = self.triangles_normals
        uv = self.uv

        print(f"Original triangles: {triangles_vertices.shape[0]}, {triangles_normals.shape[0]}, {uv.shape[0]}", end="\n")
        # split the triangles that are too big as per the max_side_triangle parameter
        # compute the size of each triangle in x and y
        # side lengths for all triangles: T x 3
        uv_good = []
        triangles_vertices_good = []
        triangles_normals_good = []

        while True:
            triangle_min_uv = np.min(uv, axis=1)
            triangle_max_uv = np.max(uv, axis=1)
            side_lengths = triangle_max_uv - triangle_min_uv
            print(f"Side lengths: {side_lengths.shape}", end="\n")
            mask_large_side = np.any(side_lengths > self.max_side_triangle, axis=1)
            # if no triangle is too large, we are done
            if not np.any(mask_large_side):
                break

            uv_good_ = uv[np.logical_not(mask_large_side)]
            triangles_vertices_good_ = triangles_vertices[np.logical_not(mask_large_side)]
            triangles_normals_good_ = triangles_normals[np.logical_not(mask_large_side)]

            # Hold on to the triangles that are good
            uv_good.append(uv_good_)
            triangles_vertices_good.append(triangles_vertices_good_)
            triangles_normals_good.append(triangles_normals_good_)

            uv_large = uv[mask_large_side]
            side_lengths = side_lengths[mask_large_side]
            triangle_min_uv = np.expand_dims(triangle_min_uv[mask_large_side], axis=1)
            triangle_max_uv = np.expand_dims(triangle_max_uv[mask_large_side], axis=1)
            triangles_vertices_large = triangles_vertices[mask_large_side]
            triangles_normals_large = triangles_normals[mask_large_side]

            mask_larger_side_x = side_lengths[:, 0] >= side_lengths[:, 1]
            mask_larger_side_y = np.logical_not(mask_larger_side_x)

            assert np.sum(mask_larger_side_x) + np.sum(mask_larger_side_y) == side_lengths.shape[0], "All triangles should be classified"

            mask_uv_x_min = uv_large[:, :, 0] == triangle_min_uv[:, :, 0]
            mask_uv_x_max = uv_large[:, :, 0] == triangle_max_uv[:, :, 0]
            mask_uv_y_min = uv_large[:, :, 1] == triangle_min_uv[:, :, 1]
            mask_uv_y_max = uv_large[:, :, 1] == triangle_max_uv[:, :, 1]

            assert np.all(np.sum(mask_uv_x_min, axis=1) >= 1), "At least one vertex should be selected per triangle"
            assert np.all(np.sum(mask_uv_x_max, axis=1) >= 1), "At least one vertex should be selected per triangle"
            assert np.all(np.sum(mask_uv_y_min, axis=1) >= 1), "At least one vertex should be selected per triangle"
            assert np.all(np.sum(mask_uv_y_max, axis=1) >= 1), "At least one vertex should be selected per triangle"

            mask_x_min = np.logical_and(mask_larger_side_x[:, None], mask_uv_x_min)
            mask_x_max = np.logical_and(mask_larger_side_x[:, None], mask_uv_x_max)
            mask_y_min = np.logical_and(mask_larger_side_y[:, None], mask_uv_y_min)
            mask_y_max = np.logical_and(mask_larger_side_y[:, None], mask_uv_y_max)

            # maximum one true value per triangle
            # Identify the first vertex that meets the condition in each triangle
            idx_x_min = np.argmax(mask_x_min, axis=1)
            idx_x_max = np.argmax(mask_x_max, axis=1)
            idx_y_min = np.argmax(mask_y_min, axis=1)
            idx_y_max = np.argmax(mask_y_max, axis=1)

            ix = np.arange(mask_x_min.shape[0])

            mask_x_min_ = np.zeros_like(mask_x_min)
            mask_x_max_ = np.zeros_like(mask_x_max)
            mask_y_min_ = np.zeros_like(mask_y_min)
            mask_y_max_ = np.zeros_like(mask_y_max)

            mask_x_min_[ix, idx_x_min] = True
            mask_x_max_[ix, idx_x_max] = True
            mask_y_min_[ix, idx_y_min] = True
            mask_y_max_[ix, idx_y_max] = True

            assert np.all(np.sum(mask_x_min_, axis=1) >= 1), "Exactly one vertex should be selected per triangle"
            assert np.all(np.sum(mask_x_max_, axis=1) >= 1), "Exactly one vertex should be selected per triangle"
            assert np.all(np.sum(mask_y_min_, axis=1) >= 1), "Exactly one vertex should be selected per triangle"
            assert np.all(np.sum(mask_y_max_, axis=1) >= 1), "Exactly one vertex should be selected per triangle"

            mask_x_min__ = np.logical_and(mask_x_min, mask_x_min_)
            mask_x_max__ = np.logical_and(mask_x_max, mask_x_max_)
            mask_y_min__ = np.logical_and(mask_y_min, mask_y_min_)
            mask_y_max__ = np.logical_and(mask_y_max, mask_y_max_)

            assert np.all(np.sum(mask_x_min__, axis=1) <= 1), "At most one vertex should be selected per triangle"
            assert np.all(np.sum(mask_x_max__, axis=1) <= 1), "At most one vertex should be selected per triangle"
            assert np.all(np.sum(mask_y_min__, axis=1) <= 1), "At most one vertex should be selected per triangle"
            assert np.all(np.sum(mask_y_max__, axis=1) <= 1), "At most one vertex should be selected per triangle"

            mask_x = np.logical_or(mask_x_min__, mask_x_max__)
            mask_y = np.logical_or(mask_y_min__, mask_y_max__)
            mask_min = np.logical_or(mask_x_min__, mask_y_min__)
            mask_max = np.logical_or(mask_x_max__, mask_y_max__)

            mask = np.logical_or(mask_x, mask_y)

            assert np.all(np.sum(mask, axis=1) == 2), "Exactly two vertices should be selected per triangle"

            # Create new vertices and normals and uvs
            new_vertices = (triangles_vertices_large[mask_min] + triangles_vertices_large[mask_max]) / 2
            new_normals = (triangles_normals_large[mask_min] + triangles_normals_large[mask_max]) / 2
            new_uv = (uv_large[mask_min] + uv_large[mask_max]) / 2

            new_triangles_vertices_0 = np.copy(triangles_vertices_large)
            new_triangles_vertices_0[mask_min] = new_vertices
            new_triangles_normals_0 = np.copy(triangles_normals_large)
            new_triangles_normals_0[mask_min] = new_normals

            new_triangles_vertices_1 = np.copy(triangles_vertices_large)
            new_triangles_vertices_1[mask_max] = new_vertices
            new_triangles_normals_1 = np.copy(triangles_normals_large)
            new_triangles_normals_1[mask_max] = new_normals

            new_uv_0 = np.copy(uv_large)
            new_uv_0[mask_min] = new_uv
            new_uv_1 = np.copy(uv_large)
            new_uv_1[mask_max] = new_uv

            # Set up for the next iteration
            triangles_vertices = np.concatenate((new_triangles_vertices_0, new_triangles_vertices_1), axis=0)
            triangles_normals = np.concatenate((new_triangles_normals_0, new_triangles_normals_1), axis=0)
            uv = np.concatenate((new_uv_0, new_uv_1), axis=0)

        self.triangles_vertices = np.concatenate(triangles_vertices_good, axis=0)
        self.triangles_normals = np.concatenate(triangles_normals_good, axis=0)
        self.uv = np.concatenate(uv_good, axis=0)

        print(f"Adjusted triangles: {self.triangles_vertices.shape[0]}, {self.triangles_normals.shape[0]}, {self.uv.shape[0]}", end="\n")
        
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
        vertices = self.triangles_vertices[triangles_mask]
        normals = self.triangles_normals[triangles_mask]
        # UV of vertices in triangles that have at least one vertice in the grid with a r-padded bounding box
        uv = self.uv[triangles_mask]

        # load grid cell from disk
        grid_cell = self.load_grid_cell(grid_index)
        if grid_cell is None:
            return None, None, None, None, None
        grid_cell = grid_cell.astype(np.float32)

        # Convert NumPy arrays to PyTorch tensors
        vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
        normals_tensor = torch.tensor(normals, dtype=torch.float32)
        uv_tensor = torch.tensor(uv, dtype=torch.float32)
        grid_cell_tensor = torch.tensor(grid_cell, dtype=torch.float32)
        grid_coord = torch.tensor(np.array(grid_index) * self.grid_size, dtype=torch.int32)

        return grid_coord, grid_cell_tensor, vertices_tensor, normals_tensor, uv_tensor

class PPMAndTextureModel(pl.LightningModule):
    def __init__(self, r=32):
        print("instantiating model")
        self.r = r
        super().__init__()

    def ppm(self, pts, tri):
        # pts T x W*H x 2
        # tri_pts T x 3 x 2
        # triangles 3
        v0 = tri[:, 2, :].unsqueeze(1) - tri[:, 0, :].unsqueeze(1)
        v1 = tri[:, 1, :].unsqueeze(1) - tri[:, 0, :].unsqueeze(1)
        v2 = pts - tri[:, 0, :].unsqueeze(1)

        dot00 = v0.pow(2).sum(dim=2)
        dot01 = (v0 * v1).sum(dim=2)
        dot11 = v1.pow(2).sum(dim=2)
        dot02 = (v2 * v0).sum(dim=2)
        dot12 = (v2 * v1).sum(dim=2)

        invDenom = 1 / (dot00 * dot11 - dot01.pow(2))
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        is_inside = (u >= 0) & (v >= 0) & ((u + v) <= 1 )

        w = 1 - u - v

        bary_coords = torch.stack([u, v, w], dim=2)
        bary_coords = normalize(bary_coords, p=1, dim=2)

        return bary_coords, is_inside
    
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

    def extract_from_image_4d(self, image, grid_index, coordinates):
        # image: T x W x H x D, coordinates: S x 4
        values = extract_from_image_4d(image, grid_index, coordinates)
        return values

    def forward(self, x):
        new_order = [2,1,0]
        # grid_cell: B x W x W x W, vertices: T x 3 x 3, normals: T x 3 x 3, uv_coords_triangles: T x 3 x 2, grid_index: T
        grid_coords, grid_cells, vertices, normals, uv_coords_triangles, grid_index = x
        
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
        # create grid points tensor: T x W*H x 2
        grid_points = self.create_grid_points_tensor(min_uv, max_diff_uv[0], max_diff_uv[1])
        del min_uv, max_uv, max_diff_uv

        # Step 3: Compute Barycentric Coordinates for all Triangles
        baryicentric_coords, is_inside = self.ppm(grid_points, uv_coords_triangles)
        # baryicentric_coords: T x W*H x 3, is_inside: T x W*H
        grid_points = grid_points[is_inside] # S x 2

        # vertices: T x 3 x 3, normals: T x 3 x 3, baryicentric_coords: T x W*H x 3
        coords = torch.einsum('ijk,isj->isk', vertices, baryicentric_coords).squeeze()
        norms = normalize(torch.einsum('ijk,isj->isk', normals, baryicentric_coords).squeeze(),dim=1)
        del vertices, normals, uv_coords_triangles
        

        # broadcast grid index to T x W*H -> S
        grid_index = grid_index.unsqueeze(-1).expand(-1, baryicentric_coords.shape[1])
        grid_index = grid_index[is_inside]
        # broadcast grid_coords to T x W*H x 3 -> S x 3
        grid_coords = grid_coords.unsqueeze(-2).expand(-1, baryicentric_coords.shape[1], -1)
        grid_coords = grid_coords[is_inside]

        del baryicentric_coords
        # coords: S x 3, norms: S x 3
        coords = coords[is_inside]
        norms = norms[is_inside]

        grid_coords_end = grid_coords + torch.tensor(grid_cells.shape[1:4], device=grid_coords.device).unsqueeze(0).expand(grid_coords.shape[0], -1)

        # Step 4: Filter out the points that are outside the grid_cells
        mask_coords = (coords[:, 0] >= grid_coords[:, 0]) & (coords[:, 0] < grid_coords_end[:, 0]) & (coords[:, 1] >= grid_coords[:, 1]) & (coords[:, 1] < grid_coords_end[:, 1]) & (coords[:, 2] >= grid_coords[:, 2]) & (coords[:, 2] < grid_coords_end[:, 2])

        # coords: S' x 3, norms: S' x 3
        coords = coords[mask_coords]
        norms = norms[mask_coords]
        grid_points = grid_points[mask_coords] # S' x 2
        grid_index = grid_index[mask_coords] # S'
        grid_coords = grid_coords[mask_coords] # S' x 3

        # Step 5: Compute the 3D coordinates for every r
        coords = coords - grid_coords
        r_arange = torch.arange(-self.r, self.r+1, device=coords.device).reshape(1, -1, 1)

        # coords_r: S x 2*r+1 x 3, grid_points: S x 2 -> S x 3
        coords = coords.unsqueeze(-2).expand(-1, 2*self.r+1, -1) + r_arange * norms.unsqueeze(-2).expand(-1, 2*self.r+1, -1)
        # Combine Coordinates and Grid Index into S' x 4
        grid_index = grid_index.unsqueeze(-1).unsqueeze(-1).expand(-1, 2*self.r+1, -1)

        # Expand and add 3rd dimension to grid points
        grid_points = grid_points.unsqueeze(-2).expand(-1, 2*self.r+1, -1)
        r_arange = self.r+r_arange.expand(grid_points.shape[0], -1, -1)
        grid_points = torch.cat((grid_points, r_arange), dim=-1)
        print(f"Coords: {coords.shape}, Grid Points: {grid_points.shape}", end="\n")
        del norms, r_arange, is_inside

        # Step 5: Extract the values from the grid cells
        values = self.extract_from_image_4d(grid_cells, grid_index, coords)
        del coords, grid_cells

        # Step 6: Return the 3D Surface Volume coordinates and the values
        values = values.reshape(-1)
        grid_points = grid_points.reshape(-1, 3)

        # Return the 3D Surface Volume coordinates and the values
        return values, grid_points
    
# Custom collation function
def custom_collate_fn(batch):
    # Initialize containers for the aggregated items
    grid_cells = []
    vertices = []
    normals = []
    uv_coords_triangles = []
    grid_index = []
    grid_coords = []

    # Loop through each batch and aggregate its items
    for i, items in enumerate(batch):
        if items is None:
            continue
        grid_coord, grid_cell, vertice, normal, uv_coords_triangle = items
        if grid_cell is None:
            continue
        grid_cells.append(grid_cell)
        vertices.append(vertice)
        normals.append(normal)
        uv_coords_triangles.append(uv_coords_triangle)
        grid_index.extend([i]*vertice.shape[0])
        grid_coord = grid_coord.unsqueeze(0).expand(vertice.shape[0], -1)
        grid_coords.extend(grid_coord)
        
    if len(grid_cells) == 0:
        return None, None, None, None, None, None
        
    # Turn the lists into tensors
    grid_cells = torch.stack(grid_cells, dim=0)
    vertices = torch.concat(vertices, dim=0)
    normals = torch.concat(normals, dim=0)
    uv_coords_triangles = torch.concat(uv_coords_triangles, dim=0)
    grid_index = torch.tensor(grid_index, dtype=torch.int32)
    grid_coords = torch.stack(grid_coords, dim=0)

    # Return a single batch containing all aggregated items
    return grid_coords, grid_cells, vertices, normals, uv_coords_triangles, grid_index
    
    
def ppm_and_texture(obj_path, grid_cell_path, grid_size=500, gpus=1, batch_size=1, r=32):
    grid_cell_template = os.path.join(grid_cell_path, "cell_yxz_{:03}_{:03}_{:03}.tif")
    dataset = MeshDataset(obj_path, grid_cell_template, grid_size=grid_size, r=r)
    num_threads = multiprocessing.cpu_count() // int(1.5 * int(gpus))
    num_treads_for_gpus = 12
    num_workers = min(num_threads, num_treads_for_gpus)
    num_workers = max(num_workers, 1)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, num_workers=num_workers, prefetch_factor=2)
    model = PPMAndTextureModel(r)
    
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
    parser.add_argument('--r', type=int, default=32)
    args = parser.parse_args()

    ppm_and_texture(args.obj, gpus=args.gpus, grid_cell_path=args.grid_cell, r=args.r)