### Julian Schilliger - ThaumatoAnakalyptor - 2024

import multiprocessing
import open3d as o3d
import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from .rendering_utils.interpolate_image_3d import extract_from_image_4d
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import tifffile
import cv2
import zarr

class MyPredictionWriter(BasePredictionWriter):
    def __init__(self, save_path, image_size, r):
        super().__init__(write_interval="batch")  # or "epoch" for end of an epoch
        self.save_path = save_path
        self.image_size = image_size
        self.r = r
        self.surface_volume_np = np.zeros((2*r+1, image_size[0], image_size[1]), dtype=np.uint16)
    
    def write_on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, prediction, batch_indices, batch, batch_idx: int, dataloader_idx: int) -> None:
        # print("Writing to Numpy")
        if prediction is None:
            return
        if len(prediction) == 0:
            return

        values, indexes_3d = prediction
        indexes_3d = indexes_3d.cpu().numpy().astype(np.int32)
        values = values.cpu().numpy().astype(np.uint16)
        if indexes_3d.shape[0] == 0:
            return

        # save into surface_volume_np
        self.surface_volume_np[indexes_3d[:, 0], indexes_3d[:, 1], indexes_3d[:, 2]] = values

        # display progress cv2.imshow
        image = (self.surface_volume_np[self.r].astype(np.float32) / 65535)
        image = cv2.resize(image, ((1000 * image.shape[1])//image.shape[1], 1000))
        image = image.T
        image = image[::-1, :]
        cv2.imshow("Surface Volume", image)
        cv2.waitKey(1)

    def write_to_disk(self, flag='jpg'):
        print("Writing prediction to disk")
        # Make folder if it does not exist
        os.makedirs(self.save_path, exist_ok=True)

        if flag == 'tif':
            self.write_tif()
        elif flag == 'jpg':
            self.write_jpg()
        elif flag == 'memmap':
            self.write_memmap()
        elif flag == 'npz':
            self.write_npz()
        elif flag == 'zarr':
            self.write_zarr()
        else:
            print("Invalid flag. Choose between 'tif', 'jpg', 'memmap', 'npz', 'zarr'")
            return
        print("Prediction written to disk")

    def write_tif(self):
        # save to disk each layer as tif
        for i in range(self.surface_volume_np.shape[0]):
            # string 0 padded to len of str(self.surface_volume_np.shape[0])
            i_str = str(i).zfill(len(str(self.surface_volume_np.shape[0])))
            image = self.surface_volume_np[i]
            image = image.T
            image = image[::-1, :]
            tifffile.imsave(os.path.join(self.save_path, f"{i_str}.tif"), image)

    def write_jpg(self, quality=60):  # You can adjust the default quality value as needed
        # save to disk each layer as jpg with specified compression quality
        for i in range(self.surface_volume_np.shape[0]):
            # string 0 padded to the length of str(self.surface_volume_np.shape[0])
            i_str = str(i).zfill(len(str(self.surface_volume_np.shape[0])))
            image = 255 * self.surface_volume_np[i].astype(np.float32) / 65535
            image = image.T
            image = image[::-1, :]
            jpg_filename = os.path.join(self.save_path, f"{i_str}.jpg")
            # Set the compression quality
            cv2.imwrite(jpg_filename, image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    def write_memmap(self):
        # save to disk as memmap
        memmap_path = os.path.join(self.save_path, "surface_volume")
        memmap = np.memmap(memmap_path, dtype='uint16', mode='w+', shape=self.surface_volume_np.shape)
        memmap[:] = self.surface_volume_np[:]
        del memmap

    def write_npz(self):
        # save to disk as npz
        npz_path = os.path.join(self.save_path, "surface_volume.npz")
        np.savez_compressed(npz_path, surface_volume=self.surface_volume_np)

    def write_zarr(self):
        # Define the chunk size
        chunk_size = (16, 16, 16)  # Example: Modify according to your needs

        # Define compression options
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)  # Example compression

        # save to disk as zarr with chunks and compression
        zarr_path = os.path.join(self.save_path, "surface_volume.zarr")
        z = zarr.open(zarr_path, mode='w', shape=self.surface_volume_np.shape, dtype='uint16', chunks=chunk_size, compressor=compressor)
        z[:] = self.surface_volume_np
        
class MeshDataset(Dataset):
    """Dataset class for rendering a mesh."""
    def __init__(self, path, grid_cell_template, grid_size=500, r=32, max_side_triangle=10):
        """Initialize the dataset."""
        self.path = path
        self.grid_cell_template = grid_cell_template
        self.r = r+1
        self.max_side_triangle = max_side_triangle
        self.load_mesh(path)
        write_path = os.path.join(os.path.dirname(path), "layers")
        self.writer = MyPredictionWriter(write_path, self.image_size, r)

        self.grid_size = grid_size
        self.grids_to_process = self.init_grids_to_process()

        self.adjust_triangle_sizes()

    def parse_mtl_for_texture_filenames(self, mtl_filepath):
        texture_filenames = []
        with open(mtl_filepath, 'r') as file:
            for line in file:
                if line.strip().startswith('map_Kd'):  # This line indicates a texture file in MTL
                    parts = line.split()
                    if len(parts) > 1:
                        texture_filenames.append(parts[1])  # The second part is the filename
        return texture_filenames
    
    def generate_mask_png(self):
        mask = np.zeros(self.image_size[::-1], dtype=np.uint8)
        for triangle in self.uv:
            triangle = triangle.astype(np.int32)
            cv2.fillPoly(mask, [triangle], 255)
        mask = mask[::-1, :]
        cv2.imwrite(os.path.join(os.path.dirname(self.path), os.path.basename(self.path) + "_mask.png"), mask)
        
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

        # Generate the mask image for foreground/background separation
        self.generate_mask_png()

        # vertices of triangles
        self.triangles_vertices = self.vertices[self.triangles]
        self.triangles_normals = self.normals[self.triangles]

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

        with tqdm(total=100, desc="Adjusting triangle sizes") as pbar:
            start = None
            while True:
                triangle_min_uv = np.min(uv, axis=1)
                triangle_max_uv = np.max(uv, axis=1)
                side_lengths = triangle_max_uv - triangle_min_uv
                max_side_lengths = np.max(side_lengths, axis=0)

                def current_progress(maxS):
                    return np.log(2) - np.log(self.max_side_triangle / maxS)

                if start is None:
                    # ln_{max_side_triangle}(max_side_lengths)
                    start = current_progress(max_side_lengths[0]) + current_progress(max_side_lengths[1])
                else:
                    now = current_progress(max_side_lengths[0]) + current_progress(max_side_lengths[1])
                    progress = max(1 - now / start, 0)
                    pbar.n = int(progress * 100)
                    pbar.refresh()

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

                mask_uv_x_min = uv_large[:, :, 0] == triangle_min_uv[:, :, 0]
                mask_uv_x_max = uv_large[:, :, 0] == triangle_max_uv[:, :, 0]
                mask_uv_y_min = uv_large[:, :, 1] == triangle_min_uv[:, :, 1]
                mask_uv_y_max = uv_large[:, :, 1] == triangle_max_uv[:, :, 1]

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

                mask_x_min__ = np.logical_and(mask_x_min, mask_x_min_)
                mask_x_max__ = np.logical_and(mask_x_max, mask_x_max_)
                mask_y_min__ = np.logical_and(mask_y_min, mask_y_min_)
                mask_y_max__ = np.logical_and(mask_y_max, mask_y_max_)

                mask_x = np.logical_or(mask_x_min__, mask_x_max__)
                mask_y = np.logical_or(mask_y_min__, mask_y_max__)
                mask_min = np.logical_or(mask_x_min__, mask_y_min__)
                mask_max = np.logical_or(mask_x_max__, mask_y_max__)

                mask = np.logical_or(mask_x, mask_y)

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

            # Show the final progress
            pbar.n = 100
            pbar.refresh()

        self.triangles_vertices = np.concatenate(triangles_vertices_good, axis=0)
        self.triangles_normals = np.concatenate(triangles_normals_good, axis=0)
        self.uv = np.concatenate(uv_good, axis=0)

        print(f"Adjusted triangles: {self.triangles_vertices.shape[0]}, {self.triangles_normals.shape[0]}, {self.uv.shape[0]}", end="\n")
        
    def init_grids_to_process(self):
        triangles_vertices = self.triangles_vertices.reshape(-1, 3)
        grids_to_process = set(map(tuple, (triangles_vertices / self.grid_size).astype(int)))

        mask_changing_vertices = np.floor(((triangles_vertices - self.r) / self.grid_size).astype(int)) != np.floor(((triangles_vertices + self.r) / self.grid_size).astype(int))
        mask_changing_vertices = np.any(mask_changing_vertices, axis=1)
        changing_vertices = triangles_vertices[mask_changing_vertices]

        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    grids_to_process.update(set(map(tuple, np.floor((changing_vertices + np.array([x*self.r, y*self.r, z*self.r])) / self.grid_size).astype(int).reshape(-1, 3))))
        
        grids_to_process = sorted(list(grids_to_process)) # Sort the blocks to process for deterministic behavior
        print(f"Number of grids to process: {len(grids_to_process)}")

        # grids_to_process = grids_to_process[:30] # Debugging

        return grids_to_process
    
    def get_writer(self):
        return self.writer
    
    def extract_triangles_mask(self, grid_index):
        # select all triangles that have at least one vertice in the grid with a r-padded bounding box
        # grid_size * grid_index - r <= vertice <= grid_size * grid_index + r
        selected_triangles_mask = np.any(np.all(np.logical_and(self.triangles_vertices >= np.array(grid_index) * self.grid_size - self.r, self.triangles_vertices <= (np.array(grid_index) + 1) * self.grid_size + self.r), axis=2), axis=1)

        return selected_triangles_mask
    
    def load_grid_cell(self, grid_index, uint8=False):
        grid_index_ = np.asarray(grid_index)[[1, 0, 2]] # swap axis for 'special' grid cell naming ...
        path = self.grid_cell_template.format(grid_index_[0]+1, grid_index_[1]+1, grid_index_[2]+1)

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
        self.new_order = [2,1,0] # [2,1,0], [2,0,1], [0,2,1], [0,1,2], [1,2,0], [1,0,2]
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

    def forward(self, x):
        # grid_coords: T x 3, grid_cell: B x W x W x W, vertices: T x 3 x 3, normals: T x 3 x 3, uv_coords_triangles: T x 3 x 2, grid_index: T
        grid_coords, grid_cells, vertices, normals, uv_coords_triangles, grid_index = x
        
        # Handle the case where the grid cells are empty
        if grid_cells is None:
            return None
                
        # Step 1: Compute AABBs for each triangle
        min_uv, _ = torch.min(uv_coords_triangles, dim=1)
        max_uv, _ = torch.max(uv_coords_triangles, dim=1)
        # Floor and ceil the UV coordinates
        min_uv = torch.floor(min_uv)
        max_uv = torch.ceil(max_uv)

        # Find largest max-min difference for each 2d axis
        max_diff_uv, _ = torch.max(max_uv - min_uv, dim=0)

        # Step 2: Generate Meshgrids for All Triangles
        # create grid points tensor: T x W*H x 2
        grid_points = self.create_grid_points_tensor(min_uv, max_diff_uv[0], max_diff_uv[1])
        del min_uv, max_uv, max_diff_uv

        # Step 3: Compute Barycentric Coordinates for all Triangles grid_points
        # baryicentric_coords: T x W*H x 3, is_inside: T x W*H
        baryicentric_coords, is_inside = self.ppm(grid_points, uv_coords_triangles)
        grid_points = grid_points[is_inside] # S x 2

        # adjust to new_order
        vertices = vertices[:, self.new_order, :]
        normals = normals[:, self.new_order, :]

        # vertices: T x 3 x 3, normals: T x 3 x 3, baryicentric_coords: T x W*H x 3
        coords = torch.einsum('ijk,isj->isk', vertices, baryicentric_coords).squeeze()
        norms = normalize(torch.einsum('ijk,isj->isk', normals, baryicentric_coords).squeeze(),dim=2)
        del vertices, normals, uv_coords_triangles

        # broadcast grid index to T x W*H -> S
        grid_index = grid_index.unsqueeze(-1).expand(-1, baryicentric_coords.shape[1])
        grid_index = grid_index[is_inside]
        # broadcast grid_coords to T x W*H x 3 -> S x 3
        grid_coords = grid_coords.unsqueeze(-2).expand(-1, baryicentric_coords.shape[1], -1)
        coords = coords - grid_coords # Reorient coordinate system origin to 0 for extraction on grid_cells
        del baryicentric_coords, grid_coords

        # coords: S x 3, norms: S x 3
        coords = coords[is_inside]
        norms = norms[is_inside]

        # Poper axis order
        coords = coords[:, self.new_order]
        norms = norms[:, self.new_order]

        # Step 4: Compute the 3D coordinates for every r slice
        r_arange = torch.arange(-self.r, self.r+1, device=coords.device).reshape(1, -1, 1)

        # coords: S x 2*r+1 x 3, grid_index: S x 2*r+1 x 1
        coords = coords.unsqueeze(-2).expand(-1, 2*self.r+1, -1) + r_arange * norms.unsqueeze(-2).expand(-1, 2*self.r+1, -1)
        grid_index = grid_index.unsqueeze(-1).unsqueeze(-1).expand(-1, 2*self.r+1, -1)

        # Expand and add 3rd dimension to grid points
        r_arange = r_arange.expand(grid_points.shape[0], -1, -1) + self.r # [0 to 2*r]
        grid_points = grid_points.unsqueeze(-2).expand(-1, 2*self.r+1, -1)
        grid_points = torch.cat((grid_points, r_arange), dim=-1)
        del r_arange, is_inside

        # Step 5: Filter out the points that are outside the grid_cells
        mask_coords = (coords[:, :, 0] >= 0) & (coords[:, :, 0] < grid_cells.shape[1]) & (coords[:, :, 1] >= 0) & (coords[:, :, 1] < grid_cells.shape[2]) & (coords[:, :, 2] >= 0) & (coords[:, :, 2] < grid_cells.shape[3])

        # coords: S' x 3, norms: S' x 3
        coords = coords[mask_coords]
        grid_points = grid_points[mask_coords] # S' x 2
        grid_index = grid_index[mask_coords] # S'

        # Step 5: Extract the values from the grid cells
        # grid_cells: T x W x H x D, coords: S' x 3, grid_index: S' x 1
        values = extract_from_image_4d(grid_cells, grid_index, coords)
        del coords, grid_cells, mask_coords

        # Step 6: Return the 3D Surface Volume coordinates and the values
        values = values.reshape(-1)
        grid_points = grid_points.reshape(-1, 3) # grid_points: S' x 3
        
        # reorder grid_points
        grid_points = grid_points[:, [2, 0, 1]]

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
    
def ppm_and_texture(obj_path, grid_cell_path, grid_size=500, gpus=1, batch_size=1, r=32, format='jpg'):
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
    writer.write_to_disk(format)
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('obj', type=str)
    parser.add_argument('grid_cell', type=str)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--r', type=int, default=32)
    parser.add_argument('--format', type=str, default='jpg')
    args = parser.parse_known_args()[0]

    ppm_and_texture(args.obj, gpus=args.gpus, grid_cell_path=args.grid_cell, r=args.r, format=args.format)