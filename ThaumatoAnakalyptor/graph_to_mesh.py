import numpy as np
import os
import shutil
import json
import open3d as o3d
import tarfile
import tempfile
from tqdm import tqdm
from scipy.interpolate import interp1d
import hdbscan
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
import concurrent.futures
import time
import multiprocessing
from multiprocessing import cpu_count, shared_memory, Process, Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Custom imports
from .instances_to_sheets import select_points
from .sheet_to_mesh import umbilicus_xz_at_y
from .sheet_computation import load_graph, ScrollGraph
from .slim_uv import Flatboi, print_array_to_file

# import colormap from matplotlib
import matplotlib.pyplot as plt
import matplotlib

import sys
sys.path.append('ThaumatoAnakalyptor/sheet_generation/build')
import pointcloud_processing

from PIL import Image
# This disables the decompression bomb protection in Pillow
Image.MAX_IMAGE_PIXELS = None

def normals_to_skew_symmetric_batch(normals):
    """
    Convert a batch of 4D normal vectors to their corresponding 4x4 skew-symmetric matrices.

    Args:
    normals (numpy.ndarray): An array of 4-dimensional normal vectors, shaped (n, 4).

    Returns:
    numpy.ndarray: A batch of 4x4 skew-symmetric matrices, shaped (n, 4, 4).
    """
    if normals.shape[1] != 3:
        raise ValueError("Each normal vector must be four-dimensional.")

    # Create an array of zeros with shape (n, 4, 4) for the batch of matrices
    n = normals.shape[0]
    skew_matrices = np.zeros((n, 4, 4))

    # Assign values to each matrix in a vectorized manner
    # skew_matrices[:, 0, 1] = -normals[:, 3]  # -w
    skew_matrices[:, 0, 2] = -normals[:, 2]  # -z
    skew_matrices[:, 0, 3] = normals[:, 1]   # y

    # skew_matrices[:, 1, 0] = normals[:, 3]   # w
    skew_matrices[:, 1, 2] = -normals[:, 0]  # -x
    skew_matrices[:, 1, 3] = normals[:, 2]   # z

    skew_matrices[:, 2, 0] = normals[:, 2]   # z
    skew_matrices[:, 2, 1] = normals[:, 0]   # x
    skew_matrices[:, 2, 3] = -normals[:, 1]  # -y

    skew_matrices[:, 3, 0] = -normals[:, 1]  # -y
    skew_matrices[:, 3, 1] = -normals[:, 2]  # -z
    skew_matrices[:, 3, 2] = normals[:, 1]   # y

    return skew_matrices

def scale_points(points, scale=4.0, axis_offset=-500):
    """
    Scale points
    """
    # Scale points
    points_ = points * scale
    # Offset points
    points_ = points_ + axis_offset # WTF SPELUFO? ;)
    # Return points
    return points_

def shuffling_points_axis(points, normals, axis_indices=[2, 0, 1]):
    """
    Rotate points by reshuffling axis
    """
    # Reshuffle axis in points and normals
    points = points[:, axis_indices]
    normals = normals[:, axis_indices]
    # Return points and normals
    return points, normals

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2=np.array([1, 0])):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def distances_points_to_line(points, line_point, line_vector):
    """
    Calculate the shortest distances from multiple points to an infinitely long line in 3D,
    defined by another point on the line and a direction vector.

    Parameters:
    points (numpy.array): The 3D points as an array of shape (n, 3).
    line_point (numpy.array): A point on the line (x, y, z).
    line_vector (numpy.array): The direction vector of the line (dx, dy, dz).

    Returns:
    numpy.array: The shortest distances from each point to the line.
    """
    # Vectors from the point on the line to the points in space
    points_to_line_point = points - line_point
    
    # Cross product of these vectors with the line's direction vector
    cross_prods = np.cross(points_to_line_point, line_vector)
    
    # Norms of the cross products give the areas of the parallelograms, divide by the length of the line_vector to get heights
    distances = np.linalg.norm(cross_prods, axis=1) / np.linalg.norm(line_vector)
    
    return distances

def closest_points_and_distances(points, line_point, line_vector):
    """
    Calculate the shortest distances from multiple points to an infinitely long line in 3D,
    and find the closest points on that line for each point.
    
    Parameters:
    points (numpy.array): The 3D points as an array of shape (n, 3).
    line_point (numpy.array): A point on the line (x, y, z).
    line_vector (numpy.array): The direction vector of the line (dx, dy, dz).
    
    Returns:
    numpy.array: The shortest distances from each point to the line.
    numpy.array: The closest points on the line for each point.
    """
    # Vectors from the point on the line to the points in space
    points_to_line_point = points - line_point
    
    # Project these vectors onto the line vector to find the closest points
    t = np.dot(points_to_line_point, line_vector) / np.dot(line_vector, line_vector)
    closest_points = line_point + np.outer(t, line_vector)
    
    # Distance calculations using the norms of the vectors from points to closest points on the line
    distances = np.linalg.norm(points - closest_points, axis=1)
    
    return distances, closest_points, t

def closest_points_and_distances_cylindrical(points, line_point, line_vector):
    """
    Calculate the shortest distances from multiple points to an infinitely long line in 3D,
    and find the closest points on that line for each point.
    
    Parameters:
    points (numpy.array): The 3D points as an array of shape (n, 3).
    line_point (numpy.array): A point on the line (x, y, z).
    line_vector (numpy.array): The direction vector of the line (dx, dy, dz).
    
    Returns:
    numpy.array: The shortest distances from each point to the line.
    numpy.array: The closest points on the line for each point.
    """
    # Vectors from the point on the line to the points in space
    points_to_line_point = points - line_point
    # Project these vectors onto the line vector to find the closest points
    t_sign = np.sign(np.dot(points_to_line_point, line_vector) / np.dot(line_vector, line_vector))
    
    radius_xy = np.sqrt(points_to_line_point[:,2]**2 + points_to_line_point[:,0]**2)
    # print(f"radius shape: {radius_xy.shape}")
    line_vector_norm = np.linalg.norm(line_vector)
    ts = t_sign * radius_xy / line_vector_norm
    closest_points = line_point + np.outer(ts, line_vector)

    # Distance calculations using the norms of the vectors from points to closest points on the line
    distances = np.linalg.norm(points - closest_points, axis=1)

    return distances, closest_points, ts

def load_ply(ply_file_path):
    """
    Load point cloud data from a .ply file.
    """
    # Check if the .ply file exists
    assert os.path.isfile(ply_file_path), f"File {ply_file_path} not found."

    # Load the .ply file
    pcd = o3d.io.read_point_cloud(ply_file_path)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    colors = np.asarray(pcd.colors)

    return points, normals, colors

def build_patch(winding_angle, subvolume_size, path, sample_ratio=1.0, align_and_flip_normals=False):
    subvolume_size = np.array(subvolume_size)
    res = load_ply(path)
    patch_points = res[0]
    patch_normals = res[1]
    patch_color = res[2]

    # Sample points from picked patch
    patch_points, patch_normals, patch_color, _ = select_points(
        patch_points, patch_normals, patch_color, patch_color, sample_ratio
    )

    # Add winding angle as 4th dimension to points
    patch_points = np.hstack((patch_points, np.ones((patch_points.shape[0], 1)) * winding_angle))
    
    return patch_points, patch_normals, patch_color

# def build_patch_tar(main_sheet_patch, subvolume_size, path, sample_ratio=1.0):
#     """
#     Load surface patch from overlapping subvolumes instances predictions.
#     """

#     # Standardize subvolume_size to a NumPy array
#     subvolume_size = np.atleast_1d(subvolume_size).astype(int)
#     if subvolume_size.shape[0] == 1:
#         subvolume_size = np.repeat(subvolume_size, 3)

#     xyz, patch_nr, winding_angle = main_sheet_patch
#     file = path + f"/{xyz[0]:06}_{xyz[1]:06}_{xyz[2]:06}"
#     tar_filename = f"{file}.tar"

#     if os.path.isfile(tar_filename):
#         with tarfile.open(tar_filename, 'r') as archive, tempfile.TemporaryDirectory() as temp_dir:
#             # Extract all .ply files at once
#             archive.extractall(path=temp_dir, members=archive.getmembers())

#             ply_file = f"surface_{patch_nr}.ply"
#             ply_file_path = os.path.join(temp_dir, ply_file)
#             ids = tuple([*map(int, tar_filename.split(".")[-2].split("/")[-1].split("_"))]+[int(ply_file.split(".")[-2].split("_")[-1])])
#             ids = (int(ids[0]), int(ids[1]), int(ids[2]), int(ids[3]))
#             res = build_patch(winding_angle, tuple(subvolume_size), ply_file_path, sample_ratio=float(sample_ratio))
#             return res

def build_patch_tar(main_sheet_patch, path, sample_ratio=1.0):
    """
    Load surface patch from overlapping subvolumes instances predictions.
    """
    xyz, patch_nr, winding_angle = main_sheet_patch
    file = os.path.join(path, f"{xyz[0]:06}_{xyz[1]:06}_{xyz[2]:06}")
    tar_filename = f"{file}.tar"

    if os.path.isfile(tar_filename):
        with tarfile.open(tar_filename, 'r') as archive:
            ply_file_name = f"surface_{patch_nr}.ply"
            member = archive.getmember(ply_file_name)
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract .ply file to temporary directory
                archive.extract(member, path=temp_dir)
                ply_file_path = os.path.join(temp_dir, member.name)
                # Load the point cloud data from the extracted file
                pcd = o3d.io.read_point_cloud(ply_file_path)
                points = np.asarray(pcd.points)
                normals = np.asarray(pcd.normals)
                colors = np.asarray(pcd.colors)

                # Process loaded data
                patch_points, patch_normals, patch_color, _ = select_points(
                    points, normals, colors, colors, sample_ratio
                )

                # Add winding angle as 4th dimension to points
                patch_points = np.hstack((patch_points, np.ones((patch_points.shape[0], 1)) * winding_angle))
                
                return patch_points, patch_normals, patch_color
    else:
        raise FileNotFoundError(f"File {tar_filename} not found.")
    
def build_patch_tar_cpp(patch_info, path, sample_ratio):
        # Wrapper for C++ function
        return pointcloud_processing.load_pointclouds([patch_info], path)

def create_shared_array(input_array, shape, dtype, name="shared_points"):
    array_size = np.prod(shape) * np.dtype(dtype).itemsize
    try:
        # Create a shared array
        shm = shared_memory.SharedMemory(create=True, size=array_size, name=name)
    except FileExistsError:
        print(f"Shared memory with name {name} already exists.")
        # Clean up the shared memory if it already exists
        shm = shared_memory.SharedMemory(create=False, size=array_size, name=name)
    except Exception as e:
        print(f"Error creating shared memory: {e}")
        raise e

    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    # Fill array with input data
    arr.fill(0)  # Initialize the array with zeros
    arr[:] = input_array[:]
    return arr, shm

def attach_shared_array(shape, dtype, name="shared_points"):
    while True:
        try:
            # Attach to an existing shared array
            shm = shared_memory.SharedMemory(name=name, create=False)
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            assert arr.shape == shape, f"Expected shape {shape} but got {arr.shape}"
            assert arr.dtype == dtype, f"Expected dtype {dtype} but got {arr.dtype}"
            # print("Attached to shared memory")
            # print(f"Memory state: size={shm.size}, name={shm.name}")
            # print(f"Array details: shape={arr.shape}, dtype={arr.dtype}, strides={arr.strides}, offset={arr.__array_interface__['data'][0] - shm.buf.tobytes().find(arr.tobytes())}")

            return arr, shm
        except FileNotFoundError:
            time.sleep(0.2)

def get_subpoints_masks(points, angle_extraction, z_height, z_size, z_padding):
    mask_angle_90 = np.logical_and(points[:,3] >= angle_extraction-90, points[:,3] < angle_extraction+90)
    mask_z_pad = np.logical_and(points[:,1] >= z_height - z_padding, points[:,1] < z_height + z_size + z_padding)
    mask_points = np.logical_and(mask_angle_90, mask_z_pad)

    mask_angle_45 = np.logical_and(points[:,3] >= angle_extraction-45, points[:,3] < angle_extraction+45)
    mask_z_strict = np.logical_and(points[:,1] >= z_height, points[:,1] < z_height + z_size)
    mask_strict_points = np.logical_and(mask_angle_45, mask_z_strict)

    print(f"Processing angle {angle_extraction} with nr indices: {np.sum(mask_points)}")
    if np.sum(mask_points) == 0:
        print(f"No points found for angle {angle_extraction}")
        return None
    
    sub_points = points[mask_points]
    print(f"Extracted subpoints with shape: {sub_points.shape}")
    # extract at z height with padding
    mask_angle_45_subpoints = np.logical_and(sub_points[:,3] >= angle_extraction-45, sub_points[:,3] < angle_extraction+45)
    mask_z_strict_subpoints = np.logical_and(sub_points[:,1] >= z_height, sub_points[:,1] < z_height + z_size)
    mask_strict_subpoints = np.logical_and(mask_angle_45_subpoints, mask_z_strict_subpoints)

    return sub_points, mask_strict_points, mask_strict_subpoints

def worker_clustering(args):
    angle_extraction, z_height, z_size, z_padding, points_shape, dtype, max_single_dist, min_cluster_size = args
    points, points_shm = attach_shared_array(points_shape, dtype, name="shared_points")
    # print(points)
    # print(f"Multithreading first and last angular value: {points[0, 3]}, {points[-1, 3]}")
    mask, mask_shm = attach_shared_array((points_shape[0],), bool, name="shared_mask")

    try:
        if points[-1,3] < angle_extraction-90:
            print(f"Angle {angle_extraction} is not in the range of the pointcloud. Breaking.")
            return False

        res_sm = get_subpoints_masks(points, angle_extraction, z_height, z_size, z_padding)
        if res_sm is None:
            return False
        
        sub_points, mask_strict_points, mask_strict_subpoints = res_sm
        
        partial_mask = filter_points_clustering_half_windings(sub_points[:,:3], max_single_dist, min_cluster_size)

        mask[mask_strict_points] = partial_mask[mask_strict_subpoints]

        print(f"MULTITHREADED: (mask: {np.sum(mask)}), Selected {np.sum(partial_mask[mask_strict_subpoints])} points from {partial_mask.shape[0]} points at angle {angle_extraction} when filtering pointcloud with max_single_dist {max_single_dist} in single linkage agglomerative clustering")
    except Exception as e:
        print(f"Error in worker: {e}. Start {angle_extraction}")
        print(f"Error in worker: {e}. Start {angle_extraction}")

    print(f"Worker {angle_extraction} finished.")
    
    points_shm.close()
    mask_shm.close()
    del points_shm, mask_shm
    print(f"Worker {angle_extraction} exited.")
    return True

def worker_clustering_copy_points(args):
    sub_points, max_single_dist, min_cluster_size = args

    try:        
        partial_mask = filter_points_clustering_half_windings(sub_points[:,:3], max_single_dist, min_cluster_size)
        print(f"MULTITHREADED: Selected {np.sum(partial_mask)} points from {partial_mask.shape[0]} points with linkage agglomerative clustering")
    except Exception as e:
        print(f"Error in worker: {e}.")
        return None

    return partial_mask

def filter_points_clustering_half_windings(points, max_single_dist=20, min_cluster_size=8000):
    """
    Filter points based on the largest connected component.

    Parameters:
    points (np.ndarray): The point cloud data (Nx3).
    normals (np.ndarray): The normals associated with the points (Nx3).
    max_single_dist (float): The maximum single linkage distance to define connectivity.

    Returns:
    np.ndarray: Points and normals belonging to the largest connected component.
    """

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=max_single_dist)
    clusters = clusterer.fit_predict(points)

    # find all clusters of size 10000 or more
    cluster_labels, cluster_counts = np.unique(clusters, return_counts=True)
    large_clusters = cluster_labels[cluster_counts >= min_cluster_size]
    large_clusters = large_clusters[large_clusters != -1]

    # filter points and normals based on the largest clusters
    mask = np.isin(clusters, large_clusters)

    print(f"Selected {np.sum(mask)} points from {len(points)} points when filtering pointcloud with max_single_dist {max_single_dist} in single linkage agglomerative clustering")

    return mask

class WalkToSheet():
    def __init__(self, graph, path):
        self.graph = graph
        self.path = path
        start_point = [3164, 3476, 3472]
        self.save_path = os.path.dirname(path) + f"/{start_point[0]}_{start_point[1]}_{start_point[2]}/" + path.split("/")[-1]
        self.lock = threading.Lock()

    def build_points(self):
        # Building the pointcloud 4D (position) + 3D (Normal) + 3D (Color, randomness) representation of the graph
        points = []
        normals = []
        colors = []

        sheet_infos = []
        for node in tqdm(self.graph.nodes, desc="Building points"):
            winding_angle = self.graph.nodes[node]['winding_angle']
            block, patch_id = node[:3], node[3]
            patch_sheet_patch_info = (block, int(patch_id), winding_angle)
            sheet_infos.append(patch_sheet_patch_info)
        time_start = time.time()
        points, normals, colors = pointcloud_processing.load_pointclouds(sheet_infos, self.path, False)
        print(f"Time to load pointclouds: {time.time() - time_start}")
        print(f"Shape of patch_points: {np.array(points).shape}")

        # print first 5 points
        for i in range(5):
            print(f"Point {i}: {points[i]}")
            print(f"Normal {i}: {normals[i]}")
            print(f"Color {i}: {colors[i]}")

        # Remove indices that are identical in the first three point dimensions
        # unique_indices = np.unique(points[:, :3], axis=0, return_index=True)[1]
        # print(f"Unique indices: {unique_indices.shape[0]}")
        # points = points[unique_indices]
        # normals = normals[unique_indices]
        # colors = colors[unique_indices]

        return points, normals, colors
    
    def build_points_reference(self, graph_path):
        graph = load_graph(graph_path)

        # Building the pointcloud 4D (position) + 3D (Normal) + 3D (Color, randomness) representation of the graph
        points = []
        normals = []
        colors = []

        sheet_infos = []
        for node in tqdm(graph.nodes, desc="Building points"):
            winding_angle = 0.0
            block, patch_id = node[:3], node[3]
            patch_sheet_patch_info = (block, int(patch_id), winding_angle)
            sheet_infos.append(patch_sheet_patch_info)

        time_start = time.time()
        points, normals, colors = pointcloud_processing.load_pointclouds(sheet_infos, self.path)
        print(f"Time to load pointclouds: {time.time() - time_start}")
        print(f"Shape of patch_points: {np.array(points).shape}")

        # print first 5 points
        for i in range(5):
            print(f"Point {i}: {points[i]}")
            print(f"Normal {i}: {normals[i]}")
            print(f"Color {i}: {colors[i]}")

        return points, normals, colors
    
    def filter_points_multiple_occurrences(self, points, normals, colors, spatial_threshold=2.0, angle_threshold=90):
        """
        Filters out points that are spatially close to each other and have significant differences
        in their fourth dimension values.
        
        Parameters:
            points (np.array): Numpy array of shape (n, 4), where each row represents a point.
            normals (np.array): Corresponding normals for each point.
            colors (np.array): Corresponding colors for each point.
            spatial_threshold (float): Distance threshold to consider points as spatially close.
            angle_threshold (float): Minimum difference in the fourth dimension to flag points for deletion.
        
        Returns:
            tuple: Filtered arrays of points, normals, and colors.
        """
        # Create a KD-tree for spatial proximity querying
        tree = cKDTree(points[:, :3])
        # Find pairs of points within the spatial threshold
        pairs = tree.query_pairs(r=spatial_threshold)
        
        # TODO: multithread this loop
        # Identify indices to delete
        delete_indices = set()
        for i, j in tqdm(pairs, desc="Filtering points from distance"):
            # Check if the difference in the fourth dimension exceeds the threshold
            if abs(points[i, 3] - points[j, 3]) > angle_threshold:
                delete_indices.update([i, j])

        # Convert to a sorted list
        delete_indices = sorted(delete_indices)

        # Filter out points, normals, and colors
        keep_indices = np.setdiff1d(np.arange(points.shape[0]), delete_indices)
        filtered_points = points[keep_indices]
        filtered_normals = normals[keep_indices]
        filtered_colors = colors[keep_indices]

        print(f"Deleted {len(delete_indices)} points from {len(points)} (before filtering) points.")
        return filtered_points, filtered_normals, filtered_colors

    def is_sorted(self, arr):
        return np.all(np.diff(arr) >= 0)

    def filter_points_clustering_multithreaded(self, points, normals, colors, max_single_dist=20, min_cluster_size=8000, z_size=400, z_padding=200):
        num_processes = cpu_count()
        # num_processes = min(num_processes, 32)
        print(f"Using {num_processes} processes for filtering points.")
        print(f"Points are sorted: {self.is_sorted(points[:, 3])}")
        shape = points.shape
        dtype = points.dtype

        shared_points, shm_points = create_shared_array(points, shape, dtype, name="shared_points")
        print(f"Shared points are sorted: {self.is_sorted(shared_points[:, 3])}")
        print(f"Shared points equal to points: {np.allclose(shared_points, points)}")
        print(f"Sorted like: {shared_points[0]}, to {shared_points[-1]}")
        mask = np.zeros((shape[0],), bool)
        shared_mask, shm_mask = create_shared_array(mask, mask.shape, bool, name="shared_mask")
        
        processes = []
        min_angle = int(np.floor(np.min(points[:, 3])))
        max_angle = int(np.ceil(np.max(points[:, 3])))
        print(f"Searchsorted middle index: {np.searchsorted(points[:,3], (min_angle + max_angle) / 2)}")
        task_queue = []
        z_min = int(np.floor(np.min(points[:, 1])))
        z_max = int(np.ceil(np.max(points[:, 1])))

        # TODO: add z_min, z_max, z_padding
        for angle_extraction in range(min_angle, max_angle, 90):
            for z_height in range(z_min, z_max, z_size):
                task_queue.append((angle_extraction, z_height))

        # with ThreadPoolExecutor(max_workers=num_processes) as executor:
        #     # Using a dictionary to identify which future is which
        #     futur_list = [executor.submit(worker_clustering, (task, z_height, z_size, z_padding, shape, dtype, max_single_dist, min_cluster_size)) for task, z_height in task_queue]
        #     length = len(futur_list)
        #     count = 0
        #     for future in as_completed(futur_list):
        #         count += 1
        #         try:
        #             result = future.result()
        #         except Exception as exc:
        #             print(f"{future} generated an exception: {exc}")
        #         else:
        #             print(result, f"({count}/{length})")

        # TODO in process get_subpoints_masks
        # with ThreadPoolExecutor(max_workers=num_processes) as executor:
        #     # Using a dictionary to identify which future is which
        #     futur_list = [executor.submit(get_subpoints_masks, (task, z_height, z_size, z_padding, shape, dtype, max_single_dist, min_cluster_size)) for task, z_height in task_queue]
        #     length = len(futur_list)
        #     count = 0
        #     for future in as_completed(futur_list):
        #         count += 1
        #         try:
        #             result = future.result()
        #         except Exception as exc:
        #             print(f"{future} generated an exception: {exc}")
        #         else:
        #             print(result, f"({count}/{length})")

        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            # Using a dictionary to identify which future is which
            futur_dict = {}
            for task, z_height in task_queue:
                res_sub = get_subpoints_masks(points, task, z_height, z_size, z_padding)
                if res_sub is None:
                    continue
                sub_points, mask_strict_points, mask_strict_subpoints = res_sub
                fut = executor.submit(worker_clustering_copy_points, (sub_points, max_single_dist, min_cluster_size))
                futur_dict[fut] = (mask_strict_points, mask_strict_subpoints)
            length = len(futur_dict)
            count = 0
            for future in as_completed(futur_dict):
                count += 1
                try:
                    result = future.result()
                    if result is None:
                        continue
                    mask_strict_points, mask_strict_subpoints = futur_dict[future]
                    shared_mask[mask_strict_points] = result[mask_strict_subpoints]
                except Exception as exc:
                    print(f"{future} generated an exception: {exc}")
                else:
                    print(f"({count}/{length})")

        # for i in range(num_processes):
        #     # self.worker_clustering(task_queue, shape, dtype, max_single_dist, min_cluster_size)
        #     p = Process(target=worker_clustering, args=(i, task_queue, shape, dtype, max_single_dist, min_cluster_size))
        #     processes.append(p)
        #     p.start()

        # for i, p in enumerate(processes):
        #     p.join()
        #     print(f"Process {i} joined.")
        print("All processes joined.")
        print(f"sum of shared mask: {np.sum(shared_mask)}")

        
        selected_points = points[shared_mask]
        selected_normals = normals[shared_mask]
        selected_colors = colors[shared_mask]

        unselected_points = points[~shared_mask]
        unselected_normals = normals[~shared_mask]
        unselected_colors = colors[~shared_mask]

        print(f"ON END FILTERING: Selected {np.sum(shared_mask)} points from {len(points)} points when filtering pointcloud with max_single_dist {max_single_dist} in single linkage agglomerative clustering")
        
        shm_points.close()
        shm_points.unlink()
        shm_mask.close()
        shm_mask.unlink()

        return (selected_points, selected_normals, selected_colors), (unselected_points, unselected_normals, unselected_colors)
    
    def filter_points_clustering(self, points, normals, colors, max_single_dist=20, min_cluster_size=8000):
        winding_angles = points[:, 3]
        min_wind = np.min(winding_angles)
        max_wind = np.max(winding_angles)

        points_list = []
        normals_list = []
        colors_list = []

        # TODO: multithread this loop
        # filter points based on the largest connected component per winding
        for angle_extraction in tqdm(np.arange(min_wind, max_wind, 90), desc="Filtering points per winding"):
            start_index, end_index = self.points_at_winding_angle(points, angle_extraction, max_angle_diff=90)
            points_ = points[start_index:end_index]
            normals_ = normals[start_index:end_index]
            colors_ = colors[start_index:end_index]

            # filter points based on the largest connected component
            mask = self.filter_points_clustering_half_windings(points_[:,:3], max_single_dist=max_single_dist, min_cluster_size=min_cluster_size)

            points_ = points_[mask]
            normals_ = normals_[mask]
            colors_ = colors_[mask]
            # only get core part
            start_index, end_index = self.points_at_winding_angle(points_, angle_extraction, max_angle_diff=45)
            points_ = points_[start_index:end_index]
            normals_ = normals_[start_index:end_index]
            colors_ = colors_[start_index:end_index]

            points_list.append(points_)
            normals_list.append(normals_)
            colors_list.append(colors_)

        points = np.concatenate(points_list, axis=0)
        normals = np.concatenate(normals_list, axis=0)
        colors = np.concatenate(colors_list, axis=0)

        return points, normals, colors

    def points_at_angle(self, points, normals, z_positions, angle, max_eucledian_distance=20):
        angle_360 = (angle + int(1 + angle//360) * 360) % 360
        angle_vector = np.array([np.cos(np.radians(angle_360)), 0.0, -np.sin(np.radians(angle_360))])
        # angle_between_control = angle_between(angle_vector[[0,2]])
        # assert np.isclose((angle + int(1 + angle//360) * 360) % 360, (angle_between_control + int(1 + angle_between_control//360) * 360) % 360), f"Angle {angle} ({(angle + int(1 + angle//360) * 360) % 360}) and angle_between_control {angle_between_control} ({(angle_between_control + int(1 + angle_between_control//360) * 360)}) are not close enough."

        # umbilicus position
        umbilicus_func = lambda z: umbilicus_xz_at_y(self.graph.umbilicus_data, z)

        ordered_pointset_dict = {}
        ordered_pointset_dict_normals = {}
        ordered_pointset_dict_ts = {}        

        # valid_mask_count = 0
        for z in z_positions:
            umbilicus_positions = umbilicus_func(z) # shape is 3
            # extract all points at most max_eucledian_distance from the line defined by the umbilicus_positions and the angle vector
            # distances = distances_points_to_line(points[:, :3], umbilicus_positions, angle_vector)
            distances2, points_on_line, t = closest_points_and_distances_cylindrical(points[:, :3], umbilicus_positions, angle_vector)
            umbilicus_distances = np.linalg.norm(points[:, :3] - umbilicus_positions, axis=1)
            # assert np.allclose(distances, distances2), f"Distance calculation is not correct: {distances} != {distances2}"
            mask = distances2 < max_eucledian_distance
            t_valid_mask = t > 0
            mask = np.logical_and(mask, np.logical_not(t_valid_mask))

            if np.sum(mask) > 0:
                # valid_mask_count += 1
                # cylindrical coordinate system
                # umbilicus_distances = umbilicus_distances[mask] # radiuses from points to umbilicus
                # mean_umbilicus_distance = np.mean(umbilicus_distances)
                # # calculate mean t value for angle vector and mean umbilicus distance
                # mean_t = mean_umbilicus_distance / np.linalg.norm(angle_vector)
                # mean_angle_vector_point = umbilicus_positions - mean_t * angle_vector
                # ordered_pointset_dict[z] = mean_angle_vector_point

                # going a little more archaic at the problem then with cylindrical coordinate system
                close_points = points_on_line[mask][:, :3]
                ordered_pointset_dict[z] = np.mean(close_points, axis=0)

                close_normals = normals[mask]
                mean_normal = np.mean(close_normals, axis=0)
                ordered_pointset_dict_normals[z] = mean_normal / np.linalg.norm(mean_normal)
                close_t = t[mask]
                ordered_pointset_dict_ts[z] = np.mean(close_t)
            else:
                ordered_pointset_dict[z] = None
                ordered_pointset_dict_normals[z] = None
                ordered_pointset_dict_ts[z] = None
        # print(f"Valid mask count: {valid_mask_count} out of {len(z_positions)}")

        # linear interpolation setup
        interpolation_positions = [z for z in z_positions if ordered_pointset_dict[z] is not None]
        if len(interpolation_positions) == 0: # bad state
            return None, None, None, angle_vector
        elif len(interpolation_positions) == 1: # almost as bad of a state
            interpolated_points, interpolated_normals, interpolated_t = np.array([ordered_pointset_dict[interpolation_positions[0]]]*len(z_positions)), np.array([ordered_pointset_dict_normals[interpolation_positions[0]]]*len(z_positions)), np.array([ordered_pointset_dict_ts[interpolation_positions[0]]]*len(z_positions))
            interpolated_points[:, 1] = z_positions
            return interpolated_points, interpolated_normals, interpolated_t, angle_vector
        interpolation_positions = np.array(interpolation_positions)
        interpolation_points = np.array([ordered_pointset_dict[z] for z in z_positions if ordered_pointset_dict[z] is not None])
        interpolation_normals = np.array([ordered_pointset_dict_normals[z] for z in z_positions if ordered_pointset_dict[z] is not None])
        interpolation_ts = np.array([ordered_pointset_dict_ts[z] for z in z_positions if ordered_pointset_dict[z] is not None])
        interpolation_function_y = interp1d(interpolation_positions, interpolation_points[:, 0], kind='linear', fill_value=(interpolation_points[0, 0], interpolation_points[-1, 0]), bounds_error=False)
        interpolation_function_x = interp1d(interpolation_positions, interpolation_points[:, 2], kind='linear', fill_value=(interpolation_points[0, 2], interpolation_points[-1, 2]), bounds_error=False)
        interpolation_function_normals_y = interp1d(interpolation_positions, interpolation_normals[:, 0], kind='linear', fill_value=(interpolation_normals[0, 0], interpolation_normals[-1, 0]), bounds_error=False)
        interpolation_function_normals_z = interp1d(interpolation_positions, interpolation_normals[:, 1], kind='linear', fill_value=(interpolation_normals[0, 1], interpolation_normals[-1, 1]), bounds_error=False)
        interpolation_function_normals_x = interp1d(interpolation_positions, interpolation_normals[:, 2], kind='linear', fill_value=(interpolation_normals[0, 2], interpolation_normals[-1, 2]), bounds_error=False)
        interpolation_function_t = interp1d(interpolation_positions, interpolation_ts, kind='linear', fill_value=(interpolation_ts[0], interpolation_ts[-1]), bounds_error=False)
        
        # generate interpolated points for all z_positions
        interpolated_points_y = interpolation_function_y(z_positions)
        interpolated_points_x = interpolation_function_x(z_positions)
        interpolated_points = np.array([np.array([y, z, x]) for y, z, x in zip(interpolated_points_y, z_positions, interpolated_points_x)])
        interpolated_normals_x = interpolation_function_normals_x(z_positions)
        interpolated_normals_y = interpolation_function_normals_y(z_positions)
        interpolated_normals_z = interpolation_function_normals_z(z_positions)
        interpolated_normals = np.array([np.array([x, y, z]) for x, y, z in zip(interpolated_normals_x, interpolated_normals_y, interpolated_normals_z)])
        interpolated_normals = np.array([normal / np.linalg.norm(normal) for normal in interpolated_normals])
        interpolated_t = interpolation_function_t(z_positions)

        return interpolated_points, interpolated_normals, interpolated_t, angle_vector
    
    def ts_at_angle(self, points, normals, z_positions, angle, max_eucledian_distance=20):
        angle_360 = (angle + int(1 + angle//360) * 360) % 360
        angle_vector = np.array([np.cos(np.radians(angle_360)), 0.0, -np.sin(np.radians(angle_360))])
        # angle_between_control = angle_between(angle_vector[[0,2]])
        # assert np.isclose((angle + int(1 + angle//360) * 360) % 360, (angle_between_control + int(1 + angle_between_control//360) * 360) % 360), f"Angle {angle} ({(angle + int(1 + angle//360) * 360) % 360}) and angle_between_control {angle_between_control} ({(angle_between_control + int(1 + angle_between_control//360) * 360)}) are not close enough."

        # umbilicus position
        umbilicus_func = lambda z: umbilicus_xz_at_y(self.graph.umbilicus_data, z)

        ordered_pointset_dict = {}
        ordered_pointset_dict_normals = {}
        ordered_pointset_dict_ts = {}        

        # valid_mask_count = 0
        for z in z_positions:
            umbilicus_positions = umbilicus_func(z) # shape is 3
            # extract all points at most max_eucledian_distance from the line defined by the umbilicus_positions and the angle vector
            # distances = distances_points_to_line(points[:, :3], umbilicus_positions, angle_vector)
            distances2, points_on_line, t = closest_points_and_distances_cylindrical(points[:, :3], umbilicus_positions, angle_vector)
            umbilicus_distances = np.linalg.norm(points[:, :3] - umbilicus_positions, axis=1)
            # assert np.allclose(distances, distances2), f"Distance calculation is not correct: {distances} != {distances2}"
            mask = distances2 < max_eucledian_distance
            t_valid_mask = t > 0
            mask = np.logical_and(mask, np.logical_not(t_valid_mask))

            if np.sum(mask) > 0:
                # valid_mask_count += 1
                # cylindrical coordinate system
                # umbilicus_distances = umbilicus_distances[mask] # radiuses from points to umbilicus
                # mean_umbilicus_distance = np.mean(umbilicus_distances)
                # # calculate mean t value for angle vector and mean umbilicus distance
                # mean_t = mean_umbilicus_distance / np.linalg.norm(angle_vector)
                # mean_angle_vector_point = umbilicus_positions - mean_t * angle_vector
                # ordered_pointset_dict[z] = mean_angle_vector_point

                # going a little more archaic at the problem then with cylindrical coordinate system
                close_points = points_on_line[mask][:, :3]
                ordered_pointset_dict[z] = np.mean(close_points, axis=0)

                close_normals = normals[mask]
                mean_normal = np.mean(close_normals, axis=0)
                ordered_pointset_dict_normals[z] = mean_normal / np.linalg.norm(mean_normal)
                close_t = t[mask]
                ordered_pointset_dict_ts[z] = np.mean(close_t)
            else:
                ordered_pointset_dict[z] = None
                ordered_pointset_dict_normals[z] = None
                ordered_pointset_dict_ts[z] = None
        # print(f"Valid mask count: {valid_mask_count} out of {len(z_positions)}")

        # linear interpolation setup
        interpolation_positions = [z for z in z_positions if ordered_pointset_dict[z] is not None]
        if len(interpolation_positions) == 0: # bad state
            return None, None, None, angle_vector
        elif len(interpolation_positions) == 1: # almost as bad of a state
            interpolated_points, interpolated_normals, interpolated_t = np.array([ordered_pointset_dict[interpolation_positions[0]]]*len(z_positions)), np.array([ordered_pointset_dict_normals[interpolation_positions[0]]]*len(z_positions)), np.array([ordered_pointset_dict_ts[interpolation_positions[0]]]*len(z_positions))
            interpolated_points[:, 1] = z_positions
            return interpolated_points, interpolated_normals, interpolated_t, angle_vector
        interpolation_positions = np.array(interpolation_positions)
        interpolation_points = np.array([ordered_pointset_dict[z] for z in z_positions if ordered_pointset_dict[z] is not None])
        interpolation_normals = np.array([ordered_pointset_dict_normals[z] for z in z_positions if ordered_pointset_dict[z] is not None])
        interpolation_ts = np.array([ordered_pointset_dict_ts[z] for z in z_positions if ordered_pointset_dict[z] is not None])
        interpolation_function_y = interp1d(interpolation_positions, interpolation_points[:, 0], kind='linear', fill_value=(interpolation_points[0, 0], interpolation_points[-1, 0]), bounds_error=False)
        interpolation_function_x = interp1d(interpolation_positions, interpolation_points[:, 2], kind='linear', fill_value=(interpolation_points[0, 2], interpolation_points[-1, 2]), bounds_error=False)
        interpolation_function_normals_y = interp1d(interpolation_positions, interpolation_normals[:, 0], kind='linear', fill_value=(interpolation_normals[0, 0], interpolation_normals[-1, 0]), bounds_error=False)
        interpolation_function_normals_z = interp1d(interpolation_positions, interpolation_normals[:, 1], kind='linear', fill_value=(interpolation_normals[0, 1], interpolation_normals[-1, 1]), bounds_error=False)
        interpolation_function_normals_x = interp1d(interpolation_positions, interpolation_normals[:, 2], kind='linear', fill_value=(interpolation_normals[0, 2], interpolation_normals[-1, 2]), bounds_error=False)
        interpolation_function_t = interp1d(interpolation_positions, interpolation_ts, kind='linear', fill_value=(interpolation_ts[0], interpolation_ts[-1]), bounds_error=False)
        
        # generate interpolated points for all z_positions
        interpolated_points_y = interpolation_function_y(z_positions)
        interpolated_points_x = interpolation_function_x(z_positions)
        interpolated_points = np.array([np.array([y, z, x]) for y, z, x in zip(interpolated_points_y, z_positions, interpolated_points_x)])
        interpolated_normals_x = interpolation_function_normals_x(z_positions)
        interpolated_normals_y = interpolation_function_normals_y(z_positions)
        interpolated_normals_z = interpolation_function_normals_z(z_positions)
        interpolated_normals = np.array([np.array([x, y, z]) for x, y, z in zip(interpolated_normals_x, interpolated_normals_y, interpolated_normals_z)])
        interpolated_normals = np.array([normal / np.linalg.norm(normal) for normal in interpolated_normals])
        interpolated_t = interpolation_function_t(z_positions)

        return interpolated_points, interpolated_normals, interpolated_t, angle_vector

    def points_at_winding_angle(self, points, winding_angle, max_angle_diff=30):
        """
        Assumes points are sorted by winding angle ascendingly
        """
        # extract all points indices with winding angle within max_angle_diff of winding_angle
        start_index = np.searchsorted(points[:, 3], winding_angle - max_angle_diff, side='left')
        end_index = np.searchsorted(points[:, 3], winding_angle + max_angle_diff, side='right')
        return start_index, end_index

    def rolled_ordered_pointset(self, points, normals, debug=False):
        # umbilicus position
        umbilicus_func = lambda z: umbilicus_xz_at_y(self.graph.umbilicus_data, z)

        # get winding angles
        winding_angles = points[:, 3]

        min_wind = np.min(winding_angles)
        max_wind = np.max(winding_angles)
        print(f"Min and max winding angles: {min_wind}, {max_wind}")

        min_z = np.min(points[:, 1])
        max_z = np.max(points[:, 1])

        z_positions = np.arange(min_z, max_z, 10)

        ordered_pointsets = []

        # test
        test_angle = -5000
        # remove test folder
        test_folder = os.path.join(self.save_path, "test_winding_angles")
        if os.path.exists(test_folder):
            shutil.rmtree(test_folder)
        os.makedirs(test_folder)
        for test_angle in range(int(min_wind), int(max_wind), 360):
            start_index, end_index = self.points_at_winding_angle(points, test_angle, max_angle_diff=180)
            points_test = points[start_index:end_index]
            colors_test = points_test[:,3]
            points_test = points_test[:,:3]
            normals_test = normals[start_index:end_index]
            print(f"extracted {len(points_test)} points at {test_angle} degrees")
            # save as ply
            ordered_pointsets_test = [(points_test, normals_test)]
            test_pointset_ply_path = os.path.join(test_folder, f"ordered_pointset_test_{test_angle}.ply")
            try:
                self.pointcloud_from_ordered_pointset(ordered_pointsets_test, test_pointset_ply_path, color=colors_test)
            except:
                print("Error saving test pointset")

        if debug:
            # save complete pointcloud
            complete_pointset_ply_path = os.path.join(test_folder, "complete_pointset.ply")
            try:
                self.pointcloud_from_ordered_pointset([(points, normals)], complete_pointset_ply_path, color=points[:,3])
            except:
                print("Error saving complete pointset")

        angles_step = 6
        # Array to store results in the order of their winding angles
        num_angles = len(np.arange(min_wind, max_wind, angles_step))
        ordered_pointsets = [None] * num_angles

        # Define the task to be executed by each thread
        def process_winding_angle(winding_angle):
            start_index, end_index = self.points_at_winding_angle(points, winding_angle)
            extracted_points = points[start_index:end_index]
            extracted_normals = normals[start_index:end_index]

            result = self.points_at_angle(extracted_points, extracted_normals, z_positions, winding_angle, max_eucledian_distance=10)
            return result

        # Using ThreadPoolExecutor to manage a pool of threads
        with ThreadPoolExecutor() as executor:
            # List to hold futures
            future_to_index = {executor.submit(process_winding_angle, wa): idx for idx, wa in enumerate(np.arange(min_wind, max_wind, angles_step))}
            
            # Collect results as they complete, respecting submission order
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing Winding Angles"):
                index = future_to_index[future]
                try:
                    result = future.result()
                    ordered_pointsets[index] = result
                except Exception as e:
                    print(f"Error processing index {index}: {e}")

        # # TODO: multithread this loop
        # # iterate through winding angles
        # for winding_angle in tqdm(np.arange(min_wind, max_wind, angles_step), desc="Winding angles"):
        #     start_index, end_index = self.points_at_winding_angle(points, winding_angle)
        #     extracted_points = points[start_index:end_index]
        #     extracted_normals = normals[start_index:end_index]

        #     res = self.points_at_angle(extracted_points, extracted_normals, z_positions, winding_angle, max_eucledian_distance=4)
        #     ordered_pointsets.append(res)

        ordered_pointsets_final = []
        # Interpolate None ordered_pointset from good neighbours, do it by t's on same z values
        indices = [o for o in range(len(ordered_pointsets)) if ordered_pointsets[o] is not None and ordered_pointsets[o][0] is not None]
        for o in range(len(ordered_pointsets)):
            if ordered_pointsets[o][0] is not None: # Good ordered_pointset entry, finalize it
                ordered_pointsets_final.append((ordered_pointsets[o][0], ordered_pointsets[o][1]))
                continue
            ordered_pointset = []
            # interpolate each height level
            angle_vector = ordered_pointsets[o][3]
            for i, z in enumerate(z_positions):
                t_vals = [ordered_pointsets[u][2][i] for u in range(len(ordered_pointsets)) if ordered_pointsets[u] is not None and ordered_pointsets[u][0] is not None]
                if len(t_vals) == 0:
                    # no good neighbours
                    continue
                # interpolate
                t_vals_interpolation = interp1d(indices, t_vals, kind='linear', fill_value=(t_vals[0], t_vals[-1]), bounds_error=False)
                t = t_vals_interpolation(o)
                # 3D position
                umbilicus_point = umbilicus_func(z)
                point_on_line = umbilicus_point + t * angle_vector
                normal = unit_vector(angle_vector)
                ordered_pointset.append((point_on_line, normal))
            ordered_pointsets_final.append((np.array([p[0] for p in ordered_pointset]), np.array([p[1] for p in ordered_pointset])))

        return ordered_pointsets_final
    
    def save_graph_pointcloud(self, graph_path):
        save_path = os.path.join(self.save_path, "graph_pointcloud.ply")
        print(save_path)
        points, normals, colors = self.build_points_reference(graph_path)

        # Scale and translate points to original coordinate system
        points = scale_points(np.array(points[:, :3]))
        # Rotate back to original axis
        points, normals = shuffling_points_axis(points, np.array(normals))
        # open3d
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        pcl.normals = o3d.utility.Vector3dVector(normals)
        o3d.io.write_point_cloud(save_path, pcl)
        
    def pointcloud_from_ordered_pointset(self, ordered_pointset, filename, color=None):
        points, normals = [], []
        for point, normal in tqdm(ordered_pointset, desc="Building pointcloud"):
            if point is not None:
                points.append(point)
                normals.append(normal)
        points = np.concatenate(points, axis=0).astype(np.float64)
        normals = np.concatenate(normals, axis=0).astype(np.float64)

        # Scale and translate points to original coordinate system
        points[:, :3] = scale_points(points[:, :3])
        # Rotate back to original axis
        points, normals = shuffling_points_axis(points, normals)
        # invert normals (layer 0 in the papyrus, layer 64 on top and outside the papyrus, layer 32 surface layer (for 64 layer surface volume))
        normals = -normals

        print(f"Saving {points.shape} points to {filename}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        if not color is None:
            # color is shape n, 1, go to rgb with hsv
            color = (color - np.min(color)) / (np.max(color) - np.min(color))
            # use colormap cold
            cmap = matplotlib.cm.get_cmap('coolwarm')
            color = cmap(color)[:, :3]
            print(color.shape, color.dtype)
            pcd.colors = o3d.utility.Vector3dVector(color)
        # Create folder if it doesn't exist
        print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save as a PLY file
        o3d.io.write_point_cloud(filename, pcd)

    def ordered_pointset_from_pointcloud(self, filename, length_ordered_pointset):
        # Load the point cloud data from the extracted file
        pcd = o3d.io.read_point_cloud(filename)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        axis_indices=[1, 2, 0]
        points, normals = shuffling_points_axis(points, normals, axis_indices=axis_indices)
        points[:, :3] = scale_points(points[:, :3], scale=0.25, axis_offset=125)
        normals = -normals
        # Rotate back to original axis

        len_outside_pointset = len(points) // length_ordered_pointset

        # Process loaded data
        ordered_pointset = []
        for i in range(0, len(points), len_outside_pointset):
            ordered_pointset.append((points[i:i+len_outside_pointset], normals[i:i+len_outside_pointset]))

        return ordered_pointset

    def mesh_from_ordered_pointset(self, ordered_pointsets):
        """
        Creates a mesh from an ordered list of point sets. Each point set is a list of entries, and each entry is a tuple of point and normal.
        Assumes each point set has the same number of entries and they are ordered consistently.

        Parameters:
        ordered_pointsets (list of lists of tuples): The ordered list of pointsets where each pointset contains tuples of (x, y, z) coordinates.

        Returns:
        o3d.geometry.TriangleMesh: The resulting mesh.
        """
        vertices = []
        normals = []
        triangles = []
        tringles_uv = []

        # First, convert all pointsets into a single list of vertices
        vertices, normals = [], []
        for point, normal in ordered_pointsets:
            if point is not None:
                vertices.append(point)
                normals.append(normal)
        # Convert to Open3D compatible format
        vertices = np.concatenate(vertices, axis=0)
        normals = np.concatenate(normals, axis=0)
        # Invert normals
        normals = -normals

        # Scale and translate vertices to original coordinate system
        vertices[:, :3] = scale_points(vertices[:, :3])
        # Rotate back to original axis
        vertices, normals = shuffling_points_axis(vertices, normals)

        
        # Number of rows and columns in the pointset grid
        num_rows = len(ordered_pointsets)
        num_cols = len(ordered_pointsets[0][0]) if num_rows > 0 else 0

        print(f"Number of rows: {num_rows}, number of columns: {num_cols}")

        # Generate triangles
        for i in tqdm(range(num_rows - 1), desc="Generating triangles"):
            for j in range(num_cols - 1):
                # Current point and the points to its right and below it
                idx = i * num_cols + j
                triangles.append([idx, idx + 1, idx + num_cols])
                tringles_uv.append(((i / num_rows, j / num_cols), (i / num_rows, (j + 1) / num_cols), ((i+1) / num_rows, j / num_cols)))
                triangles.append([idx + 1, idx + num_cols + 1, idx + num_cols])
                tringles_uv.append(((i / num_rows, (j+1) / num_cols), ((i + 1) / num_rows, (j + 1) / num_cols), ((i+1) / num_rows, j / num_cols)))

        # Create the mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.triangle_uvs = o3d.utility.Vector2dVector(np.array(tringles_uv).reshape((-1, 2)))
        # compute triangle normals and normals
        mesh = mesh.compute_triangle_normals()
        mesh = mesh.compute_vertex_normals()

        # Create a grayscale image with a specified size
        n, m = int(np.ceil(num_rows)), int(np.ceil(num_cols))  # replace with your dimensions
        uv_image = Image.new('L', (n, m), color=255)  # 255 for white background, 0 for black

        return mesh, uv_image
    
    def mesh_initial_uv(self, mesh, ordered_pointsets):
        # Create a mesh with initial UV coordinates based on the ordered pointsets
        num_vertices = len(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)

        # Create a UV map based on the ordered pointsets
        uv_map = np.zeros((num_vertices, 2), dtype=np.float64)
        for i, (points, _) in enumerate(ordered_pointsets):
            for j, point in enumerate(points):
                # find point in vertices vertices[idx] == point
                idx = np.where(np.all(vertices == point, axis=1))[0][0]
                assert np.all(vertices[idx] == point), f"Point {point} not found in vertices."
                uv_map[idx] = [i / len(ordered_pointsets), j / len(points)]

        uv_triangle_map = np.zeros((len(triangles), 3, 2), dtype=np.float64)
        for i, triangle in enumerate(triangles):
            for j, vertex in enumerate(triangle):
                uv_triangle_map[i, j] = uv_map[vertex]

        # Assign UV coordinates to the mesh
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_triangle_map.reshape(-1, 2))

        min_z = np.min(vertices[:, 2])
        max_z = np.max(vertices[:, 2])
        size_z = max_z - min_z

        # empty image
        # Create a grayscale image with a specified size
        n, m = int(np.ceil(size_z)), int(np.ceil(size_z))  # replace with your dimensions
        uv_image = Image.new('L', (n, m), color=255)  # 255 for white background, 0 for black

        return mesh, uv_image
    
    def save_mesh(self, mesh, uv_image, filename):
        """
        Save a mesh to a file.

        Parameters:
        mesh (o3d.geometry.TriangleMesh): The mesh to save.
        filename (str): The path to save the mesh to.
        """
        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save as a PLY file
        o3d.io.write_triangle_mesh(filename, mesh)

        # Save the UV image
        if uv_image is not None:
            uv_image.save(filename[:-4] + ".png")

    def save_4D_ply(self, points, normals, filename):
        """
        Save a 4D point cloud to a PLY file with skew-symmetric matrix properties.
        
        Parameters:
        points (np.ndarray): The 4D points.
        normals (np.ndarray): The normals.
        filename (str): The path to save the PLY file to.
        """
        skew_matrices = normals_to_skew_symmetric_batch(normals)
        n = points.shape[0]
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('w', 'f4')]
        
        # Append skew-symmetric matrix properties to the dtype
        for c in range(4):
            for r in range(4):
                if c != r:
                    dtype.append((f'skew_{c}_{r}', 'f4'))

        # Create structured numpy array
        structured_array = np.zeros(n, dtype=dtype)

        # Populate structured array with data
        structured_array['x'] = points[:, 0]
        structured_array['y'] = points[:, 1]
        structured_array['z'] = points[:, 2]
        structured_array['w'] = points[:, 3]

        for c in range(4):
            for r in range(4):
                if c != r:
                    structured_array[f'skew_{c}_{r}'] = skew_matrices[:, c, r]

        # Create a PlyElement instance to hold the data
        el = PlyElement.describe(structured_array, 'vertex')

        # Write to a file
        PlyData([el], text=False).write(filename)

    def flatten(self, mesh_path):
        flatboi = Flatboi(mesh_path, 5)
        harmonic_uvs, harmonic_energies = flatboi.slim(initial_condition='ordered')
        flatboi.save_img(harmonic_uvs)
        flatboi.save_obj(harmonic_uvs)
        # Get the directory of the input file
        input_directory = os.path.dirname(mesh_path)
        # Filename for the energies file
        energies_file = os.path.join(input_directory, 'energies_flatboi.txt')
        print_array_to_file(harmonic_energies, energies_file)       
        flatboi.save_mtl()

    def flatten_vc(self, mesh_path):
        # load uv mesh
        uv_mesh_path = mesh_path[:-4] + "_abf.obj"
        uv_mesh = o3d.io.read_triangle_mesh(uv_mesh_path)
        vertices = np.asarray(uv_mesh.vertices)
        uvs = vertices[:, [0, 2]]
        uvs_min_x = np.min(uvs[:, 0])
        uvs_min_y = np.min(uvs[:, 1])

        uvs = uvs - np.array([uvs_min_x, uvs_min_y])

        # load mesh
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # assign uv to mesh
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs[triangles].reshape(-1, 2))

        # save mesh
        vc_mesh_path = mesh_path[:-4] + "_vc.obj"
        
        size_x = np.max(uvs[:, 0])
        size_y = np.max(uvs[:, 1])

        # Create a grayscale image with a specified size
        n, m = int(np.ceil(size_x)), int(np.ceil(size_y))  # replace with your dimensions
        uv_image = Image.new('L', (n, m), color=255)  # 255 for white background, 0 for black

        self.save_mesh(mesh, uv_image, vc_mesh_path)

    def unroll(self, debug=False):
        # get points
        points, normals, colors = self.build_points()

        # Make directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        # Save as npz
        with open(os.path.join(self.save_path, "points.npz"), 'wb') as f:
            np.savez(f, points=points, normals=normals, colors=colors)

        # Open the npz file
        with open(os.path.join(self.save_path, "points.npz"), 'rb') as f:
            npzfile = np.load(f)
            points = npzfile['points']
            normals = npzfile['normals']
            colors = npzfile['colors']

        # # take first debug_nr_points
        # debug_nr_points = 5000000
        # index_start = np.searchsorted(points[:, 3], -5270)
        # index_end = np.searchsorted(points[:, 3], -4780)
        # print(f"Selected {index_end - index_start} points from {points.shape[0]} points.")
        # points = points[index_start:index_end]
        # normals = normals[index_start:index_end]
        # colors = colors[index_start:index_end]

        # Subsample points:
        points_subsampled, normals_subsampled, colors_subsampled, _ = select_points(points, normals, colors, colors, original_ratio=0.1)
        # random points
        # points = np.random.rand(1000, 4)
        # normals = np.random.rand(1000, 3)
        # colors = np.random.rand(1000, 3)

        # filter points
        # points, normals, colors = self.filter_points_multiple_occurrences(points, normals, colors)
        enable_clustering = False
        if enable_clustering:
            (points_selected, normals_selected, colors_selected), (points_unselected, normals_unselected, colors_unselected) = self.filter_points_clustering_multithreaded(points_subsampled, normals_subsampled, colors_subsampled)

            points_subsampled_shape = points_subsampled.shape
            del points_subsampled, normals_subsampled, colors_subsampled, normals_selected, colors_selected, normals_unselected, colors_unselected, colors

            # Extract selected points from original points
            original_selected_mask = pointcloud_processing.upsample_pointclouds(points, points_selected, points_unselected)
            points_originals_selected = points[original_selected_mask]
            normals_oiginals_selected = normals[original_selected_mask]
            print(f"Upsampled {points_originals_selected.shape[0]} points from {points.shape[0]} points. With the help of {points_selected.shape[0]} subsampled points (which got selected out of {points_subsampled_shape[0]} points).")

            del points_selected, points_unselected, points, normals
        else:
            points_originals_selected = points
            normals_oiginals_selected = normals

        # Save as npz
        with open(os.path.join(self.save_path, "points_selected.npz"), 'wb') as f:
            np.savez(f, points=points_originals_selected, normals=normals_oiginals_selected)

        # # Open the npz file
        # with open(os.path.join(self.save_path, "points_selected.npz"), 'rb') as f:
        #     npzfile = np.load(f)
        #     points_originals_selected = npzfile['points']
        #     normals_oiginals_selected = npzfile['normals']

        print(f"Shape of points_originals_selected: {points_originals_selected.shape}")
        # points_originals_selected = points_originals_selected[:1000000]
        # normals_oiginals_selected = normals_oiginals_selected[:1000000]
        # print(f"Shape of points_originals_selected: {points_originals_selected.shape}")

        # self.save_4D_ply(points, normals, os.path.join(self.save_path, "4D_points.ply"))

        pointset_ply_path = os.path.join(self.save_path, "ordered_pointset.ply")
        # get nodes
        ordered_pointsets = self.rolled_ordered_pointset(points_originals_selected, normals_oiginals_selected, debug=debug)

        # print(f"Number of ordered pointsets: {len(ordered_pointsets)}", len(ordered_pointsets[0]))
        self.pointcloud_from_ordered_pointset(ordered_pointsets, pointset_ply_path)

        # ordered_pointsets = self.ordered_pointset_from_pointcloud(pointset_ply_path, 349)

        mesh, uv_image = self.mesh_from_ordered_pointset(ordered_pointsets)

        mesh_path = os.path.join(self.save_path, "mesh.obj")
        self.save_mesh(mesh, uv_image, mesh_path)

        # Flatten mesh
        self.flatten(mesh_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Unroll a graph to a sheet')
    parser.add_argument('--path', type=str, help='Path to the instances', required=True)
    parser.add_argument('--graph', type=str, help='Path to the graph file from --Path', required=True)
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    graph_path = os.path.join(os.path.dirname(args.path), args.graph)
    graph = load_graph(graph_path)
    reference_path = graph_path.replace("evolved_graph", "subgraph")
    walk = WalkToSheet(graph, args.path)
    # walk.save_graph_pointcloud(reference_path)
    walk.unroll(debug=args.debug)