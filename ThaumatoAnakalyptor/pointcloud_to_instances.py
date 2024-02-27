### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import numpy as np
import os
import open3d as o3d
import tarfile

# surface points extraction
import torch
# show cuda devices
print(torch.cuda.device_count())
print(torch.cuda.current_device())
# show name of current device
print(torch.cuda.get_device_name(torch.cuda.current_device()))

import multiprocessing
from multiprocessing import Pool

# plotting
from tqdm import tqdm

from .mask3d.inference import batch_inference, to_surfaces, init

import json
import argparse

from .surface_fitting_utilities import get_vector_mean, rotation_matrix_to_align_z_with_v, optimize_sheet
from .grid_to_pointcloud import load_xyz_from_file, umbilicus, umbilicus_xz_at_y, fix_umbilicus_recompute

def load_ply(filename, main_drive="", alternative_drives=[]):
    """
    Load point cloud data from a .ply file.
    """
    # Check that the file exists
    i = 0
    filename_temp = filename
    while i < len(alternative_drives) and not os.path.isfile(filename_temp):
        filename_temp = filename.replace(main_drive, alternative_drives[i])
        i += 1
    filename = filename_temp
    assert os.path.isfile(filename), f"File {filename} not found."

    # Load the file and extract the points and normals
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    # Extract additional features
    colors = np.asarray(pcd.colors)

    return points, normals, colors

def save_surface_ply(surface_points, normals, colors, score, distance, coeff, n, filename):
    # Create an Open3D point cloud object and populate it
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_points.astype(np.float32))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float16))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float16))

    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save as a PLY file
    o3d.io.write_point_cloud(filename, pcd)
    
    # Save metadata as a JSON
    metadata = {
        'score': float(score),
        'distance': float(distance),
        'coeff': [float(item) for item in coeff.tolist()], # Convert numpy array to list for JSON serialization
        'n': int(n),
    }
    
    # Construct metadata filename
    base_dir, base_filename = os.path.split(filename)
    base_filename_without_extension = os.path.splitext(base_filename)[0]
    metadata_filename = os.path.join(base_dir, f"metadata_{base_filename_without_extension}.json")
    
    with open(metadata_filename, 'w') as metafile:
        json.dump(metadata, metafile)

def save_block_ply_args(args):
    save_block_ply(*args)

def save_block_ply(block_points, block_normals, block_colors, block_scores, block_name, score_threshold=0.5, distance_threshold=10.0, n=4, alpha = 1000.0, slope_alpha = 0.1, post_process=True, block_distances_precomputed=None, block_coeffs_precomputed=None, check_exist=True):
    # Check if tar file exists
    if check_exist and os.path.exists(block_name + '.tar'):
        return

    # Save to a temporary file first to ensure data integrity
    temp_block_name = block_name + "_temp"

    # Check if folder exists
    if not os.path.exists(block_name):
        # post-process the surfaces
        if post_process:
            block_points, block_normals, block_colors, block_scores, block_distances, block_coeffs = post_process_surfaces(block_points, block_normals, block_colors, block_scores, score_threshold=score_threshold, distance_threshold=distance_threshold, n=n, alpha=alpha, slope_alpha=slope_alpha)
        else:
            assert (block_distances_precomputed is not None) and (block_coeffs_precomputed is not None), "block_distances_precomputed and block_coeffs_precomputed must be provided if post_process=False"
            block_distances = block_distances_precomputed
            block_coeffs = block_coeffs_precomputed

        if sum([len(block) for block in block_points]) < 10:
            return
        
        # Create folder if it doesn't exist
        os.makedirs(os.path.dirname(temp_block_name), exist_ok=True)

        # Clean folder (if it exists)
        if os.path.exists(block_name):
            for file in os.listdir(block_name):
                os.remove(os.path.join(block_name, file))

        if os.path.exists(temp_block_name):
            for file in os.listdir(temp_block_name):
                os.remove(os.path.join(temp_block_name, file))

        # Save each surface instance
        for i in range(len(block_points)):
            if len(block_points[i]) < 10:
                continue
            save_surface_ply(block_points[i], block_normals[i], block_colors[i], block_scores[i], block_distances[i], block_coeffs[i], n, os.path.join(temp_block_name, f"surface_{i}.ply"))

    # Delete the temporary tar file if it exists
    if os.path.exists(temp_block_name + '.tar'):
        os.remove(temp_block_name + '.tar')

    used_name_block = block_name if os.path.exists(block_name) else temp_block_name

    # Tar the temp folder without including the 'temp' name inside the tar
    with tarfile.open(temp_block_name + '.tar', 'w') as tar:
        for root, _, files in os.walk(used_name_block):
            for file in files:
                full_file_path = os.path.join(root, file)
                arcname = full_file_path[len(used_name_block) + 1:]
                tar.add(full_file_path, arcname=arcname)

    # Remove the temp folder
    for root, dirs, files in os.walk(used_name_block, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(used_name_block)

    # Rename the tarball to the original filename (without '_temp')
    try:
        os.rename(temp_block_name + '.tar', block_name + '.tar')
        # print(f"Saved {block_name}.tar")
    except Exception as e:
        print(e)
        print(f"Error renaming {temp_block_name}.tar to {block_name}.tar")

def post_process_surfaces(surfaces, surfaces_normals, surfaces_colors, scores, score_threshold=0.5, distance_threshold=10.0, n=4, alpha = 1000.0, slope_alpha = 0.1):
    indices = [] # valid surfaces
    coeff_list = [] # coefficients of surfaces
    distances_list = [] # distances of surfaces
    for i in range(len(surfaces)): # loop over all surface of block
        if len(surfaces[i]) < 10: # too small surface
            continue
        if scores[i] < score_threshold: # low score
            continue

        # Calculate the normal vector of the surface
        v = get_vector_mean(surfaces_normals[i])
        R = rotation_matrix_to_align_z_with_v(v) # rotation matrix to align z-axis with normal vector
        coeff, _, points_mask, sheet_distance = optimize_sheet(surfaces[i], R, n, max_iters=2, alpha=alpha, slope_alpha=slope_alpha) # fit sheet to points
        if sheet_distance > distance_threshold: # sheet is too far away from lots of points
            continue
        indices.append(i) # take valid surfaces
        coeff_list.append(coeff) # save coefficients
        distances_list.append(sheet_distance) # save distances
        # mask sheet
        surfaces[i] = surfaces[i][points_mask]
        surfaces_normals[i] = surfaces_normals[i][points_mask]
        surfaces_colors[i] = surfaces_colors[i][points_mask]
    # Select valid surfaces
    surfaces = [surfaces[i] for i in indices]
    surfaces_normals = [surfaces_normals[i] for i in indices]
    surfaces_colors = [surfaces_colors[i] for i in indices]
    scores = [scores[i] for i in indices]

    return surfaces, surfaces_normals, surfaces_colors, scores, distances_list, coeff_list

def align_and_flip(a, b):
    """
    Modify the vectors a in-place: flip any a_i if its direction better aligns with vector b when flipped.
    """
    dot_product = np.sum(a * b, axis=-1)
    flip_mask = dot_product < 0
    a[flip_mask] = -a[flip_mask]
    return a

def normalize_volume_scale(points, grid_block_size=200):
    # Normalize volume to size 50x50x50
    points = 50.0 * points / grid_block_size
    return points

def load_single_ply(ply_file, grid_block_size, main_drive="", alternative_drives=[]):
    try:
        # Load volume
        points_, normals_, colors_ = load_ply(ply_file, main_drive, alternative_drives)
        # normalize size
        points_ = normalize_volume_scale(points_, grid_block_size=grid_block_size)
        return points_, normals_, colors_
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

def load_plys(src_folder, main_drive, alternative_drives, start, size, grid_block_size=200, num_processes=3, load_multithreaded=True):
    path_template = "cell_yxz_{:03}_{:03}_{:03}.ply"
    ply_files = []
    for x in range(start[0], start[0]+size[0]):
        for y in range(start[1], start[1]+size[1]):
            for z in range(start[2], start[2]+size[2]):
                ply_files.append(os.path.join(src_folder, path_template.format(x,y,z)))

    if load_multithreaded:
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(load_single_ply, [(ply_file, grid_block_size, main_drive, alternative_drives) for ply_file in ply_files])
    else:
        results = [load_single_ply(ply_file, grid_block_size, main_drive, alternative_drives) for ply_file in ply_files]

    # Filter out None results
    results = [res for res in results if res is not None]

    if len(results) == 0:
        return None

    # Unzip the results to get points, normals, and colors lists
    points, normals, colors = zip(*results)

    points = np.concatenate(points, axis=0)
    normals = np.concatenate(normals, axis=0)
    colors = np.concatenate(colors, axis=0)

    # Randomly shuffle the points (for data looking more like the training data during instance prediction with mask3d)
    indices = np.arange(points.shape[0])
    np.random.shuffle(indices)
    points = points[indices]
    normals = normals[indices]
    colors = colors[indices]

    return points, normals, colors

def extract_subvolume(points, normals, colors, angles, start, size=50):
    """
    Extract a subvolume of size "size" from "points" starting at "start".
    """
    # Size is int
    if isinstance(size, int):
        size = np.array([size, size, size])

    # Ensure start is in shape (a, 3)
    if start.shape == (3,):
        start = np.expand_dims(start, axis=0)

    # Ensure size is in shape (a, 3)
    if size.shape == (3,):
        size = np.expand_dims(size, axis=0)

    # Remove entries that have np.any(size[a] == 0)
    mask = np.all(size > 0, axis=1)
    start = start[mask]
    size = size[mask]

    # Check if entries exist after removing entries
    if start.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))

    # any dimension of size is 0
    if np.any(size <= 0):
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    
    # Find all points in the subvolume
    mask = np.zeros(points.shape[0], dtype=bool)
    for i in range(start.shape[0]):
        mask_i = np.all(np.logical_and(points >= start[i], points < start[i] + size[i]), axis=1)
        mask = np.logical_or(mask, mask_i)


    # No points in the subvolume
    if np.all(~mask):
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    
    subvolume_points = points[mask]
    subvolume_normals = normals[mask]
    subvolume_colors = colors[mask]
    subvolume_angles = angles[mask]

    return subvolume_points, subvolume_normals, subvolume_colors, subvolume_angles

def remove_duplicate_points(points):
    """
    Remove duplicate points from a point cloud.
    """
    unique_points = np.unique(points, axis=0)
    return unique_points

def remove_duplicate_points_normals(points, normals, colors=None, angles=None):
    """
    Remove duplicate points from a point cloud.
    """
    unique_points, indices = np.unique(points, axis=0, return_index=True)
    unique_normals = normals[indices]
    if colors is None and angles is None:
        return unique_points, unique_normals
    elif colors is None:
        unique_angles = angles[indices]
        return unique_points, unique_normals, unique_angles
    elif angles is None:
        unique_colors = colors[indices]
        return unique_points, unique_normals, unique_colors
    else:
        unique_colors = colors[indices]
        unique_angles = angles[indices]
        return unique_points, unique_normals, unique_colors, unique_angles

def detect_subvolume_surfaces(index, points_batch, normals_batch, colors_batch, names_batch, batch_size, gpus):
    points_batch_indices = [i for i, points in enumerate(points_batch) if points.shape[0] > 100]
    points_batch = [points_batch[i] for i in points_batch_indices]
    normals_batch = [normals_batch[i] for i in points_batch_indices]
    colors_batch = [colors_batch[i] for i in points_batch_indices]
    names_batch = [names_batch[i] for i in points_batch_indices]
    # Deep copy the points
    coords_batch = [np.copy(points) for points in points_batch]
    # Translate the points so that the volume starts at (0,0,0)
    min_coord_batch = [np.min(coords, axis=0) for coords in coords_batch]
    coords_batch = [coords - min_coord_batch[i] for i, coords in enumerate(coords_batch)]
    
    # Split up in batches of batch_size 
    coords_splits = [coords_batch[i:i + batch_size] for i in range(0, len(coords_batch), batch_size)]
    predictions_mask3d_batch = []
    # Predict the surfaces 
    # print()
    # for coords_split in tqdm(coords_splits):
    for coords_split in coords_splits:
        # print GPU memory usage
        res = batch_inference(coords_split, index, gpus)
        if res is None:
            print("batch_inference result is None")
            res = [{"pred_classes": []}]*len(coords_split)
        else:
            res = list(res.values())
        predictions_mask3d_batch += res

    surfaces, surfaces_normals, surfaces_colors, scores = to_surfaces(points_batch, normals_batch, colors_batch, predictions_mask3d_batch)
    return surfaces, surfaces_normals, surfaces_colors, names_batch, scores

def alpha_angles(normals):
    '''
    Calculates the turn angle of the sheet with respect to the y-axis. this indicates the orientation of the sheet. Alpha is used to position the point in the scroll. Orientation alpha' and winding number k correspond to alpha = 360*k + alpha'
    '''
    # Calculate angles in radians
    theta_rad = np.arctan2(normals[:, 2], normals[:, 0]) # Y axis is up

    # Convert radians to degrees
    theta_deg = np.degrees(theta_rad)

    return theta_deg

def extract_subvolumes_for_coord(coord, points, normals, colors, subvolume_size):
    x, y, z = coord
    x_prime, y_prime, z_prime = x, y, z
    start_coord = np.array([x_prime, y_prime, z_prime])

    subvolume_points, subvolume_normals, subvolume_colors, subvolume_angles = extract_subvolume(points, normals, colors, colors, start=start_coord, size=subvolume_size)
    
    return subvolume_points, subvolume_normals, subvolume_colors, start_coord

def subvolume_surface_instance_batch(index, points, normals, colors, start, size, path, fix_umbilicus, umbilicus_points, umbilicus_points_old, main_drive="", alternative_drives=[], subvolume_size=50, score_threshold=0.5, distance_threshold=10.0, n=4, alpha = 1000.0, slope_alpha = 0.1, batch_size=4, gpus=1, use_multiprocessing=True):
    """
    Detect surface patches from overlapping subvolumes.
    """

    # Size is int
    if isinstance(subvolume_size, int):
        subvolume_size = np.array([subvolume_size, subvolume_size, subvolume_size])

    # Iterate over all subvolumes
    start = np.array(start)
    # Swap axes
    start = start[[0,2,1]]
    min_coord = start * subvolume_size
    size = np.array(size) * subvolume_size
    ranges = (size / (subvolume_size // 2) - 1) * subvolume_size # -1 because we want to include the last subvolume for tiling operation from later calls starting at the last subvolume but not save one half filled block
    subvolumes_points = []
    subvolumes_normals = []
    subvolumes_colors = []
    start_coords = []
    block_names = []
    block_names_created = []
    
    # Find block coords that contain points
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    start = np.floor(min_coords / (subvolume_size//2)) * (subvolume_size//2)
    # Swap axes
    # Max between start and min_coord
    start = np.maximum(start, min_coord)

    stop_coord = min_coord + ranges - 1

    stop_max_coords = np.ceil(max_coords / (subvolume_size//2)) * (subvolume_size//2) - 1
    # Min between stop and max_coord
    stop = np.minimum(stop_max_coords, stop_coord)

    # Make blocks of size '50x50x50'
    for x in range(int(start[0]), int(stop[0]), subvolume_size[0] // 2):
        for y in range(int(start[1]), int(stop[1]), subvolume_size[1] // 2):
            for z in range(int(start[2]), int(stop[2]), subvolume_size[2] // 2):
                if (x + subvolume_size[0] > stop_max_coords[0]+2 or y + subvolume_size[1] > stop_max_coords[1]+2 or z + subvolume_size[2] > stop_max_coords[2]+2):
                    continue
                x_prime = x
                y_prime = y
                z_prime = z
                start_coord = np.array([x_prime,y_prime,z_prime])
                block_name = path + f"_subvolume_blocks/{start_coord[0]:06}_{start_coord[1]:06}_{start_coord[2]:06}" # nice ordering in the folder
                block_name_tar = block_name + ".tar"
                block_name_tar_alternatives = []
                for alternative_drive in alternative_drives:
                    block_name_tar_alternatives.append(block_name.replace(main_drive, alternative_drive) + ".tar")

                # Skip if the block tar already exists. Same for alternative path
                if (not (fix_umbilicus and fix_umbilicus_recompute(start_coord * 200.0 / 50.0, 200, umbilicus_points, umbilicus_points_old))) and (os.path.exists(block_name_tar) or any([os.path.exists(block_name_tar_alternative) for block_name_tar_alternative in block_name_tar_alternatives])):
                    # print(f"Block {block_name} already exists.")
                    block_names_created.append(block_name)
                    continue

                # Extract a subvolume
                subvolume_points, subvolume_normals, subvolume_colors, subvolume_angles = extract_subvolume(points, normals, colors, colors, start=start_coord, size=subvolume_size) # Second colors is just a placeholder since we dont yet have any angles which the function expects
                if len(subvolume_points) < 10:
                    # print(f"Subvolume {start_coord} has no points.")
                    continue
                
                subvolumes_points.append(subvolume_points)
                subvolumes_normals.append(subvolume_normals)
                subvolumes_colors.append(subvolume_colors)
                start_coords.append(start_coord)
                block_names.append(block_name)

    if len(subvolumes_points) == 0:
        return
    
    # Detect the surfaces in the subvolume
    surfaces, surfaces_normals, surfaces_colors, block_names, scores = detect_subvolume_surfaces(index, subvolumes_points, subvolumes_normals, subvolumes_colors, block_names, batch_size, gpus)

    # save each instance for each subvolume
    # Setting up multiprocessing, process count
    if use_multiprocessing:
        num_threads = multiprocessing.cpu_count()
        with Pool(num_threads) as pool:
            # Save the block
            pool.map(save_block_ply_args, [(surfaces[i], surfaces_normals[i], surfaces_colors[i], scores[i], block_names[i], score_threshold, distance_threshold, n, alpha, slope_alpha) for i in range(len(surfaces))])
    else:
        # single threaded version
        for i in range(len(surfaces)):
            save_block_ply(surfaces[i], surfaces_normals[i], surfaces_colors[i], scores[i], block_names[i], score_threshold, distance_threshold, n, alpha, slope_alpha)

def subvolume_computation_function(args):
    index, start, size, path, folder, dest, main_drive, alternative_drives, fix_umbilicus, umbilicus_points, umbilicus_points_old, score_threshold, batch_size, gpus, use_multiprocessing = args
    src_path = os.path.join(path, folder)
    dest_path = os.path.join(dest, folder)
    size = np.array(size) + 1 # +1 because we want to include the last subvolume for tiling operation from later calls starting at the last subvolume
    
    res = load_plys(src_path, main_drive, alternative_drives, start, size, grid_block_size=200, load_multithreaded=use_multiprocessing)
    if res is None:
        return index
    points, normals, colors = res
    points, normals, colors = remove_duplicate_points_normals(points, normals, colors)
    subvolume_surface_instance_batch(index, points, normals, colors, start, size, dest_path, fix_umbilicus, umbilicus_points, umbilicus_points_old, main_drive, alternative_drives, score_threshold=score_threshold, batch_size=batch_size, gpus=gpus, use_multiprocessing=use_multiprocessing)
    return index

def filter_umilicus_distance(start_list, size, path, folder, umbilicus_points_path, umbilicus_distance_threshold, grid_block_size=200):
    # Load umbilicus points
    umbilicus_raw_points = load_xyz_from_file(umbilicus_points_path)
    umbilicus_points = umbilicus(umbilicus_raw_points)

    # loop over all start points
    start_list_filtered = []
    for start in start_list:
        #check if start block is existing
        empty_block = True
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    if os.path.exists(os.path.join(path, folder, f"cell_yxz_{start[0]+i:03}_{start[1]+j:03}_{start[2]+k:03}.ply")):
                        empty_block = False
                        break
                if not empty_block:
                    break
            if not empty_block:
                break
        if empty_block:
            continue
        # calculate umbilicus point
        block_point = np.array(start) * grid_block_size + grid_block_size//2
        umbilicus_point = umbilicus_xz_at_y(umbilicus_points, block_point[2])
        umbilicus_point = umbilicus_point[[0, 2, 1]] # ply to corner coords
        umbilicus_point[2] = block_point[2] # if umbilicus is not in the same y plane as the block, set umbilicus y to block y (linear cast down along z axis)
        umbilicus_normal = block_point - umbilicus_point

        # distance umbilicus_normal
        distance = np.linalg.norm(umbilicus_normal)
        if (umbilicus_distance_threshold <= 0) or ((umbilicus_distance_threshold > 0) and (distance < umbilicus_distance_threshold)):
            start_list_filtered.append(start)
    
    return start_list_filtered

def update_progress_file(progress_file, indices, config):
    # Update the progress file
    with open(progress_file, 'w') as file:
        json.dump({'indices': indices, 'config': config}, file)

def subvolume_instances_multithreaded(path="/media/julian/FastSSD/scroll3_surface_points", folder="point_cloud_colorized", dest="/media/julian/HDD8TB/scroll3_surface_points", main_drive="", alternative_drives=[], fix_umbilicus=True, umbilicus_points_path="", start=[0, 0, 0], stop=[16, 17, 29], size = [3, 3, 3], umbilicus_distance_threshold=1500, score_threshold=0.5, batch_size=4, gpus=1):
    # Build start list
    start_list = []
    for x in range(start[0], stop[0], size[0]):
        for y in range(start[1], stop[1], size[1]):
            for z in range(start[2], stop[2], size[2]):
                start_list.append([x, y, z])
    print(f"Length of start list: {len(start_list)}")
    start_list = filter_umilicus_distance(start_list, size, path, folder, umbilicus_points_path, umbilicus_distance_threshold)
    print(f"Length of start list after filtering: {len(start_list)}")

    umbilicus_raw_points = load_xyz_from_file(umbilicus_points_path)
    umbilicus_points = umbilicus(umbilicus_raw_points)
    if fix_umbilicus:
        # Load old umbilicus
        umbilicus_path_old = umbilicus_points_path.replace("umbilicus", "umbilicus_old")
        # Usage
        umbilicus_raw_points_old = load_xyz_from_file(umbilicus_path_old)
        umbilicus_points_old = umbilicus(umbilicus_raw_points_old)
    else:
        umbilicus_points_old = None

    num_tasks = len(start_list)

    # init the Mask3D model
    init(gpus)

    to_compute_indices = range(num_tasks)
    computed_indices = []
    progress_file = os.path.join(dest, "progress.json")
    config = {"path": path, "folder": folder, "dest": dest, "main_drive": main_drive, "alternative_drives": alternative_drives, "fix_umbilicus": fix_umbilicus, "umbilicus_points_path": umbilicus_points_path, "start": start, "stop": stop, "size": size, "umbilicus_distance_threshold": umbilicus_distance_threshold, "score_threshold": score_threshold, "batch_size": batch_size, "gpus": gpus}
    nr_total_indices = len(to_compute_indices)
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as file:
            progress = json.load(file)
            if 'config' in progress:
                if progress['config'] != config:
                    print("Progress file found but with different config. Overwriting.")
                    
                else:
                    print("Progress file found with same config. Resuming computation.")
                    if 'indices' in progress:
                        computed_indices = progress['indices']
                        to_compute_indices = list(set(to_compute_indices) - set(computed_indices))
                        print(f"Resuming computation. {len(to_compute_indices)} blocks of {nr_total_indices} left to compute.")
                    else:
                        print("No progress file found.")


    if gpus == 1:
        # Single threaded computation
        # Initialize the tqdm progress bar
        with tqdm(total=nr_total_indices, initial=len(computed_indices)) as pbar:
            for i in to_compute_indices:
                pbar.update(1)
                result = subvolume_computation_function((i, start_list[i], size, path, folder, dest, main_drive, alternative_drives, fix_umbilicus, umbilicus_points, umbilicus_points_old, score_threshold, batch_size, gpus, True))
                index = result
                computed_indices.append(index)
                update_progress_file(progress_file, computed_indices, config)
    elif gpus > 1:
        # multithreaded computation
        num_threads = gpus
        # Initialize the tqdm progress bar
        with tqdm(total=nr_total_indices, initial=len(computed_indices)) as pbar:
            with Pool(processes=num_threads) as pool:
                a = (10 + 3) / (7.0 % 6)
                for result in pool.imap(subvolume_computation_function, [(i, start_list[i], size, path, folder, dest, main_drive, alternative_drives, fix_umbilicus, umbilicus_points, umbilicus_points_old, score_threshold, batch_size, gpus, False) for i in to_compute_indices]):
                    pbar.update(1)
                    index = result
                    computed_indices.append(index)
                    update_progress_file(progress_file, computed_indices, config)
                    
    else:
        raise ValueError("gpus must be >= 1")

def compute(path, folder, dest, main_drive, alternative_ply_drives, umbilicus_points_path, umbilicus_distance_threshold, fix_umbilicus, score_threshold, batch_size, gpus):
    import sys
    # Remove command-line arguments for later internal calls to Mask3D
    sys.argv = [sys.argv[0]]

    # Multithreaded computation
    subvolume_instances_multithreaded(path=path, folder=folder, dest=dest, main_drive=main_drive, alternative_drives=alternative_ply_drives, fix_umbilicus=fix_umbilicus, umbilicus_points_path=umbilicus_points_path, start=[0, 0, 0], stop=[100, 100, 100], size = [3, 3, 3], umbilicus_distance_threshold=umbilicus_distance_threshold, score_threshold=score_threshold, batch_size=batch_size, gpus=gpus)

def main():
    side = "_verso" # actually recto
    path = "/media/julian/SSD2/scroll3_surface_points"
    folder = f"point_cloud_colorized{side}"
    fix_umbilicus = False
    umbilicus_points_path = "/media/julian/SSD4TB/PHerc0332.volpkg/volumes/2dtifs_8um_grids/umbilicus.txt"
    dest = f"/media/julian/SSD4TB/scroll3_surface_points"
    main_drive = "SSD2"
    alternative_ply_drives = ["FastSSD", "HDD8TB"]
    # umbilicus_distance_threshold=-1 #2250 scroll 1
    # umbilicus_distance_threshold=2200 # scroll 3
    umbilicus_distance_threshold = -1
    score_threshold=0.10
    batch_size = 4

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Compute surface patches from pointcloud.")
    parser.add_argument("--path", type=str, help="Base path to the data", default=path)
    parser.add_argument("--folder", type=str, help="Folder containing the point cloud data", default=folder)
    parser.add_argument("--dest", type=str, help="Folder to save the output data", default=dest)
    parser.add_argument("--main_drive", type=str, help="Main drive that contains the input data", default=main_drive)
    parser.add_argument("--alternative_ply_drives", type=str, nargs='+', help="Alternative drives that may contain additional input data in the same path naming sceme. To split data over multiple drives if needed.", default=alternative_ply_drives)
    parser.add_argument("--umbilicus_path", type=str, help="Path to the umbilicus.txt", default=umbilicus_points_path)
    parser.add_argument("--max_umbilicus_dist", type=float, help="Maximum distance between the umbilicus and blocks that should be computed. -1.0 for no distance restriction", default=umbilicus_distance_threshold)
    parser.add_argument("--fix_umbilicus", action='store_true', help="Flag, recompute all close to the updated umbilicus (make sure to also save the old umbilicus.txt as umbilicus_old.txt)")
    parser.add_argument("--score_threshold", type=float, help="Minimum score for a surface to be saved", default=score_threshold)
    parser.add_argument("--batch_size", type=int, help="Batch size for Mask3D", default=batch_size)
    parser.add_argument("--gpus", type=int, help="Number of GPUs to use", default=1)

    # Parse the arguments
    args = parser.parse_args()
    path = args.path
    folder = args.folder
    dest = args.dest
    main_drive = args.main_drive
    alternative_ply_drives = args.alternative_ply_drives
    umbilicus_points_path = args.umbilicus_path
    umbilicus_distance_threshold = args.max_umbilicus_dist
    fix_umbilicus = args.fix_umbilicus
    score_threshold = args.score_threshold
    batch_size = args.batch_size
    gpus = args.gpus

    # Compute the surface patches
    compute(path, folder, dest, main_drive, alternative_ply_drives, umbilicus_points_path, umbilicus_distance_threshold, fix_umbilicus, score_threshold, batch_size, gpus)

    
if __name__ == "__main__":
    main()