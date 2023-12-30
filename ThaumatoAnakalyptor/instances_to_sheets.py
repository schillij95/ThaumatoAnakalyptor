### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import numpy as np
import pickle
import os
import open3d as o3d
import multiprocessing
import time
import threading
from concurrent.futures import ThreadPoolExecutor
# Global variable to store the main_sheet data
main_sheet_data = None
# Signal to indicate when the data is updated
data_updated_event = threading.Event()

# surface points extraction
from multiprocessing import Pool
from numba import jit
import pandas as pd
import datetime
import json

# plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tqdm import tqdm

import glob
from copy import deepcopy
import tarfile
import tempfile
from surface_fitting_utilities import get_vector_mean, rotation_matrix_to_align_z_with_v, fit_surface_to_points_n_regularized, plot_surface_and_points, distance_from_surface, distance_from_surface_clipped

import warnings
warnings.filterwarnings("ignore")

import argparse

# Define the maximum number of items you want in cache
# NOTE: This is not in terms of GB but rather the number of function calls.
MAX_ITEMS = 1000000

def load_ply(filename):
    """
    Load point cloud data from a .ply file inside a tarball.
    """
    # Derive tar filename from the given filename
    base_dir = os.path.dirname(filename)
    parent_dir = os.path.dirname(base_dir)
    tar_filename = os.path.join(parent_dir, f"{os.path.basename(base_dir)}.tar")
    ply_filename_inside_tar = os.path.basename(filename)

    # Check that the tar file exists
    assert os.path.isfile(tar_filename), f"File {tar_filename} not found."

    # Extract the desired ply and metadata files from the tarball to a temporary location
    with tarfile.open(tar_filename, 'r') as tar:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Construct paths to the temporary extracted files
            temp_ply_path = os.path.join(temp_dir, ply_filename_inside_tar)
            base_filename_without_extension = os.path.splitext(ply_filename_inside_tar)[0]
            metadata_filename_inside_tar = f"metadata_{base_filename_without_extension}.json"
            temp_metadata_path = os.path.join(temp_dir, metadata_filename_inside_tar)
            
            tar.extract(ply_filename_inside_tar, path=temp_dir)
            tar.extract(metadata_filename_inside_tar, path=temp_dir)
            
            # Load the ply file
            pcd = o3d.io.read_point_cloud(temp_ply_path)
            
            # Load metadata
            with open(temp_metadata_path, 'r') as metafile:
                metadata = json.load(metafile)

            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            # Extract additional features
            colors = np.asarray(pcd.colors)

    # Convert lists back to numpy arrays where needed
    if 'coeff' in metadata and metadata['coeff'] is not None:
        coeff = np.array(metadata['coeff'])
    else:
        coeff = None
    
    if 'n' in metadata and metadata['n'] is not None:
        n = int(metadata['n'])
    else:
        n = None

    score = metadata.get('score')
    distance = metadata.get('distance')

    return points, normals, colors, score, distance, coeff, n

def load_instance(path, sample_ratio=1.0):
    """
    Load instance from path.
    """
    points_list = []
    normals_list = []
    colors_list = []
    pred_score_list = []
    distance_list = []
    id_list = []

    # Deduce tar filename from given path
    tar_filename = f"{path}.tar"

    # Check that the tar file exists
    assert os.path.isfile(tar_filename), f"File {tar_filename} not found."

    # Retrieve a list of all .ply files within the tarball
    with tarfile.open(tar_filename, 'r') as archive:
        ply_files = [m.name for m in archive.getmembers() if m.name.endswith(".ply")]
        # Iterate and load all instances prediction .ply files
        for ply_file in ply_files:
            file = os.path.join(path, ply_file)
            res = load_ply(file)
            points = res[0]
            normals = res[1]
            colors = res[2]
            pred_score = res[3]
            distance = res[4]
            # Sample points from picked patch
            points, normals, colors, _ = select_points(
                points, normals, colors, colors, sample_ratio
            )
            if points.shape[0] < 10:
                continue
            points_list.append(points)
            normals_list.append(normals)
            colors_list.append(colors)
            pred_score_list.append(pred_score)
            distance_list.append(distance)
            id_list.append(tuple([*map(int, file.split("/")[-2].split("_"))]+[int(file.split("/")[-1].split(".")[-2].split("_")[-1])]))
    return points_list, normals_list, colors_list, pred_score_list, distance_list, id_list

def init_alpha_angles_for_patch(patch):
    """
    Initialize the alpha angles for a patch.
    """
    # Calculate main normal for patch
    assert len(patch["normals"]) > 0, "Patch has no normals."
    patch_normal = get_vector_mean(patch["normals"])
    # Check non-zero length in xz plane
    if np.linalg.norm(patch_normal[[0, 2]]) == 0:
        print("[WARNING] Main normal has zero length in 'XZ' (Scroll Slice) Plane! Setting patch normal to [1,0,0].")
        patch_normal = np.array([1,0,0])

    # Patch normal has alpha angle of zero
    offset = - alpha_angles(np.array([patch_normal]))[0]

    patch["anchor_points"] = [patch["points"][0]]
    patch["anchor_normals"] = [patch_normal]
    patch["anchor_angles"] = [adjust_angles_offset(alpha_angles(patch["anchor_normals"][0]), offset)]
    patch["angles"] = adjust_angles_offset(alpha_angles(patch["normals"]), offset)

def angle_to_180(angle):
    """
    Convert angle to range [-180, 180].
    """
    found_adjustables = True
    while(found_adjustables):
        mask_low = angle < -180
        mask_high = angle > 180

        angle[mask_low] = angle[mask_low] + 360
        angle[mask_high] = angle[mask_high] - 360

        found_adjustables = np.any(mask_low) or np.any(mask_high)
    return angle

def alpha_angles(normals):
    '''
    Calculates the turn angle of the sheet with respect to the y-axis. this indicates the orientation of the sheet. Alpha is used to position the point in the scroll. Orientation alpha' and winding number k correspond to alpha = 360*k + alpha'
    '''
    # Calculate angles in radians
    theta_rad = np.arctan2(normals[:, 2], normals[:, 0]) # Y axis is up

    # Convert radians to degrees
    theta_deg = np.degrees(theta_rad)

    theta_deg = angle_to_180(theta_deg)
        
    return theta_deg

def alpha_normals(angles):
    '''
    Calculates the normals of the sheet with respect to the y-axis. this indicates the orientation of the sheet.
    '''
    # Calculate angles in radians
    theta_rad = np.radians(angles)

    # Calculate normals
    normals = np.zeros((theta_rad.shape[0], 3))
    normals[:, 0] = np.cos(theta_rad)
    normals[:, 1] = 0
    normals[:, 2] = np.sin(theta_rad)

    return normals

def adjust_angles(angles, alpha_p, alpha_p_old):
    '''
    When combinding two segments -> adjusting the angles of the second segment to the first segment.
    '''
    # Offset angles by alpha_p - alpha_p_old
    offset = alpha_p - alpha_p_old
    angles = angles + offset
    
    return angles

def adjust_angles_zero(angles, offset):
    '''
    When combinding two segments -> adjusting the angles of the second segment to the first segment.
    '''
    # Angles is not a numpy array
    added_dimension = False
    if not isinstance(angles, np.ndarray):
        angles = np.array([angles])
        added_dimension = True
    # Offset angles by alpha_p - alpha_p_old
    offset_k = int(offset / 360)
    offset_r = offset - offset_k * 360
    angles_k = (angles / 360).astype(int)
    angles_r = angles - angles_k * 360
    angles = angles_r + offset_r
    angles_mask_high = angles > 180
    angles_mask_low = angles < -180
    angles[angles_mask_high] = angles[angles_mask_high] - 360
    angles[angles_mask_low] = angles[angles_mask_low] + 360
    angles = angles + offset_k * 360 + angles_k * 360

    if added_dimension:
        angles = angles[0]
    
    return angles

def adjust_angles_offset(angles, offset):
    '''
    When combinding two segments -> adjusting the angles of the second segment to the first segment.
    '''
    # Angles is not a numpy array
    added_dimension = False
    if not isinstance(angles, np.ndarray):
        angles = np.array([angles])
        added_dimension = True

    angles = angles + offset

    if added_dimension:
        angles = angles[0]
    
    return angles

def compute_angle_range(angles, angle_tolerance):
    if len(angles) == 0:
        return [(0, 0)]
    
    # Compute lower and upper bounds of the ranges
    lower_bounds = angles - angle_tolerance
    upper_bounds = angles + angle_tolerance

    # Zip the bounds and sort them
    ranges = sorted(zip(lower_bounds, upper_bounds))

    # Merge overlapping or adjacent ranges
    merged_ranges = [ranges[0]]
    for current_start, current_end in ranges[1:]:
        last_end = merged_ranges[-1][1]
        if current_start <= last_end:
            merged_ranges[-1] = (merged_ranges[-1][0], max(last_end, current_end))
        else:
            merged_ranges.append((current_start, current_end))
    return merged_ranges

def largest_group_offset(offsets, epsilon=1e-5):
    assert len(offsets) > 0, "Offsets is empty."
    # Sort the offsets
    sorted_offsets = np.sort(offsets)

    # Compute differences between consecutive sorted offsets
    diffs = np.diff(sorted_offsets)

    # Identify where the difference exceeds the epsilon indicating a new group
    group_indices = np.where(diffs > epsilon)[0] + 1
    group_starts = np.insert(group_indices, 0, 0)
    group_ends = np.append(group_indices, len(sorted_offsets))

    # Calculate group sizes and identify the largest group
    group_sizes = group_ends - group_starts
    largest_group_index = np.argmax(group_sizes)
    largest_group = sorted_offsets[group_starts[largest_group_index]:group_ends[largest_group_index]]

    # Return the mean offset value of the largest group
    assert len(largest_group) > 0, "Largest group is empty."
    return np.mean(largest_group)


def select_angles_from_subsample(points1, angles1, points2, angles2, points_subsampled):
    # make unique
    points_subsampled = remove_duplicate_points(points_subsampled)

    # Convert numpy arrays to pandas dataframes
    df1 = pd.DataFrame(points1, columns=['x', 'y', 'z'])
    df1['a1'] = angles1

    df2 = pd.DataFrame(points2, columns=['x', 'y', 'z'])
    df2['a2'] = angles2
    
    df3 = pd.DataFrame(points_subsampled, columns=['x', 'y', 'z'])
    
    # Merge dataframes on ['x', 'y', 'z'] to get the common points and normals
    common_df = pd.merge(df1, df2, on=['x', 'y', 'z'])
    common_df = pd.merge(common_df, df3, on=['x', 'y', 'z'])
    
    # Extract points and normals from the merged dataframe
    common_points = common_df[['x', 'y', 'z']].values
    common_angles1 = common_df[['a1']].values
    common_angles2 = common_df[['a2']].values
    
    return common_angles1, common_angles2

def select_tiles_points(patches_list, i, tiles):   
    patch1 = patches_list[i]

    all_points_tiles = None
    for tile in tiles:
        if tile in patch1["tiles_points"]: 
            points = patch1["tiles_points"][tile]
            # Construct all overlap points
            if all_points_tiles is None:
                all_points_tiles = points
            else:
                all_points_tiles = np.concatenate([all_points_tiles, points], axis=0)
    
    return all_points_tiles

def select_overlapping_angles(patches_list, i, j, tiles):   
    patch1 = patches_list[i]
    patch2 = patches_list[j]

    all_angles_overlap1 = None
    all_angles_overlap2 = None
    for tile in tiles:

        if tile in patch1["tiles_points"]: 
            # Tile mask overlap
            mask1 = patch1["tiles_overlap_mask"][tile][j]
            # Angles of tile points
            angles_tile1 = patch1["tiles_angles"][tile]
            # Extract overlapping angles
            # Since points are ordered in the same way, no permutation is needed. for each index, angles correspond for the overlapping points
            overlap_angles_tile1 = angles_tile1[mask1]
            # Construct all angles. overlap
            if all_angles_overlap1 is None:
                all_angles_overlap1 = overlap_angles_tile1
            else:
                all_angles_overlap1 = np.concatenate([all_angles_overlap1, overlap_angles_tile1], axis=0)

        if tile in patch2["tiles_points"]:
            # Tile mask overlap
            mask2 = patch2["tiles_overlap_mask"][tile][i]
            # Angles of tile points
            angles_tile2 = patch2["tiles_angles"][tile]
            # Extract overlapping angles
            # Since points are ordered in the same way, no permutation is needed. for each index, angles correspond for the overlapping points
            overlap_angles_tile2 = angles_tile2[mask2]
            # Construct all angles. overlap
            if all_angles_overlap2 is None:
                all_angles_overlap2 = overlap_angles_tile2
            else:
                all_angles_overlap2 = np.concatenate([all_angles_overlap2, overlap_angles_tile2], axis=0)
    
    return all_angles_overlap1, all_angles_overlap2

def select_non_overlapping_angles(patches_list, i, j, tiles):
    patch1 = patches_list[i]
    patch2 = patches_list[j]

    all_angles_non_overlap1 = None
    all_angles_non_overlap2 = None
    for tile in tiles:

        if tile in patch1["tiles_points"]:
            # Tile mask overlap
            mask1 = patch1["tiles_overlap_mask"][tile][j]
            # Non-overlapping mask
            mask1 = np.logical_not(mask1)
            # Angles of tile points
            angles_tile1 = patch1["tiles_angles"][tile]
            # Extract non-overlapping angles
            # Since points are ordered in the same way, no permutation is needed. for each index, angles correspond for the non-overlapping points
            non_overlap_angles_tile1 = angles_tile1[mask1]
            # Construct all angles. non-overlap
            if all_angles_non_overlap1 is None:
                all_angles_non_overlap1 = non_overlap_angles_tile1
            else:
                all_angles_non_overlap1 = np.concatenate([all_angles_non_overlap1, non_overlap_angles_tile1], axis=0)

        if tile in patch2["tiles_points"]:
            # Tile mask overlap
            mask2 = patch2["tiles_overlap_mask"][tile][i]
            # Non-overlapping mask
            mask2 = np.logical_not(mask2)
            # Angles of tile points
            angles_tile2 = patch2["tiles_angles"][tile]
            # Extract non-overlapping angles
            # Since points are ordered in the same way, no permutation is needed. for each index, angles correspond for the non-overlapping points
            non_overlap_angles_tile2 = angles_tile2[mask2]
            # Construct all angles. non-overlap
            if all_angles_non_overlap2 is None:
                all_angles_non_overlap2 = non_overlap_angles_tile2
            else:
                all_angles_non_overlap2 = np.concatenate([all_angles_non_overlap2, non_overlap_angles_tile2], axis=0)

    return all_angles_non_overlap1, all_angles_non_overlap2

def filter_points_by_angle(points2, angles2, angle_ranges):
    # Check which angles in angles2 are within the merged ranges
    masks = [np.logical_and(angles2 >= start, angles2 <= end) for start, end in angle_ranges]
    mask = np.any(masks, axis=0)

    # Return filtered points
    return points2[mask]

def filter_by_angle(angles2, angle_ranges):
    # Check which angles in angles2 are within the merged ranges
    masks = [np.logical_and(angles2 >= start, angles2 <= end) for start, end in angle_ranges]
    mask = np.any(masks, axis=0)

    # Return filtered points
    return mask

def subvolume_surface_patches_folder_(path, subvolume_size=50, sample_ratio=1.0):
    """
    Load surface patches from overlapping subvolumes instances predictions.
    """

    # Size is int
    if isinstance(subvolume_size, int):
        subvolume_size = np.array([subvolume_size, subvolume_size, subvolume_size])

    patches_list = []
    # Iterate over all subvolumes. glob folders in path
    for file in glob.glob(path + "*.tar"):
        surfaces, surfaces_normals, surfaces_colors, pred_scores, distances, ids = load_instance(file[:-4], sample_ratio=sample_ratio)
        # print(f"'Detected' {len(surfaces)} surfaces in subvolume {x,y,z} with id {id_} for the first instance.")
        # visualize_surfaces(surfaces)
        for i in range(len(surfaces)):
            x, y, z, id_ = ids[i]
            surface_dict = {"ids": [ids[i]], 
                            "points": surfaces[i],
                            "normals": surfaces_normals[i],
                            "colors": surfaces_colors[i], 
                            "anchor_points": [surfaces[i][0]], 
                            "anchor_normals": [surfaces_normals[i][0]],
                            "anchor_angles": [alpha_angles(np.array([surfaces_normals[i][0]]))[0]],
                            "angles": alpha_angles(surfaces_normals[i]),
                            "subvolume": [np.array([x,y,z])], 
                            "subvolume_size": [subvolume_size],
                            "iteration": 0,
                            "patch_prediction_scores": [pred_scores[i]],
                            "patch_prediction_distances": [distances[i]],
                            }
            # Add the surface dict to the list
            patches_list.append(surface_dict)

    return patches_list

def process_ply_file(ply_file, subvolume_size, path, sample_ratio, results_list):
    ids = tuple([*map(int, ply_file.split("/")[-2].split("_"))]+[int(ply_file.split("/")[-1].split(".")[-2].split("_")[-1])])
    ids = (int(ids[0]), int(ids[1]), int(ids[2]), int(ids[3]))
    main_sheet_patch = (ids[:3], ids[3], float(0.0))
    surface_dict, _ = build_patch(main_sheet_patch, tuple(subvolume_size), path, sample_ratio=float(sample_ratio))

    # Append the surface dict to the shared results_list
    results_list.append(surface_dict)

def subvolume_surface_patches_folder_(file, subvolume_size=50, sample_ratio=1.0):
    if isinstance(subvolume_size, int) or isinstance(subvolume_size, (float, np.number)):
        subvolume_size = np.array([subvolume_size, subvolume_size, subvolume_size])

    patches_list = []
    tar_filename = f"{file}.tar"
    path = os.path.dirname(tar_filename)

    if os.path.isfile(tar_filename):
        with tarfile.open(tar_filename, 'r') as archive:
            ply_files = [m.name for m in archive.getmembers() if m.name.endswith(".ply")]

            # Create a ThreadPoolExecutor to parallelize the processing of ply files
            with ThreadPoolExecutor() as executor:
                futures = []
                for ply_file_ in ply_files:
                    ply_file = os.path.join(file, ply_file_)
                    futures.append(executor.submit(process_ply_file, ply_file, subvolume_size, path, sample_ratio, patches_list))

                # Ensure all threads have finished before moving on
                for future in futures:
                    future.result()

    return patches_list

def load_ply_folder(ply_file_path):
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

    # Derive metadata file path from .ply file path
    base_filename_without_extension = os.path.splitext(os.path.basename(ply_file_path))[0]
    metadata_file_path = os.path.join(os.path.dirname(ply_file_path), f"metadata_{base_filename_without_extension}.json")

    # Initialize metadata-related variables
    coeff, n, score, distance = None, None, None, None

    if os.path.isfile(metadata_file_path):
        with open(metadata_file_path, 'r') as metafile:
            metadata = json.load(metafile)
            coeff = np.array(metadata['coeff']) if 'coeff' in metadata and metadata['coeff'] is not None else None
            n = int(metadata['n']) if 'n' in metadata and metadata['n'] is not None else None
            score = metadata.get('score')
            distance = metadata.get('distance')

    return points, normals, colors, score, distance, coeff, n

def build_patch_folder(main_sheet_patch, subvolume_size, path, sample_ratio=1.0, align_and_flip_normals=False):
    subvolume_size = np.array(subvolume_size)
    ((x, y, z), main_sheet_surface_nr, offset_angle) = main_sheet_patch
    file = path + f"/{x:06}_{y:06}_{z:06}/surface_{main_sheet_surface_nr}.ply"
    res = load_ply_folder(path)
    patch_points = res[0]
    patch_normals = res[1]
    patch_color = res[2]
    patch_score = res[3]
    patch_distance = res[4]
    patch_coeff = res[5]
    n = res[6]

    # Sample points from picked patch
    patch_points, patch_normals, patch_color, _ = select_points(
        patch_points, patch_normals, patch_color, patch_color, sample_ratio
    )

    patch_id = tuple([*map(int, file.split("/")[-2].split("_"))]+[int(file.split("/")[-1].split(".")[-2].split("_")[-1])])

    x, y, z, id_ = patch_id
    anchor_normal = get_vector_mean(patch_normals)
    anchor_angle = alpha_angles(np.array([anchor_normal]))[0]

    additional_main_patch = {"ids": [patch_id],
                    "points": patch_points,
                    "normals": patch_normals,
                    "colors": patch_color,
                    "anchor_points": [patch_points[0]], 
                    "anchor_normals": [anchor_normal],
                    "anchor_angles": [anchor_angle],
                    "angles": adjust_angles_offset(adjust_angles_zero(alpha_angles(patch_normals), - anchor_angle), offset_angle),
                    "subvolume": [(x, y, z)],
                    "subvolume_size": [subvolume_size],
                    "iteration": 0,
                    "patch_prediction_scores": [patch_score],
                    "patch_prediction_distances": [patch_distance],
                    "patch_prediction_coeff": [patch_coeff],
                    "n": [n],
                    }
    
    return additional_main_patch, offset_angle

def subvolume_surface_patches_folder(file, subvolume_size=50, sample_ratio=1.0):
    """
    Load surface patches from overlapping subvolumes instances predictions.
    """

    # Standardize subvolume_size to a NumPy array
    subvolume_size = np.atleast_1d(subvolume_size).astype(int)
    if subvolume_size.shape[0] == 1:
        subvolume_size = np.repeat(subvolume_size, 3)

    patches_list = []
    tar_filename = f"{file}.tar"

    if os.path.isfile(tar_filename):
        with tarfile.open(tar_filename, 'r') as archive, tempfile.TemporaryDirectory() as temp_dir:
            # Extract all .ply files at once
            ply_files = [m for m in archive.getmembers() if m.name.endswith(".ply")]
            archive.extractall(path=temp_dir, members=archive.getmembers())

            # Process each .ply file
            for ply_member in ply_files:
                ply_file_path = os.path.join(temp_dir, ply_member.name)
                ply_file = ply_member.name
                ids = tuple([*map(int, tar_filename.split(".")[-2].split("/")[-1].split("_"))]+[int(ply_file.split(".")[-2].split("_")[-1])])
                ids = (int(ids[0]), int(ids[1]), int(ids[2]), int(ids[3]))
                main_sheet_patch = (ids[:3], ids[3], float(0.0))
                surface_dict, _ = build_patch_folder(main_sheet_patch, tuple(subvolume_size), ply_file_path, sample_ratio=float(sample_ratio))
                patches_list.append(surface_dict)

    return patches_list

def subvolume_surface_patches_folder_old(file, subvolume_size=50, sample_ratio=1.0):
    """
    Load surface patches from overlapping subvolumes instances predictions.
    """

    # Size is int
    if isinstance(subvolume_size, int) or isinstance(subvolume_size, float) or isinstance(subvolume_size, np.float64) or isinstance(subvolume_size, np.float32) or isinstance(subvolume_size, np.int64) or isinstance(subvolume_size, np.int32) or isinstance(subvolume_size, np.int16) or isinstance(subvolume_size, np.int8) or isinstance(subvolume_size, np.uint64) or isinstance(subvolume_size, np.uint32) or isinstance(subvolume_size, np.uint16) or isinstance(subvolume_size, np.uint8):
        subvolume_size = np.array([subvolume_size, subvolume_size, subvolume_size])

    patches_list = []
    # Deduce folder filename from given path
    tar_filename = f"{file}.tar"

    # base path of tar file
    path = os.path.dirname(tar_filename)

    # Check that the tar file exists
    if os.path.isfile(tar_filename):
        # Retrieve a list of all .ply files within the tarball
        with tarfile.open(tar_filename, 'r') as archive:
            ply_files = [m.name for m in archive.getmembers() if m.name.endswith(".ply")]
            # Iterate and load all instances prediction .ply files
            for ply_file_ in ply_files:
                ply_file = os.path.join(file, ply_file_)

                ids = tuple([*map(int, ply_file.split("/")[-2].split("_"))]+[int(ply_file.split("/")[-1].split(".")[-2].split("_")[-1])])
                ids = (int(ids[0]), int(ids[1]), int(ids[2]), int(ids[3]))
                main_sheet_patch = (ids[:3], ids[3], float(0.0))
                surface_dict, _ = build_patch(main_sheet_patch, tuple(subvolume_size), path, sample_ratio=float(sample_ratio))

                # Add the surface dict to the list
                patches_list.append(surface_dict)

    return patches_list

def subvolume_surface_patches(path, subvolume_size=50):
    """
    Load surface patches from overlapping subvolumes instances predictions.
    """

    # Size is int
    if isinstance(subvolume_size, int):
        subvolume_size = np.array([subvolume_size, subvolume_size, subvolume_size])

    patches_list = []
    # Iterate over all subvolumes. glob folders in path
    for file in tqdm(glob.glob(path + "/*/")):
        surfaces, surfaces_normals, surfaces_colors, ids = load_instance(file)
        x, y, z, id_ = ids[0]
        for i in range(len(surfaces)):
            surface_dict = {"ids": [ids[i]], 
                            "points": surfaces[i],
                            "normals": surfaces_normals[i],
                            "colors": surfaces_colors[i], 
                            "anchor_points": [surfaces[i][0]], 
                            "anchor_normals": [surfaces_normals[i][0]],
                            "anchor_angles": [alpha_angles(np.array([surfaces_normals[i][0]]))[0]],
                            "angles": alpha_angles(surfaces_normals[i]),
                            "subvolume": [np.array([x,y,z])], 
                            "subvolume_size": [subvolume_size],
                            "iteration": 0,}
            # Add the surface dict to the list
            patches_list.append(surface_dict)

    return patches_list

def save_main_sheet(main_sheet, volume_blocks_scores, path="main_sheet.ta"):
    """
    Save the main sheet to a file.
    """
    save_sheet = {}
    # remove points for display
    for volume_id in main_sheet:
        save_sheet[volume_id] = {}
        for patch_id in main_sheet[volume_id]:
            save_sheet[volume_id][patch_id] = {}
            save_sheet[volume_id][patch_id]["offset_angle"] = main_sheet[volume_id][patch_id]["offset_angle"]

    sheet_segmentation = {'main_sheet': save_sheet, 'volume_blocks_scores': volume_blocks_scores}

    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(sheet_segmentation, f)

def load_main_sheet(subvolume_size=50, path="", path_ta="main_sheet.ta", sample_ratio_score=0.1, sample_ratio=0.1, add_display_points=True):
    """
    Load the main sheet from a file.
    """
    with open(path_ta, 'rb') as f:
        sheet_segmentation = pickle.load(f)

    main_sheet = sheet_segmentation['main_sheet']
    volume_blocks_scores = sheet_segmentation['volume_blocks_scores']

    # Add points and angles of main patches to main sheet
    if add_display_points:
        main_sheet_add_display_points(main_sheet, subvolume_size, path, sample_ratio)

    return main_sheet, volume_blocks_scores

def save_patches_list(patches_list, path):
    """
    Save the patches list to a file. List of dicts with lists, arrays and int, double entries
    """
    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(patches_list, f)

def load_patches_list(path):
    """
    Load patches list from a file
    """
    with open(path, 'rb') as f:
        patches_list = pickle.load(f)
    return patches_list

def load_iteration(path, iteration, subvolume_size=50):
    """
    Load iteration of devide and conquer data.
    """
    if iteration == 0:
        return subvolume_surface_patches(path, subvolume_size=subvolume_size)
    
    # path
    path = path.replace("subvolume_blocks", f"subvolume_blocks_iteration_{iteration}")
    return load_patches_list(path)

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

def select_points(points, normals, colors, angles, original_ratio):
    # indices where colors[1] < original_ratio
    indices = np.where(colors[:, 1] < original_ratio)

    # at least one point:
    if len(indices[0]) == 0:
        indices = np.array([0])
    
    points = points[indices]
    normals = normals[indices]
    colors = colors[indices]
    angles = angles[indices]
    
    return points, normals, colors, angles

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

def process_patch(args):
    """
    Process a single patch.
    Returns a tuple (patch, raw_points_count, remaining_points_count).
    """
    patch, original_ratio = args
    # Make unique
    patch['points'], patch['normals'], patch['colors'], patch['angles'] = remove_duplicate_points_normals(
        patch['points'], patch['normals'], patch['colors'], patch['angles']
    )

    doubled_points, doubled_normals, doubled_colors, doubled_angles = select_points(
        patch['points'], patch['normals'], patch['colors'], patch['angles'], original_ratio
    )

    raw_points_count = patch['points'].shape[0]
    remaining_points_count = doubled_points.shape[0]

    # Remove the points that are not in the subsampled points
    patch["points"] = doubled_points
    patch["normals"] = doubled_normals
    patch["colors"] = doubled_colors
    patch["angles"] = doubled_angles

    return (patch, raw_points_count, remaining_points_count)

def remove_points_from_patches_list_multithreading(patches_list, original_ratio=0.1):
    """
    patches_list is list containing the patches.
    original_ratio is ratio of sampling with respect to the original volume pointcloud.
    """
    total_raw = 0.0
    total_remaining = 0.0

    with Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_patch, [(patch, original_ratio) for patch in patches_list])

    # Update patches and accumulate counts
    for idx, (patch, raw_points_count, remaining_points_count) in enumerate(results):
        patches_list[idx] = patch
        total_raw += raw_points_count
        total_remaining += remaining_points_count

    print(f"Total points: {total_raw}, remaining points: {total_remaining}, ratio: {total_remaining / total_raw}")

    return patches_list

def remove_small_patches(patches_list, min_points=100):
    """
    Remove patches that have less than min_points points.
    """
    total = len(patches_list)
    i = 0
    while i < len(patches_list):
        if patches_list[i]["points"].shape[0] < min_points:
            patches_list.pop(i)
            # adjust the overlap stats
            for j in range(len(patches_list)):
                try:
                    patches_list[j]["overlapp_percentage"] = np.delete(patches_list[j]["overlapp_percentage"], i)
                    patches_list[j]["overlap"] = np.delete(patches_list[j]["overlap"], i)
                    patches_list[j]["non_overlap"] = np.delete(patches_list[j]["non_overlap"], i)
                except:
                    pass
        else:
            i += 1
    remaining = len(patches_list)
    print(f"Removed {total - remaining} patches with less than {min_points} points. {remaining} patches remaining.")
    return patches_list

def select_largest_n_patches(patches_list, n):
    """
    Select the largest n patches.
    """
    if len(patches_list) <= n:
        return patches_list
    
    # get the number of points per patch
    num_points = []
    for i in range(len(patches_list)):
        size_i = len(patches_list[i]["ids"])
        num_points.append(size_i)

    # get the indices of the n largest patches
    indices = np.argpartition(num_points, -n)[-n:]
    
    # get the n largest patches
    patches_list_largest = []
    for i in indices:
        patches_list_largest.append(patches_list[i])

    return patches_list_largest

def filter_patches_list(patches_list, iteration, min_points=100):
    """
    Filter the patches list for the iteration. Saves large enough filtered out patches to file.
    """
    total = len(patches_list)
    i = 0
    removed_patches_to_save = []
    while i < len(patches_list):
        if patches_list[i]["iteration"] < iteration:
            removed_patch = patches_list.pop(i)
            if removed_patch["points"].shape[0] >= min_points or True:
                # Save removed patch to file
                removed_patches_to_save.append(removed_patch)
        else:
            i += 1
    remaining = len(patches_list)
    print(f"Filtered {total - remaining} patches with less than {min_points} points because of patches from old iterations.")
    return patches_list, removed_patches_to_save

def contains_all(ids1, ids2):
    conts_all = True
    for id2 in ids2:
        if id2 not in ids1:
            conts_all = False
            break
    return conts_all

def remove_duplicate_patches(patches_list):
    """
    Removes patches that have the same ids.
    """
    i = 0
    while i < len(patches_list):
        # sort ids
        patches_list[i]["ids"].sort()
        ids = patches_list[i]["ids"]
        j = i + 1
        while j < len(patches_list):
            patches_list[j]["ids"].sort()
            # Check that all entries in both list are the same
            if len(ids) == len(patches_list[j]["ids"]) and np.all(np.array(ids) == np.array(patches_list[j]["ids"])):
                patches_list.pop(j)
            # Check if every entry in ids is in patches_list[j]["ids"]
            elif contains_all(patches_list[i]["ids"], patches_list[j]["ids"]):
                patches_list.pop(i)
            else:
                j += 1
        i += 1

    return patches_list

def generate_subvolumes(volume_start, volume_size, instances_block_size, nr_instances_size):
    print(f"Generating subvolumes for volume from start at {volume_start} of size {volume_size} with instances block size {instances_block_size} and nr instances size {nr_instances_size}")
    subvolumes = []
    for x in range(volume_start[0], volume_start[0] + (volume_size[0] - (nr_instances_size - 1))*instances_block_size, instances_block_size):
        for y in range(volume_start[1], volume_start[1] + (volume_size[1] - (nr_instances_size - 1))*instances_block_size, instances_block_size):
            for z in range(volume_start[2], volume_start[2] + (volume_size[2] - (nr_instances_size - 1))*instances_block_size, instances_block_size):
                subvolume = {"start": np.array([x,y,z]), "end": np.array([x+instances_block_size*nr_instances_size, y+instances_block_size*nr_instances_size, z+instances_block_size*nr_instances_size])}
                subvolumes.append(subvolume)
    print(f"Generated {len(subvolumes)} subvolumes.")
    return subvolumes

def retrieve_subvolume_patches_list(patches_list, subvolume):
    """
    Retrieve the patches that are in the subvolume.
    """
    subvolume_patches_list = []
    for patch in patches_list:
        # Check if patch is in subvolume
        if np.all(patch["points"] >= subvolume["start"]) and np.all(patch["points"] < subvolume["end"]):
            subvolume_patches_list.append(patch)
    return subvolume_patches_list

def add_overlapp_entries_to_patches_list(patches_list):
    for i in range(len(patches_list)):
        patches_list[i]["overlapp_percentage"] = np.zeros(len(patches_list))
        patches_list[i]["overlap"] = np.zeros(len(patches_list))
        patches_list[i]["non_overlap"] = np.zeros(len(patches_list))
        patches_list[i]["points_overlap"] = [None] * len(patches_list)
        patches_list[i]["scores"] = np.zeros(len(patches_list)) - 1

def compute_overlap_for_pair_(args):
    i, patches_list = args
    result_overlapps = []
    for j in range(i+1, len(patches_list)):
        # Calculate x, y, z start and end coordinates of the overlapping subvolume
        start = np.max(np.array([patches_list[i]["min"], patches_list[j]["min"]]), axis=0)
        end = np.min(np.array([patches_list[i]["max"], patches_list[j]["max"]]), axis=0)

        # Extract the overlapping subvolumes
        subvolume1_points, subvolume1_normals, subvolume1_colors, subvolume1_angles = extract_subvolume(patches_list[i]["points"], patches_list[i]["normals"], patches_list[i]["colors"], patches_list[i]["angles"], start=start, size=end - start)
        subvolume2_points, subvolume2_normals, subvolume2_colors, subvolume2_angles = extract_subvolume(patches_list[j]["points"], patches_list[j]["normals"], patches_list[j]["colors"], patches_list[j]["angles"], start=start, size=end - start)

        # Continue if there are no points in the overlapping subvolume
        if subvolume1_points.shape[0] == 0 or subvolume2_points.shape[0] == 0:
            overlapp_percentage = 0.0
            overlap = 0
            non_overlap = 0
            points_overlap = []
        else:
            # Compute the surface overlap between the two patches
            overlapp_percentage, overlap, non_overlap, points_overlap = surface_overlapp_np(subvolume1_points, subvolume1_angles, subvolume2_points, subvolume2_angles)
            # NP implementation is slower, check for same result
            # overlapp_percentage_np, overlap_np, non_overlap_np = surface_overlapp_np(subvolume1_points, subvolume2_points)
            ### Gives same results
            # assert overlapp_percentage == overlapp_percentage_np, f"Surface overlap percentage is not the same: {overlapp_percentage} != {overlapp_percentage_np}" 

        if overlapp_percentage > 0.7 and False:
            print(f"Surface overlap between patch {i} and patch {j} is {overlapp_percentage}")
        #print(f"Surface overlap between patch {i} and patch {j} is {overlapp_percentage}")

        result_overlapps.append((i, j, overlapp_percentage, overlap, non_overlap, points_overlap))
    return result_overlapps

def overlapping_tiles(patch1, patch2):
    indices1 = patch1["tiles_points"].keys()
    indices2 = patch2["tiles_points"].keys()

    overlapping_indices = []
    for index1 in indices1:
        if patch1["tiles_points"][index1].shape[0] == 0:
            continue
        if index1 in indices2:
            if patch2["tiles_points"][index1].shape[0] == 0:
                continue
            overlapping_indices.append(index1)

    if len(overlapping_indices) == 0:
        return overlapping_indices

    return overlapping_indices

def overlap_mask(patch1, patch2, index):
    # Concatenate points from both point clouds
    if index in patch1["tiles_points"] and index in patch2["tiles_points"]: # both tiles have points
        mask_shape = patch1["tiles_points"][index].shape[0]
        all_points = np.vstack([patch1["tiles_points"][index], patch2["tiles_points"][index]])
    elif index in patch1["tiles_points"]: # only patch1 has points
        mask_shape = patch1["tiles_points"][index].shape[0]
        all_points = patch1["tiles_points"][index]
    elif index in patch2["tiles_points"]: # only patch2 has points
        mask_shape = 0
        all_points = patch2["tiles_points"][index]
    else: # neither patch has points
        return np.zeros(0), np.zeros(0)
    # Find the unique points and their counts
    unique_points, indices, counts = np.unique(all_points, axis=0, return_counts=True, return_index=True)
    # Extract overlapping points from both point clouds
    mask = np.ones(all_points.shape[0], dtype=bool)
    single_count_indices = indices[counts == 1]
    mask[single_count_indices] = False

    # Split the mask into two masks for the two point clouds
    mask1 = mask[:mask_shape]
    mask2 = mask[mask_shape:]

    return mask1, mask2

def compute_overlap_for_pair(args):
    i, patches_list, epsilon, angle_tolerance = args
    result_overlapps = []
    for j in range(i+1, len(patches_list)):
        # Extract indexes of overlapping tiles
        overlap_tiles = overlapping_tiles(patches_list[i], patches_list[j])

        # Calculate overlap mask for each tile both for i and j perspective
        for tile in overlap_tiles:
            overlap_mask_i, overlap_mask_j = overlap_mask(patches_list[i], patches_list[j], tile)
            # print(f"length of tiles overlap mask of tile {tile} for patch {i} is {len(overlap_mask_i)}")
            patches_list[i]["tiles_overlap_mask"][tile][j] = overlap_mask_i
            patches_list[j]["tiles_overlap_mask"][tile][i] = overlap_mask_j        

        # Continue if there are no points in the overlapping tiles
        if len(overlap_tiles) == 0:
            overlapp_percentage = 0.0
            overlap = 0
            non_overlap = 0
            points_overlap = []
            angles_offset = None
        else:
            # Compute the surface overlap between the two patches
            overlapp_percentage, overlap, non_overlap, points_overlap, angles_offset = surface_overlapp_tiles(patches_list, i, j, overlap_tiles, epsilon=epsilon, angle_tolerance=angle_tolerance)

        result_overlapps.append((i, j, overlapp_percentage, overlap, non_overlap, points_overlap, angles_offset))
    return result_overlapps

def compute_subvolume_surfaces_patches_overlapp_multithreaded(patches_list, subvolume_size=50):
    print(f"Computing surface overlap for {len(patches_list)} patches")
    # Computing min and max coordinates as before
    for i in range(len(patches_list)):
        patches_list[i]["max"] = np.max(patches_list[i]["points"], axis=0)
        patches_list[i]["min"] = np.min(patches_list[i]["points"], axis=0)

    # Number of tasks to process
    num_tasks = len(patches_list)
    
    # Setting up multiprocessing
    with Pool(12) as pool:
        results = pool.map(compute_overlap_for_pair, [(i, patches_list) for i in range(num_tasks)])

    # Combining results
    for result in results:
        for i, j, overlapp_percentage, overlap, non_overlap, points_overlap in result:
            patches_list[i]["overlapp_percentage"][j] = overlapp_percentage
            patches_list[i]["overlap"][j] = overlap
            patches_list[i]["non_overlap"][j] = non_overlap
            patches_list[i]["points_overlap"][j] = points_overlap
            patches_list[j]["overlapp_percentage"][i] = overlapp_percentage
            patches_list[j]["overlap"][i] = overlap
            patches_list[j]["non_overlap"][i] = non_overlap
            patches_list[j]["points_overlap"][i] = points_overlap
    
    print(f"Finished computing surface overlap for {len(patches_list)} patches")

def compute_subvolume_surfaces_patches_overlapp(patches_list, subvolume_size=50, overlapp_threshold={"score_threshold": 0.25, "nr_points_min": 20.0, "nr_points_max": 500.0}):
    print(f"Computing surface overlap for {len(patches_list)} patches")
    # Computing min and max coordinates as before
    for i in range(len(patches_list)):
        patches_list[i]["max"] = np.max(patches_list[i]["points"], axis=0)
        patches_list[i]["min"] = np.min(patches_list[i]["points"], axis=0)

    # Number of tasks to process
    num_tasks = len(patches_list)
    
    # # Setting up multiprocessing
    # with Pool(12) as pool:
    #     # results = list(tqdm(pool.imap(compute_overlap_for_pair, [(i, patches_list) for i in range(num_tasks)]), total=num_tasks))
    #     results = pool.map(compute_overlap_for_pair, [(i, patches_list) for i in range(num_tasks)])

    # Single threaded
    results = []
    for i in range(num_tasks):
        results.append(compute_overlap_for_pair((i, patches_list)))

    # Combining results
    for result in results:
        for i, j, overlapp_percentage, overlap, non_overlap, points_overlap in result:
            patches_list[i]["overlapp_percentage"][j] = overlapp_percentage
            patches_list[i]["overlap"][j] = overlap
            patches_list[i]["non_overlap"][j] = non_overlap
            patches_list[i]["points_overlap"][j] = points_overlap
            patches_list[j]["overlapp_percentage"][i] = overlapp_percentage
            patches_list[j]["overlap"][i] = overlap
            patches_list[j]["non_overlap"][i] = non_overlap
            patches_list[j]["points_overlap"][i] = points_overlap
            score = overlapp_score(i, j, patches_list, overlapp_threshold=overlapp_threshold)
            patches_list[i]["scores"][j] = score
            patches_list[j]["scores"][i] = score

    # Build best scores array over all patches pairs
    best_scores = np.zeros(len(patches_list))
    best_index = np.zeros(len(patches_list), dtype=int)
    # Sort and get best scores + indices
    for i in range(len(patches_list)):
        index_max = np.argmax(patches_list[i]["scores"])
        best_index[i] = index_max
        best_scores[i] = patches_list[i]["scores"][index_max]
    print(f"Finished computing surface overlap for {len(patches_list)} patches")
    return best_scores, best_index

def filter_tiles(overlap_angles, non_overlap_agles, tiles, patch):
    overlapp_count = 0
    non_overlapp_count = 0
    for tile in tiles:
        if tile in patch["tiles_points"]: 
            # Angles of tile points
            angles_tile1 = patch["tiles_angles"][tile]
            # Check how many angles of the tile are in the overlap_angles
            angles_overlap1 = np.isin(overlap_angles, angles_tile1)
            # Check how many angles of the tile are in the non_overlap_agles
            angles_non_overlap1 = np.isin(non_overlap_agles, angles_tile1)
            if np.sum(angles_overlap1) > 0:
                overlapp_count += np.sum(angles_overlap1)
                non_overlapp_count += np.sum(angles_non_overlap1)
    
    return overlapp_count, non_overlapp_count

def surface_overlapp_tiles(patches_list, i, j, overlap_tiles, angle_tolerance=90.0, epsilon=1e-5):
    """
    Calculate the surface overlap between two point clouds.
    Points have to be exactly on the same position to be considered the same.
    """
    patch1 = patches_list[i]
    patch2 = patches_list[j]

    # Find the angles of the overlapping points
    overlap1_angles, overlap2_angles = select_overlapping_angles(patches_list, i, j, overlap_tiles)
    # Find the non-overlapping angles
    angles1_non_overlap, angles2_non_overlap = select_non_overlapping_angles(patches_list, i, j, overlap_tiles)

    # Return early if there are no overlapping points
    if len(overlap1_angles) == 0 and len(overlap2_angles) == 0:
        return 0.0, 0, 0, None, None

    assert not ((overlap1_angles is None) or (overlap2_angles is None) or (len(overlap1_angles) == 0) or (len(overlap2_angles) == 0)), f"angles for tiles {overlap_tiles} are wack: {overlap1_angles} {overlap2_angles}"
    # Calculate the angle offsets between the overlapping points
    overlap1_angle_offsets = overlap1_angles - overlap2_angles
    overlap2_angle_offsets = overlap2_angles - overlap1_angles

    # Find the offset with the highest number of occurences in the angle offsets
    # Prefilter angles for winding number (different offsets), select max overlap offset
    angles_offset1 = largest_group_offset(overlap1_angle_offsets, epsilon=epsilon)
    
    # Find winding number k with the most overlapping points
    angle1_ranges, angle2_ranges, mean_angle2_adjusted = find_patch_angle_range(angles_offset1, patch2["normals"], angle_tolerance=angle_tolerance)

    # # Ranges of angles in the first point cloud
    # angle1_ranges = compute_angle_range(overlap1_angles[filter1], angle_tolerance)
    # angle2_ranges = compute_angle_range(overlap2_angles[filter2], angle_tolerance)

    # Filter overlapping points in the first point cloud by angles within range
    filtered_overlap1_mask = filter_by_angle(overlap1_angles, angle1_ranges)
    # Filter overlapping points in the second point cloud by angles within range
    filtered_overlap2_mask = filter_by_angle(overlap2_angles, angle2_ranges)

    # Filter non-overlapping points in the first point cloud by angles within range of the second point cloud angles (+- angle_tolerance)
    filtered_non_overlap1_mask = filter_by_angle(angles1_non_overlap, angle1_ranges)

    # Filter non-overlapping points in the second point cloud by angles within range of the first point cloud angles (+- angle_tolerance)
    filtered_non_overlap2_mask = filter_by_angle(angles2_non_overlap, angle2_ranges)

    overlap1, non_overlap1 = filter_tiles(overlap1_angles[filtered_overlap1_mask], angles1_non_overlap[filtered_non_overlap1_mask], overlap_tiles, patch1)
    overlap2, non_overlap2 = filter_tiles(overlap2_angles[filtered_overlap2_mask], angles2_non_overlap[filtered_non_overlap2_mask], overlap_tiles, patch2)
    
    # Calculate the overlap percentage
    overlapp_percentage1 = 1.0 * overlap1 / (overlap1 + non_overlap1 + 0.00001)
    overlapp_percentage2 = 1.0 * overlap2 / (overlap2 + non_overlap2 + 0.00001)

    # Take the larger overlap percentage
    if overlapp_percentage1 < overlapp_percentage2:
        overlap = overlap1
        overlapp_percentage = overlapp_percentage1
        non_overlap = non_overlap1
    else:
        overlap = overlap2
        overlapp_percentage = overlapp_percentage2
        non_overlap = non_overlap2
    
    unique_points = None # Not needed anymore

    return overlapp_percentage, overlap, non_overlap, unique_points, mean_angle2_adjusted

def find_patch_angle_range(offset, patch_normals2, angle_tolerance=90):
    mean_angle2 = 0.0 # angles of patch2 already zeroed when building the patch
    filter_angle_range2 = (mean_angle2 - angle_tolerance, mean_angle2 + angle_tolerance)

    mean_angle2_adjusted = mean_angle2
    mean_angle2_adjusted += offset
    filter_angle_range1 = (mean_angle2_adjusted - angle_tolerance, mean_angle2_adjusted + angle_tolerance)
    
    return [filter_angle_range1], [filter_angle_range2], mean_angle2_adjusted

# Scho widder so es Monster ...
def surface_overlapp_np(points1, angles1, points2, angles2, angle_tolerance=30.0, epsilon=1e-5):
    """
    Calculate the surface overlap between two point clouds.
    Points have to be exactly on the same position to be considered the same.
    """
    points1, angles1 = remove_duplicate_points_normals(points1, angles1)
    points2, angles2 = remove_duplicate_points_normals(points2, angles2)

    sort_indices1 = np.lexsort(points1.T)
    points1 = points1[sort_indices1]

    sort_indices2 = np.lexsort(points2.T)
    points2 = points2[sort_indices2]

    # Concatenate points from both point clouds
    all_points = np.vstack([points1, points2])

    # Find the unique points and their counts
    unique_points, indices, counts = np.unique(all_points, axis=0, return_counts=True, return_index=True)

    # Extract overlapping points from both point clouds
    mask = np.ones(all_points.shape[0], dtype=bool)
    for i in range(unique_points.shape[0]):
        if counts[i] == 1:
            mask[indices[i]] = False

    # Return early if there are no overlapping points
    if np.all(~mask):
        return 0.0, 0, 0, np.zeros((0, 3))

    # Split the mask into two masks for the two point clouds
    mask1 = mask[:points1.shape[0]]
    mask2 = mask[points1.shape[0]:]

    # Find the overlapping points in the first point cloud
    overlap1 = points1[mask1]
    # Find the overlapping points in the second point cloud
    overlap2 = points2[mask2]

    # Find the non-overlapping points in the first point cloud
    mask1_non_overlap = np.logical_not(mask1)
    mask2_non_overlap = np.logical_not(mask2)

    # Find the angles of the overlapping points
    overlap1_angles = angles1[mask1]
    overlap2_angles = angles2[mask2]
    # Find the non-overlapping angles
    angles1_non_overlap = angles1[mask1_non_overlap]
    angles2_non_overlap = angles2[mask2_non_overlap]
    # Calculate the angle offsets between the overlapping points
    overlap1_angle_offsets = overlap1_angles - overlap2_angles
    overlap2_angle_offsets = overlap2_angles - overlap1_angles

    # Find the offset with the highest number of occurences in the angle offsets
    # Prefilter angles for winding number (different offsets), select max overlap offset
    angles_offset1 = largest_group_offset(overlap1_angle_offsets, epsilon=epsilon)
    angles_offset2 = largest_group_offset(overlap2_angle_offsets, epsilon=epsilon)

    # TODO: filter overlapping points by selected offset
    filter1 = np.logical_and(overlap1_angle_offsets >= angles_offset1 - epsilon, overlap1_angle_offsets <= angles_offset1 + epsilon)
    filter2 = np.logical_and(overlap2_angle_offsets >= angles_offset2 - epsilon, overlap2_angle_offsets <= angles_offset2 + epsilon)

    # Return early if there are no overlapping points
    if np.all(~filter1) and np.all(~filter2):
        return 0.0, 0, 0, np.zeros((0, 3))
    
    # Filtered overlapping points
    filtered_overlap1_points = overlap1[filter1]
    filtered_overlap2_points = overlap2[filter2]

    # Ranges of angles in the first point cloud
    angle1_ranges = compute_angle_range(overlap1_angles, angle_tolerance)
    angle2_ranges = compute_angle_range(overlap2_angles, angle_tolerance)

    # Non-overlapping points in the first point cloud
    non_overlap1_points = points1[mask1_non_overlap]
    # Non-overlapping points in the second point cloud
    non_overlap2_points = points2[mask2_non_overlap]

    # Filter non-overlapping points in the first point cloud by angles within range of the second point cloud angles (+- angle_tolerance)
    filtered_non_overlap1_points = filter_points_by_angle(non_overlap1_points, angles1_non_overlap+angles_offset2, angle2_ranges)
    # Filter non-overlapping points in the second point cloud by angles within range of the first point cloud angles (+- angle_tolerance)
    filtered_non_overlap2_points = filter_points_by_angle(non_overlap2_points, angles2_non_overlap+angles_offset1, angle1_ranges)

    # Calculate the overlap
    overlap1 = filtered_overlap1_points.shape[0] 
    overlap2 = filtered_overlap2_points.shape[0]

    non_overlap1 = filtered_non_overlap1_points.shape[0]
    non_overlap2 = filtered_non_overlap2_points.shape[0]

    # Calculate the overlap percentage
    overlapp_percentage1 = 1.0 * overlap1 / (overlap1 + non_overlap1 + 0.00001)
    overlapp_percentage2 = 1.0 * overlap2 / (overlap2 + non_overlap2 + 0.00001)

    # Take the larger overlap percentage
    if overlapp_percentage1 > overlapp_percentage2:
        overlap = overlap1
        overlapp_percentage = overlapp_percentage1
        non_overlap = non_overlap1
    else:
        overlap = overlap2
        overlapp_percentage = overlapp_percentage2
        non_overlap = non_overlap2
    
    return overlapp_percentage, overlap, non_overlap, unique_points

# En Koloss von ere funktion...
def repopulate_overlap_stats_(i, j, patches_list, iteration):
    """
    combining two surfaces into one surface,
    recompute the patches list when combining surfaces
    """
    # Extract the overlapping subvolumes
    patch_i = patches_list[i]["subvolume"]
    patch_j = patches_list[j]["subvolume"]

    # Combine Patches
    # Find the angles of the overlapping points
    angles_i, angles_j = select_angles_from_subsample(
        patches_list[i]["points"], patches_list[i]["angles"], 
        patches_list[j]["points"], patches_list[j]["angles"], 
        patches_list[i]["points_overlap"][j])
    # Find the angles offsets
    angles_offset = angles_i - angles_j
    # Unique angles
    def angles_almost_equal(arr, epsilon=1e-6): # numerical stability
        return np.all(np.abs(arr - arr[0]) < epsilon)
    unique_angles = np.unique(angles_offset)
    assert angles_almost_equal(unique_angles), f"Unique angles should be almost equal, but is {unique_angles}"
    assert len(unique_angles) > 0, f"Unique angles should be of length greater than 0, but is {len(unique_angles)}"
    offset = np.mean(unique_angles)
    # # Update the anchor points, normals and angles
    patches_list[i]["anchor_points"] += patches_list[j]["anchor_points"]
    patches_list[i]["anchor_normals"] += patches_list[j]["anchor_normals"]
    patches_list[i]["anchor_angles"] += [adjust_angles_offset(angle, offset) for angle in patches_list[j]["anchor_angles"]]
    # Update the angles
    patches_list[j]["angles"] = adjust_angles_offset(patches_list[j]["angles"], offset)
    # Add the j angles to the i angles
    patches_list[i]["angles"] = np.concatenate([patches_list[i]["angles"], patches_list[j]["angles"]], axis=0)

    # Add the points and normals and colors of the second surface to the first surface
    patches_list[i]["points"] = np.concatenate([patches_list[i]["points"], patches_list[j]["points"]], axis=0)
    patches_list[i]["normals"] = np.concatenate([patches_list[i]["normals"], patches_list[j]["normals"]], axis=0)
    patches_list[i]["colors"] = np.concatenate([patches_list[i]["colors"], patches_list[j]["colors"]], axis=0)

    # Get the lexicographical sort indices of the combined points
    sort_indices = np.lexsort(patches_list[i]["points"].T)

    # Rearrange all the arrays based on the lexicographical order
    patches_list[i]["angles"] = patches_list[i]["angles"][sort_indices]
    patches_list[i]["points"] = patches_list[i]["points"][sort_indices]
    patches_list[i]["normals"] = patches_list[i]["normals"][sort_indices]
    patches_list[i]["colors"] = patches_list[i]["colors"][sort_indices]

    # Remove duplicate points from points and normals and colors and angles
    patches_list[i]["points"], indices = np.unique(patches_list[i]["points"], axis=0, return_index=True)
    patches_list[i]["normals"] = patches_list[i]["normals"][indices]
    patches_list[i]["colors"] = patches_list[i]["colors"][indices]
    patches_list[i]["angles"] = patches_list[i]["angles"][indices]

    # Add the original points and normals of the second surface to the first surface
    patches_list[i]["ids"] = patches_list[i]["ids"] + patches_list[j]["ids"]

    # Add the subvolume of the second surface to the first surface
    patches_list[i]["subvolume"] = patches_list[i]["subvolume"] + patches_list[j]["subvolume"]
    # Add the subvolume size of the second surface to the first surface
    patches_list[i]["subvolume_size"] = patches_list[i]["subvolume_size"] + patches_list[j]["subvolume_size"]
    # Update the iteration of the first surface
    patches_list[i]["iteration"] = iteration

    # Remove the second surface from the list
    patches_list.pop(j)

    # Adjust the overlap stats (remove jth patch)
    for k in range(len(patches_list)):
        # Adjust the overlap stats by removing the jth patch
        patches_list[k]["overlapp_percentage"] = np.delete(patches_list[k]["overlapp_percentage"], j)
        patches_list[k]["overlap"] = np.delete(patches_list[k]["overlap"], j)
        patches_list[k]["non_overlap"] = np.delete(patches_list[k]["non_overlap"], j)
        # List index delete patches_list[k]["points_overlap"], j
        patches_list[k]["points_overlap"].pop(j)
        
    patches_list[i]["overlapp_percentage"] = np.zeros(len(patches_list))
    patches_list[i]["overlap"] = np.zeros(len(patches_list))
    patches_list[i]["non_overlap"] = np.zeros(len(patches_list))
    patches_list[i]["points_overlap"] = [None] * len(patches_list)

    best_percentage = 0.0

    # Recompute the surface overlap between all surfaces and the combined surface
    for k in range(len(patches_list)):
        if k == i:
            continue

        # Convert list of subvolumes and their sizes to NumPy arrays for i and k
        subvolumes_i = np.array(patches_list[i]["subvolume"])
        subvolumes_k = np.array(patches_list[k]["subvolume"])

        if isinstance(patches_list[i]["subvolume_size"], int):
            patches_list[i]["subvolume_size"] = [np.array([patches_list[i]["subvolume_size"], patches_list[i]["subvolume_size"], patches_list[i]["subvolume_size"]])]

        if isinstance(patches_list[k]["subvolume_size"], int):
            patches_list[k]["subvolume_size"] = [np.array([patches_list[k]["subvolume_size"], patches_list[k]["subvolume_size"], patches_list[k]["subvolume_size"]])]

        sizes_i = np.array([s_i if isinstance(s_i, np.ndarray) else np.array([s_i, s_i, s_i]) for s_i in patches_list[i]["subvolume_size"]])
        sizes_k = np.array([s_k if isinstance(s_k, np.ndarray) else np.array([s_k, s_k, s_k]) for s_k in patches_list[k]["subvolume_size"]])

        # Make sure the last dimension is 3
        if sizes_i.shape[-1] == 1:
            sizes_i = np.repeat(sizes_i, 3, axis=-1)

        if sizes_k.shape[-1] == 1:
            sizes_k = np.repeat(sizes_k, 3, axis=-1)
            
        if len(sizes_i.shape) == 1:
            sizes_i = np.expand_dims(sizes_i, axis=0)
            
        if len(sizes_k.shape) == 1:
            sizes_k = np.expand_dims(sizes_k, axis=0)

        # Calculate starts and ends for all combinations
        start_all = np.maximum(subvolumes_i[:, None], subvolumes_k)
        end_all = np.minimum(subvolumes_i[:, None] + sizes_i[:, None], subvolumes_k + sizes_k)

        # Calculate sizes for subvolumes (end - start)
        sizes_all = end_all - start_all

        overlap_starts = start_all.reshape(-1, 3)
        overlap_sizes = sizes_all.reshape(-1, 3)

        subvolume1_points, subvolume1_normals, subvolume1_colors, subvolume1_angles = extract_subvolume(patches_list[i]["points"], patches_list[i]["normals"], patches_list[i]["colors"], patches_list[i]["angles"], start=overlap_starts, size=overlap_sizes)
        subvolume2_points, subvolume2_normals, subvolume2_colors, subvolume2_angles = extract_subvolume(patches_list[k]["points"], patches_list[k]["normals"], patches_list[k]["colors"], patches_list[k]["angles"], start=overlap_starts, size=overlap_sizes)

        # There are no points in the overlapping subvolume
        if subvolume1_points.shape[0] == 0 or subvolume2_points.shape[0] == 0:
            overlapp_percentage = 0.0
            overlap = 0
            non_overlap = 0
            points_overlap = []
        else:
            # Compute the surface overlap between the two patches
            overlapp_percentage, overlap, non_overlap, points_overlap = surface_overlapp_np(subvolume1_points, subvolume1_angles, subvolume2_points, subvolume2_angles)

        if overlapp_percentage > best_percentage and overlap > 20:
            best_percentage = overlapp_percentage
        
        # Add the surface overlap to the list
        patches_list[i]["overlapp_percentage"][k] = overlapp_percentage
        patches_list[i]["overlap"][k] = overlap
        patches_list[i]["non_overlap"][k] = non_overlap
        patches_list[i]["points_overlap"][k] = points_overlap
        patches_list[k]["overlapp_percentage"][i] = overlapp_percentage
        patches_list[k]["overlap"][i] = overlap
        patches_list[k]["non_overlap"][i] = non_overlap
        patches_list[k]["points_overlap"][i] = points_overlap


    return patches_list

# En Koloss von ere funktion...
def repopulate_overlap_stats(i, j, patches_list, best_scores, best_index, iteration, overlapp_threshold={"score_threshold": 0.25, "nr_points_min": 20.0, "nr_points_max": 500.0}, return_angle_offset=False):
    """
    combining two surfaces into one surface,
    recompute the patches list when combining surfaces
    """
    # Combine Patches
    tiles = overlapping_tiles(patches_list[i], patches_list[j])
    if len(tiles) == 0: #
        print("Warning: No overlapping tiles found")
        res = [patches_list, best_scores, best_index]
        if return_angle_offset:
            offset = None
            res.append(offset)
        return tuple(res)
    # Find the angles of the overlapping points
    angles_i, angles_j = select_overlapping_angles(patches_list, i, j, tiles)
    # Find the angles offsets
    angles_offset = angles_i - angles_j
    offset = largest_group_offset(angles_offset, epsilon=overlapp_threshold["epsilon"])

    # # Update the anchor points, normals and angles
    patches_list[i]["anchor_points"] += patches_list[j]["anchor_points"]
    patches_list[i]["anchor_normals"] += patches_list[j]["anchor_normals"]
    patches_list[i]["anchor_angles"] += [adjust_angles_offset(angle, offset) for angle in patches_list[j]["anchor_angles"]]

    tiles2 = patches_list[j]["tiles"]
    for tile in tiles2:
        if patches_list[j]["tiles_points"][tile].shape[0] == 0: # skip empty second tiles
            continue
        # Combine points, normals, colors, angles and masks
        # Case 1: Tile is not in tiles of patch i
        if tile not in patches_list[i]["tiles"] or patches_list[i]["tiles_points"][tile].shape[0] == 0:
            patches_list[i]["tiles"][tile] = True
            patches_list[i]["tiles_points"][tile] = patches_list[j]["tiles_points"][tile]
            patches_list[i]["tiles_normals"][tile] = patches_list[j]["tiles_normals"][tile]
            patches_list[i]["tiles_normals_mean"][tile] = patches_list[j]["tiles_normals_mean"][tile]
            patches_list[i]["tiles_angles"][tile] = adjust_angles_offset(patches_list[j]["tiles_angles"][tile], offset)
            patches_list[i]["tiles_colors"][tile] = patches_list[j]["tiles_colors"][tile]
            patches_list[i]["tiles_overlap_mask"][tile] = patches_list[j]["tiles_overlap_mask"][tile]
        else:
            # Only add the non-overlapping points, normals and angles
            masks_i = patches_list[i]["tiles_overlap_mask"][tile]
            masks_j = patches_list[j]["tiles_overlap_mask"][tile]

            # None overlapping mask j
            non_overlapping_mask_j = masks_j[i]
            non_overlapping_mask_j = np.logical_not(non_overlapping_mask_j)

            if np.sum(non_overlapping_mask_j) == 0:
                continue

            # Combine the overlapping masks
            masks_j_overlapping_without_overlapping_to_i = masks_j[:, non_overlapping_mask_j]
            combined_masks_overlapping_maks = np.concatenate([masks_i, masks_j_overlapping_without_overlapping_to_i], axis=1)

            # Add the non-overlapping points, normals, colors and angles
            len_i_points = patches_list[i]["tiles_points"][tile].shape[0]
            len_j_points = patches_list[j]["tiles_points"][tile][non_overlapping_mask_j].shape[0]
            patches_list[i]["tiles_normals_mean"][tile] = (patches_list[i]["tiles_normals_mean"][tile] * len_i_points + patches_list[j]["tiles_normals_mean"][tile] * len_j_points) / (len_i_points + len_j_points)
            patches_list[i]["tiles_points"][tile] = np.concatenate([patches_list[i]["tiles_points"][tile], patches_list[j]["tiles_points"][tile][non_overlapping_mask_j]], axis=0)
            patches_list[i]["tiles_colors"][tile] = np.concatenate([patches_list[i]["tiles_colors"][tile], patches_list[j]["tiles_colors"][tile][non_overlapping_mask_j]], axis=0)
            patches_list[i]["tiles_normals"][tile] = np.concatenate([patches_list[i]["tiles_normals"][tile], patches_list[j]["tiles_normals"][tile][non_overlapping_mask_j]], axis=0)
            patches_list[i]["tiles_angles"][tile] = np.concatenate([patches_list[i]["tiles_angles"][tile], adjust_angles_offset(patches_list[j]["tiles_angles"][tile][non_overlapping_mask_j], offset)], axis=0)

            # Add the combined mask
            patches_list[i]["tiles_overlap_mask"][tile] = combined_masks_overlapping_maks

            # Sort the points lexicographically
            sort_indices = np.lexsort(patches_list[i]["tiles_points"][tile].T)

            # Rearrange all the arrays based on the lexicographical order
            patches_list[i]["tiles_points"][tile] = patches_list[i]["tiles_points"][tile][sort_indices]
            patches_list[i]["tiles_normals"][tile] = patches_list[i]["tiles_normals"][tile][sort_indices]
            patches_list[i]["tiles_colors"][tile] = patches_list[i]["tiles_colors"][tile][sort_indices]
            patches_list[i]["tiles_angles"][tile] = patches_list[i]["tiles_angles"][tile][sort_indices]
            patches_list[i]["tiles_overlap_mask"][tile] = patches_list[i]["tiles_overlap_mask"][tile][:, sort_indices]

    # Add the original points and normals of the second surface to the first surface
    patches_list[i]["ids"] = patches_list[i]["ids"] + patches_list[j]["ids"]

    # Add the subvolume of the second surface to the first surface
    patches_list[i]["subvolume"] = patches_list[i]["subvolume"] + patches_list[j]["subvolume"]
    # Add the subvolume size of the second surface to the first surface
    patches_list[i]["subvolume_size"] = patches_list[i]["subvolume_size"] + patches_list[j]["subvolume_size"]
    # Update the iteration of the first surface
    patches_list[i]["iteration"] = iteration
    # Update the patch prediction score
    patches_list[i]["patch_prediction_scores"] += patches_list[j]["patch_prediction_scores"]
    # Update the patch prediction distance
    patches_list[i]["patch_prediction_distances"] += patches_list[j]["patch_prediction_distances"]

    # Combine i and j overlap mask for all other patches
    for k in range(len(patches_list)):
        if k == i:
            continue
        for tile in patches_list[k]["tiles"]:
            mask_i = patches_list[k]["tiles_overlap_mask"][tile][i]
            mask_j = patches_list[k]["tiles_overlap_mask"][tile][j]
            # Combine the masks
            combined_mask = np.logical_or(mask_i, mask_j)
            # Add the combined mask
            patches_list[k]["tiles_overlap_mask"][tile][i] = combined_mask

    # Remove the second surface from the list
    patches_list.pop(j)

    # Delete the jth entry from the scores
    for k in range(len(patches_list)):
        patches_list[k]["scores"] = np.delete(patches_list[k]["scores"], j)
    best_scores = np.delete(best_scores, j)
    best_index = np.delete(best_index, j)

    # Adjust the overlap stats (remove jth patch). Adjust the masks, by deleting the jth entry
    for k in range(len(patches_list)):
        # Adjust the overlap stats by removing the jth patch
        patches_list[k]["overlapp_percentage"] = np.delete(patches_list[k]["overlapp_percentage"], j)
        patches_list[k]["overlap"] = np.delete(patches_list[k]["overlap"], j)
        patches_list[k]["non_overlap"] = np.delete(patches_list[k]["non_overlap"], j)
        # List index delete patches_list[k]["points_overlap"], j
        patches_list[k]["points_overlap"].pop(j)
        # Adjust the masks
        for tile in patches_list[k]["tiles"]:
            patches_list[k]["tiles_overlap_mask"][tile] = np.delete(patches_list[k]["tiles_overlap_mask"][tile], j, axis=0)
        
    patches_list[i]["overlapp_percentage"] = np.zeros(len(patches_list))
    patches_list[i]["overlap"] = np.zeros(len(patches_list))
    patches_list[i]["non_overlap"] = np.zeros(len(patches_list))
    patches_list[i]["points_overlap"] = [None] * len(patches_list)

    best_percentage = 0.0

    # Recompute the surface overlap between all surfaces and the combined surface
    for k in range(len(patches_list)):
        if k == i:
            continue

        overlap_tiles = overlapping_tiles(patches_list[i], patches_list[k])

        # There are no points in the overlapping subvolume
        if len(overlap_tiles) == 0:
            overlapp_percentage = 0.0
            overlap = 0
            non_overlap = 0
            points_overlap = []
        else:
            # Compute the surface overlap between the two patches
            overlapp_percentage, overlap, non_overlap, points_overlap = surface_overlapp_tiles(patches_list, i, k, overlap_tiles)

        if overlapp_percentage > best_percentage and overlap > 20:
            best_percentage = overlapp_percentage

        # Add the surface overlap to the list
        patches_list[i]["overlapp_percentage"][k] = overlapp_percentage
        patches_list[i]["overlap"][k] = overlap
        patches_list[i]["non_overlap"][k] = non_overlap
        patches_list[i]["points_overlap"][k] = points_overlap
        patches_list[i]["scores"][k] = overlapp_score(i, k, patches_list, overlapp_threshold=overlapp_threshold)
        patches_list[k]["overlapp_percentage"][i] = overlapp_percentage
        patches_list[k]["overlap"][i] = overlap
        patches_list[k]["non_overlap"][i] = non_overlap
        patches_list[k]["points_overlap"][i] = points_overlap
        patches_list[k]["scores"][i] = overlapp_score(k, i, patches_list, overlapp_threshold=overlapp_threshold)
        # Adjust the best index from deleting the jth patch
        if best_index[k] > j:
            best_index[k] -= 1
        elif best_index[k] == j:
            best_index[k] = np.argmax(patches_list[k]["scores"])
            best_scores[k] = patches_list[k]["scores"][best_index[k]]
        # Calculate the best score if the score is better than the current best score or the best index is the ith patch
        if (best_index[k] == i) or (patches_list[k]["scores"][i] > best_scores[k]):
            best_scores[k] = patches_list[k]["scores"][i]
            best_index[k] = i

    best_index[i] = np.argmax(patches_list[i]["scores"])
    best_scores[i] = patches_list[i]["scores"][best_index[i]]

    res = [patches_list, best_scores, best_index]
    if return_angle_offset:
        res.append(offset)
    return tuple(res)

def overlapp_score(i, j, patches_list, overlapp_threshold={"score_threshold": 0.25, "nr_points_min": 20.0, "nr_points_max": 500.0}, sample_ratio=0.1):
    overlapp_percentage = patches_list[i]["overlapp_percentage"][j]
    overlap = patches_list[i]["overlap"][j] / sample_ratio # Adjust for calculation based on total points

    nrp = min(max(overlapp_threshold["nr_points_min"], overlap), overlapp_threshold["nr_points_max"]) / (overlapp_threshold["nr_points_min"])
    nrp_factor = np.log(np.log(nrp) + 1.0) + 1.0

    score = overlapp_percentage

    if score >= overlapp_threshold["score_threshold"] and overlap >= overlapp_threshold["nr_points_min"] and patches_list[i]["overlap"][j] > 0:
        score = ((score - overlapp_threshold["score_threshold"])/ (1 - overlapp_threshold["score_threshold"])) ** 2
        return score * nrp_factor
    else:
        return -1.0

def combine_subvolume_surfaces(patches_list, best_scores, best_index, iteration, overlapp_threshold={"score_threshold": 0.25, "nr_points_min": 20.0, "nr_points_max": 500.0}, visualize=True):
    print("Combining surfaces ...")
    # Visualization
    # Setup the plot before the loop
    if visualize:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()

    # combine surfaces until no more surfaces can be combined
    idx = -1
    while len(patches_list) > 1:
        idx += 1
        if False and iteration > 0:
            print(f"Number of surfaces: {len(patches_list)}")
        # Find the two surfaces with the highest overlap
        if False and iteration > 0:
            print("Compute Scores")
        
        max_overlap_i = np.argmax(best_scores)
        max_overlap_j = best_index[max_overlap_i]
        max_overlap = best_scores[max_overlap_i]

        # Swap the indices if the second index is smaller than the first index
        if max_overlap_j < max_overlap_i:
            max_overlap_i, max_overlap_j = max_overlap_j, max_overlap_i
            max_overlap = best_scores[max_overlap_i]

        # If no surfaces overlap enough, stop combining surfaces
        if max_overlap < 0:
            break

        # Combine the two surfaces, remove the second surface from the list and recompute the overlap stats
        patches_list, best_scores, best_index = repopulate_overlap_stats(max_overlap_i, max_overlap_j, patches_list, best_scores, best_index, iteration, overlapp_threshold=overlapp_threshold)

        if visualize:
            # Visualize sizes
            patches_sizes = [len(patches_list[i]["points"]) for i in range(len(patches_list))]
            bins = np.linspace(0, max(patches_sizes), 30)
            ax.clear()  # Clear the previous histogram
            ax.hist(patches_sizes, bins=bins, alpha=0.75)  # Plot new histogram
            ax.set_title(f"Iteration: {idx}")
            plt.draw()  # Draw the updated plot
            plt.pause(0.1)  # Pause for a brief moment

    if visualize:
        plt.ioff()  # Turn off interactive mode
        plt.show()

    print("Finished combining surfaces")

    return patches_list


def assign_points_to_tiles(patches_list, subvolume, tiling=2):
    start = subvolume['start']
    end = subvolume['end']

    # Calculate tile size and position
    tiles_per_axis = 2 * tiling - 1
    tiles_per_axis = 1 + tiling
    tile_size = (end - start) / tiles_per_axis

    # Assign points to tiles
    for patch in patches_list:
        # Calculate tile indices
        tile_indices = np.floor((patch["points"] - start) // tile_size).astype(int)
        # assert all positive indices
        assert np.all(tile_indices >= 0), f"Tile indices should be positive, but is {tile_indices}"
        # for each unique index, add the points to the corresponding tile
        unique_indices = np.unique(tile_indices, axis=0)
        patch["tiles"] = {}
        patch["tiles_overlap_mask"] = {}
        for tile_x in range(tiles_per_axis):
            for tile_y in range(tiles_per_axis):
                for tile_z in range(tiles_per_axis):
                    index = (tile_x, tile_y, tile_z)
                    patch["tiles_overlap_mask"][index] = np.zeros((len(patches_list), 0), dtype=bool)
        patch["tiles_points"] = {}
        patch["tiles_colors"] = {}
        patch["tiles_normals"] = {}
        patch["tiles_normals_mean"] = {}
        patch["tiles_angles"] = {}
        indices_full = []
        for index in unique_indices:
            index = tuple(index)
            indices_full.append(index)
            indices = np.all(tile_indices == index, axis=1)
            patch["tiles"][index] = True
            patch["tiles_overlap_mask"][index] = np.zeros((len(patches_list), patch["points"][indices].shape[0]), dtype=bool)

            patch["tiles_points"][index] = patch["points"][indices]
            patch["tiles_colors"][index] = patch["colors"][indices]
            patch["tiles_normals"][index] = patch["normals"][indices]
            patch["tiles_angles"][index] = patch["angles"][indices]

            # sort the points in the tile by their lexicographical order
            sort_indices1 = np.lexsort(patch["tiles_points"][index].T)
            patch["tiles_points"][index] = patch["tiles_points"][index][sort_indices1]
            patch["tiles_colors"][index] = patch["tiles_colors"][index][sort_indices1]
            patch["tiles_normals"][index] = patch["tiles_normals"][index][sort_indices1]
            patch["tiles_angles"][index] = patch["tiles_angles"][index][sort_indices1]
            patch["tiles_normals_mean"][index] = np.mean(patch["tiles_normals"][index], axis=0)

def assign_points_from_tiles(patches_list):
    for patch in patches_list:
        patch["points"] = np.concatenate([patch["tiles_points"][index] for index in patch["tiles_points"]], axis=0)
        patch["normals"] = np.concatenate([patch["tiles_normals"][index] for index in patch["tiles_normals"]], axis=0)
        patch["colors"] = np.concatenate([patch["tiles_colors"][index] for index in patch["tiles_colors"]], axis=0)
        patch["angles"] = np.concatenate([patch["tiles_angles"][index] for index in patch["tiles_angles"]], axis=0)

def sheet_reconstruction(args):
    """
    Divide and conquer sheet stitching algorithm.
    """
    subvolume_patches_list, subvolume, iteration, min_sheet_size, sample_ratio, overlapp_threshold = args
    # print(f"Subvolume: ?")

    subvolume_patches_list = select_largest_n_patches(subvolume_patches_list, n=int(500))

    # Add the overlap base to the patches list that contains the points + normals + scores only before
    add_overlapp_entries_to_patches_list(subvolume_patches_list)

     # Assign points to tiles
    assign_points_to_tiles(subvolume_patches_list, subvolume, tiling=5)

    # Compute the surface overlap between all surface patches
    best_scores, best_index = compute_subvolume_surfaces_patches_overlapp(subvolume_patches_list, overlapp_threshold=overlapp_threshold)

    # Combine subvolume surfaces
    combine_subvolume_surfaces(subvolume_patches_list, best_scores, best_index, iteration, overlapp_threshold=overlapp_threshold, visualize=False)
    
    # Assign points from tiles
    assign_points_from_tiles(subvolume_patches_list)

    # Filter out old patches and save them, if they are large enough
    subvolume_patches_list_filtered, removed_patches_to_save = filter_patches_list(subvolume_patches_list, iteration, min_points=min_sheet_size*sample_ratio)

    # Filter out a subset of patches
    subvolume_patches_list_filtered = select_largest_n_patches(subvolume_patches_list_filtered, n=int(150))

    return subvolume_patches_list_filtered, removed_patches_to_save

def process_subvolume(args):
    subvolume, patches_list = args

    # Retrieve subvolume_patches_list
    subvolume_patches_list = retrieve_subvolume_patches_list(patches_list, subvolume)
    
    # Make workable copy of the patches list
    subvolume_patches_list = deepcopy(subvolume_patches_list)
    
    return subvolume_patches_list

def surrounding_volumes_main_sheet(volume_id, volume_size=50, window_size=4):
    """
    Returns the surrounding volumes of a volume
    """
    volume_id_x = volume_id[0]
    volume_id_y = volume_id[1]
    volume_id_z = volume_id[2]
    vs = (volume_size//2)
    surrounding_volumes = [None]*((2*window_size+1)**3)
    count = 0
    for x in range(-window_size, window_size+1):
        vs_x = vs * x
        vi_x = volume_id_x + vs_x
        for y in range(-window_size, window_size+1):
            vs_y = vs * y
            vi_y = volume_id_y + vs_y
            for z in range(-window_size, window_size+1):
                vs_z = vs * z
                vi_z = volume_id_z + vs_z
                surrounding_volumes[count] = (vi_x, vi_y, vi_z)
                count += 1
    return surrounding_volumes

def surrounding_volumes(volume_id, volume_size=50):
    """
    Returns the surrounding volumes of a volume
    """
    volume_id_x = volume_id[0]
    volume_id_y = volume_id[1]
    volume_id_z = volume_id[2]
    vs = (volume_size//2)
    surrounding_volumes = [None]*(3**3)
    count = 0
    for x in range(-1, 2):
        vs_x = vs * x
        vi_x = volume_id_x + vs_x
        for y in range(-1, 2):
            vs_y = vs * y
            vi_y = volume_id_y + vs_y
            for z in range(-1, 2):
                vs_z = vs * z
                vi_z = volume_id_z + vs_z
                surrounding_volumes[count] = (vi_x, vi_y, vi_z)
                count += 1
    return surrounding_volumes

def start_distance_score_factor(current_volume, angle, volume_start, volume_size=50):
    angle_ratio = 10000.0 / 360.0
    angle_distance = angle_ratio * (abs(angle) % 360)
    z_distance = abs(current_volume[1] - volume_start[1])
    distance = angle_distance + z_distance
    max_dist_inv = max(0.0, 10000.0 - distance)
    distance_score_factor =  0.25 + 0.75 * max_dist_inv / 8000.0
    return distance_score_factor

def pick_next_patches(volume_blocks_scores, volume_start, overlapp_threshold, max_patches=1, volume_size=50, sheet_angle_range=None, sheet_z_range=None):
    len_volume_blocks = len(volume_blocks_scores)
    all_scores = [None] * len_volume_blocks
    highest_scores = []

    accumulate_nr_patches = max_patches * 100
    volume_id_scale_factor = (200.0/volume_size)

    count = 0
    # Pick the highest scoring patches that hasnt a used volume in its surrounding
    for volume_id in volume_blocks_scores:
        curr_volume = volume_blocks_scores[volume_id]
        if (not sheet_z_range is None) and (volume_id[1]*volume_id_scale_factor < sheet_z_range[0] or volume_id[1]*volume_id_scale_factor > sheet_z_range[1]):
            continue
        for patch_id in curr_volume:
            curr_patch = curr_volume[patch_id]
            score = curr_patch["score"]
            offset_angle = curr_patch["offset_angle"]
            switched_winding = curr_patch["switched_winding"]
            if score < 0.0: # skip bad scores
                continue
            curr_patch_angle = curr_patch["angle"]
            if (not sheet_angle_range is None) and (curr_patch_angle < sheet_angle_range[0] or curr_patch_angle > sheet_angle_range[1]):
                continue

            if len(all_scores) <= count:
                all_scores += [None] * len_volume_blocks

            all_scores[count] = (None, (volume_id, patch_id, offset_angle, switched_winding), score, curr_patch_angle)
            count += 1

    all_scores = all_scores[:count]
    # Only pick scores that are at least final_score_min
    if "final_score_min" in overlapp_threshold and overlapp_threshold["final_score_min"] != -1.0:
        fsm = overlapp_threshold["final_score_min"]
        all_scores = [all_score for all_score in all_scores if all_score[2] >= fsm]

    for i, score_patch in enumerate(all_scores):
        score = score_patch[2]
        volume_id = score_patch[1][0]
        offset_angle = score_patch[1][2]
        curr_patch_angle = score_patch[3]
        score_distance_adjusted = score * start_distance_score_factor(volume_id, curr_patch_angle, volume_start, volume_size=volume_size)
        all_scores[i] = (score_patch[0], score_patch[1], score_distance_adjusted)
    
    all_scores.sort(key=lambda x: x[2], reverse=True)
    if len(all_scores) == 0:
        return []
    
    accumulate_nr_patches = min(accumulate_nr_patches, len(all_scores))
    highest_scores = all_scores[:accumulate_nr_patches]
    
    # Only pick scores that are at least 0.5 of the best score to not accidentally pave over close working areas with suboptimal patches
    scores_values = np.array([good_score[2] for good_score in highest_scores])
    percentile_score = np.percentile(scores_values, 95)
    best_score = percentile_score
    if "final_score_max" in overlapp_threshold and overlapp_threshold["final_score_max"] != -1.0:
        best_score = min(best_score, overlapp_threshold["final_score_max"])
        if best_score < overlapp_threshold["rounddown_best_score"]: # faster in the end
            best_score = 0.0
    highest_scores = [good_score for good_score in highest_scores if good_score[2] >= overlapp_threshold["picked_scores_similarity"] * best_score]

    # Add surrounding volumes
    for i in range(len(highest_scores)):
        volume_id = highest_scores[i][1][0]
        surrounding_volume_ids = surrounding_volumes_main_sheet(volume_id, volume_size=volume_size, window_size=overlapp_threshold["surrounding_patches_size"])
        highest_scores[i] = (surrounding_volume_ids, highest_scores[i][1], highest_scores[i][2])

    good_scores = []
    # Keep track of used volumes temporary
    volume_block_used_temp = {}
    for surrounding_volume_ids, new_patch, score in highest_scores:
        if any([volume_block_used_temp[surrounding_id] for surrounding_id in surrounding_volume_ids if surrounding_id in volume_block_used_temp]):
            continue
        good_scores.append((surrounding_volume_ids, new_patch, score))
        for surrounding_id in surrounding_volume_ids:
            volume_block_used_temp[surrounding_id] = True

    good_scores.sort(key=lambda x: x[2], reverse=True)

    if (not "final_score_max" in overlapp_threshold) or (overlapp_threshold["final_score_max"] == -1.0) or overlapp_threshold["print_scores"]:
        print(f"Highest scores: {[f'{good_score[2]:.2f}' for good_score in good_scores]}")

    # at most max_patches
    if len(good_scores) > max_patches:
        good_scores = good_scores[:max_patches]

    return good_scores

def combine_patches(patch1, patch2, overlapp_threshold, volume_start):
    i = 0
    j = 1
    iteration = 1
    if patch1 is None:
        return patch2, 0.0
    else:
        patches_list = [patch1, patch2]

    # Add the overlap base to the patches list that contains the points + normals + scores only before
    add_overlapp_entries_to_patches_list(patches_list)

    subvolume_size = patch1["subvolume_size"][0]
    subvolume = {"start": volume_start - subvolume_size//2, "end": volume_start + subvolume_size//2}

     # Assign points to tiles
    assign_points_to_tiles(patches_list, subvolume, tiling=2) # adjust tiling 

    # Compute overlap masks for the two patches
    result_0 = compute_overlap_for_pair((0, patches_list, overlapp_threshold["epsilon"], overlapp_threshold["angle_tolerance"]))
    angle_offset = result_0[0][-1]
    patch1 = add_patch(patch1, patch2, angle_offset)

    return patch1, angle_offset

def fit_sheet(patches_list, i, j, percentile, epsilon, angle_tolerance):
    # fit sheet surface to main sheet + other patch and determine how well the other patch fits into the main sheet
    main_sheet = patches_list[i]
    other_patch = patches_list[j]

    tiles = overlapping_tiles(main_sheet, other_patch)
    if len(tiles) == 0:
        return float("inf"), None
    # Find the angles of the overlapping points
    angles_i, angles_j = select_overlapping_angles(patches_list, i, j, tiles)
    points_j = select_tiles_points(patches_list, j, tiles)
    # Find the angles offsets
    angles_offset = angles_i - angles_j
    offset = largest_group_offset(angles_offset, epsilon=epsilon)

    # mean_angle_j is already 0 from build_patch
    mean_angle_j = 0.0
    mean_angle_j += offset
    filter_angle_range = (mean_angle_j - angle_tolerance, mean_angle_j + angle_tolerance)
    main_angles = main_sheet["angles"]
    main_sheet_mask = filter_by_angle(main_angles, [filter_angle_range])

    points = np.concatenate([main_sheet["points"][main_sheet_mask], other_patch["points"]], axis=0)
    points = np.unique(points, axis=0)
    normals = np.concatenate([main_sheet["normals"][main_sheet_mask], other_patch["normals"]], axis=0)
    normal_vector = get_vector_mean(normals)
    R = rotation_matrix_to_align_z_with_v(normal_vector)

    # Sheet fitting params
    n = 4
    alpha = 1000.0
    slope_alpha = 0.1

    # Fit other patch and main sheet together
    coeff_all = fit_surface_to_points_n_regularized(points, R, n, alpha=alpha, slope_alpha=slope_alpha)
    surface = (normal_vector, R, n, coeff_all)

    n_patch = 4 # TODO: make this a parameter loaded from patch
    distances_points_other = distance_from_surface(other_patch["points"], R, n, coeff_all)
    distance_percentile = np.percentile(distances_points_other, percentile)
    distance_other = np.mean(distances_points_other)

    distances_points_ = distance_from_surface(points_j, R, n_patch, coeff_all)
    if len(distances_points_) > 0:
        mean_distance_ = np.mean(distances_points_)
    else:
        mean_distance_ = 0.0

    # Fit main sheet if containing points
    if np.sum(main_sheet_mask) < 1:
        distance_main_sheet = 100.0
        distance_main_sheet_dif = 100.0
    else:
        coeff_main_sheet = fit_surface_to_points_n_regularized(main_sheet["points"][main_sheet_mask], R, n, alpha=alpha, slope_alpha=slope_alpha)
        distances_points_main_sheet = distance_from_surface(main_sheet["points"][main_sheet_mask], R, n, coeff_main_sheet)
        distance_main_sheet = np.sum(distances_points_main_sheet)

        distances_points_main_sheet_all = distance_from_surface(main_sheet["points"][main_sheet_mask], R, n, coeff_all)
        distance_main_sheet_all = np.sum(distances_points_main_sheet_all)

        distance_main_sheet_dif = (distance_main_sheet_all - distance_main_sheet) / main_sheet["points"][main_sheet_mask].shape[0]
        distance_main_sheet_dif = max(0.0, distance_main_sheet_dif)

    # Compute the distance between the two surfaces
    distance = distance_other + distance_main_sheet_dif

    return distance, distance_percentile, mean_distance_, surface

def main_sheet_add_display_points(main_sheet, subvolume_size, path, sample_ratio):
    for volume_id in main_sheet:
        for patch_id in main_sheet[volume_id]:
            main_sheet_patch = (volume_id, patch_id, main_sheet[volume_id][patch_id]["offset_angle"])
            # Get the patch
            patch, offset_angle = build_patch(main_sheet_patch, subvolume_size, path)
            # Add display points

            # Sample points from picked patch
            subsampled_points_picked_patch = select_points(
                patch['points'], patch['normals'], patch['colors'], patch['angles'], sample_ratio
            )

            # Add display points
            main_sheet[volume_id][patch_id]["displaying_points"] = subsampled_points_picked_patch

def main_sheet_recalculate_scores(main_sheet, subvolume_size, path, overlapp_threshold, sample_ratio_score, volume_size=50):
    for volume_id in main_sheet:
        surrounding_volume_ids = surrounding_volumes_main_sheet(volume_id, volume_size=volume_size, window_size=overlapp_threshold["surrounding_patches_size"]) # build larger main sheet, also adjust tiling for combine patches
        main_sheet_patches_list = build_main_sheet_patches_list(surrounding_volume_ids, main_sheet)
        # build main sheet in the specified region
        main_sheet = build_main_sheet_from_patches_list(main_sheet_patches_list, subvolume_size, path, sample_ratio_score)
        # build surrounding volumes list
        surrounding_volumes_ids = surrounding_volumes(volume_id, main_sheet)
    
        patches_list = [main_sheet]
        # Load all surrounding patches
        for surrounding_volume_id in surrounding_volumes_ids:
            path_surrounding = path + f"/{surrounding_volume_id[0]:06}_{surrounding_volume_id[1]:06}_{surrounding_volume_id[2]:06}"
            patches_list_ = subvolume_surface_patches_folder(path_surrounding, subvolume_size[0])
            # Filter out patches that are in the main sheet
            patches_list_ = [patch for patch in patches_list_ if patch["ids"][0] not in main_sheet["ids"]]
            patches_list += patches_list_

        # Add the overlap base to the patches list that contains the points + normals + scores only before
        add_overlapp_entries_to_patches_list(patches_list)

        subvolume_size = main_sheet["subvolume_size"][0]
        subvolume = {"start": volume_id - subvolume_size, "end": volume_id + subvolume_size}

        # Assign points to tiles
        assign_points_to_tiles(patches_list, subvolume, tiling=3)

        # Calculate the scores of the main sheet wrt the surrounding patches
        scores_volume = []

        # Single threaded
        results = []
        results.append(compute_overlap_for_pair((0, patches_list)))

        # Combining results
        for result in results:
            for i, j, overlapp_percentage, overlap, non_overlap, points_overlap in result:
                patches_list[i]["overlapp_percentage"][j] = overlapp_percentage
                patches_list[i]["overlap"][j] = overlap
                patches_list[i]["non_overlap"][j] = non_overlap
                patches_list[i]["points_overlap"][j] = points_overlap
                score = overlapp_score(i, j, patches_list, overlapp_threshold=overlapp_threshold, sample_ratio=sample_ratio_score)
                id = patches_list[j]["ids"][0]

                if score > 0:
                    cost_refined = fit_sheet(patches_list, i, j)
                    if cost_refined > overlapp_threshold["cost_threshold"]:
                        score = -1.0
                    else:
                        score_refined = 10.0 / (cost_refined + 0.001)
                        score *= score_refined
                
                scores_volume.append((id[:3], id[3], score))

def build_patch(main_sheet_patch, subvolume_size, path, sample_ratio=1.0, align_and_flip_normals=False):
    subvolume_size = np.array(subvolume_size)
    ((x, y, z), main_sheet_surface_nr, offset_angle) = main_sheet_patch
    file = path + f"/{x:06}_{y:06}_{z:06}/surface_{main_sheet_surface_nr}.ply"
    res = load_ply(file)
    patch_points = res[0]
    patch_normals = res[1]
    patch_color = res[2]
    patch_score = res[3]
    patch_distance = res[4]
    patch_coeff = res[5]
    n = res[6]

    # Sample points from picked patch
    patch_points, patch_normals, patch_color, _ = select_points(
        patch_points, patch_normals, patch_color, patch_color, sample_ratio
    )

    patch_id = tuple([*map(int, file.split("/")[-2].split("_"))]+[int(file.split("/")[-1].split(".")[-2].split("_")[-1])])

    x, y, z, id_ = patch_id
    anchor_normal = get_vector_mean(patch_normals)
    anchor_angle = alpha_angles(np.array([anchor_normal]))[0]

    additional_main_patch = {"ids": [patch_id],
                    "points": patch_points,
                    "normals": patch_normals,
                    "colors": patch_color,
                    "anchor_points": [patch_points[0]], 
                    "anchor_normals": [anchor_normal],
                    "anchor_angles": [anchor_angle],
                    "angles": adjust_angles_offset(adjust_angles_zero(alpha_angles(patch_normals), - anchor_angle), offset_angle),
                    "subvolume": [(x, y, z)],
                    "subvolume_size": [subvolume_size],
                    "iteration": 0,
                    "patch_prediction_scores": [patch_score],
                    "patch_prediction_distances": [patch_distance],
                    "patch_prediction_coeff": [patch_coeff],
                    "n": [n],
                    }
    
    return additional_main_patch, offset_angle

def make_unique(patch):
    # unique points
    unique_points, unique_indices = np.unique(patch["points"], axis=0, return_index=True)
    patch["points"] = patch["points"][unique_indices]
    patch["normals"] = patch["normals"][unique_indices]
    patch["colors"] = patch["colors"][unique_indices]
    patch["angles"] = patch["angles"][unique_indices]

    return patch

def add_patch(patch1, patch2, offset_angle):
    patch1["ids"] += patch2["ids"]
    patch1["points"] = np.concatenate([patch1["points"], patch2["points"]], axis=0)
    patch1["normals"] = np.concatenate([patch1["normals"], patch2["normals"]], axis=0)
    patch1["colors"] = np.concatenate([patch1["colors"], patch2["colors"]], axis=0)
    patch1["anchor_points"] += patch2["anchor_points"]
    patch1["anchor_normals"] += patch2["anchor_normals"]
    patch1["anchor_angles"] += patch2["anchor_angles"]
    patch1["angles"] =  np.concatenate([patch1["angles"], adjust_angles_offset(patch2["angles"], offset_angle)], axis=0)
    patch1["subvolume"] += patch2["subvolume"]
    patch1["subvolume_size"] += patch2["subvolume_size"]
    patch1["patch_prediction_scores"] += patch2["patch_prediction_scores"]
    patch1["patch_prediction_distances"] += patch2["patch_prediction_distances"]
    patch1["patch_prediction_coeff"] += patch2["patch_prediction_coeff"]
    patch1["n"] += patch2["n"]

    # unique points
    patch1 = make_unique(patch1)

    return patch1

def add_patch_duplicating(patch1, patch2, offset_angle):
    patch1["ids"] += patch2["ids"]
    patch1["points"] = np.concatenate([patch1["points"], patch2["points"]], axis=0)
    patch1["normals"] = np.concatenate([patch1["normals"], patch2["normals"]], axis=0)
    patch1["colors"] = np.concatenate([patch1["colors"], patch2["colors"]], axis=0)
    patch1["anchor_points"] += patch2["anchor_points"]
    patch1["anchor_normals"] += patch2["anchor_normals"]
    patch1["anchor_angles"] += patch2["anchor_angles"]
    patch1["angles"] =  np.concatenate([patch1["angles"], adjust_angles_offset(patch2["angles"], offset_angle)], axis=0)
    patch1["subvolume"] += patch2["subvolume"]
    patch1["subvolume_size"] += patch2["subvolume_size"]
    patch1["patch_prediction_scores"] += patch2["patch_prediction_scores"]
    patch1["patch_prediction_distances"] += patch2["patch_prediction_distances"]
    patch1["patch_prediction_coeff"] += patch2["patch_prediction_coeff"]
    patch1["n"] += patch2["n"]

    return patch1

def build_main_sheet_volume_from_patches_list(main_sheet_patches_list, subvolume_size, path, sample_ratio_score):
    if isinstance(subvolume_size, int) or isinstance(subvolume_size, float) or isinstance(subvolume_size, np.float64) or isinstance(subvolume_size, np.float32) or isinstance(subvolume_size, np.int64) or isinstance(subvolume_size, np.int32) or isinstance(subvolume_size, np.int16) or isinstance(subvolume_size, np.int8) or isinstance(subvolume_size, np.uint64) or isinstance(subvolume_size, np.uint32) or isinstance(subvolume_size, np.uint16) or isinstance(subvolume_size, np.uint8):
        subvolume_size = np.array([subvolume_size, subvolume_size, subvolume_size])


    main_sheet, _ = build_main_sheet_from_patches_list(main_sheet_patches_list, subvolume_size, path, sample_ratio_score)
    main_points = main_sheet["points"]
    # Coloring Red
    main_colors = np.zeros_like(main_points) + np.array([1, 0, 0])
    other_points = main_points.astype(np.float16)
    other_colors = main_colors
    other_points_list = []

    main_sheet_volumes = list(set([mid[:3] for mid in main_sheet["ids"]]))
    for volume_id in tqdm(main_sheet_volumes):
        path_surrounding = path + f"/{volume_id[0]:06}_{volume_id[1]:06}_{volume_id[2]:06}"
        patches_list = subvolume_surface_patches_folder(path_surrounding, subvolume_size)
        # Build main sheet
        for volume_sheet_patch in patches_list:
            if not volume_sheet_patch["ids"][0] in main_sheet["ids"]:
                other_point = volume_sheet_patch["points"]
                other_points_list.append(other_point)
    
    other_points = np.concatenate([other_points] + other_points_list, axis=0).astype(np.float16)
    other_colors = np.zeros_like(other_points) + 1
    unique_points, indices = np.unique(other_points, return_index=True, axis=0)
    unique_colors = other_colors[indices]
    return unique_points, unique_colors

def build_main_sheet_from_patches_list(main_sheet_patches_list, subvolume_size, path, sample_ratio_score, align_and_flip_normals=True):
    if isinstance(subvolume_size, int) or isinstance(subvolume_size, float) or isinstance(subvolume_size, np.float64) or isinstance(subvolume_size, np.float32) or isinstance(subvolume_size, np.int64) or isinstance(subvolume_size, np.int32) or isinstance(subvolume_size, np.int16) or isinstance(subvolume_size, np.int8) or isinstance(subvolume_size, np.uint64) or isinstance(subvolume_size, np.uint32) or isinstance(subvolume_size, np.uint16) or isinstance(subvolume_size, np.uint8):
        subvolume_size = np.array([subvolume_size, subvolume_size, subvolume_size])

    main_sheet = None
    main_patches = []
    
    # Build main sheet
    for main_sheet_patch in main_sheet_patches_list:
        volume_start, surface_nr, offset_angle = main_sheet_patch
        volume_start = (int(volume_start[0]), int(volume_start[1]), int(volume_start[2]))

        main_sheet_patch = (volume_start, int(surface_nr), float(offset_angle))
        additional_main_patch, offset_angle = build_patch(main_sheet_patch, tuple(subvolume_size), path, float(sample_ratio_score), align_and_flip_normals=align_and_flip_normals)
        main_patches.append((additional_main_patch, offset_angle))

        if main_sheet is None:
            main_sheet = additional_main_patch
        else:            
            main_sheet = add_patch(main_sheet, additional_main_patch, 0.0)
    
    return main_sheet, main_patches


def build_main_sheet_patches_list(main_sheet_volume_ids, main_sheet):
    main_sheet_volume_patches = []
    for volume_id in main_sheet_volume_ids:
        if volume_id in main_sheet: # slow? -> no
            for patch_id in main_sheet[volume_id]:
                main_sheet_volume_patches.append((volume_id, patch_id, main_sheet[volume_id][patch_id]["offset_angle"]))

    return main_sheet_volume_patches

def winding_switch_sheet_score_raw_precomputed_surface(patch_start, patch, overlapp_threshold):
    # Calculate distance between the two patches
    n = patch_start["n"][0]
    normals_start = patch_start['anchor_normals'][0]
    coeff_start = patch_start["patch_prediction_coeff"][0]
    normal_vector = normals_start
    R = rotation_matrix_to_align_z_with_v(normal_vector)

    # Sheet fitting params
    # n = 4
    # alpha = 1000.0
    # slope_alpha = 0.1

    # Fit other patch and main sheet together
    min_ = - overlapp_threshold["max_sheet_clip_distance"]
    max_ = overlapp_threshold["max_sheet_clip_distance"]
    points = patch["points"]
    distances_to_start_patch, patch_patch_start_direction_vector = distance_from_surface_clipped(points, R, n, coeff_start, min_, max_, return_direction=True)

    # Dot product of the normals of normal_vector and patch_patch_start_direction_vector
    dot_product = np.dot(normal_vector, patch_patch_start_direction_vector)
    # calculate patch winding direction offset (+1, -1). That determines in what direction this patch lies wrt the main sheet
    # get the sign of the dot product
    patch_winding_direction = np.sign(dot_product)
    k = patch_winding_direction
    
    score_raw = np.mean(distances_to_start_patch)

    return score_raw, k

def winding_switch_sheet_score_raw(patch_start, patch):
    # Calculate distance between the two patches
    n = patch_start["n"][0]
    points_start = patch_start["points"]
    normals_start = patch_start['normals']
    normal_vector = get_vector_mean(normals_start)
    R = rotation_matrix_to_align_z_with_v(normal_vector)

    # Sheet fitting params
    n = 4
    alpha = 1000.0
    slope_alpha = 0.1

    # Fit other patch and main sheet together
    coeff_start = fit_surface_to_points_n_regularized(points_start, R, n, alpha=alpha, slope_alpha=slope_alpha)

    points = patch["points"]
    distances_to_start_patch, patch_patch_start_direction_vector = distance_from_surface(points, R, n, coeff_start, return_direction=True)

    # Dot product of the normals of normal_vector and patch_patch_start_direction_vector
    dot_product = np.dot(normal_vector, patch_patch_start_direction_vector)
    # calculate patch winding direction offset (+1, -1). That determines in what direction this patch lies wrt the main sheet
    # get the sign of the dot product
    patch_winding_direction = np.sign(dot_product)
    k = patch_winding_direction
    
    score_raw = np.mean(distances_to_start_patch)

    return score_raw, k

def compute_winding_switch_sheet_score(patches_list_start, main_sheet, surface_nr, angle_offset, overlapp_threshold, epsilon=1e-5):
    main_sheet_angles = main_sheet["angles"]
    # Find patch of surface_nr
    patch_start = deepcopy([patch for patch in patches_list_start if patch["ids"][0][3] == surface_nr][0])

    anchor_angle_start = alpha_angles(np.array(patch_start["anchor_normals"]))[0]
    patch_start["angles"] = adjust_angles_offset(adjust_angles_zero(alpha_angles(patch_start["normals"]), - anchor_angle_start), angle_offset)
    # All other patches
    patches_list = [patch for patch in patches_list_start if patch["ids"][0][3] != surface_nr]
    # Calculate Scores
    scores = []
    for patch in patches_list:
        # recalculate patch_start angles(=overlap1_angles -> overlap2_angles) with the patch's anchor angle
        anchor_normal = patch["anchor_normals"][0]
        anchor_angle = alpha_angles(np.array([anchor_normal]))[0]
        patch_start["overlap2_angles"] = adjust_angles_offset(adjust_angles_zero(alpha_angles(patch_start["normals"]), - anchor_angle), 0.0)
        # Calculate the angle offsets
        overlap1_angles = patch_start["angles"]
        overlap2_angles = patch_start["overlap2_angles"]
        overlap1_angle_offsets = overlap1_angles - overlap2_angles
        
        # Find the offset with the highest number of occurences in the angle offsets
        # Prefilter angles for winding number (different offsets), select max overlap offset
        angles_offset1 = largest_group_offset(overlap1_angle_offsets, epsilon=epsilon)

        # Check for enough points in patch_start and patch
        invalid = (patch_start["points"].shape[0] < overlapp_threshold["min_points_winding_switch"]*overlapp_threshold["sample_ratio_score"]) or (patch["points"].shape[0] < overlapp_threshold["min_points_winding_switch"]*overlapp_threshold["sample_ratio_score"])
        # Calculate offset score
        score_raw, k = winding_switch_sheet_score_raw(patch_start, patch)
        k = k * overlapp_threshold["winding_direction"] # because of scroll's specific winding direction
        angles_offset1_winding_adjusted = angles_offset1 + k*360.0

        patch_offset_angles = adjust_angles_offset(adjust_angles_zero(alpha_angles(patch["normals"]), - anchor_angle), angles_offset1_winding_adjusted)
        # Check if patch_offset_angles with +- "angle_tolerance" are in main_sheet_angles
        # subtract every patch_offset_angles from every main_sheet_angles
        angle_diff = np.abs(patch_offset_angles[:, None] - main_sheet_angles[None, :])
        # find the minimum angle difference
        min_angle_diff = np.min(angle_diff)
        invalid_angle = min_angle_diff < overlapp_threshold["angle_tolerance"]

        invalid = invalid or invalid_angle or (score_raw < overlapp_threshold["min_winding_switch_sheet_distance"])
            
        scores.append((patch["ids"][0][:3], patch["ids"][0][3], score_raw, angles_offset1_winding_adjusted, invalid, k))

    # filter and only take the scores closest to the main patch (smallest scores) for each k in +1, -1
    direction1_scores = [score for score in scores if score[5] > 0.0]
    direction2_scores = [score for score in scores if score[5] < 0.0]

    def scores_cleaned(direction_scores):
        if len(direction_scores) == 0:
            return []
        
        score = min(direction_scores, key=lambda x: x[2])
        invalid = score[4]
        if (score[2] < 0.0) or invalid:
            return [(score[0], score[1], -1.0, score[3])]
        
        score_val = (overlapp_threshold["max_winding_switch_sheet_distance"] - (score[2] - overlapp_threshold["min_winding_switch_sheet_distance"])) / overlapp_threshold["max_winding_switch_sheet_distance"] # calculate score
        score_val = score_val * overlapp_threshold["winding_switch_sheet_score_factor"]
        score = (score[0], score[1], score_val, score[3])

        return [score]
    
    score1 = scores_cleaned(direction1_scores)
    score2 = scores_cleaned(direction2_scores)

    scores = score1 + score2 # List concatenation

    return scores

def get_patches_from_surrounding_volume(args):
    path, subvolume_size, sample_ratio_score, main_sheet_ids = args
    patches_list_ = subvolume_surface_patches_folder(path, subvolume_size, sample_ratio=sample_ratio_score)
    # Filter out patches that are in the main sheet
    return [patch for patch in patches_list_ if patch["ids"][0] not in main_sheet_ids]

def load_surrounding_patches(surrounding_volumes_ids, path, subvolume_size, sample_ratio_score, main_sheet):
    with ThreadPoolExecutor() as executor:
        args_list = [(path + f"/{id[0]:06}_{id[1]:06}_{id[2]:06}", subvolume_size[0], sample_ratio_score, main_sheet["ids"]) for id in surrounding_volumes_ids]
        patches_list = list(executor.map(get_patches_from_surrounding_volume, args_list))

    # Flatten the list of lists to a single list
    return [patch for sublist in patches_list for patch in sublist]

def check_sheet_winding_side(patch_start, angles_offset, patches_list_main_sheet, overlapp_threshold):
    # construct all patches from same volume that are in the main sheet
    patches_list_same_volume = [patch for patch in patches_list_main_sheet if patch[0]["ids"][0][:3] == patch_start["ids"][0][:3] and patch[0]["ids"][0][3] != patch_start["ids"][0][3]]
    if len(patches_list_same_volume) != 0:
        for patch_ in patches_list_same_volume:
            patch, angle_offset_patch = patch_
            score_raw, k = winding_switch_sheet_score_raw_precomputed_surface(patch_start, patch, overlapp_threshold)
            k = k * overlapp_threshold["winding_direction"] # because of scroll's specific winding direction
            valid_patch_winding_side = ((score_raw < (overlapp_threshold["cost_sheet_distance_threshold"] / 2.0)) and (abs(angles_offset - angle_offset_patch) < 30)) or ((score_raw > (overlapp_threshold["cost_sheet_distance_threshold"] / 2.0)) and not ((k < 0.0) ^ (angles_offset - angle_offset_patch > 0.0))) # xor python operator is ^
            if not valid_patch_winding_side: # False patch
                # print(f"False patch")
                return False
        # print(f"True patch")
        return True # True patch
    else:
        return True # True, because no other patches in the same volume in patches_list_main_sheet

def calculate_main_sheet_scores(main_sheet_patches_list, volume_start, surface_nr, offset_angle, switched_winding, path, overlapp_threshold, sample_ratio_score=0.1, sample_ratio=0.1):
    """
    Calculates the scores of the main sheet
    """
    # Build and load the patches_list
    # Main Sheet
    volume_start = (int(volume_start[0]), int(volume_start[1]), int(volume_start[2]))
    subvolume_size = 50
    # Size is int
    if isinstance(subvolume_size, int) or isinstance(subvolume_size, float) or isinstance(subvolume_size, np.float64) or isinstance(subvolume_size, np.float32) or isinstance(subvolume_size, np.int64) or isinstance(subvolume_size, np.int32) or isinstance(subvolume_size, np.int16) or isinstance(subvolume_size, np.int8) or isinstance(subvolume_size, np.uint64) or isinstance(subvolume_size, np.uint32) or isinstance(subvolume_size, np.uint16) or isinstance(subvolume_size, np.uint8):
        subvolume_size = np.array([subvolume_size, subvolume_size, subvolume_size])
    
    # build main sheet in the specified region
    main_sheet, main_patches = build_main_sheet_from_patches_list(main_sheet_patches_list, subvolume_size, path, sample_ratio_score)

    # picked patch
    picked_sheet_patch = (volume_start, int(surface_nr), float(0.0))
    picked_sheet, _ = build_patch(picked_sheet_patch, tuple(subvolume_size), path, float(sample_ratio_score))

    if main_sheet is None:
        main_sheet = picked_sheet
    else:
        main_sheet = add_patch(main_sheet, picked_sheet, offset_angle)
    if switched_winding:
        print("\033[94m[ThaumatoAnakalyptor]:\033[0m Switched winding of the added sheet.")

    # adjust angles of picked sheet (only important for the main sheet display points, since those points and angles only get used there)
    picked_sheet_patch = (volume_start, int(surface_nr), float(offset_angle))
    if overlapp_threshold["display"]:
        # Sample points from picked patch
        subsampled_points_picked_patch = select_points(
            picked_sheet['points'], picked_sheet['normals'], picked_sheet['colors'], picked_sheet['angles'], 1.0
        )
    else:
        subsampled_points_picked_patch = None

    # surrounding patches (immediate vicinity of the picked patch, with volume start overlap)
    surrounding_volumes_ids = surrounding_volumes(volume_start, volume_size=subvolume_size[0])
    patches_list = [main_sheet]
    
    # Load all surrounding patches
    for surrounding_volume_id in surrounding_volumes_ids:
        path_surrounding = path + f"/{surrounding_volume_id[0]:06}_{surrounding_volume_id[1]:06}_{surrounding_volume_id[2]:06}"
        patches_list_ = subvolume_surface_patches_folder(path_surrounding, subvolume_size[0], sample_ratio=sample_ratio_score)
        # Filter out patches that are in the main sheet
        patches_list_ = [patch for patch in patches_list_ if patch["ids"][0] not in main_sheet["ids"]]
        patches_list += patches_list_

    # Add the overlap base to the patches list that contains the points + normals + scores only before
    add_overlapp_entries_to_patches_list(patches_list)

    subvolume_size = main_sheet["subvolume_size"][0]
    subvolume = {"start": volume_start - 1.5 * subvolume_size, "end": volume_start + 0.5 * subvolume_size}

    # Assign points to tiles
    assign_points_to_tiles(patches_list, subvolume, tiling=3)

    # Calculate the scores of the main sheet wrt the surrounding patches
    scores_volume = {}

    # Single threaded
    results = []
    results.append(compute_overlap_for_pair((0, patches_list, overlapp_threshold["epsilon"], overlapp_threshold["angle_tolerance"])))

    # Combining results
    for result in results:
        for i, j, overlapp_percentage, overlap, non_overlap, points_overlap, angles_offset in result:
            patches_list[i]["overlapp_percentage"][j] = overlapp_percentage
            patches_list[i]["overlap"][j] = overlap
            patches_list[i]["non_overlap"][j] = non_overlap
            patches_list[i]["points_overlap"][j] = points_overlap
            score = overlapp_score(i, j, patches_list, overlapp_threshold=overlapp_threshold, sample_ratio=sample_ratio_score)
            id = patches_list[j]["ids"][0]

            if score <= 0.0:
                score = -1.0
                surface = None
            elif patches_list[j]["points"].shape[0] < overlapp_threshold["min_patch_points"] * sample_ratio_score:
                score = -1.0
                surface = None
            elif patches_list[j]["patch_prediction_scores"][0] < overlapp_threshold["min_prediction_threshold"]:
                score = -1.0
                surface = None
            elif not check_sheet_winding_side(patches_list[j], angles_offset, main_patches, overlapp_threshold): # check if all main patches are on the propper side of the proposed patch offset angle
                score = -1.0
                surface = None
            else:
                if patches_list[j]["ids"][0][:3] in [p_id[:3] for p_id in patches_list[i]["ids"]]: # Bad since it does not work with multiple windings
                    nr_instances = len([p_id for p_id in patches_list[i]["ids"] if ((p_id[:3] == patches_list[j]["ids"][0][:3]) and (True))]) # check for angles offset similar
                    score *= overlapp_threshold["multiple_instances_per_batch_factor"] ** nr_instances
                cost_refined, cost_percentile, cost_sheet_distance, surface = fit_sheet(patches_list, i, j, overlapp_threshold["cost_percentile"], overlapp_threshold["epsilon"], overlapp_threshold["angle_tolerance"])
                if cost_refined >= overlapp_threshold["cost_threshold"]:
                    score = -1.0
                elif cost_percentile >= overlapp_threshold["cost_percentile_threshold"]:
                    score = -1.0
                elif cost_sheet_distance >= overlapp_threshold["cost_sheet_distance_threshold"]:
                    score = -1.0
            
            switched_winding = False
            scores_volume[(id[:3], id[3])] = (id[:3], id[3], score, angles_offset, surface, switched_winding)

    if overlapp_threshold["enable_winding_switch"]:
        # Calculate winding switch sheet scores
        path_volume_start = path + f"/{volume_start[0]:06}_{volume_start[1]:06}_{volume_start[2]:06}"
        patches_list_start = subvolume_surface_patches_folder(path_volume_start, subvolume_size[0], sample_ratio=sample_ratio_score)
        results = compute_winding_switch_sheet_score(patches_list_start, main_sheet, surface_nr, offset_angle, overlapp_threshold)

        # Add winding switch scores to scores_volume
        for res in results:
            v = res[0]
            p = res[1]
            score = res[2]
            angles_offset = res[3]
            if (v, p) in scores_volume:
                score_volume = scores_volume[(v, p)]
                surface = score_volume[4]
                score_overlap = score_volume[2]
            else:
                surface = None
                score_overlap = -1.0
            valid_switch_patch = True
            if score_overlap > 0.0: # only add winding switch score if there is no overlap with another patch of the main sheet
                valid_switch_patch = False
            # Check if there is a patch of the main sheet close to the winding switch patch (volume id)
            for main_patch in main_sheet_patches_list:
                (_, _, offset_angle_) = main_patch
                if abs(offset_angle_ - angles_offset) < overlapp_threshold["angle_tolerance"]:
                    valid_switch_patch = False # only add winding switch score if the winding switch is not close to the main sheet
                    break

            switched_winding = True
            if not valid_switch_patch:
                score = -1.0
            scores_volume[(v, p)] = (v, p, score, angles_offset, surface, switched_winding)


    # Create list from scores_volume
    scores_volume = list(scores_volume.values())

    return offset_angle, subsampled_points_picked_patch, scores_volume

def update_main_sheet(main_sheet, patches, offset_angles, points):
    for i in range(len(patches)):
        patch = patches[i]
        volume_id, patch_id, offset_angle, switched_winding = patch

        if volume_id not in main_sheet:
            main_sheet[volume_id] = {}
        if patch_id not in main_sheet[volume_id]:
            main_sheet[volume_id][patch_id] = {}
        main_sheet[volume_id][patch_id]["offset_angle"] = offset_angles[i]
        main_sheet[volume_id][patch_id]["switched_winding"] = switched_winding
        main_sheet[volume_id][patch_id]["displaying_points"] = points[i]

def update_scores(main_sheet, volume_blocks_scores, scores, picked_patches, overlapp_threshold):
    count_good_scores = 0
    for i in range(len(scores)):
        if len(scores[i]) == 0:
            print(f"Warning no score to update {scores[i]}, i: {i}")
            continue
        volume_id, patch_id, score, angles, patch_surface, switched_winding = scores[i]
        if volume_id in main_sheet and patch_id in main_sheet[volume_id]: # only add scores for patches that are not yet in the main sheet
            continue
        if not volume_id in volume_blocks_scores:
            volume_blocks_scores[volume_id] = {}
        if not patch_id in volume_blocks_scores[volume_id]:
            volume_blocks_scores[volume_id][patch_id] = {}

        if (("switched_winding" in volume_blocks_scores[volume_id][patch_id]) and (not volume_blocks_scores[volume_id][patch_id]["switched_winding"])) or (not ("angle" in volume_blocks_scores[volume_id][patch_id])) or (not ("score" in volume_blocks_scores[volume_id][patch_id])) or (volume_blocks_scores[volume_id][patch_id]["angle"] is None) or (angles and (abs(angles - volume_blocks_scores[volume_id][patch_id]["angle"]) < overlapp_threshold["angle_tolerance"])) or (score and (score > volume_blocks_scores[volume_id][patch_id]["score"])):
            volume_blocks_scores[volume_id][patch_id]["score"] = score
            volume_blocks_scores[volume_id][patch_id]["angle"] = angles
            volume_blocks_scores[volume_id][patch_id]["surface"] = patch_surface
            volume_blocks_scores[volume_id][patch_id]["offset_angle"] = angles
            volume_blocks_scores[volume_id][patch_id]["switched_winding"] = switched_winding

        if score > 0.0:
            count_good_scores += 1

    for i in range(len(picked_patches)):
        volume_id, patch_id, offset_angle, switched_winding = picked_patches[i]
        del volume_blocks_scores[volume_id][patch_id]
        assert not patch_id in volume_blocks_scores[volume_id], f"Patch id {patch_id} still in volume {volume_id}"

        if switched_winding:
            # Remove all winding switch scores around the volume id
            surrounding_volume_ids = surrounding_volumes(volume_id)
            for surrounding_id in surrounding_volume_ids:
                vbs_keys = list(volume_blocks_scores.keys())
                if surrounding_id in vbs_keys:
                    keys_vs = list(volume_blocks_scores[surrounding_id].keys())
                    for patch_id_ in keys_vs:
                        if volume_blocks_scores[surrounding_id][patch_id_]["switched_winding"] and abs(offset_angle - volume_blocks_scores[surrounding_id][patch_id_]["angle"]) < overlapp_threshold["angle_tolerance"]:
                            del volume_blocks_scores[surrounding_id][patch_id_]
                            if len(volume_blocks_scores[surrounding_id]) == 0:
                                del volume_blocks_scores[surrounding_id]

    print(f"Updated {len(picked_patches)} picked patches and {len(scores)} scores, it had {count_good_scores} good scores.")

def update_volume_blocks_used(volume_block_used, volume_patches):
    for _, (volume_id, patch_id, _, _), _ in volume_patches:
        surrounding_volume_ids = surrounding_volumes_main_sheet(volume_id)
        for surrounding_id in surrounding_volume_ids:
            volume_block_used[surrounding_id] = False

def build_main_sheet_points(main_sheet):
    points = []
    normals = []
    angles = []
    for volume_id in main_sheet:
        for patch_id in main_sheet[volume_id]:
            offset_angle = main_sheet[volume_id][patch_id]["offset_angle"]
            displaying_points = main_sheet[volume_id][patch_id]["displaying_points"]
            points_, normals_, _, angles_ = displaying_points
            points.append(points_)
            normals.append(normals_)
            angles.append(angles_)
    
    points = np.concatenate(points, axis=0)
    normals = np.concatenate(normals, axis=0)
    angles = np.concatenate(angles, axis=0)

    return points, normals, angles

def display_main_sheet(main_sheet, nr_points_show=10000, show_normals=True):
    points, normals, angles = build_main_sheet_points(main_sheet)

    # Visualize with matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # random subsample points
    print(f"Number of points in patch {0}: {points.shape[0]}")
    subsample = min(nr_points_show, points.shape[0])
    indices = np.random.choice(points.shape[0], subsample, replace=True)
    # map to 0-1
    point_max, point_min = np.max(angles), np.min(angles)
    k = np.ceil(np.max((angles)  - np.min(angles)) / 360)
    angles = (angles - np.min(angles)) / (np.max(angles) - np.min(angles))

    # Min and max angles over all volume id, patch id:
    min_angle, max_angle = None, None
    for volume_id in main_sheet:
        for patch_id in main_sheet[volume_id]:
            if min_angle is None or min_angle > main_sheet[volume_id][patch_id]["offset_angle"]:
                min_angle = main_sheet[volume_id][patch_id]["offset_angle"]
            if max_angle is None or max_angle < main_sheet[volume_id][patch_id]["offset_angle"]:
                max_angle = main_sheet[volume_id][patch_id]["offset_angle"]
    print(f"Windings around umbilicus {k}, min angle: {min_angle}, max angle: {max_angle} and on point basis: {point_max} and {point_min}")
    
    # map to color with colormap
    color = plt.cm.cool(angles)
    # subsample
    color = color[indices]
    ax.scatter(points[indices][:,0], points[indices][:,1], points[indices][:,2], c=color, marker='o', s=3)

    # Subsample again for a few normals
    points_subsample = points[indices]
    normals_subsample = normals[indices]

    indices_normals = np.random.choice(points_subsample.shape[0], 100, replace=True)
    points_subsample = points_subsample[indices_normals]
    normals_subsample = normals_subsample[indices_normals]
    color_subsample = color[indices_normals]
    del points, normals, color

    if show_normals:
        ax.quiver(points_subsample[:, 0], points_subsample[:, 1], points_subsample[:, 2], 
                normals_subsample[:, 0], normals_subsample[:, 1], normals_subsample[:, 2], color='g', length=100, normalize=True, arrow_length_ratio=0.1, label='Vector v')
        ax.scatter(points_subsample[:,0], points_subsample[:,1], points_subsample[:,2], c=color_subsample, marker='o', s=5)

    
    # same ratio for all axes
    ax.set_aspect('equal')

    plt.show()

    del ax
    del fig

def divide_and_conquer(path, iteration, volume_start, volume_size, initial_ratio=0.1, step_ratio=0.3, min_sheet_size=100000, instances_block_size=50, overlapp_threshold={"score_threshold": 0.25, "nr_points_min": 20.0, "nr_points_max": 500.0}):
    """
    volume_start, volume_size in size of the instance blocks
    """
    # load the patches list
    patches_list = load_iteration(path, iteration)

    # iterate over all rounds
    while True:
        iteration += 1
        if iteration > 1:
            overlapp_threshold["score_threshold"] *= 0.5
        print(f"Iteration {iteration}")
        # Calculate the new sample ratio
        sample_ratio = initial_ratio * (step_ratio ** (iteration - 1))

        min_points = int(150 * sample_ratio / 0.3) + 1
        # Subsample points
        remove_points_from_patches_list_multithreading(patches_list, original_ratio=sample_ratio)
        patches_list = remove_small_patches(patches_list, min_points=min_points)

        # stop iterating when no more new patches of interest were created
        if len(patches_list) == 0:
            break

        # Generate subvolume start and sizes
        nr_instances_size = 2 ** iteration
        subvolumes = generate_subvolumes(volume_start, volume_size, instances_block_size, nr_instances_size)

        if len(subvolumes) == 0:
            break

        print(f"Extracting subvolumes patches for {len(subvolumes)} subvolumes")
        
        subvolumes_patches = []
        for subvolume in subvolumes:
            subvolume_patches_list = process_subvolume((subvolume, patches_list))
            subvolumes_patches.append(subvolume_patches_list)

        print("Done")

        iteration_patches_list = []
        removed_patches = []
        ## concurrently compute the surfaces of the combined volumes of interest
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(sheet_reconstruction, [(subvolume_patch, subvolumes[i], iteration, min_sheet_size, sample_ratio, overlapp_threshold) for i, subvolume_patch in enumerate(subvolumes_patches)])

        for subvolume_patches_list_filtered, removed_patches_to_save in results:
            iteration_patches_list += subvolume_patches_list_filtered
            removed_patches += removed_patches_to_save

        # Remove duplicate patches
        iteration_patches_list = remove_duplicate_patches(iteration_patches_list)

        # Save the combined subvolume surfaces
        subvolume_surfaces_path = path.replace("blocks", f"blocks_iteration_{iteration}") + ".pkl"
        save_patches_list(iteration_patches_list, subvolume_surfaces_path)

        # Save the removed patches
        removed_patches_path = path.replace("blocks", f"blocks_iteration_{iteration}_removed_patches") + ".pkl"
        save_patches_list(removed_patches, removed_patches_path)

        # Iterate patches list
        patches_list = iteration_patches_list

def volume_computation(args):
    main_sheet_volume_patches, new_patch, path, overlapp_threshold, sample_ratio_score, sample_ratio = args

    volume_new_patch_id, patch_id, offset_angle, switched_winding = new_patch
    patch_offset_angle, patch_points_sampled, scores_volume = calculate_main_sheet_scores(main_sheet_volume_patches, volume_new_patch_id, patch_id, offset_angle, switched_winding, path=path, overlapp_threshold=overlapp_threshold, sample_ratio_score=sample_ratio_score, sample_ratio=sample_ratio) # recalculate the scores of the volume id region for all regional patches wrt the updated main sheet
    
    return patch_offset_angle, patch_points_sampled, scores_volume, new_patch

def region_growing(path, volume_start=(800, 1300, 750), surface_nr=4, overlapp_threshold={"score_threshold": 0.8, "nr_points_min": 20.0, "nr_points_max": 500.0}, sample_ratio_score=0.1, sample_ratio=0.1, continue_segmentation=False, sheet_angle_range=None, sheet_z_range=None):
    """
    Grows a sheet from a starting patch
    """
    global main_sheet_data
    volume_block_used = {}
    volume_blocks_scores = {}
    main_sheet = {}
    # get current date as (example 20231019094556) name
    main_sheet_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = "/".join(path.split("/")[:-1]) + f"/segmentations/{main_sheet_name}/"
    # Create the save path
    os.makedirs(save_path, exist_ok=True)
    save_name = save_path + f"{main_sheet_name}.ta"
    save_progress_path = save_path + "progress/"
    # Create the save progress path
    os.makedirs(save_progress_path, exist_ok=True)
    save_progress_template = save_progress_path + f"{main_sheet_name}_"+"{}.ta"

    print(f"Main Sheet save location is: {save_path}")
    
    path_ta = path.replace("blocks", "main_sheet") + ".ta"
    if continue_segmentation and os.path.exists(path_ta):
        print("\033[94m[ThaumatoAnakalyptor]:\033[0m Continuing segmentation.")
        main_sheet, volume_blocks_scores = load_main_sheet(path=path, path_ta=path_ta, sample_ratio_score=sample_ratio_score, sample_ratio=sample_ratio)
    else:
        print("\033[94m[ThaumatoAnakalyptor]:\033[0m Starting new segmentation.")
        # Initialize main sheet and scores
        patch_offset_angle, patch_points_sampled, scores_volume = calculate_main_sheet_scores([], volume_start, surface_nr, 0.0, False, path=path, overlapp_threshold=overlapp_threshold, sample_ratio_score=sample_ratio_score, sample_ratio=sample_ratio)
        # Add initial patch to the main sheet
        main_sheet[volume_start] = {surface_nr: {"offset_angle": patch_offset_angle, "displaying_points": patch_points_sampled}}

        # Update the scores
        for score_volume in scores_volume:
            # Extract the data
            volume_id, patch_id, score_patch, offset_angle, patch_surface, switched_winding = score_volume
            # Add the score to the volume blocks scores
            if not volume_id in volume_blocks_scores:
                volume_blocks_scores[volume_id] = {}
            if not patch_id in volume_blocks_scores[volume_id]:
                volume_blocks_scores[volume_id][patch_id] = {"score": score_patch, "angle": 0.0, "offset_angle": offset_angle, "switched_winding": switched_winding, "surface": patch_surface}

    # time the while loop
    start = time.time()
    round_count = 0
    max_rounds = -1
    patches_nr_count = 0
    min_angle, max_angle = 0, 0
    max_threads = min(multiprocessing.cpu_count(), overlapp_threshold["max_threads"])
    last_patches_nr_count = 0
    while (max_rounds == -1 or round_count < max_rounds): # while patches are not assigned to the main sheet, loop
        loop_start = time.time()
        volume_block_used = {}
        max_patches = min(len(main_sheet) // 100 + 1, max_threads)
        volume_patches = pick_next_patches(volume_blocks_scores, volume_start, max_patches=max_patches, sheet_angle_range=sheet_angle_range, sheet_z_range=sheet_z_range, overlapp_threshold=overlapp_threshold) # pick good patches
        if len(volume_patches) == 0:
            break
        patches = []
        angles = []
        points = []
        scores = []
        added_patch = False
        # # Single Threaded
        # for volume_patch in volume_patches:
        #     # Build main sheet subset of interest in the volume id region
        #     main_sheet_volume_ids, new_patch, score = volume_patch
        #     if score < overlapp_threshold["score_threshold"]:
        #         continue
        #     added_patch = True
        #     print(f"Score of current picked patch: {score}")
        #     main_sheet_volume_patches = []
        #     for volume_id in main_sheet_volume_ids:
        #         if volume_id in main_sheet:
        #             for patch_id in main_sheet[volume_id]:
        #                 main_sheet_volume_patches.append((volume_id, patch_id, main_sheet[volume_id][patch_id]["offset_angle"]))
            
        #     volume_new_patch_id, patch_id = new_patch
        #     patch_offset_angle, patch_points_sampled, scores_volume = calculate_main_sheet_scores(main_sheet_volume_patches, volume_new_patch_id, patch_id, path=path, overlapp_threshold=overlapp_threshold, sample_ratio=0.1) # recalculate the scores of the volume id region for all regional patches wrt the updated main sheet

        #     patches.append(new_patch)
        #     angles.append(patch_offset_angle)
        #     points.append(patch_points_sampled)
        #     scores.extend(scores_volume)

        # # Set up multiprocessing
        main_sheet_volume_patches_list = []
        new_patch_list = []
        for volume_patch in volume_patches:
            # Build main sheet subset of interest in the volume id region
            main_sheet_volume_ids, new_patch, score = volume_patch
            if score < 0.0:
                continue
            added_patch = True
            main_sheet_volume_patches = build_main_sheet_patches_list(main_sheet_volume_ids, main_sheet)
            
            main_sheet_volume_patches_list.append(main_sheet_volume_patches)
            new_patch_list.append(new_patch)

        time_pre = time.time()

        ## concurrently combine picked patches into the main sheet
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(volume_computation, [(main_sheet_volume_patches_list[i], new_patch_list[i] , path, overlapp_threshold, sample_ratio_score, sample_ratio) for i in range(len(new_patch_list))])
        # filter out None results
        results = [result for result in results if result is not None]

        for patch_offset_angle, patch_points_sampled, scores_volume, new_patch in results:
            patches.append(new_patch)
            angles.append(patch_offset_angle)
            points.append(patch_points_sampled)
            scores.extend(scores_volume)

        if not added_patch:
            print(f"No patch added in round {round_count}!")
            break

        time_main = time.time()
        
        # Add the new patches to the main sheet
        update_main_sheet(main_sheet, patches, angles, points)

        # Update the scores
        update_scores(main_sheet, volume_blocks_scores, scores, patches, overlapp_threshold)

        # Update the volume blocks used
        update_volume_blocks_used(volume_block_used, volume_patches)

        # Min and max angles over all volume id, patch id:
        patches_nr_count = 0
        min_angle, max_angle = None, None
        for volume_id in main_sheet:
            for patch_id in main_sheet[volume_id]:
                patches_nr_count += 1
                if min_angle is None or min_angle > main_sheet[volume_id][patch_id]["offset_angle"]:
                    min_angle = main_sheet[volume_id][patch_id]["offset_angle"]
                if max_angle is None or max_angle < main_sheet[volume_id][patch_id]["offset_angle"]:
                    max_angle = main_sheet[volume_id][patch_id]["offset_angle"]
        
        round_count += 1
        if round_count % 10 == 0:
            print(f"Saving main sheet")
            save_main_sheet(main_sheet, volume_blocks_scores, path.replace("blocks", "main_sheet") + ".ta")
            save_main_sheet(main_sheet, volume_blocks_scores, save_name)

        progress_save_intervall = 1000
        if patches_nr_count // progress_save_intervall > last_patches_nr_count // progress_save_intervall:
            save_main_sheet(main_sheet, volume_blocks_scores, save_progress_template.format(round_count))
        last_patches_nr_count = patches_nr_count

        time_post = time.time()
        # print(f"Timings. Preprocess: {time_pre - loop_start:.2f}, Main: {time_main - time_pre:.2f}, Postprocess: {time_post - time_main:.2f}")
        print(f"Round {round_count}. Main Sheet consists of {patches_nr_count} patches. Total runtime is {int(time.time() - start)} seconds. Round took {(time.time() - loop_start):.1f} seconds to add {len(volume_patches)} new patches. Main patch min angle is {min_angle:.3f} and max angle is {max_angle:.3f}.")

    print(f"Finished after {round_count} rounds with total runtime of {int(time.time() - start)} seconds and {(time.time() - start) / (round_count + 0.001):.1f} seconds per round. Main Sheet consists of {patches_nr_count} patches. Min angle is {min_angle:.3f} and max angle is {max_angle:.3f}.")

    # Display the main sheet
    # display_main_sheet(main_sheet)
    main_sheet_data = main_sheet # Update this with new data
    data_updated_event.set()  # Signal that new data is available

def display_thread():
    global main_sheet_data
    while True:
        data_updated_event.wait()  # Wait until the data is updated
        print("Updating display")
        data_updated_event.clear()  # Reset the signal
        if main_sheet_data:
            try:
                display_main_sheet(main_sheet_data)
            except:
                pass

def find_starting_patch(sheet_points, path, subvolume_size=50, min_points=1000):
    subvolume_size_half = subvolume_size // 2 # is 25
    # From True Scroll to working frame. spelufo offset, axis flips, scaling
    # Spelufo offset
    sheet_points = np.array(sheet_points) + 500
    # Axis flips
    axis_indices = [1, 2, 0]
    sheet_points = sheet_points[:, axis_indices]
    # Scaling
    sheet_points = sheet_points / (200.0 / 50.0)
    # Check that all axis min max dist is less then 50 withing steps of 0-50, 25-75, 50-100, ...
    rounded_down_25_min = np.floor(np.min(sheet_points, axis=0) / subvolume_size_half) * subvolume_size_half
    rounded_up_25_max = np.ceil(np.max(sheet_points, axis=0) / subvolume_size_half) * subvolume_size_half
    min_max_dist = rounded_up_25_max - rounded_down_25_min
    assert np.all(min_max_dist <= subvolume_size), f"Min max dist is {min_max_dist} for sheet points {sheet_points}. you should pick Points closer together for starting patch selection. Continueing, but this might mean the wrong starting patch is selected."
    # centroid
    centroid = np.mean(sheet_points, axis=0)
    # calculate volume of interest
    volume_start = np.array(rounded_down_25_min).astype(int)
    # load all patches in the volume of interest
    path_surrounding = path + f"/{volume_start[0]:06}_{volume_start[1]:06}_{volume_start[2]:06}"
    patches_list = subvolume_surface_patches_folder(path_surrounding, subvolume_size)
    print(f"Length of patches list: {len(patches_list)}")
    # min dist to rounded down centroid for each patch
    dists_vectors = [patch["points"] - centroid for patch in patches_list]
    min_dists = [np.min(np.linalg.norm(dists_vector, axis=1)) for dists_vector in dists_vectors if dists_vector.shape[0] > min_points]
    filtered_indices = [i for i in range(len(dists_vectors)) if dists_vectors[i].shape[0] > min_points]
    # get argmin of min dist from interest points to min patch point dist
    min_dist_arg_filtered = np.argmin(min_dists)
    min_dist_arg = filtered_indices[min_dist_arg_filtered]
    print(f"Minimum distance to centroid for closest valid patch is {min_dists[min_dist_arg_filtered]} and has {dists_vectors[min_dist_arg].shape[0]} points.")
    # return volume id and patch id
    volume_id = np.array(patches_list[min_dist_arg]["ids"][0][:3]).astype(int)
    patch_id = int(patches_list[min_dist_arg]["ids"][0][3])
    return volume_id, patch_id

def main():
    path = "/media/julian/SSD4TB/scroll3_surface_points/point_cloud_colorized_verso_subvolume_blocks"
    iteration = 0
    volume_start = np.array([800, 1300, 750]) # 000800_001300_000750
    surface_nr=9
    # points_of_interest_org_frame = [[3483, 3325, 11500]] # scroll 1
    points_of_interest_org_frame = [[1650, 3300, 5000]] # scroll 3
    print(f"Starting patch: {volume_start}, {surface_nr}")
    volume_size = np.array([8, 8, 8])

    sample_ratio_score = 0.03 # 0.1
    sample_ratio = 0.01
    # sheet_angle_range = (-1100, 1100) # scroll 1
    sheet_angle_range = (-1100, 1100) # scroll 3
    sheet_angle_range = (-1100, 3600) # scroll 3
    # sheet_angle_range = (-420, 420) # scroll 3
    # sheet_z_range = (0, 12200) # scroll 1
    sheet_z_range = (1500, 10000) # scroll 3
    sheet_z_range = (1000, 10500) # scroll 3
    sheet_z_range = (0, 12000) # scroll 3
    continue_segmentation = True
    overlapp_threshold = {"sample_ratio_score": sample_ratio_score, "display": False, "print_scores": True, "picked_scores_similarity": 0.7, "final_score_max": 1.5, "final_score_min": 0.2, "score_threshold": 0.20, "cost_threshold": 16, "cost_percentile": 75, "cost_percentile_threshold": 12, 
                          "cost_sheet_distance_threshold": 4.0, "rounddown_best_score": 0.05,
                          "cost_threshold_prediction": 2.0, "min_prediction_threshold": 0.10, "nr_points_min": 400.0, "nr_points_max": 4000.0, "min_patch_points": 500.0, 
                          "winding_angle_range": None, "multiple_instances_per_batch_factor": 1.0,
                          "epsilon": 1e-5, "angle_tolerance": 85, "max_threads": 30,
                          "min_points_winding_switch": 2800, "min_winding_switch_sheet_distance": 7, "max_winding_switch_sheet_distance": 13, "winding_switch_sheet_score_factor": 2.5, "winding_direction": 1.0, "enable_winding_switch": True,
                          "surrounding_patches_size": 3, "max_sheet_clip_distance": 60
                          }
    # "winding_direction": -1 # scroll 1
    # "winding_direction": 1 # scroll 3
    # "winding_switch_sheet_score_factor": 1.1 # scroll 1

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Compute ThaumatoAnakalyptor Papyrus Sheets')
    parser.add_argument('--path', type=str, help='Papyrus instance patch path (containing .tar)', default=path)
    parser.add_argument('--print_scores', type=bool,help='Print scores of patches for sheet', default=overlapp_threshold["print_scores"])
    parser.add_argument('--sample_ratio_score', type=float,help='Sample ratio to apply to the pointcloud patches', default=overlapp_threshold["sample_ratio_score"])
    parser.add_argument('--score_threshold', type=float,help='Score threshold to add patches to sheet', default=overlapp_threshold["score_threshold"])
    parser.add_argument('--rounddown_best_score', type=float,help='Pick best score threshold to round down to zero from. Combats segmentation speed slowdown towards the end of segmentation.', default=overlapp_threshold["rounddown_best_score"])
    parser.add_argument('--winding_direction', type=int,help='Winding direction of sheet in scroll scan', default=overlapp_threshold["winding_direction"])
    parser.add_argument('--sheet_z_range', type=int, nargs=2,help='Z range of segmentation', default=[sheet_z_range[0], sheet_z_range[1]])
    parser.add_argument('--sheet_angle_range', type=int, nargs=2,help='Angle range of the sheet winding for segmentation', default=[sheet_angle_range[0], sheet_angle_range[1]])
    parser.add_argument('--starting_point', type=int, nargs=3,help='Starting point for a new segmentation', default=[points_of_interest_org_frame[0][0], points_of_interest_org_frame[0][1], points_of_interest_org_frame[0][2]])
    parser.add_argument('--continue_segmentation', type=int,help='Continue previous segmentation (point_cloud_colorized_subvolume_main_sheet.ta). 1 to continue, 0 to restart.', default=int(continue_segmentation))
    parser.add_argument('--enable_winding_switch', type=int,help='Enable switching of winding if two sheets lay on top of each eather. 1 enable, 0 disable.', default=int(overlapp_threshold["enable_winding_switch"]))
    parser.add_argument('--max_threads', type=int,help='Maximum number of thread to use during computation. Has a slight influence on the quality of segmentations for small segments.', default=int(overlapp_threshold["max_threads"]))
    parser.add_argument('--surrounding_patches_size', type=int,help=f'Number of surrounding half-overlapping patches in each dimension direction to consider when calculating patch pair similarity scores. Default is {overlapp_threshold["surrounding_patches_size"]}.', default=int(overlapp_threshold["surrounding_patches_size"]))

    # Take arguments back over
    args = parser.parse_args()
    path = args.path
    overlapp_threshold["print_scores"] = args.print_scores
    overlapp_threshold["sample_ratio_score"] = args.sample_ratio_score
    overlapp_threshold["score_threshold"] = args.score_threshold
    overlapp_threshold["rounddown_best_score"] = args.rounddown_best_score
    overlapp_threshold["winding_direction"] = args.winding_direction
    sheet_z_range = args.sheet_z_range
    sheet_angle_range = args.sheet_angle_range
    points_of_interest_org_frame = [args.starting_point]
    continue_segmentation = bool(int(args.continue_segmentation))
    overlapp_threshold["enable_winding_switch"] = bool(args.enable_winding_switch)
    overlapp_threshold["max_threads"] = args.max_threads
    overlapp_threshold["surrounding_patches_size"] = args.surrounding_patches_size

    volume_start, surface_nr = find_starting_patch(points_of_interest_org_frame, path)
    print(f"Starting segmentation with parameters: Path {path}, Iteration {iteration}, Volume start {volume_start}, Volume size {volume_size}, Overlapp threshold {overlapp_threshold}, Sample ratio score {sample_ratio_score}, Sample ratio {sample_ratio}, Sheet angle range {sheet_angle_range}, Sheet z range {sheet_z_range}, Continue segmentation {continue_segmentation}")
    # threading.Thread(target=display_thread).start()

    region_growing(path, volume_start=tuple(volume_start), surface_nr=surface_nr, overlapp_threshold=overlapp_threshold, sample_ratio_score=overlapp_threshold["sample_ratio_score"], sample_ratio=sample_ratio, continue_segmentation=continue_segmentation, sheet_angle_range=sheet_angle_range, sheet_z_range=sheet_z_range)
    # divide_and_conquer(path, iteration, volume_start, volume_size, overlapp_threshold=overlapp_threshold)

if __name__ == "__main__":
    main()