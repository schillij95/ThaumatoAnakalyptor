### Julian Schilliger - ThaumatoAnakalyptor - 2024

import numpy as np
from tqdm import tqdm
import pickle
import glob
import os
import tarfile
import tempfile
import open3d as o3d
import json
from multiprocessing import Pool
import time
import argparse
import yaml
import ezdxf
from ezdxf.math import Vec3
from copy import deepcopy
import matplotlib
import shutil

from .instances_to_sheets import select_points, get_vector_mean, alpha_angles, adjust_angles_zero, adjust_angles_offset, add_overlapp_entries_to_patches_list, assign_points_to_tiles, compute_overlap_for_pair, overlapp_score, fit_sheet, winding_switch_sheet_score_raw_precomputed_surface, find_starting_patch, save_main_sheet, update_main_sheet
from .sheet_to_mesh import load_xyz_from_file, scale_points, umbilicus_xz_at_y
# from .genetic_sheet_stitching import solve as solve_genetic
# from .genetic_sheet_stitching import solve_ as solve_genetic_
import sys
### C++ speed up. not yet fully implemented
sys.path.append('ThaumatoAnakalyptor/sheet_generation/build')
import pointcloud_processing
import sheet_generation
import evolve_graph
from termcolor import colored

def scale_points_(points, scale=4.0, axis_offset=-500):
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
    """
    Returns the signed angle in radians between vectors 'v1' and 'v2'.

    Examples:
        >>> angle_between(np.array([1, 0]), np.array([0, 1]))
        1.5707963267948966
        >>> angle_between(np.array([1, 0]), np.array([1, 0]))
        0.0
        >>> angle_between(np.array([1, 0]), np.array([-1, 0]))
        3.141592653589793
        >>> angle_between(np.array([1, 0]), np.array([0, -1]))
        -1.5707963267948966
    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arctan2(v2_u[1], v2_u[0]) - np.arctan2(v1_u[1], v1_u[0])
    assert angle is not None, "Angle is None."
    return angle

def surrounding_volumes(volume_id, volume_size=50):
    """
    Returns the surrounding volumes of a volume
    """
    volume_id_x = volume_id[0]
    volume_id_y = volume_id[1]
    volume_id_z = volume_id[2]
    vs = (volume_size//2)
    surrounding_volumes = [None]*3
    surrounding_volumes[0] = (volume_id_x + vs, volume_id_y, volume_id_z)
    surrounding_volumes[1] = (volume_id_x, volume_id_y + vs, volume_id_z)
    surrounding_volumes[2] = (volume_id_x, volume_id_y, volume_id_z + vs)
    return surrounding_volumes

def volumes_of_point(point, volume_size=50):
    """
    Returns all the volumes containing the point
    """
    point = np.array(point)
    size_half = volume_size//2
    volume_quadrant = np.floor(point / size_half).astype(int) * size_half
    volumes = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                volumes.append(tuple(volume_quadrant + size_half * np.array([i, j, k])))
    return volumes


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

def build_patch(main_sheet_patch, subvolume_size, path, sample_ratio=1.0, align_and_flip_normals=False):
    subvolume_size = np.array(subvolume_size)
    ((x, y, z), main_sheet_surface_nr, offset_angle) = main_sheet_patch
    file = path + f"/{x:06}_{y:06}_{z:06}/surface_{main_sheet_surface_nr}.ply"
    res = load_ply(path)
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
                # print(ply_file_path)
                # ids = tuple([*map(int, ply_member.name.split("_"))])
                ids = tuple([*map(int, tar_filename.split(".")[-2].split("/")[-1].split("_"))]+[int(ply_file.split(".")[-2].split("_")[-1])])
                ids = (int(ids[0]), int(ids[1]), int(ids[2]), int(ids[3]))
                main_sheet_patch = (ids[:3], ids[3], float(0.0))
                surface_dict, _ = build_patch(main_sheet_patch, tuple(subvolume_size), ply_file_path, sample_ratio=float(sample_ratio))
                patches_list.append(surface_dict)

    return patches_list

def build_patch_tar(main_sheet_patch, subvolume_size, path, sample_ratio=1.0):
    """
    Load surface patch from overlapping subvolumes instances predictions.
    """

    # Standardize subvolume_size to a NumPy array
    subvolume_size = np.atleast_1d(subvolume_size).astype(int)
    if subvolume_size.shape[0] == 1:
        subvolume_size = np.repeat(subvolume_size, 3)

    xyz, patch_nr, _ = main_sheet_patch
    file = path + f"/{xyz[0]:06}_{xyz[1]:06}_{xyz[2]:06}"
    tar_filename = f"{file}.tar"

    if os.path.isfile(tar_filename):
        with tarfile.open(tar_filename, 'r') as archive, tempfile.TemporaryDirectory() as temp_dir:
            # Extract all .ply files at once
            archive.extractall(path=temp_dir, members=archive.getmembers())

            ply_file = f"surface_{patch_nr}.ply"
            ply_file_path = os.path.join(temp_dir, ply_file)
            ids = tuple([*map(int, tar_filename.split(".")[-2].split("/")[-1].split("_"))]+[int(ply_file.split(".")[-2].split("_")[-1])])
            ids = (int(ids[0]), int(ids[1]), int(ids[2]), int(ids[3]))
            res = build_patch(main_sheet_patch, tuple(subvolume_size), ply_file_path, sample_ratio=float(sample_ratio))
            return res

def load_graph(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

class Graph:
    def __init__(self):
        self.edges = {}  # Stores edges with update matrices and certainty factors
        self.nodes = {}  # Stores node beliefs and fixed status

    def add_node(self, node, centroid, winding_angle=None):
        node = tuple(node)
        self.nodes[node] = {'centroid': centroid, "winding_angle": winding_angle}

    def compute_node_edges(self):
        """
        Compute the edges for each node and store them in the node dictionary.
        """
        print("Computing node edges...")
        for node in tqdm(self.nodes, desc="Adding nodes"):
            self.nodes[node]['edges'] = []
        for edge in tqdm(self.edges, desc="Computing node edges"):
            for k in self.edges[edge]:
                self.nodes[edge[0]]['edges'].append(edge)
                self.nodes[edge[1]]['edges'].append(edge)

    def remove_nodes_edges(self, nodes):
        """
        Remove nodes and their edges from the graph.
        """
        for node in tqdm(nodes, desc="Removing nodes"):
            # Delete Node Edges
            node_edges = list(self.nodes[node]['edges'])
            for edge in node_edges:
                node_ = edge[0] if edge[0] != node else edge[1]
                if node_ in self.nodes:
                    self.nodes[node_]['edges'].remove(edge)
                # Delete Edges
                if edge in self.edges:
                    del self.edges[edge]
            # Delete Node
            del self.nodes[node]
        # set length of nodes and edges
        self.nodes_length = len(self.nodes)
        self.edges_length = len(self.edges)

    def delete_edge(self, edge):
        """
        Delete an edge from the graph.
        """
        # Delete Edges
        try:
            del self.edges[edge]
        except:
            pass
        # Delete Node Edges
        node1 = edge[0]
        node2 = edge[1]
        self.nodes[node1]['edges'].remove(edge)
        self.nodes[node2]['edges'].remove(edge)

    def edge_exists(self, node1, node2):
        """
        Check if an edge exists in the graph.
        """
        node1 = tuple(node1)
        node2 = tuple(node2)
        return (node1, node2) in self.edges or (node2, node1) in self.edges

    def add_edge(self, node1, node2, certainty, sheet_offset_k=0.0, same_block=False, bad_edge=False):
        assert certainty > 0.0, "Certainty must be greater than 0."
        certainty = np.clip(certainty, 0.0, None)

        node1 = tuple(node1)
        node2 = tuple(node2)
        # Ensure node1 < node2 for bidirectional nodes
        if node2 < node1:
            node1, node2 = node2, node1
            sheet_offset_k = sheet_offset_k * (-1.0)
        sheet_offset_k = (float)(sheet_offset_k)
        if not (node1, node2) in self.edges:
            self.edges[(node1, node2)] = {}
        self.edges[(node1, node2)][sheet_offset_k] = {'certainty': certainty, 'sheet_offset_k': sheet_offset_k, 'same_block': same_block, 'bad_edge': bad_edge}

    def add_increment_edge(self, node1, node2, certainty, sheet_offset_k=0.0, same_block=False, bad_edge=False):
        assert certainty > 0.0, "Certainty must be greater than 0."
        certainty = np.clip(certainty, 0.0, None)

        node1 = tuple(node1)
        node2 = tuple(node2)
        # Ensure node1 < node2 for bidirectional nodes
        if node2 < node1:
            node1, node2 = node2, node1
            sheet_offset_k = sheet_offset_k * (-1.0)
        sheet_offset_k = (float)(sheet_offset_k)
        if not (node1, node2) in self.edges:
            self.edges[(node1, node2)] = {}
        if not sheet_offset_k in self.edges[(node1, node2)]:
            self.edges[(node1, node2)][sheet_offset_k] = {'certainty': 0.0, 'sheet_offset_k': sheet_offset_k, 'same_block': same_block, 'bad_edge': bad_edge}

        assert bad_edge == self.edges[(node1, node2)][sheet_offset_k]['bad_edge'], "Bad edge must be the same."
        if same_block != self.edges[(node1, node2)][sheet_offset_k]['same_block']:
            print("Same block must be the same.")
        self.edges[(node1, node2)][sheet_offset_k]['same_block'] = self.edges[(node1, node2)][sheet_offset_k]['same_block'] and same_block
        # Increment certainty
        self.edges[(node1, node2)][sheet_offset_k]['certainty'] += certainty

    def get_certainty(self, node1, node2, k):
        node1 = tuple(node1)
        node2 = tuple(node2)
        if node2 < node1:
            node1, node2 = node2, node1
            k = k * (-1.0)
        edge_dict = self.edges.get((node1, node2))
        if edge_dict is not None:
            edge = edge_dict.get(k)
            if edge is not None:
                return edge['certainty']
        return None
    
    def get_edge(self, node1, node2):
        if node2 < node1:
            node1, node2 = node2, node1
        return (node1, node2)
        
    def get_edge_ks(self, node1, node2):            
        k_factor = 1.0
        # Maintain bidirectional invariant
        if node2 < node1:
            node1, node2 = node2, node1
            k_factor = - 1.0
        edge_dict = self.edges.get((node1, node2))
        if edge_dict is not None:
            ks = []
            for k in edge_dict.keys():
                ks.append(k * k_factor)
            return ks
        else:
            raise KeyError(f"No edge found from {node1} to {node2}")

    def save_graph(self, path):
        print(f"Saving graph to {path} ...")
        # Save graph class object to file
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        print(f"Graph saved to {path}")

    def bfs(self, start_node):
        """
        Breadth-first search from start_node.
        """
        visited = set()
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                edges = self.nodes[node]['edges']
                for edge in edges:
                    queue.append(edge[0] if edge[0] != node else edge[1])
        return list(visited)

def score_same_block_patches(patch1, patch2, overlapp_threshold, umbilicus_distance):
    """
    Calculate the score between two patches from the same block.
    """
    # Calculate winding switch sheet scores
    distance_raw = umbilicus_distance(patch1, patch2)
    score_raw, k = np.mean(distance_raw), np.sign(distance_raw)
    k = k * overlapp_threshold["winding_direction"] # because of scroll's specific winding direction

    score_val = (overlapp_threshold["max_winding_switch_sheet_distance"] - score_raw) / (overlapp_threshold["max_winding_switch_sheet_distance"] - overlapp_threshold["min_winding_switch_sheet_distance"]) # calculate score
    
    score = score_val * overlapp_threshold["winding_switch_sheet_score_factor"]

    # Centroid distance between patches
    centroid1 = np.mean(patch1["points"], axis=0)
    centroid2 = np.mean(patch2["points"], axis=0)
    centroid_distance = np.linalg.norm(centroid1 - centroid2)

    # Check for enough points
    if centroid_distance > overlapp_threshold["max_winding_switch_sheet_distance"]:
        score = -1.0
    if centroid_distance < overlapp_threshold["min_winding_switch_sheet_distance"]:
        score = -1.0
    if score_val <= 0.0:
        score = -1.0
    if score_val >= 1.0:
        score = -1.0
    if patch1["points"].shape[0] < overlapp_threshold["min_points_winding_switch"] * overlapp_threshold["sample_ratio_score"]:
        score = -1.0
    if patch2["points"].shape[0] < overlapp_threshold["min_points_winding_switch"] * overlapp_threshold["sample_ratio_score"]:
        score = -1.0
    if score_raw < overlapp_threshold["min_winding_switch_sheet_distance"]: # whole volume went bad for winding switch
        score = -1.0
    if score_raw > overlapp_threshold["max_winding_switch_sheet_distance"]: # not good patch, but whole volume might still be good for winding switch
        score = -0.5
    if (patch1["patch_prediction_scores"][0] < overlapp_threshold["min_prediction_threshold"]) or (patch2["patch_prediction_scores"][0] < overlapp_threshold["min_prediction_threshold"]):
        score = -1.0

    return (score, k), patch1["anchor_angles"][0], patch2["anchor_angles"][0]

def process_same_block(main_block_patches_list, overlapp_threshold, umbilicus_distance):
    def scores_cleaned(direction_scores):
        if len(direction_scores) == 0:
            return []
        
        score_min = min(direction_scores, key=lambda x: x[2]) # return if there were bad scores
        if score_min[2] == -1.0:
            return [(score_min[0], score_min[1], -1.0, score_min[3], score_min[4], score_min[5])]
        
        score = max(direction_scores, key=lambda x: x[2])
        score_val = score[2]
        if score_val < 0.0:
            score_val = -1.0
        score = (score[0], score[1], score_val, score[3], score[4], score[5], score[6], score[7])

        return [score]

    # calculate switching scores of main patches
    score_switching_sheets = []
    score_bad_edges = []
    for i in range(len(main_block_patches_list)):
        score_switching_sheets_ = []
        for j in range(len(main_block_patches_list)):
            if i == j:
                continue
            (score_, k_), anchor_angle1, anchor_angle2 = score_same_block_patches(main_block_patches_list[i], main_block_patches_list[j], overlapp_threshold, umbilicus_distance)
            if score_ > 0.0:
                score_switching_sheets_.append((main_block_patches_list[i]['ids'][0], main_block_patches_list[j]['ids'][0], score_, k_, anchor_angle1, anchor_angle2, np.mean(main_block_patches_list[i]["points"], axis=0), np.mean(main_block_patches_list[j]["points"], axis=0)))

            # Add bad edges
            score_bad_edges.append((main_block_patches_list[i]['ids'][0], main_block_patches_list[j]['ids'][0], 1.0, 0.0, anchor_angle1, anchor_angle2, np.mean(main_block_patches_list[i]["points"], axis=0), np.mean(main_block_patches_list[j]["points"], axis=0)))

        # filter and only take the scores closest to the main patch (smallest scores) for each k in +1, -1
        direction1_scores = [score for score in score_switching_sheets_ if score[3] > 0.0]
        direction2_scores = [score for score in score_switching_sheets_ if score[3] < 0.0]
        
        score1 = scores_cleaned(direction1_scores)
        score2 = scores_cleaned(direction2_scores)
        score_switching_sheets += score1 + score2
    return score_switching_sheets, score_bad_edges

def score_other_block_patches(patches_list, i, j, overlapp_threshold):
    """
    Calculate the score between two patches from different blocks.
    """
    patch1 = patches_list[i]
    patch2 = patches_list[j]
    patches_list = [patch1, patch2]
    # Single threaded
    results = []
    results.append(compute_overlap_for_pair((0, patches_list, overlapp_threshold["epsilon"], overlapp_threshold["angle_tolerance"])))

    assert len(results) == 1, "Only one result should be returned."
    assert len(results[0]) == 1, "Only one pair of patches should be returned."

    # Combining results
    for result in results:
        for i, j, overlapp_percentage, overlap, non_overlap, points_overlap, angles_offset in result:
            patches_list[i]["overlapp_percentage"][j] = overlapp_percentage
            patches_list[i]["overlap"][j] = overlap
            patches_list[i]["non_overlap"][j] = non_overlap
            patches_list[i]["points_overlap"][j] = points_overlap
            score = overlapp_score(i, j, patches_list, overlapp_threshold=overlapp_threshold, sample_ratio=overlapp_threshold["sample_ratio_score"])

            if score <= 0.0:
                score = -1.0
            elif patches_list[j]["points"].shape[0] < overlapp_threshold["min_patch_points"] * overlapp_threshold["sample_ratio_score"]:
                score = -1.0
            elif patches_list[i]["patch_prediction_scores"][0] < overlapp_threshold["min_prediction_threshold"] or patches_list[j]["patch_prediction_scores"][0] < overlapp_threshold["min_prediction_threshold"]:
                score = -1.0
            elif overlapp_threshold["fit_sheet"]:
                cost_refined, cost_percentile, cost_sheet_distance, surface = fit_sheet(patches_list, i, j, overlapp_threshold["cost_percentile"], overlapp_threshold["epsilon"], overlapp_threshold["angle_tolerance"])
                if cost_refined >= overlapp_threshold["cost_threshold"]:
                    score = -1.0
                elif cost_percentile >= overlapp_threshold["cost_percentile_threshold"]:
                    score = -1.0
                elif cost_sheet_distance >= overlapp_threshold["cost_sheet_distance_threshold"]:
                    score = -1.0

    return score, patch1["anchor_angles"][0], patch2["anchor_angles"][0]

def process_block(args):
    """
    Worker function to process a single block.
    """
    file_path, path_instances, overlapp_threshold, umbilicus_data = args
    umbilicus_func = lambda z: umbilicus_xz_at_y(umbilicus_data, z)
    def umbilicus_distance(patch1, patch2):
        centroid1 = np.mean(patch1["points"], axis=0)
        centroid2 = np.mean(patch2["points"], axis=0)
        def d_(patch_centroid):
            umbilicus_point = umbilicus_func(patch_centroid[1])
            patch_centroid_vec = patch_centroid - umbilicus_point
            return np.linalg.norm(patch_centroid_vec)
        return d_(centroid1) - d_(centroid2)


    file_name = ".".join(file_path.split(".")[:-1])
    main_block_patches_list = subvolume_surface_patches_folder(file_name, sample_ratio=overlapp_threshold["sample_ratio_score"])

    patches_centroids = {}
    for patch in main_block_patches_list:
        patches_centroids[tuple(patch["ids"][0])] = np.mean(patch["points"], axis=0)

    # Extract block's integer ID
    block_id = [int(i) for i in file_path.split('/')[-1].split('.')[0].split("_")]
    block_id = np.array(block_id)
    surrounding_ids = surrounding_volumes(block_id)
    surrounding_blocks_patches_list = []
    surrounding_blocks_patches_list_ = []
    for surrounding_id in surrounding_ids:
        volume_path = path_instances + f"{file_path.split('/')[0]}/{surrounding_id[0]:06}_{surrounding_id[1]:06}_{surrounding_id[2]:06}"
        patches_temp = subvolume_surface_patches_folder(volume_path, sample_ratio=overlapp_threshold["sample_ratio_score"])
        surrounding_blocks_patches_list.append(patches_temp)
        surrounding_blocks_patches_list_.extend(patches_temp)

    # Add the overlap base to the patches list that contains the points + normals + scores only before
    patches_list = main_block_patches_list + surrounding_blocks_patches_list_
    add_overlapp_entries_to_patches_list(patches_list)
    subvolume = {"start": block_id - 50, "end": block_id + 50}

    # Assign points to tiles
    assign_points_to_tiles(patches_list, subvolume, tiling=3)

    # calculate scores between each main block patch and surrounding blocks patches
    score_sheets = []
    for i, main_block_patch in enumerate(main_block_patches_list):
        for surrounding_blocks_patches in surrounding_blocks_patches_list:
            score_sheets_patch = []
            for j, surrounding_block_patch in enumerate(surrounding_blocks_patches):
                patches_list_ = [main_block_patch, surrounding_block_patch]
                score_ = score_other_block_patches(patches_list_, 0, 1, overlapp_threshold) # score, anchor_angle1, anchor_angle2
                if score_[0] > overlapp_threshold["final_score_min"]:
                    score_sheets_patch.append((main_block_patch['ids'][0], surrounding_block_patch['ids'][0], score_[0], None, score_[1], score_[2], np.mean(main_block_patch["points"], axis=0), np.mean(surrounding_block_patch["points"], axis=0)))

            # Find the best score for each main block patch
            if len(score_sheets_patch) > 0:
                score_sheets_patch = max(score_sheets_patch, key=lambda x: x[2])
                score_sheets.append(score_sheets_patch)
    
    score_switching_sheets, score_bad_edges = process_same_block(main_block_patches_list, overlapp_threshold, umbilicus_distance)

    # Process and return results...
    return score_sheets, score_switching_sheets, score_bad_edges, patches_centroids

class ScrollGraph(Graph):
    def __init__(self, overlapp_threshold, umbilicus_path):
        super().__init__()
        self.set_overlapp_threshold(overlapp_threshold)
        self.umbilicus_path = umbilicus_path
        self.init_umbilicus(umbilicus_path)

    def init_umbilicus(self, umbilicus_path):
        # Load the umbilicus data
        umbilicus_data = load_xyz_from_file(umbilicus_path)
        # scale and swap axis
        self.umbilicus_data = scale_points(umbilicus_data, 50.0/200.0, axis_offset=0)

    def add_switch_edge(self, node_lower, node_upper, certainty, same_block):
        # Build a switch edge between two nodes
        self.add_edge(node_lower, node_upper, certainty, sheet_offset_k=1.0, same_block=same_block)

    def add_same_sheet_edge(self, node1, node2, certainty, same_block):
        # Build a same sheet edge between two nodes
        self.add_edge(node1, node2, certainty, sheet_offset_k=0.0, same_block=same_block)

    def build_other_block_edges(self, score_sheets):
        # Define a wrapper function for umbilicus_xz_at_y
        umbilicus_func = lambda z: umbilicus_xz_at_y(self.umbilicus_data, z)
        # Build edges between patches from different blocks
        for score_ in score_sheets:
            id1, id2, score, _, anchor_angle1, anchor_angle2, centroid1, centroid2 = score_
            if score < self.overlapp_threshold["final_score_min"]:
                continue
            umbilicus_point1 = umbilicus_func(centroid1[1])[[0, 2]]
            umbilicus_vector1 = umbilicus_point1 - centroid1[[0, 2]]
            angle1 = angle_between(umbilicus_vector1) * 180.0 / np.pi
            umbilicus_point2 = umbilicus_func(centroid2[1])[[0, 2]]
            umbilicus_vector2 = umbilicus_point2 - centroid2[[0, 2]]
            angle2 = angle_between(umbilicus_vector2) * 180.0 / np.pi
            if abs(angle1 - angle2) > 180.0:
                if angle1 > angle2:
                    id1, id2 = id2, id1
                self.add_switch_edge(id1, id2, score, same_block=False)
            else:
                self.add_same_sheet_edge(id1, id2, score, same_block=False)

    def build_same_block_edges(self, score_switching_sheets):
        # Define a wrapper function for umbilicus_xz_at_y
        umbilicus_func = lambda z: umbilicus_xz_at_y(self.umbilicus_data, z)
        # Build edges between patches from the same block
        disregarding_count = 0
        total_count = 0
        grand_total = 0
        for score_ in score_switching_sheets:
            grand_total += 1
            id1, id2, score, k, anchor_angle1, anchor_angle2, centroid1, centroid2 = score_
            if score < 0.0:
                continue
            total_count += 1
            umbilicus_point1 = umbilicus_func(centroid1[1])[[0, 2]]
            umbilicus_vector1 = umbilicus_point1 - centroid1[[0, 2]]
            angle1 = angle_between(umbilicus_vector1) * 180.0 / np.pi
            umbilicus_point2 = umbilicus_func(centroid2[1])[[0, 2]]
            umbilicus_vector2 = umbilicus_point2 - centroid2[[0, 2]]
            angle2 = angle_between(umbilicus_vector2) * 180.0 / np.pi
            if abs(angle1 - angle2) > 180.0:
                disregarding_count += 1
            else:
                if k < 0.0:
                    id1, id2 = id2, id1
                self.add_switch_edge(id1, id2, score, same_block=True)

    def build_bad_edges(self, score_bad_edges):
        for score_ in score_bad_edges:
            node1, node2, score, k, anchor_angle1, anchor_angle2, centroid1, centroid2 = score_
            # Build a same sheet edge between two nodes in same subvolume that is bad
            self.add_edge(node1, node2, 1.0, sheet_offset_k=0.0, same_block=True, bad_edge=True)

    def filter_blocks(self, blocks_tar_files, blocks_tar_files_int, start_block, distance):
        blocks_tar_files_int = np.array(blocks_tar_files_int)
        start_block = np.array(start_block)
        distances = np.abs(blocks_tar_files_int - start_block)
        distances = np.sum(distances, axis=1)
        filter_mask = np.sum(np.abs(blocks_tar_files_int - start_block), axis=1) < distance
        blocks_tar_files_int = blocks_tar_files_int[filter_mask]
        blocks_tar_files = [blocks_tar_files[i] for i in range(len(blocks_tar_files)) if filter_mask[i]]
        return blocks_tar_files, blocks_tar_files_int
    
    def filter_blocks_z(self, blocks_tar_files, z_min, z_max):
        # Filter blocks by z range
        blocks_tar_files_int = [[int(i) for i in x.split('/')[-1].split('.')[0].split("_")] for x in blocks_tar_files]
        blocks_tar_files_int = np.array(blocks_tar_files_int)
        filter_mask = np.logical_and(blocks_tar_files_int[:,1] >= z_min, blocks_tar_files_int[:,1] <= z_max)
        blocks_tar_files = [blocks_tar_files[i] for i in range(len(blocks_tar_files)) if filter_mask[i]]
        return blocks_tar_files
    
    def largest_connected_component(self, delete_nodes=True, min_size=None):
        print("Finding largest connected component...")        
        # walk the graph from the start node
        visited = set()
        # tqdm object showing process of visited nodes untill all nodes are visited
        tqdm_object = tqdm(total=len(self.nodes))
        components = []
        starting_index = 0
        def build_nodes(edges):
            nodes = set()
            for edge in edges:
                nodes.add(edge[0])
                nodes.add(edge[1])
            return list(nodes)
        nodes = build_nodes(self.edges)
        max_component_length = 0
        nodes_length = len(nodes)
        nodes_remaining = nodes_length
        print(f"Number of active nodes: {nodes_length}, number of total nodes: {len(self.nodes)}")
        while True:
            start_node = None
            # Pick first available unvisited node
            for node_idx in range(starting_index, nodes_length):
                node = nodes[node_idx]
                if node not in visited:
                    start_node = node
                    break
            if start_node is None:
                break    

            queue = [start_node]
            component = set()
            while queue:
                node = queue.pop(0)
                component.add(node)
                if node not in visited:
                    tqdm_object.update(1)
                    visited.add(node)
                    edges = self.nodes[node]['edges']
                    for edge in edges:
                        queue.append(edge[0] if edge[0] != node else edge[1])

            components.append(component)
            # update component length tracking
            component_length = len(component)
            max_component_length = max(max_component_length, component_length)
            nodes_remaining -= component_length
            if (min_size is None) and (nodes_remaining < max_component_length): # early stopping, already found the largest component
                print(f"breaking in early stopping")
                print()
                break

        nodes_total = len(self.nodes)
        edges_total = len(self.edges)
        print(f"number of nodes: {nodes_total}, number of edges: {edges_total}")

        # largest component
        largest_component = list(max(components, key=len))
        if min_size is not None:
            components = [component for component in components if len(component) >= min_size]
            for component in components:
                largest_component.extend(list(component))
        largest_component = set(largest_component)
        
        # Remove all other Nodes, Edges and Node Edges
        if delete_nodes:
            other_nodes = [node for node in self.nodes if node not in largest_component]
            self.remove_nodes_edges(other_nodes)
        
        print(f"Pruned {nodes_total - len(self.nodes)} nodes. Of {nodes_total} nodes.")
        print(f"Pruned {edges_total - len(self.edges)} edges. Of {edges_total} edges.")

        result = list(largest_component)
        print(f"Found largest connected component with {len(result)} nodes.")
        return result
    
    def update_graph_version(self):
        edges = self.edges
        self.edges = {}
        for edge in tqdm(edges):
            self.add_edge(edge[0], edge[1], edges[edge]['certainty'], edges[edge]['sheet_offset_k'], edges[edge]['same_block'])
        self.compute_node_edges()

    def flip_winding_direction(self):
        for edge in tqdm(self.edges):
            for k in self.edges[edge]:
                if self.edges[edge][k]['same_block']:
                    self.edges[edge][k]['sheet_offset_k'] = - self.edges[edge][k]['sheet_offset_k']

    def compute_bad_edges(self, iteration, k_factor_bad=1.0):
        print(f"Bad k factor is {k_factor_bad}.")
        # Compute bad edges
        node_cubes = {}
        for node in tqdm(self.nodes):
            if node[:3] not in node_cubes:
                node_cubes[node[:3]] = []
            node_cubes[node[:3]].append(node)

        # add bad edges between adjacent nodes
        # two indices equal, abs third is 25
        count_added_bad_edges = 0
        for node_cube in node_cubes:
            node_cube = np.array(node_cube)
            adjacent_cubes = [node_cube, node_cube + np.array([0, 0, 25]), node_cube + np.array([0, 25, 0]), node_cube + np.array([25, 0, 0]), node_cube + np.array([0, 0, -25]), node_cube + np.array([0, -25, 0]), node_cube + np.array([-25, 0, 0])]
            for adjacent_cube in adjacent_cubes:
                if tuple(adjacent_cube) in node_cubes:
                    node1s = node_cubes[tuple(node_cube)]
                    node2s = node_cubes[tuple(adjacent_cube)]
                    for node1 in node1s:
                        for node2 in node2s:
                            if node1 == node2:
                                continue
                            if not self.edge_exists(node1, node2):
                                self.add_edge(node1, node2, k_factor_bad*((iteration+1)**2), 0.0, same_block=True, bad_edge=True)
                                count_added_bad_edges += 1
                            else:
                                edge = self.get_edge(node1, node2)
                                for k in self.get_edge_ks(edge[0], edge[1]):
                                    if k == 0:
                                        continue
                                    same_block = self.edges[edge][k]['same_block']
                                    bad_edge = self.edges[edge][k]['bad_edge']
                                    if not bad_edge and same_block:
                                        if not k in self.edges[edge]:
                                            self.add_increment_edge(node1, node2, k_factor_bad*((iteration+1)**2), k, same_block=True, bad_edge=True)
                                            count_added_bad_edges += 1

            nodes = node_cubes[tuple(node_cube)]
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    self.add_edge(nodes[i], nodes[j], k_factor_bad*((iteration+1)**2), 0.0, same_block=True, bad_edge=True)

        print(f"Added {count_added_bad_edges} bad edges.")

    def adjust_edge_certainties(self):
        # Adjust same block and other block edge certainty values, such that both have the total quantity as even
        certainty_same_block = 0.0
        certainty_other_block = 0.0
        certainty_bad_block = 0.0
        for edge in tqdm(self.edges, desc="Calculating edge certainties"):
            for k in self.edges[edge]:
                if self.edges[edge][k]['bad_edge']:
                    certainty_bad_block += self.edges[edge][k]['certainty']
                elif self.edges[edge][k]['same_block']:
                    certainty_same_block += self.edges[edge][k]['certainty']
                else:
                    certainty_other_block += self.edges[edge][k]['certainty']

        factor_0 = 1.0
        factor_not_0 = 1.0*certainty_same_block / certainty_other_block
        factor_bad = certainty_same_block / certainty_bad_block
        # factor_bad = factor_bad**(2.0/3.0)
        # factor_bad = 2*factor_bad
        # factor_bad = 1.0
        print(f"Adjusted certainties: factor_0: {factor_0}, factor_not_0: {factor_not_0}, factor_bad: {factor_bad}")

        # adjust graph edge certainties
        for edge in tqdm(self.edges, desc="Adjusting edge certainties"):
            for k in self.edges[edge]:
                if self.edges[edge][k]['bad_edge']:
                    self.edges[edge][k]['certainty'] = factor_bad * self.edges[edge][k]['certainty']
                elif self.edges[edge][k]['same_block']:
                    self.edges[edge][k]['certainty'] = factor_0 * self.edges[edge][k]['certainty']
                else:
                    self.edges[edge][k]['certainty'] = factor_not_0 * self.edges[edge][k]['certainty']

        self.compute_node_edges()
        return (factor_0, factor_not_0, factor_bad)

    def build_graph(self, path_instances, start_point, num_processes=4, prune_unconnected=False):
        blocks_tar_files = glob.glob(path_instances + '/*.tar')

        #from original coordinates to instance coordinates
        start_block, patch_id = find_starting_patch([start_point], path_instances)
        self.start_block, self.patch_id, self.start_point = start_block, patch_id, start_point

        print(f"Found {len(blocks_tar_files)} blocks.")
        print("Building graph...")

        # Create a pool of worker processes
        with Pool(num_processes) as pool:
            # Map the process_block function to each file
            zipped_args = list(zip(blocks_tar_files, [path_instances] * len(blocks_tar_files), [self.overlapp_threshold] * len(blocks_tar_files), [self.umbilicus_data] * len(blocks_tar_files)))
            results = list(tqdm(pool.imap(process_block, zipped_args), total=len(zipped_args)))

        print(f"Number of results: {len(results)}")

        count_res = 0
        patches_centroids = {}
        # Process results from each worker
        for score_sheets, score_switching_sheets, score_bad_edges, volume_centroids in results:
            count_res += len(score_sheets)
            # Calculate scores, add patches edges to graph, etc.
            self.build_other_block_edges(score_sheets)
            self.build_same_block_edges(score_switching_sheets)
            self.build_bad_edges(score_bad_edges)
            patches_centroids.update(volume_centroids)
        print(f"Number of results: {count_res}")

        # Define a wrapper function for umbilicus_xz_at_y
        umbilicus_func = lambda z: umbilicus_xz_at_y(self.umbilicus_data, z)

        # Add patches as nodes to graph
        edges_keys = list(self.edges.keys())
        nodes_from_edges = set()
        for edge in tqdm(edges_keys, desc="Initializing nodes"):
            nodes_from_edges.add(edge[0])
            nodes_from_edges.add(edge[1])
        for node in nodes_from_edges:
            try:
                umbilicus_point = umbilicus_func(patches_centroids[node][1])[[0, 2]]
                umbilicus_vector = umbilicus_point - patches_centroids[node][[0, 2]]
                angle = angle_between(umbilicus_vector) * 180.0 / np.pi
                self.add_node(node, patches_centroids[node], winding_angle=angle)
            except:
                del self.edges[edge]

        # working but slow, version over it not yet tested should work tho (3. Mai 2024)
        # for edge in tqdm(edges_keys, desc="Initializing nodes"):
        #     try:
        #         umbilicus_point0 = umbilicus_func(patches_centroids[edge[0]][1])[[0, 2]]
        #         umbilicus_vector0 = umbilicus_point0 - patches_centroids[edge[0]][[0, 2]]
        #         angle0 = angle_between(umbilicus_vector0) * 180.0 / np.pi
        #         self.add_node(edge[0], patches_centroids[edge[0]], winding_angle=angle0)

        #         umbilicus_point1 = umbilicus_func(patches_centroids[edge[1]][1])[[0, 2]]
        #         umbilicus_vector1 = umbilicus_point1 - patches_centroids[edge[1]][[0, 2]]
        #         angle1 = angle_between(umbilicus_vector1) * 180.0 / np.pi
        #         self.add_node(edge[1], patches_centroids[edge[1]], winding_angle=angle1)
        #     except: 
        #         del self.edges[edge] # one node might be outside computed volume

        node_id = tuple((*start_block, patch_id))
        print(f"Start node: {node_id}, nr nodes: {len(self.nodes)}, nr edges: {len(self.edges)}")
        self.compute_node_edges()
        if prune_unconnected:
            print("Prunning unconnected nodes...")
            self.largest_connected_component()

        print(f"Nr nodes: {len(self.nodes)}, nr edges: {len(self.edges)}")

        return start_block, patch_id
    
    def naked_graph(self):
        """
        Return a naked graph with only nodes.
        """
        # Remove all nodes and edges
        graph = ScrollGraph(self.overlapp_threshold, self.umbilicus_path)
        # add all nodes
        for node in self.nodes:
            graph.add_node(node, self.nodes[node]['centroid'], winding_angle=self.nodes[node]['winding_angle'])
        return graph
    
    def set_overlapp_threshold(self, overlapp_threshold):
        if hasattr(self, "overlapp_threshold"):
            # compare if the new threshold is different from the old one in any aspect or subaspect
            different = False
            if not different and set(overlapp_threshold.keys()) != set(self.overlapp_threshold.keys()):
                different = True

            # Check if all values for each key are the same
            if not different:
                for key in overlapp_threshold:
                    if overlapp_threshold[key] != self.overlapp_threshold[key]:
                        different = True
                        break

            if not different:
                print("Overlapping threshold is the same. Not updating.")
                return

        print("Setting overlapping threshold...")
        self.overlapp_threshold = overlapp_threshold
        print("Overlapping threshold set.")

    def create_dxf_with_colored_polyline(self, filename, color=1, min_z=None, max_z=None):
        # Create a new DXF document.
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()

        for edge in tqdm(self.edges, desc="Creating DXF..."):
            ks = self.get_edge_ks(edge[0], edge[1])
            for k in ks:
                same_block = self.edges[edge][k]['same_block']
                c = color
                if same_block:
                    c = 2
                elif k != 0.0:
                    c = 5
                # Create polyline points 
                polyline_points = []
                polyline_points_raw = []
                to_add = True
                for pi, point in enumerate(edge):
                    centroid = self.nodes[point]['centroid']
                    if min_z is not None and centroid[1] < min_z:
                        to_add = False
                    if max_z is not None and centroid[1] > max_z:
                        to_add = False
                    if not to_add:
                        break
                    polyline_points.append(Vec3(int(centroid[0]), int(centroid[1]), int(centroid[2])))
                    polyline_points_raw.append(Vec3(int(centroid[0]), int(centroid[1]), int(centroid[2])))
                    
                    # Add an indicator which node of the edge has the smaller k value
                    if same_block:
                        if k > 0.0 and pi == 0:
                            polyline_points = [Vec3(int(centroid[0]), int(centroid[1]) + 10, int(centroid[2]))] + polyline_points
                        elif k < 0.0 and pi == 1:
                            polyline_points += [Vec3(int(centroid[0]), int(centroid[1]) + 10, int(centroid[2]))]

                if same_block and k == 0.0:
                    c = 4

                # Add an indicator which node of the edge has the smaller k value
                if k != 0.0:
                    # Add indicator for direction
                    start_point = Vec3(polyline_points_raw[0].x+1, polyline_points_raw[0].y+1, polyline_points_raw[0].z+1)
                    end_point = Vec3(polyline_points_raw[1].x+1, polyline_points_raw[1].y+1, polyline_points_raw[1].z+1)
                    if k < 0.0:
                        start_point, end_point = end_point, start_point
                    indicator_vector = (end_point - start_point) * 0.5
                    indicator_end = start_point + indicator_vector
                    if c == 2:
                        indicator_color = 7
                    else:
                        indicator_color = 6  # Different color for the indicator
                    
                    # Add the indicator line
                    msp.add_line(start_point, indicator_end, dxfattribs={'color': indicator_color})

                if to_add:
                    # Add the 3D polyline to the model space
                    msp.add_polyline3d(polyline_points, dxfattribs={'color': c})

        # Save the DXF document
        doc.saveas(filename)
    
    def compare_polylines_graph(self, other, filename, color=1, min_z=None, max_z=None):
        # Create a new DXF document.
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()

        for edge in tqdm(self.edges):
            ks = self.get_edge_ks(edge[0], edge[1])
            for k in ks:
                same_block = self.edges[edge][k]['same_block']
                c = color
                if same_block:
                    c = 2
                elif k != 0.0:
                    c = 5
                if edge not in other.edges:
                    if c == 2:
                        c = 4
                    elif c == 5:
                        c = 6
                    else:
                        c = 3
                
                # Create polyline points 
                polyline_points = []
                to_add = True
                for pi, point in enumerate(edge):
                    centroid = self.nodes[point]['centroid']
                    if min_z is not None and centroid[1] < min_z:
                        to_add = False
                    if max_z is not None and centroid[1] > max_z:
                        to_add = False
                    if not to_add:
                        break
                    polyline_points.append(Vec3(int(centroid[0]), int(centroid[1]), int(centroid[2])))

                    # Add an indicator which node of the edge has the smaller k value
                    if same_block:
                        if k > 0.0 and pi == 0:
                            polyline_points = [Vec3(int(centroid[0]), int(centroid[1]) + 10, int(centroid[2]))] + polyline_points
                        elif k < 0.0 and pi == 1:
                            polyline_points += [Vec3(int(centroid[0]), int(centroid[1]) + 10, int(centroid[2]))]

                if to_add:
                    # Add the 3D polyline to the model space
                    msp.add_polyline3d(polyline_points, dxfattribs={'color': c})

        # Save the DXF document
        doc.saveas(filename)

    def extract_subgraph(self, min_z=None, max_z=None, umbilicus_max_distance=None, add_same_block_edges=False, tolerated_nodes=None, min_x=None, max_x=None, min_y=None, max_y=None):
        # Define a wrapper function for umbilicus_xz_at_y
        umbilicus_func = lambda z: umbilicus_xz_at_y(self.umbilicus_data, z)
        # Extract subgraph with nodes within z range
        subgraph = ScrollGraph(self.overlapp_threshold, self.umbilicus_path)
        # starting block info
        subgraph.start_block = self.start_block
        subgraph.patch_id = self.patch_id
        for node in tqdm(self.nodes, desc="Extracting subgraph..."):
            centroid = self.nodes[node]['centroid']
            winding_angle = self.nodes[node]['winding_angle']
            if (tolerated_nodes is not None) and (node in tolerated_nodes):
                subgraph.add_node(node, centroid, winding_angle=winding_angle)
                continue
            elif (min_z is not None) and (centroid[1] < min_z):
                continue
            elif (max_z is not None) and (centroid[1] > max_z):
                continue
            elif (min_x is not None) and (centroid[2] < min_x):
                continue
            elif (max_x is not None) and (centroid[2] > max_x):
                continue
            elif (min_y is not None) and (centroid[0] < min_y):
                continue
            elif (max_y is not None) and (centroid[0] > max_y):
                continue
            elif (umbilicus_max_distance is not None) and np.linalg.norm(umbilicus_func(centroid[1]) - centroid) > umbilicus_max_distance:
                continue
            else:
                subgraph.add_node(node, centroid, winding_angle=winding_angle)
        for edge in self.edges:
            node1, node2 = edge
            if (tolerated_nodes is not None) and (node1 in tolerated_nodes) and (node2 in tolerated_nodes): # dont add useless edges
                continue
            if node1 in subgraph.nodes and node2 in subgraph.nodes:
                for k in self.get_edge_ks(node1, node2):
                    if add_same_block_edges or (not self.edges[edge][k]['same_block']):
                        subgraph.add_edge(node1, node2, self.edges[edge][k]['certainty'], k, self.edges[edge][k]['same_block'], bad_edge=self.edges[edge][k]['bad_edge'])
        subgraph.compute_node_edges()
        return subgraph
    
class WeightedUnionFind():
    def __init__(self):
        self.parent = {}
        self.size = {}
        self.weight = {}
        self.prefered = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.weight[x] = 0
            self.size[x] = 1
            return x, 0.0
        
        root = x
        path_weight = self.weight[root]
        while root != self.parent[root]:
            root = self.parent[root]
            path_weight += self.weight[root]
        total_weight = path_weight
        while x != root:
            next_ = self.parent[x]
            self.parent[x] = root
            total_weight_ = total_weight - self.weight[x]
            self.weight[x] = total_weight
            total_weight = total_weight_
            x = next_
        return root, path_weight

    def merge(self, x, y, k, prefered_x=False, prefered_y=False):
        if x not in self.parent:
            self.parent[x] = x
            self.weight[x] = 0
            self.size[x] = 1
        if y not in self.parent:
            self.parent[y] = y
            self.weight[y] = 0
            self.size[y] = 1

        self.prefered[x] = prefered_x
        self.prefered[y] = prefered_y

        rootX, weightX = self.find(x)
        rootY, weightY = self.find(y)
        prefered_rootX = self.prefered[rootX] if rootX in self.prefered else False
        prefered_rootY = self.prefered[rootY] if rootY in self.prefered else False

        if rootX == rootY:
            connection_weight = weightX - weightY
            same_bool = connection_weight == k
            if not same_bool:
                print(f"Connection weight: {connection_weight}, k: {k}")
            return same_bool
        
        if prefered_rootY and (not prefered_rootX):
            self.parent[rootX] = rootY
            self.weight[rootX] = weightY + k - weightX
            self.size[rootY] += self.size[rootX]
            self.size[rootX] = 0
        elif prefered_rootX and (not prefered_rootY):
            self.parent[rootY] = rootX
            self.weight[rootY] = weightX - k - weightY
            self.size[rootX] += self.size[rootY]
            self.size[rootY] = 0
        elif self.size[rootX] < self.size[rootY]:
            self.parent[rootX] = rootY
            self.weight[rootX] = weightY + k - weightX
            self.size[rootY] += self.size[rootX]
            self.size[rootX] = 0
        else:
            self.parent[rootY] = rootX
            self.weight[rootY] = weightX - k - weightY
            self.size[rootX] += self.size[rootY]
            self.size[rootY] = 0
        return True
    
    def connected(self, x, y):
        rootX, _ = self.find(x)
        rootY, _ = self.find(y)
        return rootX == rootY
    
    def connection_weight(self, x, y):
        rootX, weightX = self.find(x)
        rootY, weightY = self.find(y)
        if rootX == rootY:
            return weightX - weightY
        return None
    
    def get_roots(self):
        roots = []
        for x in self.parent:
            if x == self.parent[x]:
                roots.append(x)
        return roots
    
    def get_components(self):
        components = {}
        for x in self.parent:
            root, _ = self.find(x)
            if root not in components:
                components[root] = []
            components[root].append(x)
        components_list = list(components.values())
        return components_list
    
    def contains(self, x):
        return x in self.parent
        
class EvolutionaryGraphEdgesSelection():
    def __init__(self, graph, path, save_path, min_z=None, max_z=None):
        self.graph = graph
        self.path = path
        self.save_path = save_path
        self.min_z = min_z
        self.max_z = max_z
    
    def build_graph_data_block(self, graph, min_z=None, max_z=None, helper_graph=None, min_x=None, max_x=None, min_y=None, max_y=None, iteration=0):
        if iteration == 0:
            return self.build_graph_data(graph, min_z=min_z, max_z=max_z, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, strict_edges=True)
        """
        Builds a dictionary with the node data.
        """
        if helper_graph is None:
            helper_graph = graph

        def in_range(centroid, min_z_, max_z_, min_x_=None, max_x_=None, min_y_=None, max_y_=None):
            if min_z_ is not None and centroid[1] <= min_z_:
                return False
            if max_z_ is not None and centroid[1] > max_z_:
                return False
            if min_x_ is not None and centroid[2] <= min_x_:
                return False
            if max_x_ is not None and centroid[2] > max_x_:
                return False
            if min_y_ is not None and centroid[0] <= min_y_:
                return False
            if max_y_ is not None and centroid[0] > max_y_:
                return False
            return True
        
        def changing_side(centroid1_, centroid2_, min_z_, max_z_, min_x_, max_x_, min_y_, max_y_):
            if ((min_z_ is not None) and (max_z_ is not None)) and ((2*(centroid1_[1]-min_z_))//(max_z_ - min_z_) != (2*(centroid2_[1]-min_z_))//(max_z_ - min_z_)):
                return True
            if ((min_x_ is not None) and (max_x_ is not None)) and ((2*(centroid1_[2]-min_x_))//(max_x_ - min_x_) != (2*(centroid2_[2]-min_x_))//(max_x_ - min_x_)):
                return True
            if ((min_y_ is not None) and (max_y_ is not None)) and ((2*(centroid1_[0]-min_y_))//(max_y_ - min_y_) != (2*(centroid2_[0]-min_y_))//(max_y_ - min_y_)):
                return True
            return False
        
        node_data = {}
        for node in graph.nodes:
            node_data[node] = graph.nodes[node]

        nodes = node_data
        nodes_index_dict = {node: i for i, node in enumerate(nodes)}
        index_nodes_dict = {i: node for i, node in enumerate(nodes)}

        nonone_certainty_count = 0
        edges_by_indices  = []
        edges_by_subvolume_indices = []
        initial_component = {}
        for edge in graph.edges:
            node1, node2 = edge
            ks = graph.get_edge_ks(node1, node2)
            centroid1 = graph.nodes[node1]['centroid']
            centroid2 = graph.nodes[node2]['centroid']
            for k in ks:
                same_block = graph.edges[edge][k]['same_block']
                to_add = True
                
                if (not in_range(centroid1, min_z, max_z, min_x, max_x, min_y, max_y)) or (not in_range(centroid2, min_z, max_z, min_x, max_x, min_y, max_y)):
                    to_add = False
                elif (not same_block) and (not changing_side(centroid1, centroid2, min_z, max_z, min_x, max_x, min_y, max_y)):
                    to_add = False
                if not to_add:
                    continue
                certainty = graph.edges[edge][k]['certainty']
                assert certainty >= 0.0, f"Invalid certainty: {certainty} for edge: {edge}"
                if certainty > 0.0 and certainty != 1.0:
                    nonone_certainty_count += 1
                edges_by_indices.append((nodes_index_dict[node1], nodes_index_dict[node2], k, 1 + int(100*certainty)))

                if ("assigned_k" in helper_graph.nodes[node1]) and ("assigned_k" in helper_graph.nodes[node2]):
                    assigned_k1 = helper_graph.nodes[node1]["assigned_k"]
                    assigned_k2 = helper_graph.nodes[node2]["assigned_k"]
                    if assigned_k2 - assigned_k1 == k:
                        edges_by_subvolume_indices.append((nodes_index_dict[node1], nodes_index_dict[node2], k, 1 + int(100*certainty), node1[0], node1[1], node1[2], node2[0], node2[1], node2[2], assigned_k1, assigned_k2))
                    else:
                        print(f"Assigned k mismatch: {assigned_k1}, {assigned_k2}, {k}")
        print(f"None one certainty edges: {nonone_certainty_count} out of {len(edges_by_indices)} edges.")
        
        edges_by_indices = np.array(edges_by_indices).astype(np.int32)
        edges_by_subvolume_indices = np.array(edges_by_subvolume_indices).astype(np.int32)
        if len(initial_component) == 0:
            print("Building Initial component with 0 nodes.")
            initial_component = np.zeros((0,2), dtype=np.int32)
        else:
            print(f"Building Initial component with {len(initial_component)} nodes.")
            initial_component = [(int(node_index), int(k)) for node_index, k in initial_component.items()]
            initial_component = np.array(initial_component, dtype=np.int32)
        return edges_by_indices, edges_by_subvolume_indices, initial_component, index_nodes_dict

    def build_graph_data(self, graph, min_z=None, max_z=None, strict_edges=True, helper_graph=None, min_x=None, max_x=None, min_y=None, max_y=None):
        """
        Builds a dictionary with the node data.
        """
        if helper_graph is None:
            helper_graph = graph
        def in_range(centroid, min_z_, max_z_, min_x_=None, max_x_=None, min_y_=None, max_y_=None):
            if min_z_ is not None and centroid[1] <= min_z_:
                return False
            if max_z_ is not None and centroid[1] > max_z_:
                return False
            if min_x_ is not None and centroid[2] <= min_x_:
                return False
            if max_x_ is not None and centroid[2] > max_x_:
                return False
            if min_y_ is not None and centroid[0] <= min_y_:
                return False
            if max_y_ is not None and centroid[0] > max_y_:
                return False
            return True
        
        node_data = {}
        for node in graph.nodes:
            node_data[node] = graph.nodes[node]

        nodes = node_data
        nodes_index_dict = {node: i for i, node in enumerate(nodes)}
        index_nodes_dict = {i: node for i, node in enumerate(nodes)}

        nonone_certainty_count = 0
        edges_by_indices  = []
        edges_by_indices_k_bool  = []
        edges_by_indices_bad = []
        edges_by_subvolume_indices = []
        edges_by_subvolume_indices_k_bool = []
        edges_by_subvolume_indices_bad = []
        initial_component = {}
        for edge in graph.edges:
            node1, node2 = edge
            centroid1 = graph.nodes[node1]['centroid']
            centroid2 = graph.nodes[node2]['centroid']
            to_add = True
            if not strict_edges:
                if (not in_range(centroid1, min_z, max_z, min_x, max_x, min_y, max_y)) and (not in_range(centroid2, min_z, max_z, min_x, max_x, min_y, max_y)):
                    to_add = False
                elif not in_range(centroid1, min_z, max_z, min_x, max_x, min_y, max_y):
                    # assign known k to initial component for evolutionary algorithm
                    if "assigned_k" in helper_graph.nodes[node1]:
                        initial_component[nodes_index_dict[node1]] = int(helper_graph.nodes[node1]["assigned_k"])
                    else:
                        to_add = False
                elif not in_range(centroid2, min_z, max_z, min_x, max_x, min_y, max_y):
                    # assign known k to initial component for evolutionary algorithm
                    if "assigned_k" in helper_graph.nodes[node2]:
                        initial_component[nodes_index_dict[node2]] = int(helper_graph.nodes[node2]["assigned_k"])
                    else:
                        to_add = False
            elif strict_edges and ((not in_range(centroid1, min_z, max_z, min_x, max_x, min_y, max_y)) or (not in_range(centroid2, min_z, max_z, min_x, max_x, min_y, max_y))):
                to_add = False
            if not to_add:
                continue
            ks = graph.get_edge_ks(node1, node2)
            for k in ks:
                certainty = graph.edges[edge][k]['certainty']
                assert certainty >= 0.0, f"Invalid certainty: {certainty} for edge: {edge}"
                if certainty > 0.0 and certainty != 1.0:
                    nonone_certainty_count += 1
                if not graph.edges[edge][k]['bad_edge']:
                    edges_by_indices.append((nodes_index_dict[node1], nodes_index_dict[node2], k, 1 + int(100*certainty)))
                    edges_by_indices_k_bool.append(graph.edges[edge][k]['same_block'])
                else:
                    edges_by_indices_bad.append((nodes_index_dict[node1], nodes_index_dict[node2], k, 1 + int(100*certainty)))

                if ("assigned_k" in helper_graph.nodes[node1]) and ("assigned_k" in helper_graph.nodes[node2]):
                    assigned_k1 = helper_graph.nodes[node1]["assigned_k"]
                    assigned_k2 = helper_graph.nodes[node2]["assigned_k"]
                    if assigned_k2 - assigned_k1 == k:
                        if not graph.edges[edge][k]['bad_edge']:
                            edges_by_subvolume_indices.append((nodes_index_dict[node1], nodes_index_dict[node2], k, 1 + int(100*certainty), node1[0], node1[1], node1[2], node2[0], node2[1], node2[2], assigned_k1, assigned_k2))
                            edges_by_subvolume_indices_k_bool.append(graph.edges[edge][k]['same_block'])
                        else:
                            edges_by_subvolume_indices_bad.append((nodes_index_dict[node1], nodes_index_dict[node2], k, 1 + int(100*certainty), node1[0], node1[1], node1[2], node2[0], node2[1], node2[2], assigned_k1, assigned_k2))
                    else:
                        print(f"Assigned k mismatch: {assigned_k1}, {assigned_k2}, {k}")
        
        edges_by_indices = np.array(edges_by_indices).astype(np.int32)
        edges_by_indices_k_bool = np.array(edges_by_indices_k_bool).astype(bool)
        if len(edges_by_indices_bad) == 0:
            edges_by_indices_bad = np.zeros((0,4), dtype=np.int32)
        else:
            edges_by_indices_bad = np.array(edges_by_indices_bad).astype(np.int32)
        edges_by_subvolume_indices = np.array(edges_by_subvolume_indices).astype(np.int32)
        edges_by_subvolume_indices_k_bool = np.array(edges_by_subvolume_indices_k_bool).astype(bool)
        if len(edges_by_subvolume_indices_bad) == 0:
            edges_by_subvolume_indices_bad = np.zeros((0,12), dtype=np.int32)
        else:
            edges_by_subvolume_indices_bad = np.array(edges_by_subvolume_indices_bad).astype(np.int32)
        if len(initial_component) == 0:
            initial_component = np.zeros((0,2), dtype=np.int32)
        else:
            initial_component = [(int(node_index), int(k)) for node_index, k in initial_component.items()]
            initial_component = np.array(initial_component, dtype=np.int32)
        return (edges_by_indices, edges_by_indices_k_bool, edges_by_indices_bad), (edges_by_subvolume_indices, edges_by_subvolume_indices_k_bool, edges_by_subvolume_indices_bad), initial_component, index_nodes_dict
    
    def build_graph_data_fast(self, graph, side_length, padding):
        """
        Builds a dictionary with the node data.
        """
        graph_z = np.array([graph.nodes[node]['centroid'][1] for node in graph.nodes])
        graph_z_min = int(np.floor(np.min(graph_z)))
        graph_z_max = int(np.ceil(np.max(graph_z)))
        graph_x = np.array([graph.nodes[node]['centroid'][2] for node in graph.nodes])
        graph_x_min = int(np.floor(np.min(graph_x)))
        graph_x_max = int(np.ceil(np.max(graph_x)))
        graph_y = np.array([graph.nodes[node]['centroid'][0] for node in graph.nodes])
        graph_y_min = int(np.floor(np.min(graph_y)))
        graph_y_max = int(np.ceil(np.max(graph_y)))
        
        def subblocks_from_centroid(node):
            centroid = graph.nodes[node]['centroid']
            x1 = int((centroid[2] - graph_x_min) // side_length)
            x2 = int((centroid[2] - graph_x_min + padding) // side_length)
            x3 = int((centroid[2] - graph_x_min - padding) // side_length)
            x = set([x1, x2, x3])
            y1 = int((centroid[0] - graph_y_min) // side_length)
            y2 = int((centroid[0] - graph_y_min + padding) // side_length)
            y3 = int((centroid[0] - graph_y_min - padding) // side_length)
            y = set([y1, y2, y3])
            z1 = int((centroid[1] - graph_z_min) // side_length)
            z2 = int((centroid[1] - graph_z_min + padding) // side_length)
            z3 = int((centroid[1] - graph_z_min - padding) // side_length)
            z = set([z1, z2, z3])

            # Adjust to graph limits
            x = set([graph_x_min + xi*side_length for xi in x])
            y = set([graph_y_min + yi*side_length for yi in y])
            z = set([graph_z_min + zi*side_length for zi in z])

            # remove values that are too small
            x = set([xi for xi in x if xi >= graph_x_min])
            y = set([yi for yi in y if yi >= graph_y_min])
            z = set([zi for zi in z if zi >= graph_z_min])

            # remove values that are too large
            x = set([xi for xi in x if xi < graph_x_max])
            y = set([yi for yi in y if yi < graph_y_max])
            z = set([zi for zi in z if zi < graph_z_max])

            subblocks = []
            for xi in x:
                for yi in y:
                    for zi in z:
                        subblocks.append((xi, yi, zi))
            assert len(subblocks) <= 8, f"Too many subblocks: {len(subblocks)}, subblocks: {subblocks}, centroid: {centroid}, x: {x}, y: {y}, z: {z}, min adjusted centroid: {centroid[2] - graph_x_min}, {centroid[0] - graph_y_min}, {centroid[1] - graph_z_min}, padding: {padding}, side_length: {side_length}"

            return subblocks

        
        node_data = {}
        for node in graph.nodes:
            node_data[node] = graph.nodes[node]

        nodes = node_data
        nodes_index_dict = {node: i for i, node in enumerate(nodes)}
        index_nodes_dict = {i: node for i, node in enumerate(nodes)}

        def add_subblock(subblock, data_dict):
            if subblock not in data_dict:
                data_dict[subblock] = {'edges_by_indices': [], 'edges_by_indices_k_bool': [], 'edges_by_indices_bad': [], 'edges_by_subvolume_indices': [], 'edges_by_subvolume_indices_k_bool': [], 'edges_by_subvolume_indices_bad': [], 'initial_component': {}}
            return data_dict

        subblock_graph_data_dict = {}
        for edge in tqdm(graph.edges, desc="Building subblock graph data..."):
            node1, node2 = edge
            subblocks1 = subblocks_from_centroid(node1)
            subblocks2 = subblocks_from_centroid(node2)
            intersection_subblocks = list(set(subblocks1).intersection(set(subblocks2)))
            
            ks = graph.get_edge_ks(node1, node2)
            for subblock in intersection_subblocks:
                subblock_graph_data_dict = add_subblock(subblock, subblock_graph_data_dict) # housekeeping

                for k in ks:
                    certainty = graph.edges[edge][k]['certainty']
                    assert certainty >= 0.0, f"Invalid certainty: {certainty} for edge: {edge}"

                    if not graph.edges[edge][k]['bad_edge']:
                        subblock_graph_data_dict[subblock]['edges_by_indices'].append((nodes_index_dict[node1], nodes_index_dict[node2], k, 1 + int(100*certainty)))
                        subblock_graph_data_dict[subblock]['edges_by_indices_k_bool'].append(graph.edges[edge][k]['same_block'])
                    else:
                        subblock_graph_data_dict[subblock]['edges_by_indices_bad'].append((nodes_index_dict[node1], nodes_index_dict[node2], k, 1 + int(100*certainty)))

                    if ("assigned_k" in graph.nodes[node1]) and ("assigned_k" in graph.nodes[node2]):
                        assigned_k1 = graph.nodes[node1]["assigned_k"]
                        assigned_k2 = graph.nodes[node2]["assigned_k"]
                        if assigned_k2 - assigned_k1 == k:
                            if not graph.edges[edge][k]['bad_edge']:
                                subblock_graph_data_dict[subblock]['edges_by_subvolume_indices'].append((nodes_index_dict[node1], nodes_index_dict[node2], k, 1 + int(100*certainty), node1[0], node1[1], node1[2], node2[0], node2[1], node2[2], assigned_k1, assigned_k2))
                                subblock_graph_data_dict[subblock]['edges_by_subvolume_indices_k_bool'].append(graph.edges[edge][k]['same_block'])
                            else:
                                subblock_graph_data_dict[subblock]['edges_by_subvolume_indices_bad'].append((nodes_index_dict[node1], nodes_index_dict[node2], k, 1 + int(100*certainty), node1[0], node1[1], node1[2], node2[0], node2[1], node2[2], assigned_k1, assigned_k2))
                        else:
                            print(f"Assigned k mismatch: {assigned_k1}, {assigned_k2}, {k}")
        print(f"Found {len(subblock_graph_data_dict)} subblocks.")

        for subblock in subblock_graph_data_dict:
            subblock_graph_data_dict[subblock]['edges_by_indices'] = np.array(subblock_graph_data_dict[subblock]['edges_by_indices']).astype(np.int32)
            subblock_graph_data_dict[subblock]['edges_by_indices_k_bool'] = np.array(subblock_graph_data_dict[subblock]['edges_by_indices_k_bool']).astype(bool)
            if len(subblock_graph_data_dict[subblock]['edges_by_indices_bad']) == 0:
                subblock_graph_data_dict[subblock]['edges_by_indices_bad'] = np.zeros((0,4), dtype=np.int32)
            else:
                subblock_graph_data_dict[subblock]['edges_by_indices_bad'] = np.array(subblock_graph_data_dict[subblock]['edges_by_indices_bad']).astype(np.int32)
            subblock_graph_data_dict[subblock]['edges_by_subvolume_indices'] = np.array(subblock_graph_data_dict[subblock]['edges_by_subvolume_indices']).astype(np.int32)
            subblock_graph_data_dict[subblock]['edges_by_subvolume_indices_k_bool'] = np.array(subblock_graph_data_dict[subblock]['edges_by_subvolume_indices_k_bool']).astype(bool)
            if len(subblock_graph_data_dict[subblock]['edges_by_subvolume_indices_bad']) == 0:
                subblock_graph_data_dict[subblock]['edges_by_subvolume_indices_bad'] = np.zeros((0,12), dtype=np.int32)
            else:
                subblock_graph_data_dict[subblock]['edges_by_subvolume_indices_bad'] = np.array(subblock_graph_data_dict[subblock]['edges_by_subvolume_indices_bad']).astype(np.int32)
            if len(subblock_graph_data_dict[subblock]['initial_component']) == 0:
                subblock_graph_data_dict[subblock]['initial_component'] = np.zeros((0,2), dtype=np.int32)
            else:
                subblock_graph_data_dict[subblock]['initial_component'] = [(int(node_index), int(k)) for node_index, k in subblock_graph_data_dict[subblock]['initial_component'].items()]
                subblock_graph_data_dict[subblock]['initial_component'] = np.array(subblock_graph_data_dict[subblock]['initial_component'], dtype=np.int32)
        return subblock_graph_data_dict, index_nodes_dict
    
    # def solve_call_(self, input, initial_component=None, problem='k_assignment'):
    #     input = input.astype(np.int32)
    #     if initial_component is None:
    #         initial_component = np.zeros((0,2), dtype=np.int32)
    #     initial_component = initial_component.astype(np.int32)
    #     # easily switch between dummy and real computation
    #     valid_mask, valid_edges_count = solve_genetic(input, initial_component=initial_component, problem=problem)
    #     valid_mask = valid_mask > 0
    #     return valid_mask, valid_edges_count
    
    def solve_call(self, input, initial_component=None, problem='k_assignment', k_factors=None, iteration=0, last_iteration=False):
        input_graph = input[0].astype(np.int32)
        input_k_bool = input[1].astype(bool)
        input_bad = input[2].astype(np.int32)
        if initial_component is None:
            initial_component = np.zeros((0,2), dtype=np.int32)
        initial_component = initial_component.astype(np.int32)
        def calculate_fitness_k_factors(graph, graph_k_bool, graph_bad):
            # sum of k == 0 and sum of k != 0
            assert np.sum(graph[:, 3] <= 0) == 0, "There should be no edges with zero,negative certainty"
            k_0 = np.sum(graph[~graph_k_bool, 3])
            k_not_0 = np.sum(graph[graph_k_bool, 3])
            k_bad = np.sum(graph_bad[:, 3])
            factor_0 = 1.00
            factor_not_0 = 1.0 * k_0 / (1 + k_not_0)
            factor_bad = 1.0 * k_0 / (1 + k_bad)
            # third root of k bad
            # factor_bad = factor_bad**(1/2)
            print(f"Factor 0: {factor_0}, Factor not 0: {factor_not_0}, Factor bad: {factor_bad}, with k_0: {k_0}, k_not_0: {k_not_0} and k_bad: {k_bad}")
            print(f"Maximum possible fitness: {2*k_0}")

            return factor_0, factor_not_0, factor_bad
        
        def find_nr_nodes(input_edges):
            nodes1 = np.unique(input_edges[:, 0])
            nodes2 = np.unique(input_edges[:, 1])
            nodes = np.unique(np.concatenate((nodes1, nodes2)))
            return len(nodes)

        max_valid_edges_factor = 0.5
        if k_factors is not None:
            factor_0, factor_not_0, factor_bad = k_factors
            print(f"Factor 0: {factor_0}, Factor not 0: {factor_not_0} and Factor bad: {factor_bad} from previous iteration.")
        else:
            factor_0, factor_not_0, factor_bad = calculate_fitness_k_factors(input_graph, input_k_bool, input_bad)
        nr_active_nodes = find_nr_nodes(input_graph)
        # print(f"There are {nr_active_nodes} unique active nodes in the graph. Max possible fittnes is {2*nr_active_nodes + nr_active_nodes*np.log(nr_active_nodes)}")
        # easily switch between dummy and real computation
        debug=True
        if not debug:
            population_size = 500
            generations = 600
        else:
            population_size = 250 # 500
            generations = 300 # 200
        population_size *= iteration + 1
        generations *= iteration + 1
        generations -= 1 # because of fixing in the genes. you dont want to fix the genes in the last generation

        if problem == 'k_assignment':
            use_edge_ignoring = iteration==0
            use_edge_ignoring = True
            use_edge_ignoring = use_edge_ignoring and (not last_iteration)
            valid_edges_count, valid_mask, solution_weights = evolve_graph.evolution_solve_k_assignment(population_size, generations, input_graph.shape[0], input_graph, input_k_bool, input_bad.shape[0], input_bad, factor_0, factor_not_0, factor_bad, initial_component.shape[0], initial_component, use_edge_ignoring)
        elif problem == 'patch_selection':
            valid_edges_count, valid_mask, solution_weights = evolve_graph.evolution_solve_patches(population_size, generations, input_graph.shape[0], input_graph, input_k_bool, input_bad.shape[0], input_bad, factor_0, factor_not_0, factor_bad)

        valid_mask = valid_mask > 0
        return valid_mask, valid_edges_count, (factor_0, factor_not_0, factor_bad)
    
    def build_graph_blocks_old_working_slow(self, graph, side_length=200, padding=50):
        graph_z = np.array([graph.nodes[node]['centroid'][1] for node in graph.nodes])
        graph_z_min = int(np.floor(np.min(graph_z)))
        graph_z_max = int(np.ceil(np.max(graph_z)))
        graph_z_size = graph_z_max - graph_z_min
        graph_x = np.array([graph.nodes[node]['centroid'][2] for node in graph.nodes])
        graph_x_min = int(np.floor(np.min(graph_x)))
        graph_x_max = int(np.ceil(np.max(graph_x)))
        graph_x_size = graph_x_max - graph_x_min
        graph_y = np.array([graph.nodes[node]['centroid'][0] for node in graph.nodes])
        graph_y_min = int(np.floor(np.min(graph_y)))
        graph_y_max = int(np.ceil(np.max(graph_y)))
        graph_y_size = graph_y_max - graph_y_min

        graph_blocks = []
        for x_start in tqdm(range(graph_x_min, graph_x_max, side_length)):
            for y_start in range(graph_y_min, graph_y_max, side_length):
                for z_start in range(graph_z_min, graph_z_max, side_length):
                    print(f"Block: {x_start}, {y_start}, {z_start}, end: {x_start + side_length}, {y_start + side_length}, {z_start + side_length}")
                    min_z = z_start - padding
                    max_z = z_start + side_length + padding
                    min_x = x_start - padding
                    max_x = x_start + side_length + padding
                    min_y = y_start - padding
                    max_y = y_start + side_length + padding
                    edges_by_indices, edges_by_subvolume_indices, initial_component, index_nodes_dict = self.build_graph_data(graph, min_z=min_z, max_z=max_z, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, strict_edges=True)
                    if len(edges_by_indices[0]) > 0:
                        graph_blocks.append((edges_by_indices, edges_by_subvolume_indices, initial_component, index_nodes_dict, (min_x, max_x, min_y, max_y, min_z, max_z, padding)))

        # check that no node is in two different blocks
        nodes_others = set()
        # print(colored("Testing blocks integrity", 'yellow'))
        # for edges_by_indices, _, _, index_nodes_dict, _ in graph_blocks:
        #     nodes = set()
        #     for edge in edges_by_indices[0]:
        #         node1 = index_nodes_dict[edge[0]]
        #         node2 = index_nodes_dict[edge[1]]
        #         nodes.add(node1)
        #         nodes.add(node2)
        #     for node in nodes:
        #         if node in nodes_others:
        #             print(colored(f"Node {node} is in two different blocks.", 'red'))
        #         nodes_others.add(node)

        return graph_blocks

    def build_graph_blocks(self, graph, side_length=200, padding=50):
        subblock_graph_data_dict, index_nodes_dict = self.build_graph_data_fast(graph, side_length=side_length, padding=padding)

        graph_blocks = []
        for subblock in subblock_graph_data_dict:
            edges_by_indices = subblock_graph_data_dict[subblock]['edges_by_indices']
            edges_by_indices_k_bool = subblock_graph_data_dict[subblock]['edges_by_indices_k_bool']
            edges_by_indices_bad = subblock_graph_data_dict[subblock]['edges_by_indices_bad']
            edges_by_subvolume_indices = subblock_graph_data_dict[subblock]['edges_by_subvolume_indices']
            edges_by_subvolume_indices_k_bool = subblock_graph_data_dict[subblock]['edges_by_subvolume_indices_k_bool']
            edges_by_subvolume_indices_bad = subblock_graph_data_dict[subblock]['edges_by_subvolume_indices_bad']
            initial_component = subblock_graph_data_dict[subblock]['initial_component']
            min_x = subblock[0] - padding
            max_x = subblock[0] + side_length + padding
            min_y = subblock[1] - padding
            max_y = subblock[1] + side_length + padding
            min_z = subblock[2] - padding
            max_z = subblock[2] + side_length + padding

            if len(edges_by_indices) > 0:
                graph_blocks.append(((edges_by_indices, edges_by_indices_k_bool, edges_by_indices_bad), (edges_by_subvolume_indices, edges_by_subvolume_indices_k_bool, edges_by_subvolume_indices_bad), initial_component, index_nodes_dict, (min_x, max_x, min_y, max_y, min_z, max_z, padding)))

        return graph_blocks
    
    def solve_graph_blocks(self, graph_blocks, k_factors=None, iteration=0, last_iteration=False):
        results = []
        k_factor_0 = 0.0
        k_factor_not_0 = 0.0
        k_factor_bad = 0.0
        for block_nr, (edges_by_indices, edges_by_subvolume_indices, initial_component, index_nodes_dict, _) in enumerate(graph_blocks):
            print(f"Solving block {block_nr + 1}/{len(graph_blocks)}")
            # print(f"Number of edges: {len(edges_by_indices)}")
            valid_mask, valid_edges_count, k_factors_ = self.solve_call(edges_by_indices, initial_component=initial_component, problem='k_assignment', k_factors=k_factors, iteration=iteration, last_iteration=last_iteration)
            k_factor_0 += k_factors_[0]
            k_factor_not_0 += k_factors_[1]
            k_factor_bad += k_factors_[2]
            # print(f"Valid edges count: {valid_edges_count}")
            valid_mask = np.copy(valid_mask > 0)
            results.append((valid_mask, valid_edges_count))

        k_factors_mean = (k_factor_0 / len(graph_blocks), k_factor_not_0 / len(graph_blocks), k_factor_bad / len(graph_blocks))

        return results, k_factors_mean

    def contract_graph_blocks(self, original_graph, solution_graph, graph_blocks, solutions, last_iteration=False, side_length=200):
        edges_indices_list = []
        edges_mask_list = []
        index_nodes_dict_list = []
        for i in range(len(solutions)):
            (edges_by_indices, _, _), _, initial_component, index_nodes_dict, (min_x, max_x, min_y, max_y, min_z, max_z, padding) = graph_blocks[i]
            valid_mask, valid_edges_count = solutions[i]
            solution_graph = self.transition_graph_from_edge_selection(edges_by_indices, original_graph, valid_mask, index_nodes_dict, graph_blocks[i], graph=solution_graph)
            # In the end also add the remaining same block edges that make up the connected graph
            if last_iteration:
                print("Last iteration, adding k switching edges too")
                solution_graph = self.same_block_graph_from_edge_selection(edges_by_indices, original_graph, valid_mask, index_nodes_dict, graph_blocks[i], graph=solution_graph)

            edges_indices_list.append(edges_by_indices)
            edges_mask_list.append(valid_mask)
            index_nodes_dict_list.append(index_nodes_dict)

        solution_graph.compute_node_edges()
        solution_graph.largest_connected_component(delete_nodes=True, min_size=5)
        solution_graph.compute_node_edges()

        # contracted_graph = self.contracted_graph_from_edge_selection(edges_indices_list, original_graph, edges_mask_list, index_nodes_dict_list, graph_blocks)
        contracted_graph = self.contracted_graph_from_solution_graph(solution_graph, original_graph, side_length=side_length)
        if not last_iteration:
            self.check_contracted_and_solution_graph(contracted_graph, solution_graph)

        return contracted_graph, solution_graph
    
    def solve_divide_and_conquer(self, start_block_side_length=200, iteration=None, k_factors_original=(1.0, 1.0, 1.0)):
        graph = self.graph
        solution_graph = None

        graph_z = np.array([graph.nodes[node]['centroid'][1] for node in graph.nodes])
        graph_z_min = int(np.floor(np.min(graph_z)))
        graph_z_max = int(np.ceil(np.max(graph_z)))
        graph_z_size = graph_z_max - graph_z_min
        graph_x = np.array([graph.nodes[node]['centroid'][2] for node in graph.nodes])
        graph_x_min = int(np.floor(np.min(graph_x)))
        graph_x_max = int(np.ceil(np.max(graph_x)))
        graph_x_size = graph_x_max - graph_x_min
        graph_y = np.array([graph.nodes[node]['centroid'][0] for node in graph.nodes])
        graph_y_min = int(np.floor(np.min(graph_y)))
        graph_y_max = int(np.ceil(np.max(graph_y)))
        graph_y_size = graph_y_max - graph_y_min

        graph_size_max = np.max([graph_z_size, graph_x_size, graph_y_size])
        print(f"Graph size max: {graph_size_max}")
        blocking_factor = 2
        iteration_resume = iteration is not None
        if iteration is None:
            print("starting iterations from the beginning")
            iteration = 0
        side_length = start_block_side_length * (blocking_factor ** iteration)
        padding = 50 * (blocking_factor ** iteration)
        k_factors = (1.0, 1.0, 1.0)
        # k_factors = None
        k_f_org = k_factors
        
        # Divide and conquer block size. start small and work way up.
        while True:
            break_criteria = side_length > graph_size_max
            if iteration_resume:
                iteration_resume = False
                # load iteration
                graph, solution_graph, graph_blocks, results, k_factors = self.load_iteration(iteration)
                if iteration==0:
                    graph = self.graph
                    k_factors = k_f_org
                print(f"[Loading]: K factors: {k_factors}")
                # k_factors = None
            else:
                # break_criteria = True
                print(colored(f"Side length: {side_length} maximum graph size: {graph_size_max}", 'yellow'))
                graph.compute_bad_edges(iteration, k_factors_original[2])
                graph_blocks_ = self.build_graph_blocks(graph, side_length=side_length, padding=padding)
                if len(graph_blocks_) <= 0: # no more blocks to process
                    break
                graph_blocks = graph_blocks_ # to avoid downstream 0 length graph_blocks
                break_criteria = break_criteria or (len(graph_blocks) <= 1)

                results, _ = self.solve_graph_blocks(graph_blocks, k_factors=k_factors, iteration=iteration, last_iteration=break_criteria)
                # save iteration
                self.save_iteration(graph, solution_graph, graph_blocks, results, k_factors, iteration)

            print(f"Iteration {iteration} with {len(graph_blocks)} blocks.")
            graph, solution_graph = self.contract_graph_blocks(graph, solution_graph, graph_blocks, results, last_iteration=break_criteria, side_length=side_length)
            self.build_points(solution_graph, iteration)

            # Debug
            # graph.create_dxf_with_colored_polyline(self.save_path.replace("blocks", "graph_solution") + ".dxf", min_z=800, max_z=900)

            if break_criteria:
                break
            
            side_length *= blocking_factor
            padding *= blocking_factor
            iteration += 1

        print("Finishing up Graph Evolution...")
        # select largest connected component
        solution_graph.compute_node_edges()
        solution_graph.largest_connected_component(min_size=20)
        edges_by_indices = graph_blocks[0][0][0]
        valid_mask = results[0][0]
        self.update_ks(solution_graph, edges_by_indices=edges_by_indices, valid_mask=valid_mask, update_winding_angles=True) # update winding angles here at the very end (only once)
        print("Solved graph with genetic algorithm.")

        return solution_graph
    
    def pointcloud_from_ordered_pointset(self, ordered_pointset, filename, color=None):
        points, normals = [], []
        for point, normal in ordered_pointset:
            if point is not None:
                points.append(point)
                normals.append(normal)
        points = np.concatenate(points, axis=0).astype(np.float64)
        normals = np.concatenate(normals, axis=0).astype(np.float64)

        # Scale and translate points to original coordinate system
        points[:, :3] = scale_points_(points[:, :3])
        # Rotate back to original axis
        points, normals = shuffling_points_axis(points, normals)
        # invert normals (layer 0 in the papyrus, layer 64 on top and outside the papyrus, layer 32 surface layer (for 64 layer surface volume))
        normals = -normals

        # print(f"Saving {points.shape} points to {filename}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        if not color is None:
            # color is shape n, 1, go to rgb with hsv
            color = (color - np.min(color)) / (np.max(color) - np.min(color))
            # use colormap cold
            cmap = matplotlib.cm.get_cmap('coolwarm')
            color = cmap(color)[:, :3]
            # print(color.shape, color.dtype)
            pcd.colors = o3d.utility.Vector3dVector(color)
        # Create folder if it doesn't exist
        # print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save as a PLY file
        o3d.io.write_point_cloud(filename, pcd)

    def build_points(self, graph, iteration):
        debug_folder = os.path.join(self.save_path, f"debug_iteration_{iteration}")
        # rm folder and create again
        if os.path.exists(debug_folder):
            shutil.rmtree(debug_folder)
        os.makedirs(debug_folder, exist_ok=True)

        # find nodes and ks
        nodes, ks = self.bfs_ks(graph)
        unique_ks, ks_counts = np.unique(ks, return_counts=True)

        # sort by highest counts to lowest
        argsort_count = np.argsort(ks_counts)[::-1]
        unique_ks = unique_ks[argsort_count]

        # Building the pointcloud 4D (position) + 3D (Normal) + 3D (Color, randomness) representation of the graph

        for k_index, k in enumerate(tqdm(unique_ks, desc="Building debugging pointclouds...")):
            sheet_infos = []
            for i, node in enumerate(nodes):
                if ks[i] != k:
                    continue
                winding_angle = graph.nodes[node]['winding_angle']
                block, patch_id = node[:3], node[3]
                patch_sheet_patch_info = (block, int(patch_id), winding_angle)
                sheet_infos.append(patch_sheet_patch_info)

            points, normals, colors = pointcloud_processing.load_pointclouds(sheet_infos, self.path, False)
            # save as ply
            ordered_pointsets_debug = [(points[:,:3], normals)]
            colors_debug = points[:,3]
            debug_pointset_ply_path = os.path.join(debug_folder, f"ordered_pointset_debug_{k_index}.ply")
            self.pointcloud_from_ordered_pointset(ordered_pointsets_debug, debug_pointset_ply_path, color=colors_debug)
    
    def save_iteration(self, graph, solution_graph, graph_blocks, results, k_factors, iteration):
        # save graphs
        graph_iteration_path = self.save_path.replace("blocks", f"iteration_{iteration}_graph") + ".pkl"
        solution_graph_iteration_path = self.save_path.replace("blocks", f"iteration_{iteration}_solution_graph") + ".pkl"
        # Save graph class object to file
        with open(graph_iteration_path, 'wb') as file:
            pickle.dump(graph, file)
        with open(solution_graph_iteration_path, 'wb') as file:
            pickle.dump(solution_graph, file)
        # save blocks
        graph_blocks_path = self.save_path.replace("blocks", f"iteration_{iteration}_graph_blocks") + ".pkl"
        with open(graph_blocks_path, 'wb') as f:
            pickle.dump(graph_blocks, f)
        # save results
        results_path = self.save_path.replace("blocks", f"iteration_{iteration}_results") + ".pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        # save k factors
        k_factors_path = self.save_path.replace("blocks", f"iteration_{iteration}_k_factors") + ".pkl"
        with open(k_factors_path, 'wb') as f:
            pickle.dump(k_factors, f)

    def load_iteration(self, iteration):
        # load graphs
        graph_iteration_path = self.save_path.replace("blocks", f"iteration_{iteration}_graph") + ".pkl"
        solution_graph_iteration_path = self.save_path.replace("blocks", f"iteration_{iteration}_solution_graph") + ".pkl"
        graph = load_graph(graph_iteration_path)
        solution_graph = load_graph(solution_graph_iteration_path)
        # load blocks
        graph_blocks_path = self.save_path.replace("blocks", f"iteration_{iteration}_graph_blocks") + ".pkl"
        with open(graph_blocks_path, 'rb') as f:
            graph_blocks = pickle.load(f)
        # load results
        results_path = self.save_path.replace("blocks", f"iteration_{iteration}_results") + ".pkl"
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        # load k factors
        k_factors_path = self.save_path.replace("blocks", f"iteration_{iteration}_k_factors") + ".pkl"
        with open(k_factors_path, 'rb') as f:
            k_factors = pickle.load(f)
        return graph, solution_graph, graph_blocks, results, k_factors

    
    def solve_subloop(self, pbar, graph_extraction_start, z_height_steps, start_node, evolved_graph):
        # Extract subgraph data for evolutionary algorithm 
        self.edges_by_indices, _, initial_component, index_nodes_dict = self.build_graph_data(self.graph, min_z=graph_extraction_start, max_z=graph_extraction_start+z_height_steps, strict_edges=False, helper_graph=evolved_graph)
        print(f"Graph nodes length {len(self.graph.nodes)}, edges length: {len(self.graph.edges)}")
        print("Number of edges: ", len(self.edges_by_indices))
        print("Initial component shape: ", initial_component.shape)

        # Solve with genetic algorithm
        valid_mask, valid_edges_count = self.solve_call(self.edges_by_indices, initial_component=initial_component, problem='k_assignment')

        # Build graph from genetic algorithm solution (selected edges from the graph)
        evolved_graph_temp = deepcopy(evolved_graph)
        evolved_graph_temp = self.graph_from_edge_selection(self.edges_by_indices, self.graph, valid_mask, index_nodes_dict, evolved_graph_temp)
        # select largest connected component in solution
        largest_component = evolved_graph_temp.largest_connected_component(delete_nodes=False)

        # Update start node for Breadth First Search. In two steps: first for the unfiltered graph, then for the filtered graph
        if start_node is None:
            # start_node_graph = self.graph_from_edge_selection(self.edges_by_indices, self.graph, valid_mask)
            start_node_temp = largest_component[0]
        else:
            start_node_temp = start_node
        
        # Compute ks by simple bfs to filter based on ks and subvolume
        self.update_ks(evolved_graph_temp, start_node=start_node_temp, edges_by_indices=self.edges_by_indices, valid_mask=valid_mask, update_winding_angles=False) # do not update winding angles since this would result in an inconsistent graph

        # Filter PointCloud for max 1 patch per subvolume (again evolutionary algorithm)
        evolved_graph = self.filter(evolved_graph_temp, graph=evolved_graph, min_z=graph_extraction_start, max_z=graph_extraction_start+z_height_steps)

        # Extract largest connected component for filtered graph (start_node_temp when start_node==None might have been removed)
        largest_component = evolved_graph.largest_connected_component(delete_nodes=False)

        if start_node is None:
            # start_node_graph = self.graph_from_edge_selection(self.edges_by_indices, self.graph, valid_mask)
            start_node = largest_component[0]

        # Compute final ks by simple bfs (for this subgraph)
        self.update_ks(evolved_graph, start_node=start_node, edges_by_indices=self.edges_by_indices, valid_mask=valid_mask, update_winding_angles=False) # do not update winding angles since this would result in an inconsistent graph

        pbar.update(1)
        
        return start_node, evolved_graph, valid_mask

    def solve(self, z_height_steps=200):
        graph_centroids = np.array([self.graph.nodes[node]['centroid'] for node in self.graph.nodes])
        graph_centroids_min = int(np.floor(np.min(graph_centroids, axis=0))[1])
        graph_centroids_max = int(np.ceil(np.max(graph_centroids, axis=0))[1])
        graph_centroids_middle = int(np.round(np.mean(graph_centroids, axis=0))[1])
        print(f"Graph centroids min: {np.floor(np.min(graph_centroids, axis=0))}, max: {np.ceil(np.max(graph_centroids, axis=0))}")

        evolved_graph = self.graph.naked_graph()
        start_node = None
        with tqdm(total=2 + (graph_centroids_max - graph_centroids_min) // z_height_steps, desc="Evolving valid graph") as pbar:
            for graph_extraction_start in range(graph_centroids_middle, graph_centroids_max, z_height_steps):
                if abs(graph_extraction_start - graph_centroids_middle) // z_height_steps > 1:
                    break
                # Extract all the nodes and connections of them from one z height cutout in the graph
                start_node, evolved_graph, valid_mask = self.solve_subloop(pbar, graph_extraction_start, z_height_steps, start_node, evolved_graph)
            for graph_extraction_start in range(graph_centroids_middle-z_height_steps, graph_centroids_min, -z_height_steps):
                if abs(graph_extraction_start - graph_centroids_middle) // z_height_steps > 1:
                    break
                # Extract all the nodes and connections of them from one z height cutout in the graph
                start_node, evolved_graph, valid_mask = self.solve_subloop(pbar, graph_extraction_start, z_height_steps, start_node, evolved_graph)

        print("Finishing up Graph Evolution...")
        # select largest connected component
        evolved_graph.largest_connected_component()
        self.update_ks(evolved_graph, edges_by_indices=self.edges_by_indices, valid_mask=valid_mask, update_winding_angles=True) # update winding angles here at the very end (only once)
        print("Solved graph with genetic algorithm.")
        return evolved_graph
    
    def filter(self, graph_to_filter, min_z=None, max_z=None, graph=None):
        print("Filtering patches with genetic algorithm...")
        # Filter edges with genetic algorithm
        _, self.edges_by_subvolume_indices, _, index_nodes_dict = self.build_graph_data(graph_to_filter, min_z=min_z, max_z=max_z, strict_edges=False)
        # Solve with genetic algorithm
        valid_mask, valid_edges_count = self.solve_call(self.edges_by_subvolume_indices, problem="patch_selection")
        # Build graph selected nodes. maintains connectivity from the unfiltered graph
        filtered_graph = self.graph_from_node_selection(self.edges_by_subvolume_indices, graph_to_filter, valid_mask, index_nodes_dict, graph=graph)
        return filtered_graph
    
    def graph_from_node_selection(self, edges_indices, input_graph, edges_mask, index_nodes_dict, graph=None, min_z=None, max_z=None):
        """
        Creates a graph from the DP table.
        """
        if graph is None:
            graph = ScrollGraph(input_graph.overlapp_threshold, input_graph.umbilicus_path)
        # start block and patch id
        graph.start_block = input_graph.start_block
        graph.patch_id = input_graph.patch_id

        # Collect selected nodes
        selected_nodes = set()
        for i in tqdm(range(len(edges_mask))):
            if edges_mask[i]:
                edge = edges_indices[i]
                node0_index, node1_index, k = edge[:3]
                node1 = index_nodes_dict[node0_index]
                node2 = index_nodes_dict[node1_index]
                certainty = input_graph.edges[(node1, node2)][k]['certainty']

                centroid1 = input_graph.nodes[node1]['centroid']
                centroid2 = input_graph.nodes[node2]['centroid']

                if (min_z is not None) and ((centroid1[1] < min_z) or (centroid2[1] < min_z)):
                    continue
                if (max_z is not None) and ((centroid1[1] > max_z) or (centroid2[1] > max_z)):
                    continue

                assert certainty > 0.0, f"Invalid certainty: {certainty} for edge: {edge}"
                assert ('assigned_k' in input_graph.nodes[node1]) == ('assigned_k' in input_graph.nodes[node2]), f"Invalid assigned k: {input_graph.nodes[node1]} != {input_graph.nodes[node2]}"
                if not ('assigned_k' in input_graph.nodes[node1]):
                    continue
                # Add selected nodes
                selected_nodes.add(node1)
                selected_nodes.add(node2)

        selected_nodes = list(selected_nodes)
        
        # graph add nodes
        nr_winding_angles = 0
        nodes = list(input_graph.nodes.keys())
        print(f"input nodes length: {len(nodes)} vs selected nodes length: {len(selected_nodes)}")
        for node in selected_nodes:
            if input_graph.nodes[node]['winding_angle'] is not None:
                nr_winding_angles += 1
            graph.add_node(node, input_graph.nodes[node]['centroid'], winding_angle=input_graph.nodes[node]['winding_angle'])
            # Assign k
            graph.nodes[node]['assigned_k'] = input_graph.nodes[node]['assigned_k']
        added_edges_count = 0
        print(f"Number of winding angles: {nr_winding_angles} of output {len(selected_nodes)} nodes.")

        start_node = selected_nodes[0]
        selected_nodes = selected_nodes[1:]
        k_start = input_graph.nodes[start_node]['assigned_k']
        # Connect every selected node with the start node
        for node in selected_nodes:
            k_node = input_graph.nodes[node]['assigned_k']
            k = k_node - k_start
            graph.add_edge(start_node, node, 1.0, k, False)
            added_edges_count += 1
            
        print(f"Added {added_edges_count} edges to the graph.")
        graph.compute_node_edges()
        print(f"Filtered graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
        return graph
    
    def check_contracted_and_solution_graph(self, contracted_graph, solution_graph):
        contracted_graph.compute_node_edges()
        solution_graph.compute_node_edges()
        # for each node in contracted graph as start node do bfs in solution graph
        for node in contracted_graph.nodes:
            start_node = node
            bfs_nodes = solution_graph.bfs(start_node)
            
            for i in range(len(bfs_nodes)):
                bfs_node = bfs_nodes[i]
                if bfs_node in contracted_graph.nodes:
                    assert start_node == bfs_node, f"Node {start_node} != {bfs_node} in bfs_nodes: {bfs_nodes}"        
    
    def contracted_graph_from_edge_selection_adjacent(self, edges_indices_list, input_graph, edges_mask_list, index_nodes_dict_list, graph_blocks, graph=None):
        """
        Creates a contracted graph from the DP table.
        """
        if graph is None:
            graph = ScrollGraph(input_graph.overlapp_threshold, input_graph.umbilicus_path)
        # start block and patch id
        graph.start_block = input_graph.start_block
        graph.patch_id = input_graph.patch_id

        # Build up list of contracted nodes
        block_id_dict = {}
        # Initialize contracted nodes
        for pos in range(len(edges_indices_list)):
            edges_indices = edges_indices_list[pos]
            edges_mask = edges_mask_list[pos]
            index_nodes_dict = index_nodes_dict_list[pos]

            for i in tqdm(range(len(edges_mask))):
                edge = edges_indices[i]
                node0_index, node1_index = edge[:2]
                k = edge[2]
                node1 = index_nodes_dict[node0_index]
                node2 = index_nodes_dict[node1_index]
                # store block id
                block_id_dict[node1] = pos
                block_id_dict[node2] = pos
                # Add selected nodes to contracted nodes
                if edges_mask[i]:
                    same_block = input_graph.edges[(node1, node2)][k]['same_block']
                    if same_block:
                        continue
                    # ????

        uf = WeightedUnionFind()

        # Contract other block edges
        for pos in range(len(edges_indices_list)):
            edges_indices = edges_indices_list[pos]
            edges_mask = edges_mask_list[pos]
            index_nodes_dict = index_nodes_dict_list[pos]
            for i in tqdm(range(len(edges_mask))):
                if edges_mask[i]:
                    edge = edges_indices[i]
                    node0_index, node1_index = edge[:2]
                    node1 = index_nodes_dict[node0_index]
                    node2 = index_nodes_dict[node1_index]
                    k = edge[2]
                    # assert k == k_, f"Invalid k in graph construction: {k} != {k_}"
                    same_block = input_graph.edges[(node1, node2)][k]['same_block']
                    assert not input_graph.edges[(node1, node2)][k]['bad_edge'], f"Invalid bad edge: {node1}, {node2}, {k}. All edges here should be good edges by construction."
                    # contract nodes that have an evolved connected edge (only if it was not same block connection)
                    if not same_block:
                        assert uf.merge(node1, node2, k), f"Invalid merge: {node1}, {node2}, {k}"

        # Build up list of contracted edges (same block edges)
        contracted_edges = {}
        # for pos in range(len(edges_indices_list)):
        #     edges_indices = edges_indices_list[pos]
        #     edges_mask = edges_mask_list[pos]
        #     index_nodes_dict = index_nodes_dict_list[pos]
        #     for i in tqdm(range(len(edges_mask))):
        #         if edges_mask[i]:
        #             edge = edges_indices[i]
        #             node0_index, node1_index = edge[:2]
        #             node1 = index_nodes_dict[node0_index]
        #             node2 = index_nodes_dict[node1_index]
        for edge in input_graph.edges:
            node1, node2 = edge
            if (node1 in uf.parent) and (node2 in uf.parent):
                ks = input_graph.get_edge_ks(node1, node2)
                for k in ks:
                    if input_graph.edges[edge][k]['bad_edge']:
                        continue
                    certainty = input_graph.edges[(node1, node2)][k]['certainty']
                    same_block = input_graph.edges[(node1, node2)][k]['same_block']
                    # only look at same block edges
                    if not same_block:
                        continue

                    node1_root, k1 = uf.find(node1)
                    node2_root, k2 = uf.find(node2)

                    # discard self loops
                    if node1_root == node2_root:
                        continue

                    # proper order of nodes
                    first_node = node2_root if node2_root < node1_root else node1_root
                    second_node = node1_root if node2_root < node1_root else node2_root
                    k_adjusted = int(k1 - k - k2)
                    k_adjusted = k_adjusted if node2_root < node1_root else -k_adjusted

                    if not (first_node, second_node) in contracted_edges:
                        contracted_edges[(first_node, second_node)] = {}

                    if not k_adjusted in contracted_edges[(first_node, second_node)]:
                        contracted_edges[(first_node, second_node)][k_adjusted] = 0.0
                    contracted_edges[(first_node, second_node)][k_adjusted] += certainty

        # Add contracted edges information to graph
        added_edges_count = 0
        for edge in contracted_edges:
            node1, node2 = edge
            for k in contracted_edges[edge]:
                certainty = contracted_edges[edge][k]

                assert certainty > 0.0, f"Invalid certainty: {certainty} for edge: {edge}"
                graph.add_edge(node1, node2, certainty, k, True) # "Same Block" edge
                if not node1 in graph.nodes:
                    graph.add_node(node1, input_graph.nodes[node1]['centroid'], winding_angle=input_graph.nodes[node1]['winding_angle'])
                if not node2 in graph.nodes:
                    graph.add_node(node2, input_graph.nodes[node2]['centroid'], winding_angle=input_graph.nodes[node2]['winding_angle'])
                added_edges_count += 1

        # Define all nodes
        nodes = set()
        for node in uf.get_roots():
            nodes.add(node)

        nodes = list(nodes)
        print(f"nodes length: {len(nodes)}")

        # graph add nodes
        nr_winding_angles = 0
        for node in nodes:
            if input_graph.nodes[node]['winding_angle'] is not None:
                nr_winding_angles += 1
            graph.add_node(node, input_graph.nodes[node]['centroid'], winding_angle=input_graph.nodes[node]['winding_angle'])
        print(f"Number of winding angles: {nr_winding_angles} of {len(nodes)} nodes.")

        # update roots of every node
        count_total = len(uf.parent)
        count_roots = len(uf.get_roots())
        print(colored(f"Number of roots: {count_roots}, Number of total: {count_total}", 'yellow'))

        # Compile larger scale transitioning edges between different "pos" ids
        contracted_edges_transitioning = {}
        for edge in input_graph.edges:
            node1, node2 = edge
            if (node1 in uf.parent) and (node2 in uf.parent):
                ks = input_graph.get_edge_ks(node1, node2)
                for k in ks:
                    if input_graph.edges[edge][k]['bad_edge']:
                        continue
                    same_block = input_graph.edges[(node1, node2)][k]['same_block']
                    if same_block: # only add transitioning edges
                        continue
                    if (node1 in block_id_dict) and (node2 in block_id_dict) and (block_id_dict[node1] != block_id_dict[node2]):
                        node1_root, k1 = uf.find(node1)
                        node2_root, k2 = uf.find(node2)
                        if node1_root != node2_root:
                            certainty = input_graph.edges[edge][k]['certainty']
                        
                            first_node = node2_root if node2_root < node1_root else node1_root
                            second_node = node1_root if node2_root < node1_root else node2_root
                            k_adjusted = int(k1 - k - k2)
                            k_adjusted = k_adjusted if node2_root < node1_root else -k_adjusted

                            if not (first_node, second_node) in contracted_edges_transitioning:
                                contracted_edges_transitioning[(first_node, second_node)] = {}

                            if not k_adjusted in contracted_edges_transitioning[(first_node, second_node)]:
                                contracted_edges_transitioning[(first_node, second_node)][k_adjusted] = 0.0

                            contracted_edges_transitioning[(first_node, second_node)][k_adjusted] += certainty

        # Add transitioning edges to graph
        for edge in contracted_edges_transitioning:
            node1, node2 = edge
            for k in contracted_edges_transitioning[edge]:
                certainty = contracted_edges_transitioning[edge][k]

                assert certainty > 0.0, f"Invalid certainty: {certainty} for edge: {edge}"
                graph.add_edge(node1, node2, certainty, k, False) # "Transitioning Block" edge
                if not node1 in graph.nodes:
                    graph.add_node(node1, input_graph.nodes[node1]['centroid'], winding_angle=input_graph.nodes[node1]['winding_angle'])
                if not node2 in graph.nodes:
                    graph.add_node(node2, input_graph.nodes[node2]['centroid'], winding_angle=input_graph.nodes[node2]['winding_angle'])
                added_edges_count += 1

        # Add bad edges to contracted graph
        for pos in range(len(edges_indices_list)):
            edges_indices = edges_indices_list[pos]
            edges_mask = edges_mask_list[pos]
            index_nodes_dict = index_nodes_dict_list[pos]

            success_, ks_list = self.bfs_ks_indices(edges_indices, valid_mask_int=edges_mask)
            if not success_:
                continue

            nodes_indices_pos = set()
            for i in tqdm(range(len(edges_mask))):
                if edges_mask[i]:
                    edge = edges_indices[i]
                    node1_index, node2_index = edge[:2]
                    nodes_indices_pos.add(node1_index)
                    nodes_indices_pos.add(node2_index)
            
            nodes_indices_pos = list(nodes_indices_pos)
            for i in range(len(nodes_indices_pos)):
                    for j in range(i+1, len(nodes_indices_pos)):
                        node1_index = nodes_indices_pos[i]
                        node2_index = nodes_indices_pos[j]
                        node1 = index_nodes_dict[node1_index]
                        node2 = index_nodes_dict[node2_index]

                        if (node1 in uf.parent) and (node2 in uf.parent):
                            node1_root, k1 = uf.find(node1)
                            node2_root, k2 = uf.find(node2)

                            # discard self loops
                            if node1_root == node2_root:
                                continue
                            # discard contracted edges
                            if node1_root != node1:
                                continue
                            if node2_root != node2:
                                continue

                            for ks in ks_list:
                                # check if node0_index, node1_index in same connected component
                                if (node1_index in ks) and (node2_index in ks):
                                    if abs(ks[node1_index] - ks[node2_index]) < 3:
                                        break # this is a good edge, not adding bad edge here
                                    graph.add_edge(node1, node2, 1.0, 0.0, same_block=True, bad_edge=True)
                                    added_edges_count += 1

        print(f"Added {added_edges_count} edges to the graph.")
        graph.compute_node_edges()
        print(f"Filtered graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
        return graph
    
    def contracted_graph_from_edge_selection(self, edges_indices_list, input_graph, edges_mask_list, index_nodes_dict_list, graph_blocks, graph=None):
        """
        Creates a contracted graph from the DP table.
        """
        def core_node(graph_block, node): # finds node in contracted graph
            _, _, _, _, (min_x, max_x, min_y, max_y, min_z, max_z, padding) = graph_block
            if (input_graph.nodes[node]['centroid'][2] <= min_x + padding):
                return False
            if (input_graph.nodes[node]['centroid'][2] > max_x - padding):
                return False
            if (input_graph.nodes[node]['centroid'][0] <= min_y + padding):
                return False
            if (input_graph.nodes[node]['centroid'][0] > max_y - padding):
                return False
            if (input_graph.nodes[node]['centroid'][1] <= min_z + padding):
                return False
            if (input_graph.nodes[node]['centroid'][1] > max_z - padding):
                return False
            return True
        
        def contains_node(graph_block, node):
            _, _, _, _, (min_x, max_x, min_y, max_y, min_z, max_z, padding) = graph_block
            if (input_graph.nodes[node]['centroid'][2] <= min_x):
                return False
            if (input_graph.nodes[node]['centroid'][2] > max_x):
                return False
            if (input_graph.nodes[node]['centroid'][0] <= min_y):
                return False
            if (input_graph.nodes[node]['centroid'][0] > max_y):
                return False
            if (input_graph.nodes[node]['centroid'][1] <= min_z):
                return False
            if (input_graph.nodes[node]['centroid'][1] > max_z):
                return False
            return True
        
        if graph is None:
            graph = ScrollGraph(input_graph.overlapp_threshold, input_graph.umbilicus_path)
        # start block and patch id
        graph.start_block = input_graph.start_block
        graph.patch_id = input_graph.patch_id

        # Build up list of contracted nodes
        block_id_dict = {}
        # Initialize contracted nodes
        for pos in range(len(edges_indices_list)):
            edges_indices = edges_indices_list[pos]
            edges_mask = edges_mask_list[pos]
            index_nodes_dict = index_nodes_dict_list[pos]
            graph_block = graph_blocks[pos]

            for i in range(len(edges_mask)):
                edge = edges_indices[i]
                node0_index, node1_index = edge[:2]
                k = edge[2]
                node1 = index_nodes_dict[node0_index]
                node2 = index_nodes_dict[node1_index]
                if (not core_node(graph_block, node1)) or (not core_node(graph_block, node2)):
                    continue
                # store block id
                block_id_dict[node1] = pos
                block_id_dict[node2] = pos
                # Add selected nodes to contracted nodes
                if edges_mask[i]:
                    same_block = input_graph.edges[(node1, node2)][k]['same_block']
                    if same_block:
                        continue
                    # ????

        ufs = []
        uf_cores = []

        # Contract other block edges
        for pos in range(len(edges_indices_list)):
            ufs.append(WeightedUnionFind())
            uf_cores.append(WeightedUnionFind())
            edges_indices = edges_indices_list[pos]
            edges_mask = edges_mask_list[pos]
            index_nodes_dict = index_nodes_dict_list[pos]
            graph_block = graph_blocks[pos]
            for i in tqdm(range(len(edges_mask))):
                if edges_mask[i]:
                    edge = edges_indices[i]
                    node0_index, node1_index = edge[:2]
                    node1 = index_nodes_dict[node0_index]
                    node2 = index_nodes_dict[node1_index]
                    k = edge[2]
                    # assert k == k_, f"Invalid k in graph construction: {k} != {k_}"
                    same_block = input_graph.edges[(node1, node2)][k]['same_block']
                    assert not input_graph.edges[(node1, node2)][k]['bad_edge'], f"Invalid bad edge: {node1}, {node2}, {k}. All edges here should be good edges by construction."
                    # contract nodes that have an evolved connected edge (only if it was not same block connection)
                    if not same_block:
                        assert ufs[pos].merge(node1, node2, k, core_node(graph_block, node1), core_node(graph_block, node2)), f"Invalid merge: {node1}, {node2}, {k}"
                    # Add all connections to core nodes
                    assert uf_cores[pos].merge(node1, node2, k, core_node(graph_block, node1), core_node(graph_block, node2)), f"Invalid merge: {node1}, {node2}, {k}"

        # Build up list of contracted edges (same block edges)
        contracted_edges = {}
        
        # Union Find Approach
        added_edges_count = 0
        non_core_roots = 0
        size_non_core = 0
        core_roots = 0
        size_core = 0
        non_connected_roots = 0
        connected_roots = 0
        size_unconnected = 0
        size_connected = 0
        total_roots = 0
        for pos in range(len(edges_indices_list)):
            roots = ufs[pos].get_roots()
            graph_block = graph_blocks[pos]
            total_roots += len(roots)
            for i in range(len(roots)):
                node1_root = roots[i]
                if not core_node(graph_block, node1_root):
                        size_non_core += ufs[pos].size[node1_root]
                        non_core_roots += 1
                else:
                    size_core += ufs[pos].size[node1_root]
                    core_roots += 1
            for i in range(len(roots)):
                node1_root = roots[i]
                if not core_node(graph_block, node1_root):
                    continue
                for j in range(i+1,len(roots)):
                    node2_root = roots[j]
                    if not core_node(graph_block, node2_root):
                        continue
                    if not uf_cores[pos].connected(node1_root, node2_root):
                        non_connected_roots += 1
                        size_unconnected += ufs[pos].size[node1_root] + ufs[pos].size[node2_root]
                        continue
                    connected_roots += 1
                    size_connected += ufs[pos].size[node1_root] + ufs[pos].size[node2_root]

                    k_adjusted = uf_cores[pos].connection_weight(node1_root, node2_root)
                    first_node = node1_root if node1_root < node2_root else node2_root
                    second_node = node2_root if node1_root < node2_root else node1_root
                    k_adjusted = k_adjusted if node1_root < node2_root else -k_adjusted
                    if abs(k_adjusted) > 1: # difference in k is too large, add bad edge (edge that gives negative score for evolution if those two nodes belong to the same k)
                        graph.add_edge(first_node, second_node, 1.0, 0.0, same_block=True, bad_edge=True)
                        added_edges_count += 1
                    else:
                        if not (first_node, second_node) in contracted_edges:
                            contracted_edges[(first_node, second_node)] = {}
                        if not k_adjusted in contracted_edges[(first_node, second_node)]:
                            contracted_edges[(first_node, second_node)][k_adjusted] = 0.0
                        size_1 = ufs[pos].size[node1_root]
                        size_2 = ufs[pos].size[node2_root]
                        contracted_edges[(first_node, second_node)][k_adjusted] += 100.0 / (1 + abs(k_adjusted)) # min(size_1, size_2)

        print(f"nr roots: {total_roots}")
        print(f"core_roots: {core_roots}, size_core: {size_core},")
        print(f"non_core_roots: {non_core_roots}, size_non_core: {size_non_core},")
        print(f"non_connected_roots: {non_connected_roots}, size_unconnected: {size_unconnected}")
        print(f"connected_roots: {connected_roots}, size_connected: {size_connected}")

        # # original bad edges approach
        # # Build up list of contracted edges (same block edges)
        # for edge in input_graph.edges:
        #     node1, node2 = edge
        #     if not (node1 in block_id_dict) or not (node2 in block_id_dict):
        #         continue
        #     pos1 = block_id_dict[node1]
        #     pos2 = block_id_dict[node2]
        #     if pos2 != pos1:
        #         continue
        #     if (node1 in ufs[pos1].parent) and (node2 in ufs[pos1].parent):
        #         ks = input_graph.get_edge_ks(node1, node2)
        #         for k in ks:
        #             if input_graph.edges[edge][k]['bad_edge']:
        #                 continue
        #             certainty = input_graph.edges[(node1, node2)][k]['certainty']
        #             same_block = input_graph.edges[(node1, node2)][k]['same_block']
        #             # only look at same block edges
        #             if not same_block:
        #                 continue

        #             node1_root, k1 = ufs[pos1].find(node1)
        #             node2_root, k2 = ufs[pos1].find(node2)

        #             if not core_node(graph_blocks[pos1], node1_root) or not core_node(graph_blocks[pos1], node2_root):
        #                 continue

        #             # discard self loops
        #             if node1_root == node2_root:
        #                 continue

        #             # proper order of nodes
        #             first_node = node2_root if node2_root < node1_root else node1_root
        #             second_node = node1_root if node2_root < node1_root else node2_root
        #             k_adjusted = - int(- k1 + k + k2)
        #             k_adjusted = k_adjusted if node2_root < node1_root else -k_adjusted

        #             if not (first_node, second_node) in contracted_edges:
        #                 contracted_edges[(first_node, second_node)] = {}

        #             if not k_adjusted in contracted_edges[(first_node, second_node)]:
        #                 contracted_edges[(first_node, second_node)][k_adjusted] = 0.0
        #             contracted_edges[(first_node, second_node)][k_adjusted] += certainty

        print(f"contracted_edges length: {len(contracted_edges)}")

        # Add contracted edges information to graph
        for edge in contracted_edges:
            node1, node2 = edge
            # max_k = max(contracted_edges[edge], key=lambda k: contracted_edges[edge][k])
            # k = max_k
            for k in contracted_edges[edge]:
                certainty = contracted_edges[edge][k]

                if len(contracted_edges[edge]) > 1:
                    print(f"adding edge: {node1}, {node2}, {k}, {certainty}, len: {len(contracted_edges[edge])}")

                assert certainty > 0.0, f"Invalid certainty: {certainty} for edge: {edge}"
                if k == 0:
                    graph.add_edge(node1, node2, 100.0, k, False)
                else:
                    graph.add_edge(node1, node2, certainty, k, True) # "Same Block" edge
                if not node1 in graph.nodes:
                    graph.add_node(node1, input_graph.nodes[node1]['centroid'], winding_angle=input_graph.nodes[node1]['winding_angle'])
                if not node2 in graph.nodes:
                    graph.add_node(node2, input_graph.nodes[node2]['centroid'], winding_angle=input_graph.nodes[node2]['winding_angle'])
                added_edges_count += 1

        # Define all nodes
        nodes = set()
        for pos in range(len(edges_indices_list)):
            for node in ufs[pos].get_roots():
                if core_node(graph_blocks[pos], node):
                    nodes.add(node)

        nodes = list(nodes)
        print(f"nodes length: {len(nodes)}")

        # graph add nodes
        nr_winding_angles = 0
        for node in nodes:
            if input_graph.nodes[node]['winding_angle'] is not None:
                nr_winding_angles += 1
            graph.add_node(node, input_graph.nodes[node]['centroid'], winding_angle=input_graph.nodes[node]['winding_angle'])
        print(f"Number of winding angles: {nr_winding_angles} of {len(nodes)} nodes.")

        # update roots of every node
        # count_total = len(uf.parent)
        # count_roots = len(uf.get_roots())
        # print(colored(f"Number of roots: {count_roots}, Number of total: {count_total}", 'yellow'))
        def overlapp_score(overlap, overlapp_percentage, overlapp_threshold={"score_threshold": 0.10, "nr_points_min": 3.0, "nr_points_max": 40.0}):
            nrp = min(max(overlapp_threshold["nr_points_min"], overlap), overlapp_threshold["nr_points_max"]) / (overlapp_threshold["nr_points_min"])
            nrp_factor = np.log(np.log(nrp) + 1.0) + 1.0

            score = overlapp_percentage

            if score >= overlapp_threshold["score_threshold"] and overlap >= overlapp_threshold["nr_points_min"] and overlap > 0:
                score = ((score - overlapp_threshold["score_threshold"])/ (1 - overlapp_threshold["score_threshold"])) ** 2
                return score * nrp_factor
            else:
                return -1.0

        # Compile larger scale transitioning edges between different "pos" ids
        # contracted_edges_transitioning = {}
        # # TODO: speed up!
        # # TODO: rework score function
        # for pos1 in range(len(edges_indices_list)):
        #     for pos2 in range(pos1+1, len(edges_indices_list)):
        #         graph_block1 = graph_blocks[pos1]
        #         graph_block2 = graph_blocks[pos2]
        #         # only compute on overlapping blocks
        #         _, _, _, _, (min_x1, max_x1, min_y1, max_y1, min_z1, max_z1, padding1) = graph_block1
        #         _, _, _, _, (min_x2, max_x2, min_y2, max_y2, min_z2, max_z2, padding2) = graph_block2
        #         x_overlap_bool = (min_x1 < max_x2) and (max_x1 > min_x2)
        #         y_overlap_bool = (min_y1 < max_y2) and (max_y1 > min_y2)
        #         z_overlap_bool = (min_z1 < max_z2) and (max_z1 > min_z2)
        #         overlap_bool = x_overlap_bool and y_overlap_bool and z_overlap_bool
        #         if not overlap_bool:
        #             continue

        #         ufs1 = ufs[pos1]
        #         ufs2 = ufs[pos2]
        #         # calculate connected components overlapp
        #         connected_components1 = ufs1.get_components()
        #         connected_components2 = ufs2.get_components()

        #         for i in range(len(connected_components1)):
        #             for j in range(len(connected_components2)):
        #                 component1, component2 = connected_components1[i], connected_components2[j]
        #                 root_pos1, _ = ufs1.find(component1[0])
        #                 root_pos2, _ = ufs2.find(component2[0])

        #                 if (not core_node(graph_block1, root_pos1)) or (not core_node(graph_block2, root_pos2)):
        #                     continue

        #                 component_overlap_score = {}
        #                 total_found = 0
        #                 component_missing_score = 0.0
        #                 for node1 in component1:
        #                     # node1 centroid in both blocks
        #                     if contains_node(graph_block1, node1) and contains_node(graph_block2, node1):
        #                         if node1 in component2:
        #                             root_pos1_, k_pos1 = ufs[pos1].find(node1)
        #                             root_pos2_, k_pos2 = ufs[pos2].find(node1)
        #                             assert root_pos1_ == root_pos1, f"Invalid root_pos1: {root_pos1_} != {root_pos1}"
        #                             assert root_pos2_ == root_pos2, f"Invalid root_pos2: {root_pos2_} != {root_pos2}"

        #                             k = - (k_pos1 - k_pos2) # root_pos1 -> node1 -> root_pos2
        #                             if not (k in component_overlap_score):
        #                                 component_overlap_score[k] = 0
        #                             component_overlap_score[k] += 1
        #                             total_found += 1
        #                         else: # not in both blocks
        #                             component_missing_score += 1
        #                 for node2 in component2:
        #                     # node2 centroid not in both blocks
        #                     if contains_node(graph_block1, node2) and contains_node(graph_block2, node2):
        #                         if not (node2 in component1):
        #                             component_missing_score += 1
                        
        #                 for k in component_overlap_score:
        #                     percentage = component_overlap_score[k] / (total_found + component_missing_score)
        #                     if percentage <= 0.0:
        #                         continue

        #                     score = 1000 * overlapp_score(component_overlap_score[k], percentage)
        #                     if score <= 0.0:
        #                         continue
        #                     # score = 100 * percentage
        #                     # print(f"Overlap score: {score} for k: {k} between {root_pos1} and {root_pos2}")

        #                     first_node = root_pos1 if root_pos1 < root_pos2 else root_pos2
        #                     second_node = root_pos2 if root_pos1 < root_pos2 else root_pos1
        #                     k_adjusted = k if root_pos1 < root_pos2 else -k
        #                     if not (first_node, second_node) in contracted_edges_transitioning:
        #                         contracted_edges_transitioning[(first_node, second_node)] = {}
        #                     if not k in contracted_edges_transitioning[(first_node, second_node)]:
        #                         contracted_edges_transitioning[(first_node, second_node)][k] = 0.0
        #                     contracted_edges_transitioning[(first_node, second_node)][k] += score

        # print(f"Candidates for transitioning edges: {len(contracted_edges_transitioning)}")
        # # Add transitioning edges to graph
        # for edge in contracted_edges_transitioning:
        #     node1, node2 = edge
        #     for k in contracted_edges_transitioning[edge]:
        #         certainty = contracted_edges_transitioning[edge][k]

        #         assert certainty > 0.0, f"Invalid certainty: {certainty} for edge: {edge}"
        #         graph.add_edge(node1, node2, certainty, k, False) # "Transitioning Block" edge
        #         if not node1 in graph.nodes:
        #             graph.add_node(node1, input_graph.nodes[node1]['centroid'], winding_angle=input_graph.nodes[node1]['winding_angle'])
        #         if not node2 in graph.nodes:
        #             graph.add_node(node2, input_graph.nodes[node2]['centroid'], winding_angle=input_graph.nodes[node2]['winding_angle'])
        #         added_edges_count += 1

        # contract other block edges into the root nodes when building the graph
        count_found_pos = 0
        count_missing_pos = 0
        for pos in range(len(edges_indices_list)):
            graph_block = graph_blocks[pos]
            edges_indices = edges_indices_list[pos]
            edges_mask = edges_mask_list[pos]
            index_nodes_dict = index_nodes_dict_list[pos]
            graph_block = graph_blocks[pos]
            for i in range(len(edges_mask)):
                if edges_mask[i]:
                    edge = edges_indices[i]
                    node0_index, node1_index = edge[:2]
                    node1 = index_nodes_dict[node0_index]
                    node2 = index_nodes_dict[node1_index]
                    k = edge[2]

                    if not core_node(graph_block, node1):
                        node = node2
                        other_node = node1
                        k = -k
                    else:
                        node = node1
                        other_node = node2
                    # first node is core node
                    if not core_node(graph_block, node):
                        continue
                    # Only increment-contract edges that switch between core and non-core nodes
                    if core_node(graph_block, other_node):
                        continue

                    node_root = ufs[pos].find(node)[0]
                    
                    if not other_node in block_id_dict: # edge going to an
                        count_missing_pos += 1
                        continue
                    count_found_pos += 1

                    other_pos = block_id_dict[other_node]
                    assert other_pos != pos, f"Invalid other_pos: {other_pos} == {pos}"

                    # root in other subblock
                    other_root = ufs[other_pos].find(other_node)[0]

                    # calculate k for node_root -> other_root by the way of (node_root -> node -> other_node -> other_root)
                    k_node_root_node = ufs[pos].connection_weight(node_root, node)
                    k_other_node_other_root = ufs[other_pos].connection_weight(other_node, other_root)

                    certainty = input_graph.get_certainty(node, other_node, k)
                    assert certainty >= 0.0, f"Invalid certainty: {certainty} for edge: {node}, {other_node}, {k}"
                    
                    k_adjusted = k_node_root_node + k + k_other_node_other_root
                    graph.add_increment_edge(node_root, other_root, certainty, k_adjusted, False) # Transitioning edge
                    if not node_root in graph.nodes:
                        graph.add_node(node_root, input_graph.nodes[node_root]['centroid'], winding_angle=input_graph.nodes[node_root]['winding_angle'])
                    if not other_root in graph.nodes:
                        graph.add_node(other_root, input_graph.nodes[other_root]['centroid'], winding_angle=input_graph.nodes[other_root]['winding_angle'])
                    added_edges_count += 1
        print(f"Found {count_found_pos} edges between different blocks. And missed {count_missing_pos} edges.")

        # Add bad edges to contracted graph
        # for pos in range(len(edges_indices_list)):
        #     edges_indices = edges_indices_list[pos]
        #     edges_mask = edges_mask_list[pos]
        #     index_nodes_dict = index_nodes_dict_list[pos]
        #     graph_block = graph_blocks[pos]

        #     success_, ks_list = self.bfs_ks_indices(edges_indices, valid_mask_int=edges_mask)
        #     if not success_:
        #         continue

        #     nodes_indices_pos = set()
        #     for i in tqdm(range(len(edges_mask))):
        #         if edges_mask[i]:
        #             edge = edges_indices[i]
        #             node1_index, node2_index = edge[:2]
        #             nodes_indices_pos.add(node1_index)
        #             nodes_indices_pos.add(node2_index)
            
        #     nodes_indices_pos = list(nodes_indices_pos)
        #     for i in range(len(nodes_indices_pos)):
        #             for j in range(i+1, len(nodes_indices_pos)):
        #                 node1_index = nodes_indices_pos[i]
        #                 node2_index = nodes_indices_pos[j]
        #                 node1 = index_nodes_dict[node1_index]
        #                 node2 = index_nodes_dict[node2_index]

        #                 if (node1 in uf.parent) and (node2 in uf.parent):
        #                     node1_root, k1 = uf.find(node1)
        #                     node2_root, k2 = uf.find(node2)

        #                     # discard self loops
        #                     if node1_root == node2_root:
        #                         continue
        #                     # discard contracted edges
        #                     if node1_root != node1:
        #                         continue
        #                     if node2_root != node2:
        #                         continue

        #                     for ks in ks_list:
        #                         # check if node0_index, node1_index in same connected component
        #                         if (node1_index in ks) and (node2_index in ks):
        #                             if abs(ks[node1_index] - ks[node2_index]) < 3:
        #                                 break # this is a good edge, not adding bad edge here
        #                             graph.add_edge(node1, node2, 1.0, 0.0, same_block=True, bad_edge=True)
        #                             added_edges_count += 1

        print(f"Added {added_edges_count} edges to the graph.")
        graph.compute_node_edges()
        print(f"Filtered graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
        return graph
    
    def contracted_graph_from_solution_graph(self, solution_graph, original_graph, side_length):
        original_graph.compute_node_edges()
        print(f"[Info]: Side length is {side_length}")
        graph_z = np.array([original_graph.nodes[node]['centroid'][1] for node in original_graph.nodes])
        graph_z_min = int(np.floor(np.min(graph_z)))
        graph_x = np.array([original_graph.nodes[node]['centroid'][2] for node in original_graph.nodes])
        graph_x_min = int(np.floor(np.min(graph_x)))
        graph_y = np.array([original_graph.nodes[node]['centroid'][0] for node in original_graph.nodes])
        graph_y_min = int(np.floor(np.min(graph_y)))

        def subblock_from_centroid(node):
            centroid = original_graph.nodes[node]['centroid']
            x = int((centroid[2] - graph_x_min) // side_length)
            y = int((centroid[0] - graph_y_min) // side_length)
            z = int((centroid[1] - graph_z_min) // side_length)

            return (x, y, z)

        def same_cores(node1, node2): # finds node in contracted graph
            subblock1 = subblock_from_centroid(node1)
            subblock2 = subblock_from_centroid(node2)
            
            return subblock1 == subblock2
        
        graph = ScrollGraph(solution_graph.overlapp_threshold, solution_graph.umbilicus_path)
        graph.start_block = solution_graph.start_block
        graph.patch_id = solution_graph.patch_id

        uf = WeightedUnionFind()
        count_uf_merged = 0
        for edge in solution_graph.edges:
            node1, node2 = edge
            ks = solution_graph.get_edge_ks(node1, node2)
            for k in ks:
                if solution_graph.edges[edge][k]['bad_edge']:
                    continue
                same_block = solution_graph.edges[edge][k]['same_block']
                if same_block:
                    continue
                
                assert uf.merge(node1, node2, k), f"Invalid merge: {node1}, {node2}, {k}"
                count_uf_merged += 1

        print(f"Merged {count_uf_merged} nodes in the union find structure. Nr edges: {len(solution_graph.edges)}")

        # contract to roots
        count_contracted_edges = 0
        contracted_same_block_edges = {}
        contracted_other_block_edges = {}
        contracted_bad_edges = {}
        for edge in original_graph.edges:
            node1, node2 = edge
            if (not uf.contains(node1)) or (not uf.contains(node2)):
                continue

            node1_root, k1 = uf.find(node1)
            node2_root, k2 = uf.find(node2)

            if node1_root == node2_root:
                continue

            same_cores_bool = same_cores(node1_root, node2_root)
            ks = original_graph.get_edge_ks(node1, node2)

            for k in ks:
                same_block = original_graph.edges[edge][k]['same_block']
                bad_edge = original_graph.edges[edge][k]['bad_edge']
                
                certainty = original_graph.edges[edge][k]['certainty']

                k_ = - k1 + k + k2

                if node1_root < node2_root:
                    n1r, n2r = node2_root, node1_root
                    k_ = -k_
                else:
                    n1r, n2r = node1_root, node2_root

                if bad_edge:
                    if not (n1r, n2r) in contracted_bad_edges:
                        contracted_bad_edges[(n1r, n2r)] = {}
                    if not k_ in contracted_bad_edges[(n1r, n2r)]:
                        contracted_bad_edges[(n1r, n2r)][k_] = 0.0
                    contracted_bad_edges[(n1r, n2r)][k_] += certainty
                else:
                    if same_block:
                        if not (n1r, n2r) in contracted_same_block_edges:
                            contracted_same_block_edges[(n1r, n2r)] = {}
                        if not k_ in contracted_same_block_edges[(n1r, n2r)]:
                            contracted_same_block_edges[(n1r, n2r)][k_] = 0.0
                        contracted_same_block_edges[(n1r, n2r)][k_] += certainty
                    else:
                        if same_cores_bool:
                            continue # Dont add other block edges in the same block (they were unselected by the evolutionary algorithm)
                        if not (n1r, n2r) in contracted_other_block_edges:
                            contracted_other_block_edges[(n1r, n2r)] = {}
                        if not k_ in contracted_other_block_edges[(n1r, n2r)]:
                            contracted_other_block_edges[(n1r, n2r)][k_] = 0.0
                        contracted_other_block_edges[(n1r, n2r)][k_] += certainty

                graph.add_node(n1r, original_graph.nodes[n1r]['centroid'], winding_angle=original_graph.nodes[n1r]['winding_angle'])
                graph.add_node(n2r, original_graph.nodes[n2r]['centroid'], winding_angle=original_graph.nodes[n2r]['winding_angle'])

        edges = set()
        for edge in contracted_same_block_edges:
            edges.add(edge)
        for edge in contracted_other_block_edges:
            edges.add(edge)
        for edge in contracted_bad_edges:
            edges.add(edge)

        count_bad_contactions = 0
        count_same_block_contactions = 0
        count_other_block_contactions = 0
        for edge in edges:
                ks = set()
                for k in contracted_same_block_edges.get(edge, {}):
                    ks.add(k)
                for k in contracted_other_block_edges.get(edge, {}):
                    ks.add(k)
                for k in contracted_bad_edges.get(edge, {}):
                    ks.add(k)
                
                for k in ks:
                    certainty_bad = 0.0
                    certainty_same_block = 0.0
                    certainty_other_block = 0.0
                    if edge in contracted_bad_edges:
                        certainty_bad = contracted_bad_edges[edge].get(k, 0.0)
                    if edge in contracted_same_block_edges:
                        certainty_same_block = contracted_same_block_edges[edge].get(k, 0.0)
                    if edge in contracted_other_block_edges:
                        certainty_other_block = contracted_other_block_edges[edge].get(k, 0.0)

                    assert certainty_bad + certainty_same_block + certainty_other_block > 0.0, f"Invalid certainty: {certainty_bad}, {certainty_same_block}, {certainty_other_block} for edge: {edge}, {k}"

                    # pick the mode of the edge with the highest certainty
                    if certainty_same_block > certainty_other_block and certainty_same_block > certainty_bad:
                        certainty = certainty_same_block
                        same_block = True
                        bad_edge = False
                        count_same_block_contactions += 1
                    elif certainty_other_block > certainty_same_block and certainty_other_block > certainty_bad:
                        certainty = certainty_other_block
                        same_block = False
                        bad_edge = False
                        count_other_block_contactions += 1
                    else:
                        certainty = certainty_bad
                        same_block = True
                        bad_edge = True
                        count_bad_contactions += 1

                    node1, node2 = edge
                    graph.add_increment_edge(node1, node2, certainty, k, same_block, bad_edge)
                    count_contracted_edges += 1
        print(f"Contracted {count_contracted_edges} edges to the graph. Same block: {count_same_block_contactions}, Other block: {count_other_block_contactions}, Bad edges: {count_bad_contactions}")

        return graph
                

    def same_block_graph_from_edge_selection(self, edges_indices, input_graph, edges_mask, index_nodes_dict, graph_block, graph=None):
        """
        Creates a graph from the DP table.
        """
        nodes = list(input_graph.nodes.keys())
        # print(f"nodes length: {len(nodes)}")
        if graph is None:
            graph = ScrollGraph(input_graph.overlapp_threshold, input_graph.umbilicus_path)
        # start block and patch id
        graph.start_block = input_graph.start_block
        graph.patch_id = input_graph.patch_id
        # graph add nodes
        nr_winding_angles = 0
        for node in nodes:
            if input_graph.nodes[node]['winding_angle'] is not None:
                nr_winding_angles += 1
            graph.add_node(node, input_graph.nodes[node]['centroid'], winding_angle=input_graph.nodes[node]['winding_angle'])
        added_edges_count = 0
        # print(f"Number of winding angles: {nr_winding_angles} of {len(nodes)} nodes.")

        for i in tqdm(range(len(edges_mask)), desc="Adding same edges to graph"):
            if edges_mask[i]:
                edge = edges_indices[i]
                node0_index, node1_index = edge[:2]
                node1 = index_nodes_dict[node0_index]
                node2 = index_nodes_dict[node1_index]
                _, _, _, _, (min_x, max_x, min_y, max_y, min_z, max_z, padding) = graph_block
                if (input_graph.nodes[node1]['centroid'][2] <= min_x + padding) or (input_graph.nodes[node2]['centroid'][2] <= min_x + padding):
                    continue
                if (input_graph.nodes[node1]['centroid'][2] >= max_x - padding) or (input_graph.nodes[node2]['centroid'][2] >= max_x - padding):
                    continue
                if (input_graph.nodes[node1]['centroid'][0] <= min_y + padding) or (input_graph.nodes[node2]['centroid'][0] <= min_y + padding):
                    continue
                if (input_graph.nodes[node1]['centroid'][0] >= max_y - padding) or (input_graph.nodes[node2]['centroid'][0] >= max_y - padding):
                    continue
                if (input_graph.nodes[node1]['centroid'][1] <= min_z + padding) or (input_graph.nodes[node2]['centroid'][1] <= min_z + padding):
                    continue
                if (input_graph.nodes[node1]['centroid'][1] >= max_z - padding) or (input_graph.nodes[node2]['centroid'][1] >= max_z - padding):
                    continue
                
                # ks = input_graph.get_edge_ks(node1, node2)
                k = edge[2]
                # assert k == k_, f"Invalid k: {k} != {k_}"
                certainty = input_graph.edges[(node1, node2)][k]['certainty']
                same_block = input_graph.edges[(node1, node2)][k]['same_block']
                bad_edge = input_graph.edges[(node1, node2)][k]['bad_edge']
                assert bad_edge == False, f"Invalid bad edge: {bad_edge} for edge: {edge}"
                if not same_block: # only add same block edges
                    continue
                if bad_edge: # do not add bad edges
                    continue

                assert certainty > 0.0, f"Invalid certainty: {certainty} for edge: {edge}"
                graph.add_edge(node1, node2, certainty, k, True)
                added_edges_count += 1
            
        # print(f"Added {added_edges_count} edges to the graph.")
        return graph
    
    def transition_graph_from_edge_selection(self, edges_indices, input_graph, edges_mask, index_nodes_dict, graph_block, graph=None):
        """
        Creates a graph from the DP table.
        """
        nodes = list(input_graph.nodes.keys())
        # print(f"nodes length: {len(nodes)}")
        if graph is None:
            graph = ScrollGraph(input_graph.overlapp_threshold, input_graph.umbilicus_path)
        # start block and patch id
        graph.start_block = input_graph.start_block
        graph.patch_id = input_graph.patch_id
        # graph add nodes
        nr_winding_angles = 0
        for node in nodes:
            if input_graph.nodes[node]['winding_angle'] is not None:
                nr_winding_angles += 1
            graph.add_node(node, input_graph.nodes[node]['centroid'], winding_angle=input_graph.nodes[node]['winding_angle'])
        added_edges_count = 0
        # print(f"Number of winding angles: {nr_winding_angles} of {len(nodes)} nodes.")

        for i in tqdm(range(len(edges_mask)), desc="Adding transitioning edges to graph"):
            if edges_mask[i]:
                edge = edges_indices[i]
                node0_index, node1_index = edge[:2]
                node1 = index_nodes_dict[node0_index]
                node2 = index_nodes_dict[node1_index]
                _, _, _, _, (min_x, max_x, min_y, max_y, min_z, max_z, padding) = graph_block
                if (input_graph.nodes[node1]['centroid'][2] <= min_x + padding) or (input_graph.nodes[node2]['centroid'][2] <= min_x + padding):
                    continue
                if (input_graph.nodes[node1]['centroid'][2] > max_x - padding) or (input_graph.nodes[node2]['centroid'][2] > max_x - padding):
                    continue
                if (input_graph.nodes[node1]['centroid'][0] <= min_y + padding) or (input_graph.nodes[node2]['centroid'][0] <= min_y + padding):
                    continue
                if (input_graph.nodes[node1]['centroid'][0] > max_y - padding) or (input_graph.nodes[node2]['centroid'][0] > max_y - padding):
                    continue
                if (input_graph.nodes[node1]['centroid'][1] <= min_z + padding) or (input_graph.nodes[node2]['centroid'][1] <= min_z + padding):
                    continue
                if (input_graph.nodes[node1]['centroid'][1] > max_z - padding) or (input_graph.nodes[node2]['centroid'][1] > max_z - padding):
                    continue
                
                # ks = input_graph.get_edge_ks(node1, node2)
                k = edge[2]
                # assert k == k_, f"Invalid k in graph construction: {k} != {k_}"
                certainty = input_graph.edges[(node1, node2)][k]['certainty']
                same_block = input_graph.edges[(node1, node2)][k]['same_block']
                bad_edge = input_graph.edges[(node1, node2)][k]['bad_edge']
                assert bad_edge == False, f"Invalid bad edge: {bad_edge} for edge: {edge}"
                if same_block: # only add transitioning edges
                    continue
                if bad_edge: # do not add bad edges
                    continue

                # if k != 0:
                #     print(f"Selected edge: {node1}, {node2}, {k}, {certainty}")

                assert certainty > 0.0, f"Invalid certainty: {certainty} for edge: {edge}"
                graph.add_edge(node1, node2, certainty, k, False)
                added_edges_count += 1
            
        # print(f"Added {added_edges_count} edges to the graph.")
        return graph

    def graph_from_edge_selection(self, edges_indices, input_graph, edges_mask, index_nodes_dict, graph=None, min_z=None, max_z=None):
        """
        Creates a graph from the DP table.
        """
        nodes = list(input_graph.nodes.keys())
        print(f"nodes length: {len(nodes)}")
        if graph is None:
            graph = ScrollGraph(input_graph.overlapp_threshold, input_graph.umbilicus_path)
        # start block and patch id
        graph.start_block = input_graph.start_block
        graph.patch_id = input_graph.patch_id
        # graph add nodes
        nr_winding_angles = 0
        for node in nodes:
            if input_graph.nodes[node]['winding_angle'] is not None:
                nr_winding_angles += 1
            graph.add_node(node, input_graph.nodes[node]['centroid'], winding_angle=input_graph.nodes[node]['winding_angle'])
        added_edges_count = 0
        print(f"Number of winding angles: {nr_winding_angles} of {len(nodes)} nodes.")

        for i in tqdm(range(len(edges_mask))):
            if edges_mask[i]:
                edge = edges_indices[i]
                node0_index, node1_index = edge[:2]
                k = edge[2]
                node1 = index_nodes_dict[node0_index]
                node2 = index_nodes_dict[node1_index]
                # ks = input_graph.get_edge_ks(node1, node2)
                certainty = input_graph.edges[(node1, node2)][k]['certainty']

                centroid1 = input_graph.nodes[node1]['centroid']
                centroid2 = input_graph.nodes[node2]['centroid']

                if (min_z is not None) and ((centroid1[1] <= min_z) or (centroid2[1] <= min_z)):
                    continue
                if (max_z is not None) and ((centroid1[1] > max_z) or (centroid2[1] > max_z)):
                    continue

                assert certainty > 0.0, f"Invalid certainty: {certainty} for edge: {edge}"
                graph.add_edge(node1, node2, certainty, k, False)
                added_edges_count += 1
            
        print(f"Added {added_edges_count} edges to the graph.")
        graph.compute_node_edges()
        print(f"Filtered graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
        return graph
    
    def update_ks(self, graph, start_node=None, edges_by_indices=None, valid_mask=None, update_winding_angles=False):
        self.bfs_ks_indices(edges_by_indices, valid_mask_int=valid_mask)
        # Compute nodes, ks
        nodes, ks = self.bfs_ks(graph, start_node=start_node)
        # Update the ks for the extracted nodes
        self.update_winding_angles(graph, nodes, ks, update_winding_angles=update_winding_angles)

    def bfs_ks(self, graph, start_node=None):
        unused_nodes = set([node for node in graph.nodes])
        ks_list = []
        nodes_list = []
        len_visited = 0
        nr_no_start_nodes = 0
        while len(unused_nodes) > 0:
            # Use BFS to traverse the graph and compute the ks
            nr_no_start_nodes += 1
            start_node = list(unused_nodes)[0]
            visited = {start_node: True}
            queue = [start_node]
            ks = {start_node: 0}
            while queue:
                node = queue.pop(0)
                unused_nodes.remove(node)
                node_k = ks[node]
                for edge in graph.nodes[node]['edges']:
                    if edge[0] == edge[1]:
                        print(f"Self edge: {edge}, strange ... ?")
                        continue
                    if edge[0] == node:
                        other_node = edge[1]
                    else:
                        other_node = edge[0]
                    if other_node in visited:
                        # Assert for correct k
                        ks_edge = graph.get_edge_ks(node, other_node)
                        assert (ks[other_node] - node_k) in ks_edge, f"Invalid k: {ks[other_node]} != {node_k} + {ks_edge}"
                        continue
                    visited[other_node] = True
                    ks_edge = graph.get_edge_ks(node, other_node)
                    assert len(ks_edge) == 1, f"Invalid ks_edge: {ks_edge}"
                    k = ks_edge[0]
                    ks[other_node] = node_k + k
                    queue.append(other_node)

            nodes = [node for node in visited]
            ks = np.array([ks[node] for node in nodes]) # to numpy
            ks = ks - np.min(ks) # 0 to max
            ks_list.append(ks)
            nodes_list.append(nodes)
            len_visited += len(visited)

        # assign unique k ranges to each entry
        start_k = 0
        ks_list_assigned = []
        nodes_assigned = []
        for i in range(len(ks_list)):
            nodes = nodes_list[i]
            ks = ks_list[i]
            min_k = np.min(ks)
            ks = ks - min_k + start_k
            start_k = np.max(ks) + 10
            ks_list_assigned.append(ks)
            nodes_assigned.extend(nodes)

        # concat to numpy
        ks_list = np.concatenate(ks_list_assigned)

        print(f"{nr_no_start_nodes} times no start node provided, using first node in the graph.")
        print(f"Visited {len_visited} nodes. During Breadth First Search.")

        return nodes_assigned, ks_list
    
    def bfs_ks_indices_transition(self, input_graph, graph_block, index_nodes_dict, edges_indices, valid_mask_int):
        valid_mask = valid_mask_int > 0

        edges = {}
        valid_edges = edges_indices[valid_mask]
        valid_nodes = set()
        for edge in valid_edges:
            node1 = index_nodes_dict[edge[0]]
            node2 = index_nodes_dict[edge[1]]
            k = edge[2]
            same_block = input_graph.edges[(node1, node2)][k]['same_block']
            if same_block:
                continue

            if edge[0] not in edges:
                edges[edge[0]] = set()
            edges[edge[0]].add((edge[0], edge[1], edge[2], edge[3]))
            if edge[1] not in edges:
                edges[edge[1]] = set()
            edges[edge[1]].add((edge[0], edge[1], edge[2], edge[3]))
            valid_nodes.add(edge[0])
            valid_nodes.add(edge[1])
        
        # Use BFS to traverse the graph and compute the ks

        ks_list = []
        start_nodes = []

        # while elements in valid_nodes
        while len(valid_nodes) > 0:
            val_nodes_list = list(valid_nodes)
            start_node = None
            for i in range(len(val_nodes_list)):
                start_node = list(valid_nodes)[i]
                # get rid of padded nodes as start nodes
                _, _, _, _, (min_x, max_x, min_y, max_y, min_z, max_z, padding) = graph_block
                if (input_graph.nodes[node1]['centroid'][2] < min_x + padding) or (input_graph.nodes[node2]['centroid'][2] < min_x + padding):
                    continue
                if (input_graph.nodes[node1]['centroid'][2] > max_x - padding) or (input_graph.nodes[node2]['centroid'][2] > max_x - padding):
                    continue
                if (input_graph.nodes[node1]['centroid'][0] < min_y + padding) or (input_graph.nodes[node2]['centroid'][0] < min_y + padding):
                    continue
                if (input_graph.nodes[node1]['centroid'][0] > max_y - padding) or (input_graph.nodes[node2]['centroid'][0] > max_y - padding):
                    continue
                if (input_graph.nodes[node1]['centroid'][1] < min_z + padding) or (input_graph.nodes[node2]['centroid'][1] < min_z + padding):
                    continue
                if (input_graph.nodes[node1]['centroid'][1] > max_z - padding) or (input_graph.nodes[node2]['centroid'][1] > max_z - padding):
                    continue
                break
            if start_node is None:
                print("No start node found, breaking.")
                break
            visited = {start_node: True}
            queue = [start_node]
            ks = {start_node: 0}
            start_nodes.append(start_node)
            while queue:
                node = queue.pop(0)
                # remove node from valid nodes
                valid_nodes.remove(node)
                node_k = ks[node]
                for edge in edges[node]:
                    node1, node2, k, certainty = edge
                    same_block = input_graph.edges[(node1, node2)][k]['same_block']
                    if same_block: # only use transitioning edges
                        continue
                    if node1 == node:
                        other_node = node2
                    else:
                        other_node = node1
                        k = -k # flip k if edge direction is flipped
                    if other_node in visited:
                        # Assert for correct k
                        condition= ks[other_node] == node_k + k
                        assert condition, f"Invalid k: {ks[other_node]} != {node_k + k}, edges_indices: {edges_indices}, valid_mask: {valid_mask}, valid_mask_int: {valid_mask_int}"
                        if not condition:
                            return False, None
                        continue
                    visited[other_node] = True
                    ks[other_node] = node_k + k
                    queue.append(other_node)
            ks_list.append(ks)
        return True, ks_list, start_nodes
    
    def bfs_ks_indices(self, edges_indices, valid_mask_int):
        valid_mask = valid_mask_int > 0

        edges = {}
        valid_edges = edges_indices[valid_mask]
        valid_nodes = set()
        for edge in valid_edges:
            if edge[0] not in edges:
                edges[edge[0]] = set()
            edges[edge[0]].add((edge[0], edge[1], edge[2], edge[3]))
            if edge[1] not in edges:
                edges[edge[1]] = set()
            edges[edge[1]].add((edge[0], edge[1], edge[2], edge[3]))
            valid_nodes.add(edge[0])
            valid_nodes.add(edge[1])
        
        # Use BFS to traverse the graph and compute the ks

        ks_list = []

        # while elements in valid_nodes
        while len(valid_nodes) > 0:
            start_node = list(valid_nodes)[0]
            visited = {start_node: True}
            queue = [start_node]
            ks = {start_node: 0}
            while queue:
                node = queue.pop(0)
                # remove node from valid nodes
                valid_nodes.remove(node)
                node_k = ks[node]
                for edge in edges[node]:
                    node1, node2, k, certainty = edge
                    if node1 == node:
                        other_node = node2
                    else:
                        other_node = node1
                        k = -k # flip k if edge direction is flipped
                    if other_node in visited:
                        # Assert for correct k
                        condition= ks[other_node] == node_k + k
                        assert condition, f"Invalid k: {ks[other_node]} != {node_k + k}, edges_indices: {edges_indices}, valid_mask: {valid_mask}, valid_mask_int: {valid_mask_int}"
                        if not condition:
                            return False, None
                        continue
                    visited[other_node] = True
                    ks[other_node] = node_k + k
                    queue.append(other_node)
            ks_list.append(ks)
        return True, ks_list
    
    def update_winding_angles(self, graph, nodes, ks, update_winding_angles=False):
        # Update winding angles
        for i, node in enumerate(nodes):
            graph.nodes[node]['assigned_k'] = ks[i]
            if update_winding_angles:
                graph.nodes[node]['winding_angle'] = - ks[i]*360 + graph.nodes[node]['winding_angle']

def compute(overlapp_threshold, start_point, path, recompute=False, compute_cpp_translation=False, stop_event=None, continue_segmentation=False, iteration=None):
    umbilicus_path = os.path.dirname(path) + "/umbilicus.txt"
    start_block, patch_id = find_starting_patch([start_point], path)

    save_path = os.path.dirname(path) + f"/{start_point[0]}_{start_point[1]}_{start_point[2]}/" + path.split("/")[-1]
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Configs: {overlapp_threshold}")

    recompute_path = path.replace("blocks", "scroll_graph") + ".pkl"
    recompute = recompute or not os.path.exists(recompute_path)

    continue_segmentation_path = path.replace("blocks", "scroll_graph_progress") + ".pkl"

    update_graph = False
    if update_graph:
        # Build graph
        if recompute:
            scroll_graph = ScrollGraph(overlapp_threshold, umbilicus_path)
            start_block, patch_id = scroll_graph.build_graph(path, num_processes=30, start_point=start_point, prune_unconnected=False)
            print("Saving built graph...")
            scroll_graph.save_graph(recompute_path)
        elif continue_segmentation:
            print("Loading graph...")
            scroll_graph = load_graph(continue_segmentation_path)
            start_block, patch_id = find_starting_patch([start_point], path)
        else:
            print("Loading graph...")
            scroll_graph = load_graph(recompute_path)
            start_block, patch_id = find_starting_patch([start_point], path)

        # redo graph
        # scroll_graph.update_graph_version()
        # scroll_graph.compute_node_edges()
        # scroll_graph.save_graph(recompute_path)

        # min_x, max_x, min_y, max_y = None, None, None, None
        # min_z, max_z, umbilicus_max_distance = None, None, None
        # min_z, max_z, umbilicus_max_distance = 600, 1000, 200
        # min_x, max_x, min_y, max_y, min_z, max_z, umbilicus_max_distance = 700, 800, 650, 850, 800, 900, None # 2 blocks without middle
        # min_x, max_x, min_y, max_y, min_z, max_z, umbilicus_max_distance = 650, 850, 550, 950, 750, 950, None # 2 blocks without middle # start block length 200
        # min_x, max_x, min_y, max_y, min_z, max_z, umbilicus_max_distance = 475, 875, 525, 925, 600, 1000, None # 4x4x4 blocks with middle
        # min_x, max_x, min_y, max_y, min_z, max_z, umbilicus_max_distance = 450, 750, 625, 925, 800, 900, None # 3x3x1 blocks with middle
        # min_x, max_x, min_y, max_y, min_z, max_z, umbilicus_max_distance = 350, 850, 525, 1025, 800, 900, None # 5x5x1 blocks with middle
        # min_x, max_x, min_y, max_y, min_z, max_z, umbilicus_max_distance = None, None, None, None, 800, 900, None # all x all x 1 blocks with middle
        # min_x, max_x, min_y, max_y, min_z, max_z, umbilicus_max_distance = 575, 775, 625, 825, 700, 900, None # 2x2x2 blocks with middle
        # min_x, max_x, min_y, max_y, min_z, max_z, umbilicus_max_distance = 575, 775, 725, 825, 800, 900, None # 2 blocks with middle
        # min_x, max_x, min_y, max_y, min_z, max_z, umbilicus_max_distance = 600, 1000, 550, 950, 600, 1000, None
        min_x, max_x, min_y, max_y, min_z, max_z, umbilicus_max_distance = None, None, None, None, 800, 1000, None # scroll 1 whole slab
        # min_z, max_z, umbilicus_max_distance = None, None, 160
        subgraph = scroll_graph.extract_subgraph(min_z=min_z, max_z=max_z, umbilicus_max_distance=umbilicus_max_distance, add_same_block_edges=True, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
        subgraph.save_graph(save_path.replace("blocks", "subgraph") + ".pkl")
    subgraph = load_graph(save_path.replace("blocks", "subgraph") + ".pkl")

    # TODO: remove after building the graph correctly with winding direction -1.0 (scroll 3)
    # subgraph = scroll_graph
    # subgraph.flip_winding_direction() # only because working on scroll 3 and i built the graph with winding direction 1.0
    subgraph.compute_bad_edges(iteration=0)
    k_factors = subgraph.adjust_edge_certainties()

    # subgraph.create_dxf_with_colored_polyline(save_path.replace("blocks", "subgraph") + ".dxf", min_z=min_z, max_z=max_z)
    graph_filter = EvolutionaryGraphEdgesSelection(subgraph, path, save_path)
    # evolved_graph = graph_filter.solve()
    evolved_graph = graph_filter.solve_divide_and_conquer(start_block_side_length=100, iteration=iteration, k_factors_original=k_factors)

    # save graph
    evolved_graph.save_graph(save_path.replace("blocks", "evolved_graph") + ".pkl")
    # Visualize the graph
    # evolved_graph.create_dxf_with_colored_polyline(save_path.replace("blocks", "evolved_graph") + ".dxf", min_z=min_z, max_z=max_z)
    # subgraph.compare_polylines_graph(evolved_graph, save_path.replace("blocks", "evolved_graph_comparison") + ".dxf", min_z=min_z, max_z=max_z)

    debug_display = False
    if debug_display:
        scroll_graph.create_dxf_with_colored_polyline(save_path.replace("blocks", "graph") + ".dxf", min_z=975, max_z=1000)

        # Solved Graph
        solved_graph_path = save_path.replace("blocks", "scroll_graph_progress") + ".pkl"
        solved_graph = load_graph(solved_graph_path)
        solved_graph.create_dxf_with_colored_polyline(save_path.replace("blocks", "solved_graph") + ".dxf", min_z=975, max_z=1000)
        scroll_graph.compare_polylines_graph(solved_graph, save_path.replace("blocks", "solved_graph_comparison") + ".dxf", min_z=975, max_z=1000)
 
def random_walks():
    path = "/media/julian/SSD4TB/scroll3_surface_points/point_cloud_colorized_verso_subvolume_blocks"
    # sample_ratio_score = 0.03 # 0.1
    start_point=[1650, 3300, 5000] # seg 1
    start_point=[1450, 3500, 5000] # seg 2
    start_point=[1350, 3600, 5000] # seg 3
    start_point=[3164, 3476, 3472] # test scroll 3
    continue_segmentation = 0
    overlapp_threshold = {"sample_ratio_score": 0.03, "display": False, "print_scores": True, "picked_scores_similarity": 0.7, "final_score_max": 1.5, "final_score_min": 0.0005, "score_threshold": 0.005, "fit_sheet": False, "cost_threshold": 17, "cost_percentile": 75, "cost_percentile_threshold": 14, 
                          "cost_sheet_distance_threshold": 4.0, "rounddown_best_score": 0.005,
                          "cost_threshold_prediction": 2.5, "min_prediction_threshold": 0.15, "nr_points_min": 200.0, "nr_points_max": 4000.0, "min_patch_points": 300.0, 
                          "winding_angle_range": None, "multiple_instances_per_batch_factor": 1.0,
                          "epsilon": 1e-5, "angle_tolerance": 85, "max_threads": 30,
                          "min_points_winding_switch": 1000, "min_winding_switch_sheet_distance": 3, "max_winding_switch_sheet_distance": 10, "winding_switch_sheet_score_factor": 1.5, "winding_direction": 1.0, "enable_winding_switch": False, "enable_winding_switch_postprocessing": False,
                          "surrounding_patches_size": 3, "max_sheet_clip_distance": 60, "sheet_z_range": (-5000, 400000), "sheet_k_range": (-1, 2), "volume_min_certainty_total_percentage": 0.0, "max_umbilicus_difference": 30,
                          "walk_aggregation_threshold": 100, "walk_aggregation_max_current": -1,
                          "bad_edge_threshold": 3
                          }
    # Scroll 1: "winding_direction": 1.0
    # Scroll 3: "winding_direction": -1.0

    max_nr_walks = 10000
    max_steps = 101
    min_steps = 16
    max_tries = 6
    min_end_steps = 4
    max_unchanged_walks = 30 * max_nr_walks
    recompute = 0
    compute_cpp_translation = False
    
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Cut out ThaumatoAnakalyptor Papyrus Sheet')
    parser.add_argument('--path', type=str, help='Papyrus instance patch path (containing .tar)', default=path)
    parser.add_argument('--recompute', type=int,help='Recompute graph', default=recompute)
    parser.add_argument('--print_scores', type=bool,help='Print scores of patches for sheet', default=overlapp_threshold["print_scores"])
    parser.add_argument('--sample_ratio_score', type=float,help='Sample ratio to apply to the pointcloud patches', default=overlapp_threshold["sample_ratio_score"])
    parser.add_argument('--score_threshold', type=float,help='Score threshold to add patches to sheet', default=overlapp_threshold["score_threshold"])
    parser.add_argument('--min_prediction_threshold', type=float,help='Min prediction threshold to add patches to sheet', default=overlapp_threshold["min_prediction_threshold"])
    parser.add_argument('--final_score_min', type=float,help='Final score min threshold to add patches to sheet', default=overlapp_threshold["final_score_min"])
    parser.add_argument('--rounddown_best_score', type=float,help='Pick best score threshold to round down to zero from. Combats segmentation speed slowdown towards the end of segmentation.', default=overlapp_threshold["rounddown_best_score"])
    parser.add_argument('--winding_direction', type=int,help='Winding direction of sheet in scroll scan', default=overlapp_threshold["winding_direction"])
    parser.add_argument('--sheet_z_range', type=int, nargs=2,help='Z range of segmentation', default=[overlapp_threshold["sheet_z_range"][0], overlapp_threshold["sheet_z_range"][1]])
    parser.add_argument('--sheet_k_range', type=int, nargs=2,help='Angle range (as k 1k = 360 deg, k is int) of the sheet winding for segmentation', default=[overlapp_threshold["sheet_k_range"][0], overlapp_threshold["sheet_k_range"][1]])
    parser.add_argument('--starting_point', type=int, nargs=3,help='Starting point for a new segmentation', default=start_point)
    parser.add_argument('--continue_segmentation', type=int,help='Continue previous segmentation (point_cloud_colorized_subvolume_main_sheet.ta). 1 to continue, 0 to restart.', default=int(continue_segmentation))
    parser.add_argument('--enable_winding_switch', type=int,help='Enable switching of winding if two sheets lay on top of each eather. 1 enable, 0 disable.', default=int(overlapp_threshold["enable_winding_switch"]))
    parser.add_argument('--enable_winding_switch_postprocessing', type=int,help='Enable postprocessing switching of winding if two sheets lay on top of each eather. 1 enable, 0 disable.', default=int(overlapp_threshold["enable_winding_switch_postprocessing"]))
    parser.add_argument('--max_threads', type=int,help='Maximum number of thread to use during computation. Has a slight influence on the quality of segmentations for small segments.', default=int(overlapp_threshold["max_threads"]))
    parser.add_argument('--surrounding_patches_size', type=int,help=f'Number of surrounding half-overlapping patches in each dimension direction to consider when calculating patch pair similarity scores. Default is {overlapp_threshold["surrounding_patches_size"]}.', default=int(overlapp_threshold["surrounding_patches_size"]))
    parser.add_argument('--max_nr_walks', type=int,help=f'Maximum number of random walks to perform. Default is {max_nr_walks}.', default=int(max_nr_walks))
    parser.add_argument('--min_steps', type=int,help=f'Minimum number of steps for a random walk to be considered valid. Default is {min_steps}.', default=int(min_steps))
    parser.add_argument('--min_end_steps', type=int,help=f'Minimum number of steps for a random walk to be considered valid at the end of a random walk. Default is {min_end_steps}.', default=int(min_end_steps))
    parser.add_argument('--max_unchanged_walks', type=int,help=f'Maximum number of random walks to perform without updating the graph before finishing the segmentation. Default is {max_unchanged_walks}.', default=int(max_unchanged_walks))
    parser.add_argument('--min_certainty_p', type=float,help=f'Minimum percentage of certainty of a volume to be considered for random walks. Default is {overlapp_threshold["volume_min_certainty_total_percentage"]}.', default=overlapp_threshold["volume_min_certainty_total_percentage"])
    parser.add_argument('--max_umbilicus_dif', type=float,help=f'Maximum difference in umbilicus distance between two patches to be considered valid. Default is {overlapp_threshold["max_umbilicus_difference"]}.', default=overlapp_threshold["max_umbilicus_difference"])
    parser.add_argument('--walk_aggregation_threshold', type=int,help=f'Number of random walks to aggregate before updating the graph. Default is {overlapp_threshold["walk_aggregation_threshold"]}.', default=int(overlapp_threshold["walk_aggregation_threshold"]))
    parser.add_argument('--walk_aggregation_max_current', type=int,help=f'Maximum number of random walks to aggregate before updating the graph. Default is {overlapp_threshold["walk_aggregation_max_current"]}.', default=int(overlapp_threshold["walk_aggregation_max_current"]))
    parser.add_argument('--iteration', type=int,help='Iteration number', default=None)

    # Take arguments back over
    args = parser.parse_args()
    print(f"Args: {args}")

    path = args.path
    recompute = bool(int(args.recompute))
    overlapp_threshold["print_scores"] = args.print_scores
    overlapp_threshold["sample_ratio_score"] = args.sample_ratio_score
    overlapp_threshold["score_threshold"] = args.score_threshold
    overlapp_threshold["min_prediction_threshold"] = args.min_prediction_threshold
    overlapp_threshold["final_score_min"] = args.final_score_min
    overlapp_threshold["rounddown_best_score"] = args.rounddown_best_score
    overlapp_threshold["winding_direction"] = args.winding_direction
    overlapp_threshold["sheet_z_range"] = [z_range_ /(200.0 / 50.0) for z_range_ in args.sheet_z_range]
    overlapp_threshold["sheet_k_range"] = args.sheet_k_range
    start_point = args.starting_point
    continue_segmentation = bool(args.continue_segmentation)
    overlapp_threshold["enable_winding_switch"] = bool(args.enable_winding_switch)
    overlapp_threshold["enable_winding_switch_postprocessing"] = bool(args.enable_winding_switch_postprocessing)
    overlapp_threshold["max_threads"] = args.max_threads
    overlapp_threshold["surrounding_patches_size"] = args.surrounding_patches_size
    min_steps = args.min_steps
    max_nr_walks = args.max_nr_walks
    min_end_steps = args.min_end_steps
    overlapp_threshold["volume_min_certainty_total_percentage"] = args.min_certainty_p
    overlapp_threshold["max_umbilicus_difference"] = args.max_umbilicus_dif
    overlapp_threshold["walk_aggregation_threshold"] = args.walk_aggregation_threshold
    overlapp_threshold["walk_aggregation_max_current"] = args.walk_aggregation_max_current
    overlapp_threshold["max_nr_walks"] = max_nr_walks
    overlapp_threshold["max_unchanged_walks"] = max_unchanged_walks
    overlapp_threshold["continue_walks"] = continue_segmentation
    overlapp_threshold["max_steps"] = max_steps
    overlapp_threshold["max_tries"] = max_tries
    overlapp_threshold["min_steps"] = min_steps
    overlapp_threshold["min_end_steps"] = min_end_steps
    iteration = args.iteration

    # Compute
    compute(overlapp_threshold=overlapp_threshold, start_point=start_point, path=path, recompute=recompute, compute_cpp_translation=compute_cpp_translation, continue_segmentation=continue_segmentation, iteration=iteration)

if __name__ == '__main__':
    random_walks()