### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

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
import matplotlib.pyplot as plt

from .instances_to_sheets import select_points, get_vector_mean, alpha_angles, adjust_angles_zero, adjust_angles_offset, add_overlapp_entries_to_patches_list, assign_points_to_tiles, compute_overlap_for_pair, overlapp_score, fit_sheet, winding_switch_sheet_score_raw_precomputed_surface, find_starting_patch, save_main_sheet, update_main_sheet
from .sheet_to_mesh import load_xyz_from_file, scale_points, umbilicus_xz_at_y
import sys
### C++ speed up. not yet fully implemented
# sys.path.append('sheet_generation/build')
# import sheet_generation

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

    def add_node(self, node, centroid):
        node = tuple(node)
        self.nodes[node] = {'centroid': centroid}

    def compute_node_edges(self):
        """
        Compute the edges for each node and store them in the node dictionary.
        """
        print("Computing node edges...")
        for node in self.nodes:
            self.nodes[node]['edges'] = []
        for edge in tqdm(self.edges):
            self.nodes[edge[0]]['edges'].append(edge)
            self.nodes[edge[1]]['edges'].append(edge)

    def remove_nodes_edges(self, nodes):
        """
        Remove nodes and their edges from the graph.
        """
        for node in nodes:
            # Delete Node
            del self.nodes[node]
            # Delete Edges
            for edge in list(self.edges.keys()):
                if node in edge:
                    del self.edges[edge]
            # Delete Node Edges
            for node_ in self.nodes:
                node_edges_ = list(self.nodes[node_]['edges'])
                for edge in node_edges_:
                    if node in edge:
                        self.nodes[node_]['edges'].remove(edge)
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

    def add_edge(self, node1, node2, certainty, sheet_offset_k=0.0, same_block=False):
        assert certainty > 0.0, "Certainty must be greater than 0."
        certainty = np.clip(certainty, 0.0, None)

        node1 = tuple(node1)
        node2 = tuple(node2)
        # Ensure node1 < node2 for bidirectional nodes
        if node2 < node1:
            node1, node2 = node2, node1
            sheet_offset_k = sheet_offset_k * (-1.0)
        self.edges[(node1, node2)] = {'certainty': certainty, 'sheet_offset_k': sheet_offset_k, 'same_block': same_block}
        
    def get_edge_data(self, node1, node2):
        # Maintain bidirectional invariant
        if node2 < node1:
            node1, node2 = node2, node1
        edge_data = self.edges.get((node1, node2))
        if edge_data is not None:
            return node1, node2, edge_data['certainty'], edge_data['sheet_offset_k']
        else:
            raise KeyError(f"No edge found from {node1} to {node2}")
        
    def get_edge_k(self, node1, node2):            
        k_factor = 1.0
        # Maintain bidirectional invariant
        if node2 < node1:
            node1, node2 = node2, node1
            k_factor = - 1.0
        edge_dict = self.edges.get((node1, node2))
        if edge_dict is not None:
            return edge_dict['sheet_offset_k'] * k_factor
        else:
            raise KeyError(f"No edge found from {node1} to {node2}")

    def save_graph(self, path):
        # Save graph class object to file
        with open(path, 'wb') as file:
            pickle.dump(self, file)

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
        score = (score[0], score[1], score_val, score[3], score[4], score[5])

        return [score]

    # calculate switching scores of main patches
    score_switching_sheets = []
    for i in range(len(main_block_patches_list)):
        score_switching_sheets_ = []
        for j in range(len(main_block_patches_list)):
            if i == j:
                continue
            (score_, k_), anchor_angle1, anchor_angle2 = score_same_block_patches(main_block_patches_list[i], main_block_patches_list[j], overlapp_threshold, umbilicus_distance)
            if score_ > 0.0:
                score_switching_sheets_.append((main_block_patches_list[i]['ids'][0], main_block_patches_list[j]['ids'][0], score_, k_, anchor_angle1, anchor_angle2))

        # filter and only take the scores closest to the main patch (smallest scores) for each k in +1, -1
        direction1_scores = [score for score in score_switching_sheets_ if score[3] > 0.0]
        direction2_scores = [score for score in score_switching_sheets_ if score[3] < 0.0]
        
        score1 = scores_cleaned(direction1_scores)
        score2 = scores_cleaned(direction2_scores)
        score_switching_sheets += score1 + score2
    return score_switching_sheets

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
                    score_sheets_patch.append((main_block_patch['ids'][0], surrounding_block_patch['ids'][0], score_[0], None, score_[1], score_[2]))

            # Find the best score for each main block patch
            if len(score_sheets_patch) > 0:
                score_sheets_patch = max(score_sheets_patch, key=lambda x: x[2])
                score_sheets.append(score_sheets_patch)
    
    score_switching_sheets = process_same_block(main_block_patches_list, overlapp_threshold, umbilicus_distance)

    # Process and return results...
    return score_sheets, score_switching_sheets, patches_centroids

class ScrollGraph(Graph):
    def __init__(self, overlapp_threshold,umbilicus_path):
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
        # Build edges between patches from different blocks
        for score_ in score_sheets:
            id1, id2, score, _, anchor_angle1, anchor_angle2 = score_
            if score < self.overlapp_threshold["final_score_min"]:
                continue
            if abs(anchor_angle1 - anchor_angle2) > 180.0:
                if anchor_angle1 > anchor_angle2:
                    id1, id2 = id2, id1
                self.add_switch_edge(id1, id2, score, same_block=False)
            else:
                self.add_same_sheet_edge(id1, id2, score, same_block=False)

    def build_same_block_edges(self, score_switching_sheets):
        # Build edges between patches from the same block
        disregarding_count = 0
        total_count = 0
        grand_total = 0
        for score_ in score_switching_sheets:
            grand_total += 1
            id1, id2, score, k, anchor_angle1, anchor_angle2 = score_
            if score < 0.0:
                continue
            total_count += 1
            if abs(anchor_angle1 - anchor_angle2) > 180.0:
                disregarding_count += 1
            else:
                if k < 0.0:
                    id1, id2 = id2, id1
                self.add_switch_edge(id1, id2, score, same_block=True)

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

    def largest_connected_component(self):
        print("Finding largest connected component...")        
        # walk the graph from the start node
        visited = set()
        # tqdm object showing process of visited nodes untill all nodes are visited
        tqdm_object = tqdm(total=len(self.nodes))
        components = []
        starting_index = 0
        nodes = list(self.nodes.keys())
        max_component_length = 0
        nodes_length = len(nodes)
        nodes_remaining = nodes_length
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
            if nodes_remaining < max_component_length: # early stopping, already found the largest component
                break
        
        nodes_total = len(self.nodes)
        edges_total = len(self.edges)
        print(f"number of nodes: {nodes_total}, number of edges: {edges_total}")


        # largest component
        largest_component = max(components, key=len)
        other_nodes = [node for node in self.nodes if node not in largest_component]
        # Remove all other Nodes, Edges and Node Edges
        self.remove_nodes_edges(other_nodes)
        
        print(f"Pruned {nodes_total - len(self.nodes)} nodes. Of {nodes_total} nodes.")
        print(f"Pruned {edges_total - len(self.edges)} edges. Of {edges_total} edges.")

    def build_graph(self, path_instances, start_point, num_processes=4, prune_unconnected=False):
        blocks_tar_files = glob.glob(path_instances + '/*.tar')

        # debug
        blocks_tar_files = self.filter_blocks_z(blocks_tar_files, 950, 1000)

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
        for score_sheets, score_switching_sheets, volume_centroids in results:
            count_res += len(score_sheets)
            # Calculate scores, add patches edges to graph, etc.
            self.build_other_block_edges(score_sheets)
            self.build_same_block_edges(score_switching_sheets)
            patches_centroids.update(volume_centroids)
        print(f"Number of results: {count_res}")

        # Add patches as nodes to graph
        edges_keys = list(self.edges.keys())
        for edge in edges_keys:
            try:
                self.add_node(edge[0], patches_centroids[edge[0]])
                self.add_node(edge[1], patches_centroids[edge[1]])
            except: 
                del self.edges[edge] # one node might be outside computed volume

        node_id = tuple((*start_block, patch_id))
        print(f"Start node: {node_id}, nr nodes: {len(self.nodes)}, nr edges: {len(self.edges)}")
        self.compute_node_edges()
        if prune_unconnected:
            print("Prunning unconnected nodes...")
            self.largest_connected_component()

        print(f"Nr nodes: {len(self.nodes)}, nr edges: {len(self.edges)}")

        return start_block, patch_id
    
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

        for edge in tqdm(self.edges):
            same_block = self.edges[edge]['same_block']
            k = self.get_edge_k(edge[0], edge[1])
            c = color
            if same_block:
                c = 2
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
        
class RandomWalkSolver:
    def __init__(self, graph, umbilicus_path):
        self.graph = graph
        self.init_edges_count()
        self.umbilicus_path = umbilicus_path
        self.init_umbilicus(umbilicus_path)

        if not hasattr(self.graph, "overlapp_threshold_filename"):
            # Save overlap_threshold to a YAML file
            overlapp_threshold_filename = 'overlapp_threshold.yaml'
            self.graph.overlapp_threshold_filename = overlapp_threshold_filename

    def init_edges_count(self):
        self.edges_count = {}
        for edge in self.graph.edges:
            self.edges_count[edge] = {'good': 0, 'bad': 0}

    def init_umbilicus(self, umbilicus_path):
        # Load the umbilicus data
        umbilicus_data = load_xyz_from_file(umbilicus_path)
        # scale and swap axis
        umbilicus_data = scale_points(umbilicus_data, 50.0/200.0, axis_offset=0)
        # Define a wrapper function for umbilicus_xz_at_y
        self.umbilicus_func = lambda z: umbilicus_xz_at_y(umbilicus_data, z)

    def save_overlapp_threshold(self):
        with open(self.graph.overlapp_threshold_filename, 'w') as file:
            yaml.dump(self.graph.overlapp_threshold, file)

    def solve(self, path, max_nr_walks=100, max_steps=100, max_tries=6, min_steps=10, stop_event=None):
        # Load the walk
        picked_nrs = np.array([0], dtype=float)
        last_bad_walk_nodes = []
        max_bad_length = 1000
        good_nodes_count = 0
        bad_nodes_count = 0

        # Parameters for the random walk
        nr_walks_total = 0
        nr_unupdated_walks = 0
        failed_dict = {}
        time_pick = 0.0
        time_walk = 0.0
        time_postprocess = 0.0
        nr_unchanged_walks = 0

        # Making random walks as long as this loop is not broken
        while max_nr_walks > 0 and (stop_event is None or not stop_event.is_set()):
            # Timing
            time_start = time.time()
            # Pick a starting node for the random walk
            sn = self.pick_start_node(last_bad_walk_nodes) # use all nodes and special weights on nodes from recent bad walks
            time_pick += time.time() - time_start
            time_start = time.time()
            # Perform a random walk
            walk, k, good = self.random_walk(sn, 0, max_steps=max_steps, max_tries=max_tries, min_steps=min_steps)
            time_walk += time.time() - time_start
            time_start = time.time()
            
            if walk is None:
                # Keep track of failures
                nr_unchanged_walks += 1
                if not k in failed_dict:
                    failed_dict[k] = 0
                failed_dict[k] += 1
            elif good:
                # Track good edges in walk
                for i in range(len(walk) - 1):
                    edge = (walk[i], walk[i + 1])
                    if edge[1] < edge[0]:
                        edge = (edge[1], edge[0])
                    self.edges_count[edge]['good'] += 1
                    good_nodes_count += 1
            else:
                # Track bad edges in walk
                for i in range(len(walk) - 1):
                    edge = (walk[i], walk[i + 1])
                    if edge[1] < edge[0]:
                        edge = (edge[1], edge[0])
                    self.edges_count[edge]['bad'] += 1
                    bad_nodes_count += 1
                # Add the last bad walk nodes to the list
                last_bad_walk_nodes += walk[:-1]
                if len(last_bad_walk_nodes) > max_bad_length:
                    last_bad_walk_nodes = last_bad_walk_nodes[-max_bad_length:]


            # update the walks + save progress
            nr_walks_total += 1
            if nr_walks_total % (max_nr_walks * 2) == 0:
                # Postprocess the walks
                # Delete edges that are too bad
                self.remove_bad_edges()
                # periodically clear unconnected nodes/extract largest connected component
                self.graph.largest_connected_component()
                # Clean last bad walk nodes
                last_bad_walk_nodes = self.clean_last_bad_walk_nodes(last_bad_walk_nodes)

                if nr_walks_total % (max_nr_walks * 2 * 100) == 0:
                    self.graph.save_graph(path.replace("blocks", "scroll_graph_progress") + ".pkl")
                print(f"Walks: {nr_walks_total}, unupdated walks: {nr_unupdated_walks}, good: {good_nodes_count}, bad {bad_nodes_count}, failed because: \n{failed_dict}")
                print(f"Time pick: {time_pick}, time walk: {time_walk}, time postprocess: {time_postprocess}, step size: {min_steps}, walk_aggregation_threshold: {self.graph.overlapp_threshold['walk_aggregation_threshold']}, k_range: {self.graph.overlapp_threshold['sheet_k_range']}")
                time_pick, time_walk, time_postprocess = 0.0, 0.0, 0.0
            time_postprocess += time.time() - time_start

        # Finishing the random walks and returning final solution
        print(f"Walks: {nr_walks_total}, sucessful: {good_nodes_count}, failed because: \n{failed_dict}")
        # mean and std and median of picked_nrs
        mean = np.mean(picked_nrs)
        std = np.std(picked_nrs)
        median = np.median(picked_nrs)
        print(f"Mean: {mean}, std: {std}, median: {median} of picked_nrs")

    def remove_bad_edges(self):
        print("Removing bad edges...")
        # Remove edges that are too bad
        for edge in list(self.edges_count.keys()):
            if self.edges_count[edge]['bad'] - self.edges_count[edge]['good'] > self.graph.overlapp_threshold["bad_edge_threshold"]:
                self.graph.delete_edge(edge)
                del self.edges_count[edge]
        print("Bad edges removed.")

    def clean_last_bad_walk_nodes(self, last_bad_walk_nodes):
        # Remove non-existing nodes from last_bad_walk_nodes
        cleaned_bad_nodes = [node for node in last_bad_walk_nodes if node in self.graph.nodes]
        # Remove 10% of the nodes, but at least 50 nodes
        remove_count = max(50, int(len(cleaned_bad_nodes) * 0.1))
        if len(cleaned_bad_nodes) > remove_count:
            cleaned_bad_nodes = cleaned_bad_nodes[remove_count:]
        else:
            cleaned_bad_nodes = []
        # remove duplicates
        cleaned_bad_nodes = list(dict.fromkeys(cleaned_bad_nodes))
        return cleaned_bad_nodes
    
    def sheet_switching_nodes(self, node, k, volume_dict, ks):
        """
        Get the nodes that are connected to the given node with a sheet switching edge. (edge to same volume, but different sheet)
        """
        overlapp_threshold = self.graph.overlapp_threshold
        nodes_ = []
        ks_ = []
        edges = self.graph.nodes[node]['edges']
        for edge in edges:
            if edge[0] == node:
                other_node = edge[1]
            else:
                other_node = edge[0]
            if other_node[:3] == node[:3]:
                # node not in volume_dict and k not in ks for volume_dict volume id
                if (not other_node[:3] in volume_dict) or (not other_node[3] in volume_dict[other_node[:3]]):
                    k_other = k + self.graph.get_edge_k(node, other_node)
                    if (node[:3] in volume_dict) and (k_other in [ks[volume_dict[node[:3]][key]] for key in volume_dict[node[:3]]]):
                        continue
                    if k_other < overlapp_threshold["sheet_k_range"][0] or k_other > overlapp_threshold["sheet_k_range"][1]:
                        continue
                    nodes_.append(other_node)
                    ks_.append(k_other)

        return nodes_, ks_
    
    def get_next_valid_volumes(self, node, volume_min_certainty_total_percentage=0.15):
        edges = self.graph.nodes[node]['edges']
        volumes = {}
        for edge in edges:
            certainty = self.graph.edges[edge]['certainty']
            if certainty <= 0.0:
                continue
            if edge[0] == node:
                next_node = edge[1]
            else:
                next_node = edge[0]
            next_volume = next_node[:3]
            if (not self.graph.overlapp_threshold["enable_winding_switch"]) and (next_volume == node[:3]): # same_volume, winding switch disregarded for random walks if enable_winding_switch is False
                continue

            if next_volume == node[:3]:
                k = self.graph.get_edge_k(node, next_node)
            else:
                k = None
            next_volume = (next_volume[0], next_volume[1], next_volume[2], k)
            if next_volume not in volumes:
                volumes[next_volume] = 0.0
            volumes[next_volume] = max(certainty, volumes[next_volume])
        
        total_certainty = sum(volumes.values())
        # filter out volumes with low certainty
        volumes = [volume for volume, certainty in volumes.items() if certainty / total_certainty > volume_min_certainty_total_percentage]
        return volumes
    
    def pick_next_volume(self, node):
        volumes = self.get_next_valid_volumes(node, volume_min_certainty_total_percentage=self.graph.overlapp_threshold["volume_min_certainty_total_percentage"])
        if len(volumes) == 0:
            return None
        # pick random volume
        volume = volumes[np.random.randint(len(volumes))]
        return volume
    
    def pick_best_node(self, node, next_volume):
        next_node = None
        next_certainty = 0.0
        next_node_k = 0.0
        edges = self.graph.nodes[node]['edges']
        for edge in edges:
            # check if edge goes into next volume
            if edge[0] == node:
                node_ = edge[1]
                k_factor = 1.0
            else:
                node_ = edge[0]
                k_factor = - 1.0
            if tuple(node_[:3]) != tuple(next_volume[:3]):
                continue
            if not next_volume[3] is None:
                if next_volume[3] != self.graph.get_edge_k(node, node_):
                    continue
            # check if edge has higher certainty
            certainty = self.graph.edges[edge]['certainty']
            if certainty <= next_certainty:
                continue

            next_node = node_
            next_certainty = certainty
            next_node_k = k_factor * self.graph.edges[edge]['sheet_offset_k']
        return next_node, next_node_k
    
    def pick_next_node(self, node):
        edges = self.graph.nodes[node]['edges']
        if len(edges) == 0:
            return None, None
        edge = edges[np.random.randint(len(edges))]
        node2 = edge[0] if edge[0] != node else edge[1]
        k = self.graph.get_edge_k(node, node2)
        return node2, k
    
    def display_umbilicus_angles(self, filename, z):
        # Display the umbilicus angles at z level.
        # displayed angles are 0 (white), 90 (red), 180 (green), 270 (blue)
        # using self.umbilicus_func

        # Get the angles
        zero_angle_vec = np.array([1.0, 0.0, 0.0])
        ninethy_angle_vec = np.array([0.0, 0.0, 1.0])
        oneeighty_angle_vec = np.array([-1.0, 0.0, 0.0])
        twohundredseventy_angle_vec = np.array([0.0, 0.0, -1.0])
        # Get umbilicus at z level
        umbilicus = self.umbilicus_func(z)

        # Display the angles, write dxfs
        # Create a new DXF document.
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()

        # Create polyline points 
        polyline_points = [Vec3(int(umbilicus[0]), int(umbilicus[1]), int(umbilicus[2])), Vec3(int(umbilicus[0] + zero_angle_vec[0] * 500), int(umbilicus[1] + zero_angle_vec[1] * 500), int(umbilicus[2] + zero_angle_vec[2] * 500))]

        # Add the 3D polyline to the model space
        msp.add_polyline3d(polyline_points, dxfattribs={'color': 1})

        polyline_points = [Vec3(int(umbilicus[0]), int(umbilicus[1]), int(umbilicus[2])), Vec3(int(umbilicus[0] + ninethy_angle_vec[0] * 500), int(umbilicus[1] + ninethy_angle_vec[1] * 500), int(umbilicus[2] + ninethy_angle_vec[2] * 500))]
        msp.add_polyline3d(polyline_points, dxfattribs={'color': 2})

        polyline_points = [Vec3(int(umbilicus[0]), int(umbilicus[1]), int(umbilicus[2])), Vec3(int(umbilicus[0] + oneeighty_angle_vec[0] * 500), int(umbilicus[1] + oneeighty_angle_vec[1] * 500), int(umbilicus[2] + oneeighty_angle_vec[2] * 500))]
        msp.add_polyline3d(polyline_points, dxfattribs={'color': 3})

        print(f"Alpha angles: 0 (red): {alpha_angles(np.array([zero_angle_vec]))}, 90 (yellow): {alpha_angles(np.array([ninethy_angle_vec]))}, 180 (green): {alpha_angles(np.array([oneeighty_angle_vec]))}, 270 (cyan): {alpha_angles(np.array([twohundredseventy_angle_vec]))}")

        # Save the DXF document
        doc.saveas(filename)
        
    
    def umbilicus_distance(self, node):
        if "umbilicus_distance" in self.graph.nodes[node]:
            return self.graph.nodes[node]["umbilicus_distance"]
        else:
            patch_centroid = self.graph.nodes[node]["centroid"]
            umbilicus_point = self.umbilicus_func(patch_centroid[1])
            patch_centroid_vec = patch_centroid - umbilicus_point
            self.graph.nodes[node]["umbilicus_distance"] = np.linalg.norm(patch_centroid_vec)
            return self.graph.nodes[node]["umbilicus_distance"]
        
    def check_overlapp_walk(self, walk, ks, step_size=20, away_dist_check=500):
        # build volume dict from walk
        volume_dict = {}
        for i, node in enumerate(walk):
            k = ks[i]
            volume = node[:3]
            if not volume in volume_dict:
                volume_dict[volume] = {}
            volume_dict[volume][node[3]] = k

        for i, node in enumerate(walk):
            k = ks[i]
            patch_centroid = self.graph.nodes[node]["centroid"]

            umbilicus_point = self.umbilicus_func(patch_centroid[1])
            dist = self.umbilicus_distance(node)
            nr_steps = int(dist // step_size)
            umbilicus_vec_step = (umbilicus_point - patch_centroid) / nr_steps
            umbilicus_steps = int(dist // step_size)
            for j in range(-away_dist_check//step_size, umbilicus_steps):
                step_point = patch_centroid + j * umbilicus_vec_step
                centroid_volumes = volumes_of_point(step_point)
                for volume in centroid_volumes:
                    if volume in volume_dict:
                        for patch in volume_dict[volume]:
                            k_ = volume_dict[volume][patch]
                            if k != k_:
                                continue
                            elif self.graph.overlapp_threshold["max_umbilicus_difference"] > 0.0:
                                dist_ = self.umbilicus_distance((volume[0], volume[1], volume[2], patch))
                                if abs(dist - dist_) > self.graph.overlapp_threshold["max_umbilicus_difference"]:
                                    return False
        return True
    
    def centroid_vector(self, node):
        if "centroid_vector" in node:
            return node["centroid_vector"]
        else:
            patch_centroid = node["centroid"]
            umbilicus_point = self.umbilicus_func(patch_centroid[1])
            node["centroid_vector"] = patch_centroid - umbilicus_point
            return node["centroid_vector"]
        
    def random_walk(self, start_node, start_k, max_steps=20, max_tries=6, min_steps=5):
        start_node = tuple(start_node)
        node = start_node
        steps = 0
        walk = [start_node]
        ks = [start_k]
        current_k  = ks[0]
        ks_dict = {start_node[:3]: [current_k]}
        patch_dict = {(start_node[:3], current_k): [start_node[3]]}
        steps_dict = {start_node: 0}
        while steps < max_steps:
            steps += 1
            tries = 0
            while True: # try to find next node
                if tries >= max_tries:
                    return None, "exeeded max_tries", False
                tries += 1
                node_, k = self.pick_next_node(node)
                # not found node
                if node_ is None:
                    continue
                # respect z and k range
                if (node_[1] < self.graph.overlapp_threshold["sheet_z_range"][0]) or (node_[1] > self.graph.overlapp_threshold["sheet_z_range"][1]):
                    continue
                if (current_k + k < self.graph.overlapp_threshold["sheet_k_range"][0]) or (current_k + k > self.graph.overlapp_threshold["sheet_k_range"][1]):
                    continue
                # node already visited
                if node_ in steps_dict:
                    node_walk_index = steps_dict[node_]
                    if ks[node_walk_index] != current_k + k: # different k, walk failure
                        walk.append(node_)
                        return walk, "small loop closure failed", False
                    elif steps - steps_dict[node_] > min_steps: # has enough steps, return subwalk
                        start_idx = steps_dict[node_]
                        walk.append(node_)
                        ks.append(current_k + k)
                        walk = walk[start_idx:]
                        ks = ks[start_idx:]

                        # Check for no bad overlaps
                        if not self.check_overlapp_walk(walk, ks):
                            return walk, "bad overlapp walk", False

                        return walk, "found good walk", True
                    continue # go find another next node
                elif (node_[:3] in ks_dict) and (current_k + k in ks_dict[node_[:3]]): # k already visited for this volume
                    # check if umbilicus distance is close
                    dist1 = self.umbilicus_distance(node_)
                    for patch2 in patch_dict[(node_[:3], current_k + k)]:
                        node2 = (node_[0], node_[1], node_[2], patch2)
                        dist2 = self.umbilicus_distance(node2)
                        if abs(dist1 - dist2) > self.graph.overlapp_threshold["max_umbilicus_difference"]:
                            walk.append(node_)
                            return walk, "already visited volume at current k", False
                    break
                else: # found valid next node
                    break

            # add node to walk
            node = node_
            steps_dict[tuple(node)] = steps
            if node is None:
                return None, "no next volume", False
            walk.append(node)
            current_k += k
            ks.append(current_k)
            if not node[:3] in ks_dict:
                ks_dict[node[:3]] = []
            ks_dict[node[:3]].append(current_k)
            if not (node[:3], current_k) in patch_dict:
                patch_dict[(node[:3], current_k)] = []
            patch_dict[(node[:3], current_k)].append(node[3])
            
        return None, "loop not closed in max_steps", False
    
    def pick_start_node(self, last_bad_walk_nodes):
        len_bad_walk_nodes = len(last_bad_walk_nodes)
        if len_bad_walk_nodes > 0 and np.random.rand() < 0.5:
            # pick from last bad walk nodes
            node = last_bad_walk_nodes[np.random.randint(0, len_bad_walk_nodes)]
        else:
            # randomly pick from all nodes
            len_nodes = self.graph.nodes_length
            node = list(self.graph.nodes.keys())[np.random.randint(0, len_nodes)]

        return node
    
    def save_solution(self, path, nodes, ks):
        print(f"\033[94m[ThaumatoAnakalyptor]:\033[0m Saving random walk computation in 2 seconds...")
        time.sleep(2)
        print(f"\033[94m[ThaumatoAnakalyptor]:\033[0m Saving random walk computation")
        # Save the solution to a file
        # save graph ks and nodes
        np.save(path.replace("blocks", "graph_RW") + "_ks.npy", ks)
        np.save(path.replace("blocks", "graph_RW") + "_nodes.npy", nodes)
        # self.graph.save_graph(path.replace("blocks", "graph_RW_solved") + ".pkl")
        print(f"\033[94m[ThaumatoAnakalyptor]:\033[0m Saved")

class WalkToSheet():
    def __init__(self, graph, nodes, ks, path, save_path, overlapp_threshold):
        self.graph = graph
        self.path = path
        self.save_path = save_path
        self.nodes = list(map(tuple, nodes))
        self.ks  = ks
        self.overlapp_threshold = overlapp_threshold

    def create_sheet(self):
        print(f"Min and max k: {np.min(self.ks)}, {np.max(self.ks)}")
        start_block, patch_id = self.graph.start_block, self.graph.patch_id
        # get start node
        main_sheet_patch_info = (start_block, int(patch_id), float(0.0))
        main_sheet_patch, _ = build_patch_tar(main_sheet_patch_info, (50, 50, 50), self.path, sample_ratio=1.0)
        anchor_angle = main_sheet_patch["anchor_angles"][0]
        main_sheet = {}
        main_sheet[tuple(start_block)] = {patch_id: {"offset_angle": anchor_angle, "displaying_points": []}}
        # visit all connected nodes
        print(F"Creating sheet with {len(self.nodes)} patches.")
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            k = self.ks[i]
            # build patch and add to sheet
            sheet_patch = (node[:3], int(node[3]), float(0.0))
            patch, _ = build_patch_tar(sheet_patch, (50, 50, 50), self.path, sample_ratio=self.overlapp_threshold["sample_ratio_score"])
            angle_offset = patch["anchor_angles"][0] - k * 360.0
            patches = [(node[:3], int(node[3]), angle_offset, None)]
            offset_angles = [angle_offset]
            update_main_sheet(main_sheet, patches, offset_angles, [[]])

        return main_sheet
    
    def save(self, main_sheet):
        save_main_sheet(main_sheet, {}, self.save_path.replace("blocks", "main_sheet_RW") + ".ta")

def compute(overlapp_threshold, start_point, path, recompute=False, compute_cpp_translation=False, stop_event=None, continue_segmentation=False):
    umbilicus_path = os.path.dirname(path) + "/umbilicus.txt"
    start_block, patch_id = find_starting_patch([start_point], path)

    save_path = os.path.dirname(path) + f"/{start_point[0]}_{start_point[1]}_{start_point[2]}/" + path.split("/")[-1]
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Configs: {overlapp_threshold}")

    recompute_path = path.replace("blocks", "scroll_graph") + ".pkl"
    recompute = recompute or not os.path.exists(recompute_path)

    continue_segmentation_path = path.replace("blocks", "scroll_graph_progress") + ".pkl"
    
    # Build graph
    if recompute:
        scroll_graph = ScrollGraph(overlapp_threshold, umbilicus_path)
        start_block, patch_id = scroll_graph.build_graph(path, num_processes=30, start_point=start_point, prune_unconnected=True)
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

    rw_solver = RandomWalkSolver(scroll_graph, umbilicus_path)

    debug_display = False
    if debug_display:
        scroll_graph.create_dxf_with_colored_polyline(save_path.replace("blocks", "graph") + ".dxf", min_z=975, max_z=1000)
        rw_solver.display_umbilicus_angles(save_path.replace("blocks", "umbilicus") + ".dxf", 985)
        
    scroll_graph.set_overlapp_threshold(overlapp_threshold)
    scroll_graph.start_block, scroll_graph.patch_id = start_block, patch_id
    # start_block, patch_id = scroll_graph.start_block, scroll_graph.patch_id

    solver = RandomWalkSolver(scroll_graph, umbilicus_path)
    solver.save_overlapp_threshold()
    rw_solver.solve(path=save_path, max_nr_walks=overlapp_threshold["max_nr_walks"], max_steps=overlapp_threshold["max_steps"], max_tries=overlapp_threshold["max_tries"], min_steps=overlapp_threshold["min_steps"], stop_event=stop_event)
    
    exit()
    
    # save graph
    scroll_graph.save_graph(save_path.replace("blocks", "graph_RW_solved") + ".pkl")
    
    if True:
        print("Creating sheet...")
        w2s = WalkToSheet(scroll_graph, nodes, ks, path, save_path, overlapp_threshold)
        main_sheet = w2s.create_sheet()
        print("Saving sheet...")
        w2s.save(main_sheet)

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

    # Compute
    compute(overlapp_threshold=overlapp_threshold, start_point=start_point, path=path, recompute=recompute, compute_cpp_translation=compute_cpp_translation, continue_segmentation=continue_segmentation)

if __name__ == '__main__':
    random_walks()