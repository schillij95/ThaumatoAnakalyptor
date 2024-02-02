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

    def add_node(self, node, cardinality, centroid, fixed=False):
        node = tuple(node)
        belief = np.zeros(cardinality)
        belief[1:-1] = 1.0 / (cardinality - 2)
        self.nodes[node] = {'belief': belief, 'centroid': centroid, 'fixed': fixed}

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

    def add_edge(self, node1, node2, update_matrix1, update_matrix2, certainty, sheet_offset_k=0.0, add_transition_matrices=True):
        assert certainty > 0.0, "Certainty must be greater than 0."
        certainty = np.clip(certainty, 0.0, None)

        node1 = tuple(node1)
        node2 = tuple(node2)
        if len(update_matrix1) != len(update_matrix1[0]):
            raise ValueError("Update matrix must be square.")
        if len(update_matrix2) != len(update_matrix2[0]):
            raise ValueError("Update matrix must be square.")
        update_matrix1 = np.array(update_matrix1).astype(float)
        update_matrix2 = np.array(update_matrix2).astype(float)
        # Ensure node1 < node2 for bidirectional nodes
        if node2 < node1:
            node1, node2 = node2, node1
            update_matrix1, update_matrix2 = update_matrix2, update_matrix1
            sheet_offset_k = sheet_offset_k * (-1.0)
        self.edges[(node1, node2)] = {'certainty': certainty, 'sheet_offset_k': sheet_offset_k}
        if add_transition_matrices:
            self.edges[(node1, node2)]['matrix1'] = np.array(update_matrix1)
            self.edges[(node1, node2)]['matrix2'] = np.array(update_matrix2)


    def get_edge_data(self, node1, node2):
        # Maintain bidirectional invariant
        if node2 < node1:
            node1, node2 = node2, node1
        edge_data = self.edges.get((node1, node2))
        if edge_data is not None:
            return node1, node2, edge_data['matrix1'], edge_data['matrix2'], edge_data['certainty'], edge_data['sheet_offset_k']
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

    def is_node_fixed(self, node):
        return self.nodes[node]['fixed']

    def get_node_belief(self, node):
        return self.nodes[node]['belief']
    
    def get_node_value(self, node):
        current_node = self.nodes[node]
        return current_node['belief'], current_node['fixed']

    def set_node_belief(self, node, belief):
        if node not in self.nodes:
            raise KeyError(f"Node {node} does not exist.")
        belief = np.array(belief).astype(float)
        if self.is_node_fixed(node):
            raise ValueError(f"Node {node} belief is fixed and cannot be updated.")
        if len(belief) != len(self.nodes[node]['belief']):
            raise ValueError(f"New belief must have length {len(self.nodes[node]['belief'])}.")
        self.nodes[node]['belief'] = belief

    def set_node_fixed(self, node):
        self.nodes[node]['fixed'] = True

    def save_graph(self, path):
        # Save graph class object to file
        with open(path, 'wb') as file:
            pickle.dump(self, file)

def score_same_block_patches(patch1, patch2, overlapp_threshold):
    """
    Calculate the score between two patches from the same block.
    """
    # Calculate winding switch sheet scores
    score_raw, k = winding_switch_sheet_score_raw_precomputed_surface(patch1, patch2, overlapp_threshold)
    k = k * overlapp_threshold["winding_direction"] # because of scroll's specific winding direction

    score_val = (overlapp_threshold["max_winding_switch_sheet_distance"] - score_raw) / (overlapp_threshold["max_winding_switch_sheet_distance"] - overlapp_threshold["min_winding_switch_sheet_distance"]) # calculate score
    
    score = score_val * overlapp_threshold["winding_switch_sheet_score_factor"]

    # Check for enough points
    if score <= 0.0:
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

def process_same_block(main_block_patches_list, overlapp_threshold):
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
            (score_, k_), anchor_angle1, anchor_angle2 = score_same_block_patches(main_block_patches_list[i], main_block_patches_list[j], overlapp_threshold)
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
            id = patches_list[j]["ids"][0]

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
    file_path, path_instances, overlapp_threshold = args
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
    for surrounding_id in surrounding_ids:
        volume_path = path_instances + f"{file_path.split('/')[0]}/{surrounding_id[0]:06}_{surrounding_id[1]:06}_{surrounding_id[2]:06}"
        surrounding_blocks_patches_list.extend(subvolume_surface_patches_folder(volume_path, sample_ratio=overlapp_threshold["sample_ratio_score"]))

    # Add the overlap base to the patches list that contains the points + normals + scores only before
    patches_list = main_block_patches_list + surrounding_blocks_patches_list
    add_overlapp_entries_to_patches_list(patches_list)
    subvolume = {"start": block_id - 50, "end": block_id + 50}

    # Assign points to tiles
    assign_points_to_tiles(patches_list, subvolume, tiling=3)

    # calculate scores between each main block patch and surrounding blocks patches
    score_sheets = []
    for i, main_block_patch in enumerate(main_block_patches_list):
        for j, surrounding_block_patch in enumerate(surrounding_blocks_patches_list):
            patches_list_ = [main_block_patch, surrounding_block_patch]
            score_ = score_other_block_patches(patches_list_, 0, 1, overlapp_threshold) # score, anchor_angle1, anchor_angle2
            if score_[0] > overlapp_threshold["final_score_min"]:
                score_sheets.append((main_block_patch['ids'][0], surrounding_block_patch['ids'][0], score_[0], None, score_[1], score_[2]))
    
    score_switching_sheets = process_same_block(main_block_patches_list, overlapp_threshold)

    # Process and return results...
    return score_sheets, score_switching_sheets, patches_centroids

class ScrollGraph(Graph):
    def __init__(self, cardinality, overlapp_threshold, limit_stickiness=0.5, add_transition_matrices=False):
        super().__init__()
        self.cardinality = cardinality + 2  # Add two extra values for the upper and lower limits
        self.limit_stickiness = limit_stickiness
        self.add_transition_matrices = add_transition_matrices
        self.set_overlapp_threshold(overlapp_threshold)

    def add_switch_edge(self, node_lower, node_upper, certainty):
        # Build a switch edge between two nodes
        update_matrix_down = np.zeros((self.cardinality, self.cardinality), dtype=float)
        update_matrix_up = np.zeros((self.cardinality, self.cardinality), dtype=float)
        for i in range(self.cardinality):
            if i < self.cardinality - 1:
                update_matrix_down[i, i + 1] = 1.0
            if i > 0:
                update_matrix_up[i, i - 1] = 1.0
        
        update_matrix_down[0, 0] = 1.0
        update_matrix_down[0, 1] = 1.0

        update_matrix_up[self.cardinality - 1, self.cardinality - 1] = 1.0
        update_matrix_up[self.cardinality - 1, self.cardinality - 2] = 1.0

        self.add_edge(node_lower, node_upper, update_matrix_up, update_matrix_down, certainty, sheet_offset_k=1.0, add_transition_matrices=self.add_transition_matrices)

    def add_same_sheet_edge(self, node1, node2, certainty):
        # Build a same sheet edge between two nodes
        update_matrix = np.zeros((self.cardinality, self.cardinality), dtype=float)
        for i in range(self.cardinality):
            update_matrix[i, i] = 1
        self.add_edge(node1, node2, update_matrix, update_matrix, certainty, sheet_offset_k=0.0, add_transition_matrices=self.add_transition_matrices)

    def build_other_block_edges(self, score_sheets):
        # Build edges between patches from different blocks
        for score_ in score_sheets:
            id1, id2, score, _, anchor_angle1, anchor_angle2 = score_
            if score < self.overlapp_threshold["final_score_min"]:
                continue
            if abs(anchor_angle1 - anchor_angle2) > 180.0:
                if anchor_angle1 > anchor_angle2:
                    id1, id2 = id2, id1
                self.add_switch_edge(id1, id2, score)
            else:
                self.add_same_sheet_edge(id1, id2, score)

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
                self.add_switch_edge(id1, id2, score)

    def filter_blocks(self, blocks_tar_files, blocks_tar_files_int, start_block, distance):
        blocks_tar_files_int = np.array(blocks_tar_files_int)
        start_block = np.array(start_block)
        distances = np.abs(blocks_tar_files_int - start_block)
        distances = np.sum(distances, axis=1)
        filter_mask = np.sum(np.abs(blocks_tar_files_int - start_block), axis=1) < distance
        blocks_tar_files_int = blocks_tar_files_int[filter_mask]
        blocks_tar_files = [blocks_tar_files[i] for i in range(len(blocks_tar_files)) if filter_mask[i]]
        return blocks_tar_files, blocks_tar_files_int
    
    def prune_unconnected_nodes(self, start_node):
        # Prune nodes that are not connected to the start node
        # walk the graph from the start node
        visited = set()
        queue = [start_node]
        # tqdm object showing process of visited nodes untill all nodes are visited
        tqdm_object = tqdm(total=len(self.nodes))
        while queue:
            node = queue.pop(0)
            if node not in visited:
                tqdm_object.update(1)
                visited.add(node)
                edges = self.nodes[node]['edges']
                for edge in edges:
                    queue.append(edge[0] if edge[0] != node else edge[1])
        nodes_total = len(self.nodes)
        edges_total = len(self.edges)
        print(f"number of nodes: {nodes_total}, number of edges: {edges_total}")
        # remove unvisited nodes
        for node in list(self.nodes.keys()):
            if node not in visited:
                del self.nodes[node]

        # remove unvisited edges
        for edge in list(self.edges.keys()):
            if edge[0] not in visited or edge[1] not in visited:
                del self.edges[edge]
        
        print(f"Pruned {nodes_total - len(self.nodes)} nodes. Of {nodes_total} nodes.")
        print(f"Pruned {edges_total - len(self.edges)} edges. Of {edges_total} edges.")

    def build_graph(self, path_instances, start_point, distance, num_processes=4, prune_unconnected=False):
        blocks_tar_files = glob.glob(path_instances + '/*.tar')
        blocks_tar_files_int = [[int(i) for i in x.split('/')[-1].split('.')[0].split("_")] for x in blocks_tar_files]

        #from original coordinates to instance coordinates
        start_block, patch_id = find_starting_patch([start_point], path_instances)
        self.start_block, self.patch_id, self.start_point = start_block, patch_id, start_point

        distance = distance / (200.0 / 50.0)
        self.distance = distance

        if distance and distance > 0.0:
            blocks_tar_files, blocks_tar_files_int = self.filter_blocks(blocks_tar_files, blocks_tar_files_int, start_block, distance)

        print(f"Found {len(blocks_tar_files)} blocks.")
        print("Building graph...")

        # Create a pool of worker processes
        with Pool(num_processes) as pool:
            # Map the process_block function to each file
            zipped_args = list(zip(blocks_tar_files, [path_instances] * len(blocks_tar_files), [self.overlapp_threshold] * len(blocks_tar_files)))
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
        for edge in self.edges:
            self.add_node(edge[0], self.cardinality, patches_centroids[edge[0]])
            self.add_node(edge[1], self.cardinality, patches_centroids[edge[1]])

        node_id = tuple((*start_block, patch_id))
        print(f"Start node: {node_id}, nr nodes: {len(self.nodes)}, nr edges: {len(self.edges)}")
        belief = np.zeros(self.cardinality)
        belief[belief.shape[0]//2] = 1.0
        self.set_node_belief(node_id, belief)
        # Set node fixed
        self.set_node_fixed(node_id)
        self.compute_node_edges()
        if prune_unconnected:
            print("Prunning unconnected nodes...")
            self.prune_unconnected_nodes(node_id)

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
        # update nodes entries
        for node in self.nodes:
            if "volume_precomputation" in self.nodes[node]:
                del self.nodes[node]["volume_precomputation"]
        print("Overlapping threshold set.")
        
class RandomWalkSolver:
    def __init__(self, graph, umbilicus_path):
        self.graph = graph
        self.walk_aggregation = {"current_found_nodes": 0}
        self.umbilicus_path = umbilicus_path
        self.init_umbilicus(umbilicus_path)

        if not hasattr(self.graph, "overlapp_threshold_filename"):
            # Save overlap_threshold to a YAML file
            overlapp_threshold_filename = 'overlapp_threshold.yaml'
            self.graph.overlapp_threshold_filename = overlapp_threshold_filename

    def init_umbilicus(self, umbilicus_path):
        # Load the umbilicus data
        umbilicus_data = load_xyz_from_file(umbilicus_path)
        # scale and swap axis
        umbilicus_data = scale_points(umbilicus_data, 50.0/200.0, axis_offset=0)
        # Define a wrapper function for umbilicus_xz_at_y
        self.umbilicus_func = lambda z: umbilicus_xz_at_y(umbilicus_data, z)

    def translate_data_to_cpp(self, recompute_translation=True):
        """
        Prepares graph data, solver parameters, and overlap threshold for C++ processing.

        :param graph: A graph object with nodes and edges.
        :param solver_parameters: Dictionary of solver parameters like max_nr_walks, max_steps, etc.
        :param overlapp_threshold: Dictionary of overlap threshold parameters.
        :return: Data structures suitable for C++ processing.
        """
        if not recompute_translation and hasattr(self.graph, "cpp_translation"):

            res = [self.graph.overlapp_threshold_filename]
            for i in range(1, 7):
                path_numpy_array = self.graph.cpp_translation.replace("blocks", "graph_RW") + "_nodes_cpp_" + str(i) + ".npy"
                print("load path:", path_numpy_array)
                numpy_array = np.load(path_numpy_array)
                res.append(numpy_array)
            return res
        
        # Prepare nodes data
        ids = []
        centroids = []
        umbilicusDirections = []
        next_nodes = []
        k_values = []
        validIndices = []
        for node, attributes in tqdm(self.graph.nodes.items()):
            next_ns = {}
            valid_volumes = self.get_next_valid_volumes(node, volume_min_certainty_total_percentage=self.graph.overlapp_threshold["volume_min_certainty_total_percentage"])
            valid_volumes = [tuple(volume[:3]) for volume in valid_volumes]
            for edge in attributes['edges']:
                if edge[0][:3] == edge[1][:3]: # disregard loopback volume edges
                    continue
                other_node = edge[0] if edge[0] != node else edge[1]
                if not tuple(other_node[:3]) in valid_volumes: # only look at valid next volumes
                    continue
                k = self.graph.get_edge_k(node, other_node)
                certainty = self.graph.edges[edge]['certainty']
                if certainty <= 0:
                    continue
                if not other_node[:3] in next_ns:
                    next_ns[other_node[:3]] = (other_node, certainty, k)
                elif next_ns[other_node[:3]][1] < certainty:
                    next_ns[other_node[:3]] = (other_node, certainty, k)

            k_vals = [next_ns[key][2] for key in next_ns]
            next_ns = [list(next_ns[key][0]) for key in next_ns]
            validIndices.append(len(next_ns))
            if len(next_ns) < 6:
                next_ns = next_ns + [[-1, -1, -1, -1]] * (6 - len(next_ns))
                k_vals = k_vals + [0.0] * (6 - len(k_vals))

            assert len(next_ns) == 6, f"Number of next nodes must be 6. Got {len(next_ns)}."
            for nens in next_ns:
                assert len(nens) == 4, f"Each next node must have 4 values. Got {len(nens)}."
            assert len(k_vals) == 6, f"Number of k values must be 6. Got {len(k_vals)}."

            ids.append(list(node))
            centroids.append(attributes['centroid'])
            umbilicusDirections.append(self.centroid_vector(self.graph.nodes[node]))
            next_nodes.append(next_ns)
            k_values.append(k_vals)

        ids_array = np.array(ids).astype(int)
        centroids_array = np.array(centroids).astype(float)
        umbilicusDirections_array = np.array(umbilicusDirections, dtype=np.float64).astype(float)
        next_nodes_array = np.array(next_nodes).astype(int)
        k_values_array = np.array(k_values).astype(int)
        validIndices_array = np.array(validIndices).astype(int)

        assert ids_array.shape[0] == len(self.graph.nodes), f"Number of nodes must be {len(self.graph.nodes)}. Got {ids_array.shape[0]}."
        assert centroids_array.shape[0] == len(self.graph.nodes), f"Number of nodes must be {len(self.graph.nodes)}. Got {centroids_array.shape[0]}."
        assert umbilicusDirections_array.shape[0] == len(self.graph.nodes), f"Number of nodes must be {len(self.graph.nodes)}. Got {umbilicusDirections_array.shape[0]}."
        assert next_nodes_array.shape[0] == len(self.graph.nodes), f"Number of nodes must be {len(self.graph.nodes)}. Got {next_nodes_array.shape[0]}."
        assert k_values_array.shape[0] == len(self.graph.nodes), f"Number of nodes must be {len(self.graph.nodes)}. Got {k_values_array.shape[0]}."
        assert validIndices_array.shape[0] == len(self.graph.nodes), f"Number of nodes must be {len(self.graph.nodes)}. Got {validIndices_array.shape[0]}."

        # Save overlap_threshold to a YAML file
        overlapp_threshold_filename = 'overlapp_threshold.yaml'
        self.graph.overlapp_threshold_filename = overlapp_threshold_filename
        self.save_overlapp_threshold()

        return overlapp_threshold_filename, ids_array, next_nodes_array, validIndices_array, k_values_array, umbilicusDirections_array, centroids_array

    def save_overlapp_threshold(self):
        with open(self.graph.overlapp_threshold_filename, 'w') as file:
            yaml.dump(self.graph.overlapp_threshold, file)

    def translate_data_to_python(self, nodes_array, ks_array):
        # Prepare nodes data
        nodes = []
        ks = []
        for i in range(nodes_array.shape[0]):
            node = tuple(nodes_array[i])
            nodes.append(node)
            ks.append(ks_array[i])

        return nodes, ks

    def solve_cpp(self, path, starting_node, max_nr_walks=100, max_unchanged_walks=10000, max_steps=100, max_tries=6, min_steps=10, min_end_steps=4, continue_walks=False, nodes=None, ks=None):
        ### C++ RW, not yet fully implemented on the cpp side
        # try:
        #     nodes_array, ks_array = sheet_generation.solve_random_walk(*starting_node, *self.translate_data_to_cpp(recompute_translation=False))
        # except Exception as e:
        #     print(f"Error: {e}")
        #     raise e
        # return nodes_array, ks_array
        return None, None

    def solve(self, path, starting_node, max_nr_walks=100, max_unchanged_walks=10000, max_steps=100, max_tries=6, min_steps=10, min_end_steps=4, continue_walks=False, nodes=None, ks=None, k_step_size=8):
        k_step_size_half = k_step_size // 2
        overlapp_threshold = self.graph.overlapp_threshold
        k_min = overlapp_threshold["sheet_k_range"][0]
        k_max = overlapp_threshold["sheet_k_range"][1]
        # call solve with some range of the k values to facilitate efficient solving
        for k_i in range(max(k_min, -k_step_size_half), k_max, k_step_size_half):
            overlapp_threshold["sheet_k_range"] = (k_i, min(k_max, k_i + k_step_size))
            nodes, ks = self.solve_(path, starting_node, max_nr_walks=max_nr_walks, max_unchanged_walks=max_unchanged_walks, max_steps=max_steps, max_tries=max_tries, min_steps=min_steps, min_end_steps=min_end_steps, continue_walks=continue_walks, nodes=nodes, ks=ks)
            continue_walks = True
        if k_min < -k_step_size_half:
            for k_i in range(min(k_max, k_step_size_half), k_min, -k_step_size_half):
                overlapp_threshold["sheet_k_range"] = (max(k_min, k_i-k_step_size), k_i)
                nodes, ks = self.solve_(path, starting_node, max_nr_walks=max_nr_walks, max_unchanged_walks=max_unchanged_walks, max_steps=max_steps, max_tries=max_tries, min_steps=min_steps, min_end_steps=min_end_steps, continue_walks=continue_walks, nodes=nodes, ks=ks)
                continue_walks = True
        overlapp_threshold["sheet_k_range"] = (k_min, k_max)
        return nodes, ks

    def solve_(self, path, starting_node, max_nr_walks=100, max_unchanged_walks=10000, max_steps=100, max_tries=6, min_steps=10, min_end_steps=4, continue_walks=False, nodes=None, ks=None):
        walk_aggregation_threshold = self.graph.overlapp_threshold["walk_aggregation_threshold"]
        min_steps_start = min_steps
        if not continue_walks:
            volume_dict = {starting_node[:3]: {starting_node[3]: 0}}
            nodes = np.array([starting_node], dtype=int)
            nodes_neighbours_count = np.array([0], dtype=int)
            picked_nrs = np.array([0], dtype=float)
            pick_prob = np.array([0.0], dtype=float)
            ks = np.array([0])
            old_nodes = None
            old_ks = None
            nr_previous_nodes = 0
        else: # continue_walks
            nodes_valid_mask_z = np.logical_and(nodes[:,1] >= self.graph.overlapp_threshold["sheet_z_range"][0], nodes[:,1] <= self.graph.overlapp_threshold["sheet_z_range"][1])
            nodes_valid_mask_k = np.logical_and(ks >= self.graph.overlapp_threshold["sheet_k_range"][0], ks <= self.graph.overlapp_threshold["sheet_k_range"][1])
            nodes_valid_mask = np.logical_and(nodes_valid_mask_z, nodes_valid_mask_k)

            if np.sum(nodes_valid_mask) == 0:
                print(f"\033[94m[ThaumatoAnakalyptor]: No valid initial nodes found.\033[0m")
                return nodes, ks

            nr_previous_nodes = nodes.shape[0]
            old_nodes = nodes[~nodes_valid_mask]
            old_ks = ks[~nodes_valid_mask]

            nodes = nodes[nodes_valid_mask]
            ks = ks[nodes_valid_mask]

            volume_dict = {}
            for i in range(len(nodes)):
                node = tuple(nodes[i])
                if not node[:3] in volume_dict:
                    volume_dict[node[:3]] = {}
                volume_dict[node[:3]][node[3]] = i
            picked_nrs = np.zeros(len(nodes))
            pick_prob = np.zeros(len(nodes))
            nodes_neighbours_count = np.zeros(len(nodes))
            # update neighbours count for all nodes
            for index_node in range(len(nodes)):
                node = nodes[index_node]
                for edge in self.graph.nodes[tuple(node)]['edges']:
                    neighbour_node = edge[0] if edge[0] != tuple(node) else edge[1]
                    if neighbour_node[:3] in volume_dict and neighbour_node[3] in volume_dict[neighbour_node[:3]]:
                        index_node_neighbour = volume_dict[neighbour_node[:3]][neighbour_node[3]]
                        nodes_neighbours_count[index_node_neighbour] += 1
                        nodes_neighbours_count[index_node] += 1

        nr_walks = 0
        nr_walks_total = 0
        nr_unupdated_walks = 0
        failed_dict = {}
        time_pick = 0.0
        time_walk = 0.0
        time_postprocess = 0.0
        nr_unchanged_walks = 0
        raw_sucessful_walks = 0
        # while nr_walks < max_nr_walks or max_nr_walks < 0:
        while max_nr_walks > 0:
            time_start = time.time()
            sn, sk = self.pick_start_node(nodes, nodes_neighbours_count, ks, picked_nrs, pick_prob)
            time_pick += time.time() - time_start
            time_start = time.time()
            walk, k = self.random_walk(sn, sk, volume_dict, nodes, ks, max_steps=max_steps, max_tries=max_tries, min_steps=min_steps)
            time_walk += time.time() - time_start
            time_start = time.time()
            if walk is None:
                nr_unchanged_walks += 1
                if not k in failed_dict:
                    failed_dict[k] = 0
                failed_dict[k] += 1
            else:
                new_nodes = []
                new_ks = []
                existing_nodes = []
                new_walk = False
                for i in range(len(walk)):
                    walk_node = walk[i]
                    if (walk_node[:3] not in volume_dict) or (walk_node[3] not in volume_dict[walk_node[:3]]):
                        new_nodes.append(walk[i])
                        new_ks.append(k[i])
                        new_walk = True
                    else:
                        existing_nodes.append(walk[i])
                if new_walk:
                    if not self.check_overlapp_walk(walk, k, volume_dict, ks):
                        walk, k = None, "wrong patch overlapp"
                        if not k in failed_dict:
                            failed_dict[k] = 0
                        failed_dict[k] += 1

            if walk is not None:
                raw_sucessful_walks += 1
                new_nodes, new_ks = self.walk_aggregation_func(new_nodes, new_ks, volume_dict, ks)

                if len(new_nodes) != 0:
                    nr_walks += 1
                    nr_unchanged_walks = 0
                if not new_walk:
                    nr_unupdated_walks += 1
                    nr_unchanged_walks += 1
                    # add to picked_nrs since in this region, the patches picking is saturated
                    for node in walk:
                        index_ = volume_dict[node[:3]][node[3]]
                        picked_nrs[index_] += 5
                else:
                    # update picking probability for existing nodes
                    for node in existing_nodes:
                        index_ = volume_dict[node[:3]][node[3]]
                        picked_nrs[index_] = max(picked_nrs[index_]-5, 0)

                if (nr_unchanged_walks > max_unchanged_walks) and (self.walk_aggregation["current_found_nodes"] != 0 or nr_unchanged_walks > 2*max_unchanged_walks):
                    print(f"Walks: {nr_walks_total}, unupdated walks: {nr_unupdated_walks}, sucessful: {nr_walks}, raw_sucessful_walks: {raw_sucessful_walks}, failed because: \n{failed_dict}")
                    print(f"Time pick: {time_pick}, time walk: {time_walk}, time postprocess: {time_postprocess}")
                    picked_nrs = np.zeros_like(picked_nrs)
                    nr_unchanged_walks = 0
                    if min_steps > 1 and min_steps//2 >= min_end_steps:
                        min_steps = min_steps // 2
                        print(f"\033[94m[ThaumatoAnakalyptor]: Max unchanged walks reached. Adjusting min_steps to {min_steps}.\033[0m")
                    elif self.graph.overlapp_threshold["walk_aggregation_threshold"] > 1:
                        min_steps = min_steps_start
                        self.graph.overlapp_threshold["walk_aggregation_threshold"] = self.graph.overlapp_threshold["walk_aggregation_threshold"] // 2
                        print(f"\033[94m[ThaumatoAnakalyptor]: Max unchanged walks reached. Adjusting walk_aggregation_threshold to {self.graph.overlapp_threshold['walk_aggregation_threshold']}.\033[0m")
                    else:
                        self.graph.overlapp_threshold["walk_aggregation_threshold"] = walk_aggregation_threshold
                        print(f"\033[94m[ThaumatoAnakalyptor]: Max unchanged walks reached. Finishing the random walks.\033[0m")
                        break

                length_nodes = len(nodes)
                # Add new nodes to nodes
                if len(new_nodes) != 0:
                    nodes = np.concatenate((nodes, np.array(new_nodes)))
                    ks = np.concatenate((ks, np.array(new_ks)))
                    picked_nrs = np.concatenate((picked_nrs, np.zeros(len(new_nodes))))
                    pick_prob = np.concatenate((pick_prob, np.zeros(len(new_nodes))))
                    nodes_neighbours_count = np.concatenate((nodes_neighbours_count, np.zeros(len(new_nodes))))


                for new_index in range(len(new_nodes)):
                    if not new_nodes[new_index][:3] in volume_dict:
                        volume_dict[new_nodes[new_index][:3]] = {}
                    volume_dict[new_nodes[new_index][:3]][new_nodes[new_index][3]] = length_nodes + new_index
                
                # update neighbours count for new nodes
                for index_new_node in range(len(new_nodes)):
                    node = new_nodes[index_new_node]
                    for edge in self.graph.nodes[node]['edges']:
                        neighbour_node = edge[0] if edge[0] != node else edge[1]
                        if neighbour_node[:3] in volume_dict and neighbour_node[3] in volume_dict[neighbour_node[:3]]:
                            index_node = volume_dict[neighbour_node[:3]][neighbour_node[3]]
                            nodes_neighbours_count[index_node] += 1
                            nodes_neighbours_count[length_nodes + index_new_node] += 1
                
            nr_walks_total += 1
            if nr_walks_total % (max_nr_walks * 2) == 0:
                if nr_walks_total % (max_nr_walks * 2 * 100) == 0:
                    nodes_save = nodes
                    ks_save = ks
                    if old_nodes is not None:
                        nodes_save = np.concatenate((old_nodes, nodes))
                        ks_save = np.concatenate((old_ks, ks))
                    self.save_solution(path, nodes_save, ks_save)
                print(f"Previous nodes: {nr_previous_nodes}, Current Nodes: {nodes.shape[0] + (old_nodes.shape[0] if old_nodes is not None else 0) - nr_previous_nodes}, Walks: {nr_walks_total}, unupdated walks: {nr_unupdated_walks}, sucessful: {nr_walks}, raw_sucessful_walks: {raw_sucessful_walks}, failed because: \n{failed_dict}")
                print(f"Time pick: {time_pick}, time walk: {time_walk}, time postprocess: {time_postprocess}, step size: {min_steps}, walk_aggregation_threshold: {self.graph.overlapp_threshold['walk_aggregation_threshold']}, k_range: {self.graph.overlapp_threshold['sheet_k_range']}")
                time_pick, time_walk, time_postprocess = 0.0, 0.0, 0.0
            time_postprocess += time.time() - time_start

        print(f"Walks: {nr_walks_total}, sucessful: {nr_walks}, raw_sucessful_walks: {raw_sucessful_walks}, failed because: \n{failed_dict}")
        # mean and std and median of picked_nrs
        mean = np.mean(picked_nrs)
        std = np.std(picked_nrs)
        median = np.median(picked_nrs)
        print(f"Mean: {mean}, std: {std}, median: {median} of picked_nrs")
        self.nodes_solution = nodes
        self.ks_solution = ks

        if old_nodes is not None:
            nodes = np.concatenate((nodes, old_nodes))
            ks = np.concatenate((ks, old_ks))

        return nodes, ks
    
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
    
    def walk_aggregation_func(self, nodes, ks, volume_dict, ks_original):
        if not nodes:
            return nodes, ks
        overlapp_threshold = self.graph.overlapp_threshold
        new_nodes = []
        new_ks = []
        found_aggregated_nodes = False
        for i, node in enumerate(nodes):
            k = ks[i]
            key = tuple((node, k))
            if not key in self.walk_aggregation:
                self.walk_aggregation[key] = 0
            self.walk_aggregation[key] += 1
            if self.walk_aggregation[key] >= overlapp_threshold["walk_aggregation_threshold"]:
                found_aggregated_nodes = True
                if not node in new_nodes:
                    new_nodes.append(node)
                    new_ks.append(k)
        if found_aggregated_nodes:
            if overlapp_threshold["enable_winding_switch_postprocessing"]:
                nodes_switched = []
                ks_switched = []
                for node in new_nodes:
                    nodes_node, ks_node = self.sheet_switching_nodes(node, k, volume_dict, ks_original)
                    for i, node_ in enumerate(nodes_node):
                        k_ = ks_node[i]
                        if node_ in new_nodes or node_ in nodes_switched:
                            continue
                        continue_flag = False
                        for j, new_node in enumerate(new_nodes):
                            if tuple(node_[:3]) == tuple(new_node[:3]):
                                if k_ == new_ks[j]:
                                    continue_flag = True
                                    break
                        if continue_flag:
                            continue
                        
                        nodes_switched.append(node_)
                        ks_switched.append(k_)

                new_nodes += nodes_switched
                new_ks += ks_switched

            self.walk_aggregation["current_found_nodes"] += len(new_nodes)
            if self.walk_aggregation["current_found_nodes"] >= overlapp_threshold["walk_aggregation_max_current"] and overlapp_threshold["walk_aggregation_max_current"] > 0:
                self.walk_aggregation = {"current_found_nodes": 0}
            else:
                for i in range(len(new_nodes)):
                    new_node = new_nodes[i]
                    new_k = new_ks[i]
                    new_key = tuple((new_node, new_k))
                    if new_key in self.walk_aggregation:
                        del self.walk_aggregation[new_key]
        return new_nodes, new_ks


    def get_next_valid_volumes(self, node, volume_min_certainty_total_percentage=0.15):
        if "volume_precomputation" in self.graph.nodes[node]:
            if "valid_volumes" in self.graph.nodes[node]["volume_precomputation"]:
                return self.graph.nodes[node]["volume_precomputation"]["valid_volumes"]
        else:
            self.graph.nodes[node]["volume_precomputation"] = {}
        
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
        
        self.graph.nodes[node]["volume_precomputation"]["valid_volumes"] = volumes
        return volumes
    
    def pick_next_volume(self, node):
        volumes = self.get_next_valid_volumes(node, volume_min_certainty_total_percentage=self.graph.overlapp_threshold["volume_min_certainty_total_percentage"])
        if len(volumes) == 0:
            return None
        # pick random volume
        volume = volumes[np.random.randint(len(volumes))]
        return volume
    
    def pick_best_node(self, node, next_volume):
        if "volume_precomputation" in self.graph.nodes[node]:
            if "next_volume" in self.graph.nodes[node]["volume_precomputation"]:
                if next_volume in self.graph.nodes[node]["volume_precomputation"]["next_volume"]:
                    return self.graph.nodes[node]["volume_precomputation"]["next_volume"][next_volume]
        else:
            self.graph.nodes[node]["volume_precomputation"] = {"next_volume": {}}
        
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
        
        if not ("next_volume" in self.graph.nodes[node]["volume_precomputation"]):
            self.graph.nodes[node]["volume_precomputation"]["next_volume"] = {}
        self.graph.nodes[node]["volume_precomputation"]["next_volume"][next_volume] = [next_node, next_node_k]
        return next_node, next_node_k
    
    def pick_next_node(self, node):
        next_volume = self.pick_next_volume(node)
        if next_volume is None:
            return None, None
        
        return self.pick_best_node(node, next_volume)
    
    def umbilicus_distance(self, node):
        if "umbilicus_distance" in self.graph.nodes[node]:
            return self.graph.nodes[node]["umbilicus_distance"]
        else:
            patch_centroid = self.graph.nodes[node]["centroid"]
            umbilicus_point = self.umbilicus_func(patch_centroid[1])
            patch_centroid_vec = patch_centroid - umbilicus_point
            self.graph.nodes[node]["umbilicus_distance"] = np.linalg.norm(patch_centroid_vec)
            return self.graph.nodes[node]["umbilicus_distance"]
    
    def check_overlapp_walk(self, walk, ks, volume_dict, ks_nodes, step_size=20, away_dist_check=500):
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
                            index_ = volume_dict[volume][patch]
                            k_ = ks_nodes[index_]
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
        
    def random_walk(self, start_node, start_k, volume_dict, nodes, ks_nodes, max_steps=20, max_tries=6, min_steps=5):
        start_node = tuple(start_node)
        node = start_node
        steps = 0
        walk = [start_node]
        ks = [start_k]
        current_k  = ks[0]
        ks_dict = {start_node[:3]: [current_k]}
        while steps < max_steps:
            steps += 1
            tries = 0
            while True: # try to find next node
                if tries >= max_tries:
                    return None, "exeeded max_tries"
                tries += 1
                node_, k = self.pick_next_node(node)
                # not found node
                if not node_:
                    continue
                # respect z and k range
                if (node_[1] < self.graph.overlapp_threshold["sheet_z_range"][0]) or (node_[1] > self.graph.overlapp_threshold["sheet_z_range"][1]):
                    continue
                if (current_k + k < self.graph.overlapp_threshold["sheet_k_range"][0]) or (current_k + k > self.graph.overlapp_threshold["sheet_k_range"][1]):
                    continue
                # node already visited
                if node_ in walk and not (node_ == start_node):
                    node_walk_index = walk.index(node_)
                    if ks[node_walk_index] != current_k + k:
                        return None, "small loop closure failed"
                    continue
                elif (node_[:3] in ks_dict) and (current_k + k in ks_dict[node_[:3]]): # k already visited for this volume
                    # start node volume
                    if (node_[:3] == start_node[:3]) and (node_[3] == start_node[3]) and (current_k + k == ks[0]):
                        if steps < min_steps:
                            continue
                        else:
                            break
                    return None, "already visited volume at current k"
                else: # found valid next node
                    break
            node = node_
            if node is None:
                return None, "no next volume"
            walk.append(node)
            current_k += k
            ks.append(current_k)
            if not node[:3] in ks_dict:
                ks_dict[node[:3]] = []
            ks_dict[node[:3]].append(current_k)
            if node[:3] == start_node[:3]:
                if node[3] == start_node[3] and current_k == start_k:
                    if steps < min_steps:
                        return None, "too few steps"
                    else:
                        if self.check_walk(walk): # check if walk is valid over inverse loop closure
                            return walk, ks
                        else:
                            return None, "inverse loop closure failed"
                
            if node[:3] in volume_dict:
                for key_volume in volume_dict[node[:3]].keys():
                    index_ = volume_dict[node[:3]][key_volume]
                    if ks_nodes[index_] == current_k: # same winding
                        if node[3] == key_volume: # same patch
                            if steps >= min_steps: # has enough steps
                                if self.check_walk(walk): # check if walk is valid over inverse loop closure
                                    return walk, ks
                                else:
                                    return None, "inverse loop closure failed"
                        else:
                            return None, "loop closure failed with different nodes for same volume id and k"
                    else:
                        if node[3] == key_volume: # other winding but same patch
                            return None, "loop closure failed with already existing node"
            
        return None, "loop not closed in max_steps"
    
    def check_walk(self, walk):
        for i in range(len(walk) - 1, 0, -1):
            node = walk[i]
            node_next = walk[i-1]
            k = self.graph.get_edge_k(node, node_next) if tuple(node[:3]) == tuple(node_next[:3]) else None
            next_volume = (node_next[0], node_next[1], node_next[2], k)
            bn, _ = self.pick_best_node(walk[i], next_volume)
            if (not bn) or (not all([walk[i-1][u] == bn[u] for u in range(len(bn))])):
                return False
        return True
    
    def pick_start_node(self, nodes, nodes_neighbours_count, ks, picked_nrs, pick_prob):
        mean_ = np.mean(picked_nrs)
        min_ = np.min(picked_nrs)
        min_mean_abs = mean_ - min_
        threshold = min_ + min_mean_abs * 0.25
        mask_threshold = picked_nrs <= threshold
        mask = mask_threshold
        valid_indices = np.where(mask)[0]  # Get indices where mask is True

        assert valid_indices.shape[0] > 0, "No nodes to pick from."

        rand_index = np.random.choice(valid_indices)
        node = nodes[rand_index]
        k = ks[rand_index]
        picked_nrs[rand_index] += 1
        return node, k
    
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

def random_walks():
    path = "/media/julian/SSD4TB/scroll3_surface_points/point_cloud_colorized_verso_subvolume_blocks"
    sample_ratio_score = 0.03 # 0.1
    start_point=[1650, 3300, 5000] # seg 1
    start_point=[1450, 3500, 5000] # seg 2
    start_point=[1350, 3600, 5000] # seg 3
    continue_segmentation = 0
    overlapp_threshold = {"sample_ratio_score": sample_ratio_score, "display": False, "print_scores": True, "picked_scores_similarity": 0.7, "final_score_max": 1.5, "final_score_min": 0.0005, "score_threshold": 0.005, "fit_sheet": False, "cost_threshold": 17, "cost_percentile": 75, "cost_percentile_threshold": 14, 
                          "cost_sheet_distance_threshold": 4.0, "rounddown_best_score": 0.005,
                          "cost_threshold_prediction": 2.5, "min_prediction_threshold": 0.15, "nr_points_min": 200.0, "nr_points_max": 4000.0, "min_patch_points": 300.0, 
                          "winding_angle_range": None, "multiple_instances_per_batch_factor": 1.0,
                          "epsilon": 1e-5, "angle_tolerance": 85, "max_threads": 30,
                          "min_points_winding_switch": 3800, "min_winding_switch_sheet_distance": 9, "max_winding_switch_sheet_distance": 20, "winding_switch_sheet_score_factor": 1.5, "winding_direction": -1.0, "enable_winding_switch": False, "enable_winding_switch_postprocessing": False,
                          "surrounding_patches_size": 3, "max_sheet_clip_distance": 60, "sheet_z_range": (-5000, 400000), "sheet_k_range": (-1, 2), "volume_min_certainty_total_percentage": 0.0, "max_umbilicus_difference": 30,
                          "walk_aggregation_threshold": 100, "walk_aggregation_max_current": -1
                          }
    # scroll 3
    # overlapp_threshold = {"sample_ratio_score": sample_ratio_score, "display": False, "print_scores": True, "picked_scores_similarity": 0.7, "final_score_max": 1.5, "final_score_min": 0.01, "score_threshold": 0.010, "fit_sheet": False, "cost_threshold": 16, "cost_percentile": 75, "cost_percentile_threshold": 12, 
    #                     "cost_sheet_distance_threshold": 4.0, "rounddown_best_score": 0.05,
    #                     "cost_threshold_prediction": 2.0, "min_prediction_threshold": 0.50, "nr_points_min": 300.0, "nr_points_max": 8000.0, "min_patch_points": 400.0, 
    #                     "winding_angle_range": None, "multiple_instances_per_batch_factor": 1.0,
    #                     "epsilon": 1e-5, "angle_tolerance": 85, "max_threads": 30,
    #                     "min_points_winding_switch": 3800, "min_winding_switch_sheet_distance": 9, "max_winding_switch_sheet_distance": 20, "winding_switch_sheet_score_factor": 1.5, "winding_direction": 1.0, "enable_winding_switch": False, "enable_winding_switch_postprocessing": False,
    #                     "surrounding_patches_size": 3, "max_sheet_clip_distance": 60, "sheet_z_range": (-5000, 400000), "sheet_k_range": (-1, 2), "volume_min_certainty_total_percentage": 0.015, "max_umbilicus_difference": 20,
    #                     "walk_aggregation_threshold": 100, "walk_aggregation_max_current": -1
    #                     }
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
    # if args.max_unchanged_walks != max_unchanged_walks:
    #     max_unchanged_walks = args.max_unchanged_walks
    # else:
    #     max_unchanged_walks = 30 * max_nr_walks
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


    umbilicus_path = os.path.dirname(path) + "/umbilicus.txt"
    start_block, patch_id = find_starting_patch([start_point], path)

    save_path = os.path.dirname(path) + f"/{start_point[0]}_{start_point[1]}_{start_point[2]}/" + path.split("/")[-1]
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Configs: {overlapp_threshold}")
    print(f"Args: {args}")
    
    # Build graph
    if recompute:
        scroll_graph = ScrollGraph(7, overlapp_threshold, limit_stickiness=0.6)
        start_block, patch_id = scroll_graph.build_graph(path, num_processes=30, start_point=start_point, distance=-1)
        print("Saving built graph...")
        scroll_graph.save_graph(path.replace("blocks", "graph_raw") + ".pkl")

    if recompute and compute_cpp_translation:
        scroll_graph = load_graph(path.replace("blocks", "graph_raw") + ".pkl")
        solver = RandomWalkSolver(scroll_graph, umbilicus_path)
        print("Computing cpp translation...")
        res = solver.translate_data_to_cpp(recompute_translation=True)
        # Save data to graph object
        solver.graph.cpp_translation = path
        solver.graph.save_graph(path.replace("blocks", "graph_raw") + ".pkl")
        for i in range(1, len(res)):
            # save np
            np.save(path.replace("blocks", "graph_RW") + "_nodes_cpp_" + str(i) + ".npy", res[i])
        print("Saved cpp translation.")
        
    solve_graph = True
    if solve_graph:
        scroll_graph = load_graph(path.replace("blocks", "graph_raw") + ".pkl")
        if continue_segmentation:
            nodes = np.load(save_path.replace("blocks", "graph_RW") + "_nodes.npy")
            ks = np.load(save_path.replace("blocks", "graph_RW") + "_ks.npy")
            # scroll_graph = load_graph(save_path.replace("blocks", "graph_RW_solved") + ".pkl")
        else:
            nodes = None
            ks = None
        scroll_graph.set_overlapp_threshold(overlapp_threshold)
        scroll_graph.start_block, scroll_graph.patch_id = start_block, patch_id
        # start_block, patch_id = scroll_graph.start_block, scroll_graph.patch_id
        starting_node = tuple((*start_block, patch_id))

        solver = RandomWalkSolver(scroll_graph, umbilicus_path)
        solver.save_overlapp_threshold()
        nodes, ks = solver.solve_(path=save_path, starting_node=starting_node, max_nr_walks=max_nr_walks, max_unchanged_walks=max_unchanged_walks, max_steps=max_steps, max_tries=max_tries, min_steps=min_steps, min_end_steps=min_end_steps, continue_walks=continue_segmentation, nodes=nodes, ks=ks)
        # save graph ks and nodes
        np.save(save_path.replace("blocks", "graph_RW") + "_ks.npy", ks)
        np.save(save_path.replace("blocks", "graph_RW") + "_nodes.npy", nodes)
        scroll_graph.save_graph(save_path.replace("blocks", "graph_RW_solved") + ".pkl")
    else:
        ks = np.load(path.replace("blocks", "graph_RW") + "_ks.npy")
        nodes = np.load(path.replace("blocks", "graph_RW") + "_nodes.npy")
        scroll_graph = load_graph(path.replace("blocks", "graph_RW_solved") + ".pkl")
        scroll_graph.overlapp_threshold = overlapp_threshold
    
    if True:
        print("Creating sheet...")
        w2s = WalkToSheet(scroll_graph, nodes, ks, path, save_path, overlapp_threshold)
        main_sheet = w2s.create_sheet()
        print("Saving sheet...")
        w2s.save(main_sheet)


if __name__ == '__main__':
    random_walks()