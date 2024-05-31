import numpy as np
import os
import shutil
import open3d as o3d
from tqdm import tqdm
import time
import threading
import pickle
from copy import deepcopy

# Custom imports
from .Random_Walks import load_graph, ScrollGraph
from .slim_uv import Flatboi, print_array_to_file

# import colormap from matplotlib
import matplotlib

import sys
sys.path.append('ThaumatoAnakalyptor/sheet_generation/build')
import pointcloud_processing

from PIL import Image
# This disables the decompression bomb protection in Pillow
Image.MAX_IMAGE_PIXELS = None

def load_graph(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

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

class WalkToSheet():
    def __init__(self, graph, path, start_point=[3164, 3476, 3472], scale_factor=1.0):
        self.scale_factor = scale_factor
        self.graph = graph
        self.path = path
        self.save_path = os.path.dirname(path) + f"/{start_point[0]}_{start_point[1]}_{start_point[2]}/" + path.split("/")[-1]
        self.lock = threading.Lock()

    def build_points(self):
        # Building the pointcloud 4D (position) + 3D (Normal) + 3D (Color, randomness) representation of the graph
        points = []
        normals = []
        colors = []

        sheet_infos = []
        for node in tqdm(self.graph.nodes, desc="Building points"):
            if 'assigned_k' not in self.graph.nodes[node]:
                continue
            winding_angle = self.graph.nodes[node]['winding_angle']
            block, patch_id = node[:3], node[3]
            patch_sheet_patch_info = (block, int(patch_id), winding_angle)
            sheet_infos.append(patch_sheet_patch_info)
        time_start = time.time()
        points, normals, colors = pointcloud_processing.load_pointclouds(sheet_infos, self.path, True)
        print(f"Time to load pointclouds: {time.time() - time_start}")
        print(f"Shape of patch_points: {np.array(points).shape}")

        # print first 5 points
        for i in range(5):
            print(f"Point {i}: {points[i]}")
            print(f"Normal {i}: {normals[i]}")
            print(f"Color {i}: {colors[i]}")

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

    def points_at_winding_angle(self, points, winding_angle, max_angle_diff=30):
        """
        Assumes points are sorted by winding angle ascendingly
        """
        # extract all points indices with winding angle within max_angle_diff of winding_angle
        start_index = np.searchsorted(points[:, 3], winding_angle - max_angle_diff, side='left')
        end_index = np.searchsorted(points[:, 3], winding_angle + max_angle_diff, side='right')
        return start_index, end_index
    
    def extract_all_same_vector(self, angle_vector, vector):
        indices = []
        for i in range(len(angle_vector)):
            if np.allclose(angle_vector[i], vector):
                indices.append(i)
        return indices
    
    # Calculate initial means of each list, handle empty lists by setting means to None
    def calculate_means(self, ts_lists):
        res = []
        for ts in ts_lists:
            # for t in ts:
            #     if len(t) > 0:
            #         print(f"t: {t}")
            res_ = [np.mean(t) if len(t) > 0 else None for t in ts]
            res.append(res_)
        return res
    
    def compute_means_adjacent(self, adjacent_ts, adjacent_normals, winding_direction):
        # Copy to preserve original lists
        original_ts = deepcopy(adjacent_ts)

        # Create dictionaries to map each t to its corresponding normal
        t_normals_dict_list = []
        for ts, normals in zip(adjacent_ts, adjacent_normals):
            dict_list = []
            for t, normal in zip(ts, normals):
                dict_ts = {}
                for t_, normal_ in zip(t, normal):
                    dict_ts[t_] = normal_
                dict_list.append(dict_ts)
            t_normals_dict_list.append(dict_list)
        
        t_means = self.calculate_means(adjacent_ts)

        # Function to refine means based on adjacent means
        def refine_means(t_means):
            for u in range(len(original_ts)):
                for i in range(len(original_ts[u])):
                    # if t_means[u][i] is None:
                    #     continue

                    # Determine the valid next mean
                    next_mean = next((t_means[j][i] for j in range(u-1, -1, -1) if t_means[j][i] is not None), None)
                    # Determine the valid previous mean
                    prev_mean = next((t_means[j][i] for j in range(u+1, len(t_means)) if t_means[j][i] is not None), None)
                    # Filter t values based on previous and next means
                    if not winding_direction:
                        prev_mean, next_mean = next_mean, prev_mean
                    adjacent_ts[u][i] = [t for t in original_ts[u][i] if ((prev_mean is None or t > prev_mean) or (u == len(t_means) - 1) or (u > 0 and len(original_ts) == 0)) and ((next_mean is None or t < next_mean) or (u == 0) or (u < len(original_ts) - 1 and len(original_ts[u+1][i]) == 0))]

            # Recalculate means after filtering
            return self.calculate_means(adjacent_ts)
        
        # Function to refine means based on adjacent means
        def final_refine_means(t_means):
            for u in range(len(original_ts)):
                for i in range(len(original_ts[u])):
                    # Determine the valid next mean
                    next_mean = next((t_means[j][i] for j in range(u-1, -1, -1) if t_means[j][i] is not None), None)
                    # Determine the valid previous mean
                    prev_mean = next((t_means[j][i] for j in range(u+1, len(t_means)) if t_means[j][i] is not None), None)
                    # Filter t values based on previous and next means
                    if not winding_direction:
                        prev_mean, next_mean = next_mean, prev_mean
                    adjacent_ts[u][i] = [t for t in original_ts[u][i] if ((prev_mean is not None and t > prev_mean) or (u == len(t_means) - 1) or (u > 0 and len(original_ts) == 0)) and ((next_mean is not None and t < next_mean) or (u == 0) or (u < len(original_ts) - 1 and len(original_ts[u+1][i]) == 0))]

            # Recalculate means after filtering
            return self.calculate_means(adjacent_ts)

        # Iteratively refine means until they are in the correct order
        max_iterations = 10  # Limit iterations to prevent infinite loop
        for _ in range(max_iterations):
            previous_means = t_means[:]
            t_means = refine_means(t_means)
            if t_means == previous_means:  # Stop if no change
                break
        t_means = final_refine_means(t_means)

        normals_means = []
        for i, ts in enumerate(adjacent_ts):
            normals_means.append([])
            for e, t in enumerate(ts):
                filtered_normals = [t_normals_dict_list[i][e][t_] for t_ in t]
                normals_means[i].append(np.mean(filtered_normals, axis=0) if len(filtered_normals) > 0 else None)

        # Check for all good t values
        z_len = len(t_means[0])
        for u in range(z_len):
            last_t = None
            for i in range(len(t_means)):
                if t_means[i][u] is not None:
                    if last_t is not None:
                        if winding_direction and t_means[i][u] >= last_t:
                            print(f"Something is wrong with t values: {t_means[i][u]} < {last_t}")
                            t_means[i][u] = None
                            continue
                        elif not winding_direction and t_means[i][u] <= last_t:
                            print(f"Something is wrong with t values: {t_means[i][u]} > {last_t}")
                            t_means[i][u] = None
                            continue
                    last_t = t_means[i][u]

        return t_means, normals_means     
    
    def find_inner_outermost_winding_direction(self, t_means, angle_vector):
        # Fill out all the None values in t_means and normals_means
        # compute mean ts value of furthest out t values
        mean_outermost_ts = 0.0
        count_outermost = 0
        mean_innermost_ts = 0.0
        count_innermost = 0
        start_outer = angle_vector[0]
        for i in range(len(t_means)):
            if (i != 0) and np.allclose(np.array(angle_vector[i]), np.array(start_outer)):
                break
            for u in range(len(t_means[i])):
                if t_means[i][u] is not None:
                    mean_outermost_ts += t_means[i][u]
                    count_outermost += 1
        start_inner = angle_vector[-1]
        for i in range(len(t_means)-1, -1, -1):
            if (i != len(t_means)-1) and np.allclose(np.array(angle_vector[i]), np.array(start_inner)):
                break
            for u in range(len(t_means[i])):
                if t_means[i][u] is not None:
                    mean_innermost_ts += t_means[i][u]
                    count_innermost += 1
        if count_outermost > 0:
            mean_outermost_ts /= count_outermost
        else:
            print("No outermost t values found.")
        if count_innermost > 0:
            mean_innermost_ts /= count_innermost
        else:
            print("No innermost t values found.")

        winding_direction = True
        if mean_innermost_ts > mean_outermost_ts:
            mean_innermost_ts, mean_outermost_ts = mean_outermost_ts, mean_innermost_ts
            winding_direction = False
            print("Winding direction is reversed.")
        else:
            print("Winding direction is normal.")

        print(f"Mean innermost: {mean_innermost_ts}, mean outermost: {mean_outermost_ts}")
        return mean_innermost_ts, mean_outermost_ts, winding_direction

    def initial_full_pointset(self, t_means, normals_means, angle_vector, mean_innermost_ts, mean_outermost_ts, winding_direction):
        mean_dist = mean_innermost_ts - mean_outermost_ts
        
        def adjacent_means(position, t_mean):
            # Determine the valid next mean
            next_mean = None
            next_ind = None
            for i in range(position-1, -1, -1):
                if t_mean[i] is not None:
                    next_mean = t_mean[i]
                    next_ind = i
                    break
            # Determine the valid previous mean
            prev_mean = None
            prev_ind = None
            for i in range(position+1, len(t_mean)):
                if t_mean[i] is not None:
                    prev_mean = t_mean[i]
                    prev_ind = i
                    break

            # Reverse if winding direction is reversed
            if not winding_direction:
                prev_mean, next_mean = next_mean, prev_mean
                prev_ind, next_ind = next_ind, prev_ind

            length_ts = len(t_mean)
            # use outer_mean and inner_mean if prev and next are both none
            if prev_mean is None and next_mean is None:
                prev_mean = mean_innermost_ts
                prev_ind = length_ts - 1 if winding_direction else 0
                next_mean = mean_outermost_ts
                next_ind = 0 if winding_direction else length_ts - 1
            # Filter based on previous mean
            elif prev_mean is None:
                prev_mean = next_mean + mean_dist
                if prev_mean > 0.0:
                    prev_mean = 0.0
                prev_ind = length_ts - 1 if winding_direction else 0
            # Filter based on next mean
            elif next_mean is None:
                next_mean = prev_mean - mean_dist
                if next_mean > 0.0:
                    next_mean = 0.0
                next_ind = 0 if winding_direction else length_ts - 1
            
            return prev_mean, prev_ind, next_mean, next_ind

        computed_vectors_set = set()
        interpolated_ts = []
        interpolated_normals = []
        fixed_points = []
        for i in range(len(t_means)):
            interpolated_ts.append([None]*len(t_means[i]))
            interpolated_normals.append([None]*len(t_means[i]))
            fixed_points.append([False]*len(t_means[i]))
        
        for i in range(len(t_means)):
            curve_angle_vector = angle_vector[i]
            if not tuple(curve_angle_vector) in computed_vectors_set:
                computed_vectors_set.add(tuple(curve_angle_vector))
            else:
                continue
            # get all indices with the same angle vector
            same_vector_indices = self.extract_all_same_vector(angle_vector, curve_angle_vector)
            normal = unit_vector(curve_angle_vector)
            same_vector_ts = [t_means[j] for j in same_vector_indices]
            for e, j in enumerate(same_vector_indices):
                for z in range(len(t_means[j])):
                    if t_means[j][z] is not None:
                        interpolated_ts[j][z] = t_means[j][z]
                        interpolated_normals[j][z] = normals_means[j][z]
                        fixed_points[j][z] = True
                        continue
                    adjacent_vector_ts = [same_vector_ts[a][z] for a in range(len(same_vector_ts))]
                    prev_mean, prev_ind, next_mean, next_ind = adjacent_means(e, adjacent_vector_ts)
                    
                    # Interpolate between the previous and next mean
                    if (next_ind - prev_ind) != 0:
                        interpolated_t = (next_mean - prev_mean) / (next_ind - prev_ind) * (e - prev_ind) + prev_mean
                    else:
                        interpolated_t = next_mean
                    interpolated_ts[j][z] = interpolated_t
                    interpolated_normals[j][z] = normal
                    fixed_points[j][z] = False

        # Check interpolated
        for i in range(len(interpolated_ts)):
            curve_angle_vector = angle_vector[i]
            # get all indices with the same angle vector
            same_vector_indices = self.extract_all_same_vector(angle_vector, curve_angle_vector)
            i_pos_in_same_vector = same_vector_indices.index(i)
            for j in range(len(interpolated_ts[i])):
                if interpolated_ts[i][j] is None:
                    print(f"Interpolated ts is None at {i}, {j}")
                if winding_direction:
                    if i_pos_in_same_vector > 0 and interpolated_ts[i][j] >= interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]:
                        print(f"low side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not >= {interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]}")
                    if i_pos_in_same_vector < len(same_vector_indices) - 1 and interpolated_ts[i][j] <= interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]:
                        print(f"high side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not <= {interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]}")
                else:
                    if i_pos_in_same_vector > 0 and interpolated_ts[i][j] <= interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]:
                        print(f"low side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not <= {interpolated_ts[same_vector_indices[i_pos_in_same_vector-1]][j]}")
                    if i_pos_in_same_vector < len(same_vector_indices) - 1 and interpolated_ts[i][j] >= interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]:
                        print(f"high side: Interpolated ts is not sorted at {i}, {j} with {interpolated_ts[i][j]} not >= {interpolated_ts[same_vector_indices[i_pos_in_same_vector+1]][j]}")

        return interpolated_ts, interpolated_normals, fixed_points
    
    def deduct_ordered_pointset_neighbours(self, ordered_pointset, angle_vector, winding_direction):
        # create a dictionary with the indices of the points in the ordered pointset as keys and a list of the indices of the neighbouring points as values
        neighbours_dict = {}
        angle_set = set()
        for i in range(len(ordered_pointset)):
            if tuple(angle_vector[i]) in angle_set:
                continue
            angle_set.add(tuple(angle_vector[i]))
            same_vector_indices = self.extract_all_same_vector(angle_vector, angle_vector[i])
            for e, j in enumerate(same_vector_indices):
                for k in range(len(ordered_pointset[j])):
                    assert ordered_pointset[j][k] is not None, f"Point at {j}, {k} is None"
                    dict_key = (j, k)
                    if dict_key not in neighbours_dict:
                        neighbours_dict[dict_key] = {"front_back": [], "top_bottom": [], "left": None, "right": None}
                    # append top and bottom neighbours
                    for l in range(-1, 2):
                        if l == 0:
                            continue
                        if k + l >= 0 and k + l < len(ordered_pointset[j]):
                            assert ordered_pointset[j][k+l] is not None, f"Point at {j}, {k+l} is None"
                            neighbours_dict[dict_key]["top_bottom"].append((j, k+l))
                    # append left and right neighbours
                    for l in range(-1, 2):
                        if l == 0:
                            continue
                        if e + l >= 0 and e + l < len(same_vector_indices):
                            assert ordered_pointset[same_vector_indices[e+l]][k] is not None, f"Point at {same_vector_indices[e+l]}, {k} is None"
                            if l == -1:
                                neighbours_dict[dict_key]["left"] = (same_vector_indices[e+l], k)
                            else:
                                neighbours_dict[dict_key]["right"] = (same_vector_indices[e+l], k)
                    # swap left and right if winding direction is reversed
                    if not winding_direction:
                        neighbours_dict[dict_key]["left"], neighbours_dict[dict_key]["right"] = neighbours_dict[dict_key]["right"], neighbours_dict[dict_key]["left"]
                    # append front and back neighbours
                    for l in range(-1, 2):
                        if l == 0:
                            continue
                        if j + l >= 0 and j + l < len(ordered_pointset):
                            assert ordered_pointset[j+l][k] is not None, f"Point at {j+l}, {k} is None"
                            neighbours_dict[dict_key]["front_back"].append((j+l, k))
        return neighbours_dict
    
    def value_from_key(self, key, ordered_pointset):
        if key is None:
            return None
        else:
            return ordered_pointset[key[0]][key[1]]
                    
    def calculate_ms(self, ordered_pointset, neighbours_dict):
        m_r = [None]*len(ordered_pointset)
        m_l = [None]*len(ordered_pointset)
        m_ts = [None]*len(ordered_pointset)
        r = [None]*len(ordered_pointset)
        l = [None]*len(ordered_pointset)
        for i in range(len(ordered_pointset)):
            m_r[i] = [None]*len(ordered_pointset[i])
            m_l[i] = [None]*len(ordered_pointset[i])
            m_ts[i] = [None]*len(ordered_pointset[i])
            r[i] = [None]*len(ordered_pointset[i])
            l[i] = [None]*len(ordered_pointset[i])
        for i in range(len(ordered_pointset)):
            for k in range(len(ordered_pointset[i])):
                dict_key = (i, k)
                neighbours = neighbours_dict[dict_key]
                l[i][k] = neighbours["left"]
                r[i][k] = neighbours["right"]
                l[i][k] = self.value_from_key(l[i][k], ordered_pointset)
                assert l[i][k] is None or l[i][k] <= 0.0, f"l[i][k] is not less than 0: {l[i][k]}"
                r[i][k] = self.value_from_key(r[i][k], ordered_pointset)
                assert r[i][k] is None or r[i][k] <= 0.0, f"r[i][k] is not less than 0: {r[i][k]}"
                fb = neighbours["front_back"]
                tb = neighbours["top_bottom"]
                same_sheet_neighbours = fb + tb
                m_r_ = 0.0
                m_l_ = 0.0
                m_ts_ = 0.0
                count_r = 0
                count_l = 0
                for n in same_sheet_neighbours:
                    ts_n = ordered_pointset[n[0]][n[1]]
                    assert ts_n <= 0.0, f"ts_n is not less than 0: {ts_n}"
                    m_ts_ += ts_n

                    l_n = neighbours_dict[n]["left"]
                    r_n = neighbours_dict[n]["right"]

                    l_n = self.value_from_key(l_n, ordered_pointset)
                    r_n = self.value_from_key(r_n, ordered_pointset)

                    if l_n is not None:
                        m_l_ += (l_n - ts_n)
                        count_l += 1
                    if r_n is not None:
                        m_r_ += (r_n - ts_n)
                        count_r += 1

                if count_r == 0:
                    m_r_ = None
                else:
                    m_r_ /= count_r
                if count_l == 0:
                    m_l_ = None
                else:
                    m_l_ /= count_l
                m_ts_ /= len(same_sheet_neighbours)
                m_r[i][k] = m_r_
                m_l[i][k] = m_l_
                m_ts[i][k] = m_ts_

        return r, l, m_r, m_l, m_ts
    
    def solve_for_t_individual(self, r, l, m_r, m_l, m_ts, a=0.01):
        t_ts = m_ts
        t_total = t_ts
        count_total = a
        if (r is not None) and (m_r is not None):
            t_r = r - m_r
            t_total += t_r
            count_total += 1.0
        if (l is not None) and (m_l is not None):
            t_l = l - m_l
            t_total += t_l
            count_total += 1.0
        t_total /= count_total

        if t_total > 0.0:
            t_total = 0.0

        return t_total
    
    def respect_non_overlapping(self, i, j, neighbours_dict, interpolated_ts, new_ts_d):
        # respect the non-overlapping invariant of the ordered pointset
        def side_of(ts_, n):
            # calculate what side ts_ is wrt to n
            return ts_ > n, ts_ == n
        old_ts = interpolated_ts[i][j]
        l = neighbours_dict[(i, j)]["left"]
        r = neighbours_dict[(i, j)]["right"]
        l = self.value_from_key(l, interpolated_ts)
        r = self.value_from_key(r, interpolated_ts)
        if l is not None:
            side_old, invalid = side_of(old_ts, l)
            assert not invalid, f"old_ts: {old_ts}, l: {l}, side: {side_old}"
            side_new, invalid = side_of(old_ts + new_ts_d, l)
            if side_old != side_new:
                new_ts_d = l - old_ts
                # only go 50% of the way
                new_ts_d *= 0.5
        if r is not None:
            side_old, invalid = side_of(old_ts, r)
            assert not invalid, f"old_ts: {old_ts}, r: {r}, side: {side_old}"
            side_new, invalid = side_of(old_ts + new_ts_d, r)
            if side_old != side_new:
                new_ts_d = r - old_ts
                # only go 50% of the way
                new_ts_d *= 0.5
        return new_ts_d
    
    def compute_interpolated_adjacent(self, neighbours_dict, interpolated_ts, fixed_points):
        r, l, m_r, m_l, m_ts = self.calculate_ms(interpolated_ts, neighbours_dict)
        error_val = 0.0

        new_interpolated_ts = deepcopy(interpolated_ts)

        # Compute interpolated values for all points
        for i in range(len(interpolated_ts)):
            for j in range(len(interpolated_ts[i])):
                if not fixed_points[i][j]:
                    t = self.solve_for_t_individual(r[i][j], l[i][j], m_r[i][j], m_l[i][j], m_ts[i][j], a=1.0)
                    assert t <= 0.0, f"t is not less than 0: {t}"
                    d_t = t - interpolated_ts[i][j]
                    # the raw d_t might be breaking the non-overlapping invariant
                    d_t = self.respect_non_overlapping(i, j, neighbours_dict, interpolated_ts, d_t)
                    error_val += abs(d_t)
                    new_interpolated_ts[i][j] = interpolated_ts[i][j] + 0.1 * d_t
        
        return new_interpolated_ts, error_val

    def interpolate_ordered_pointset(self, ordered_pointset, ordered_normals, ordered_umbilicus_points, angle_vector, winding_direction):
        computed_vectors_set = set()
        result_ts = []
        result_normals = []
        for i in range(len(ordered_pointset)):
            result_ts.append([None]*len(ordered_pointset[i]))
            result_normals.append([None]*len(ordered_pointset[i]))
            
        for i in tqdm(range(len(ordered_pointset))):
            curve_angle_vector = angle_vector[i]
            if tuple(curve_angle_vector) in computed_vectors_set:
                continue
            computed_vectors_set.add(tuple(curve_angle_vector))
            # get all indices with the same angle vector
            same_vector_indices = self.extract_all_same_vector(angle_vector, curve_angle_vector)
            same_vector_ts = [ordered_pointset[j] for j in same_vector_indices]
    
            same_vector_normals = [ordered_normals[j] for j in same_vector_indices]
            t_means, normals_means = self.compute_means_adjacent(same_vector_ts, same_vector_normals, winding_direction)

            # print(f"length of t_means: {len(t_means)}")
            for e, j in enumerate(same_vector_indices):
                result_ts[j] = t_means[e]
                result_normals[j] = normals_means[e]
                # # Only for debugging without subsequent initial pointset and optimization
                # for o in range(len(t_means[e])):
                #     if t_means[e][o] is None:
                #         result_ts[j][o] = 0.0
                #         result_normals[j][o] = np.array([1.0, 0.0, 0.0])

        return result_ts, result_normals
    
    def ordered_pointset_to_points(self, ordered_pointset, ordered_normals, ordered_umbilicus_points, angle_vector):
        pointset = []

        for i in tqdm(range(len(ordered_pointset))):
            ordered_curve = ordered_pointset[i]
            ordered_normals_curve = ordered_normals[i]
            ordered_umbilicus_curve = ordered_umbilicus_points[i]
            curve_angle_vector = angle_vector[i]
            for j in range(len(ordered_curve)):
                umbilicus_p = np.array(ordered_umbilicus_curve[j])
                np_angle_vec = np.array(curve_angle_vector)
                # print(f"shape of umbilicus_p: {umbilicus_p.shape}, shape of curve_angle_vector: {np_angle_vec.shape}")
                point = umbilicus_p + ordered_curve[j] * np_angle_vec
                point = np.array(point)
                normal = ordered_normals_curve[j]
                normal = np.array(normal)
                pointset.append(([point], [normal]))

        # remove test folder
        test_folder = os.path.join(self.save_path, "test_cpp")
        if os.path.exists(test_folder):
            shutil.rmtree(test_folder)
        os.makedirs(test_folder)
        test_pointset_ply_path = os.path.join(test_folder, f"ordered_pointset_test_cpp.ply")
        self.pointcloud_from_ordered_pointset(pointset, os.path.join(self.save_path, test_pointset_ply_path))
            
    def rolled_ordered_pointset(self, points, normals, debug=False):
        # get winding angles
        winding_angles = points[:, 3]

        min_wind = np.min(winding_angles)
        max_wind = np.max(winding_angles)
        print(f"Min and max winding angles: {min_wind}, {max_wind}")

        # some debugging visualization of seperate pointcloud windings
        produce_test_pointclouds_windings = False
        if produce_test_pointclouds_windings:
            # remove test folder
            test_folder = os.path.join(self.save_path, "test_winding_angles")
            if os.path.exists(test_folder):
                shutil.rmtree(test_folder)
            os.makedirs(test_folder)
            # test
            for test_angle in range(int(min_wind), int(max_wind), 360):
                start_index, end_index = self.points_at_winding_angle(points, test_angle+180, max_angle_diff=180)
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
        
        print("Using Cpp rolled_ordered_pointset")
        # Set to false to load precomputed partial results during development
        fresh_start = True
        if fresh_start:
            result = pointcloud_processing.create_ordered_pointset(points, normals, self.graph.umbilicus_data)
            # save result as pkl
            result_pkl_path = os.path.join(self.save_path, "ordered_pointset.pkl")
            with open(result_pkl_path, 'wb') as f:
                pickle.dump(result, f)
        else:
            result_pkl_path = os.path.join(self.save_path, "ordered_pointset.pkl")
            with open(result_pkl_path, 'rb') as f:
                result = pickle.load(f)

        print("length of result: ", len(result))
        ordered_pointset, ordered_normals, ordered_umbilicus_points, angle_vector = zip(*result)
        print("length of ordered_pointset: ", len(ordered_pointset))

        # determine winding direction
        t_means = self.calculate_means(ordered_pointset)
        mean_innermost_ts, mean_outermost_ts, winding_direction = self.find_inner_outermost_winding_direction(t_means, angle_vector)

        # Set to false to load precomputed partial results during development
        fresh_start2 = True
        if fresh_start2:

            result_ts, result_normals = self.interpolate_ordered_pointset(ordered_pointset, ordered_normals, ordered_umbilicus_points, angle_vector, winding_direction)
            interpolated_ts, interpolated_normals = result_ts, result_normals
            # save result as pkl
            result_pkl_path = os.path.join(self.save_path, "results_ts_normals.pkl")
            with open(result_pkl_path, 'wb') as f:
                pickle.dump((result_ts, result_normals), f)
        else:
            result_pkl_path = os.path.join(self.save_path, "results_ts_normals.pkl")
            with open(result_pkl_path, 'rb') as f:
                (result_ts, result_normals) = pickle.load(f)

        # interpolate initial full pointset. After this step there exists an "ordered pointset" prototype without any None values
        interpolated_ts, interpolated_normals, fixed_points = self.initial_full_pointset(result_ts, result_normals, angle_vector, mean_innermost_ts, mean_outermost_ts, winding_direction)

        # Calculate for each point in the ordered pointset its neighbouring indices (3d in a 2d list). on same sheet, top bottom, front back, adjacent sheets neighbours: left right
        neighbours_dict = self.deduct_ordered_pointset_neighbours(interpolated_ts, angle_vector, winding_direction)

        # Optimize the full pointset for smooth surface with best guesses for interpolated t values
        error_val_d = 1.0
        last_error_val = None
        for i in tqdm(range(10000), desc="Optimizing full pointset"):
            interpolated_ts, error_val = self.compute_interpolated_adjacent(neighbours_dict, interpolated_ts, fixed_points)
            print(f"Error value: {error_val}")
            if last_error_val is not None and abs(last_error_val - error_val) < error_val_d:
                break
            last_error_val = error_val

        # go from interpolated t values to ordered pointset (3D points)
        interpolated_points = []
        for i in range(len(interpolated_ts)):
            interpolated_points.append([])
            for j in range(len(interpolated_ts[i])):
                if interpolated_ts[i][j] is not None:
                    interpolated_points[i].append(np.array(ordered_umbilicus_points[i][j]) + interpolated_ts[i][j] * np.array(angle_vector[i]))
                else:
                    raise ValueError("Interpolated t is None")

        print("Finished Cpp rolled_ordered_pointset")

        # Creating ordered pointset output format
        ordered_pointsets_final = []
        for i in range(len(interpolated_points)):
            ordered_pointsets_final.append((np.array(interpolated_points[i]), np.array(interpolated_normals[i])))

        self.ordered_pointset_to_points(interpolated_ts, interpolated_normals, ordered_umbilicus_points, angle_vector)

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
        # scale vertices back to CT scan coordinates
        vertices[:, :3] = self.scale_factor * vertices[:, :3]
        
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

    def unroll(self, debug=False):
        # Set to false to load precomputed partial results during development
        start_fresh = True
        if start_fresh: 
            # Set to false to load precomputed partial results during development
            start_fresh_build_points = True
            if start_fresh_build_points:
                # get points
                points, normals, colors = self.build_points()

                # Make directory if it doesn't exist
                os.makedirs(self.save_path, exist_ok=True)
                # Save as npz
                with open(os.path.join(self.save_path, "points.npz"), 'wb') as f:
                    np.savez(f, points=points, normals=normals, colors=colors)
            else:
                # Open the npz file
                with open(os.path.join(self.save_path, "points.npz"), 'rb') as f:
                    npzfile = np.load(f)
                    points = npzfile['points']
                    normals = npzfile['normals']
                    colors = npzfile['colors']

            points_originals_selected = points
            normals_originals_selected = normals

            # Save as npz
            with open(os.path.join(self.save_path, "points_selected.npz"), 'wb') as f:
                np.savez(f, points=points_originals_selected, normals=normals_originals_selected)
        else:
            # Open the npz file
            with open(os.path.join(self.save_path, "points_selected.npz"), 'rb') as f:
                npzfile = np.load(f)
                points_originals_selected = npzfile['points']
                normals_originals_selected = npzfile['normals']

        print(f"Shape of points_originals_selected: {points_originals_selected.shape}")

        pointset_ply_path = os.path.join(self.save_path, "ordered_pointset.ply")
        # get nodes
        ordered_pointsets = self.rolled_ordered_pointset(points_originals_selected, normals_originals_selected, debug=debug)

        self.pointcloud_from_ordered_pointset(ordered_pointsets, pointset_ply_path)

        mesh, uv_image = self.mesh_from_ordered_pointset(ordered_pointsets)

        mesh_path = os.path.join(self.save_path, "mesh.obj")
        self.save_mesh(mesh, uv_image, mesh_path)

        # Flatten mesh
        self.flatten(mesh_path)

if __name__ == '__main__':
    start_point = [3164, 3476, 3472]
    import argparse
    parser = argparse.ArgumentParser(description='Unroll a graph to a sheet')
    parser.add_argument('--path', type=str, help='Path to the instances', required=True)
    parser.add_argument('--graph', type=str, help='Path to the graph file from --Path', required=True)
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--start_point', type=int, nargs=3, default=start_point, help='Start point for the unrolling')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Scale factor for the mesh')

    args = parser.parse_args()

    graph_path = os.path.join(os.path.dirname(args.path), args.graph)
    graph = load_graph(graph_path)
    reference_path = graph_path.replace("evolved_graph", "subgraph")
    start_point = args.start_point
    scale_factor = args.scale_factor
    walk = WalkToSheet(graph, args.path, start_point, scale_factor)
    # walk.save_graph_pointcloud(reference_path)
    walk.unroll(debug=args.debug)