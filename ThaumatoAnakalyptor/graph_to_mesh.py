import numpy as np
import os
import json
import open3d as o3d
import tarfile
import tempfile
from tqdm import tqdm
from scipy.interpolate import interp1d
import hdbscan
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement

# Custom imports
from .instances_to_sheets import select_points
from .sheet_to_mesh import umbilicus_xz_at_y
from .sheet_computation import load_graph, ScrollGraph

# import colormap from matplotlib
import matplotlib.pyplot as plt
import matplotlib

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

def build_patch_tar(main_sheet_patch, subvolume_size, path, sample_ratio=1.0):
    """
    Load surface patch from overlapping subvolumes instances predictions.
    """

    # Standardize subvolume_size to a NumPy array
    subvolume_size = np.atleast_1d(subvolume_size).astype(int)
    if subvolume_size.shape[0] == 1:
        subvolume_size = np.repeat(subvolume_size, 3)

    xyz, patch_nr, winding_angle = main_sheet_patch
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
            res = build_patch(winding_angle, tuple(subvolume_size), ply_file_path, sample_ratio=float(sample_ratio))
            return res

class WalkToSheet():
    def __init__(self, graph, path):
        self.graph = graph
        self.path = path
        start_point = [3164, 3476, 3472]
        self.save_path = os.path.dirname(path) + f"/{start_point[0]}_{start_point[1]}_{start_point[2]}/" + path.split("/")[-1]

    def build_points(self):
        # Building the pointcloud 4D (position) + 3D (Normal) + 3D (Color, randomness) representation of the graph
        points = []
        normals = []
        colors = []
        # TODO: multithread this loop
        for node in tqdm(self.graph.nodes, desc="Building points"):
            winding_angle = self.graph.nodes[node]['winding_angle']
            block, patch_id = node[:3], node[3]
            patch_sheet_patch_info = (block, int(patch_id), winding_angle)
            patch_points, patch_normals, patch_color = build_patch_tar(patch_sheet_patch_info, (50, 50, 50), self.path, sample_ratio=1.0)
            points.append(patch_points)
            normals.append(patch_normals)
            colors.append(patch_color)
        
        points = np.concatenate(points, axis=0)
        normals = np.concatenate(normals, axis=0)
        colors = np.concatenate(colors, axis=0)

        # Remove indices that are identical in the first three point dimensions
        unique_indices = np.unique(points[:, :3], axis=0, return_index=True)[1]
        print(f"Unique indices: {unique_indices.shape[0]}")
        points = points[unique_indices]
        normals = normals[unique_indices]
        colors = colors[unique_indices]

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
            indices = self.points_at_winding_angle(points, angle_extraction, max_angle_diff=90)
            points_ = points[indices]
            normals_ = normals[indices]
            colors_ = colors[indices]

            # filter points based on the largest connected component
            mask = self.filter_points_clustering_half_windings(points_[:,:3], max_single_dist=max_single_dist, min_cluster_size=min_cluster_size)

            points_ = points_[mask]
            normals_ = normals_[mask]
            colors_ = colors_[mask]
            # only get core part
            indices = self.points_at_winding_angle(points_, angle_extraction, max_angle_diff=45)
            points_ = points_[indices]
            normals_ = normals_[indices]
            colors_ = colors_[indices]

            points_list.append(points_)
            normals_list.append(normals_)
            colors_list.append(colors_)

        points = np.concatenate(points_list, axis=0)
        normals = np.concatenate(normals_list, axis=0)
        colors = np.concatenate(colors_list, axis=0)

        return points, normals, colors
        

    def filter_points_clustering_half_windings(self, points, max_single_dist=20, min_cluster_size=8000):
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

        for z in z_positions:
            umbilicus_positions = umbilicus_func(z) # shape is 3
            # extract all points at most max_eucledian_distance from the line defined by the umbilicus_positions and the angle vector
            distances = distances_points_to_line(points[:, :3], umbilicus_positions, angle_vector)
            distances2, points_on_line, t = closest_points_and_distances(points[:, :3], umbilicus_positions, angle_vector)
            assert np.allclose(distances, distances2), f"Distance calculation is not correct: {distances} != {distances2}"
            mask = distances2 < max_eucledian_distance
            t_valid_mask = t >= 0
            mask = np.logical_and(mask, np.logical_not(t_valid_mask))

            if np.sum(mask) > 0:
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
        # extract all points indices with winding angle within max_angle_diff of winding_angle
        indices = np.where(np.abs(points[:, 3] - winding_angle) < max_angle_diff)
        return indices

    def rolled_ordered_pointset(self, points, normals):
        # umbilicus position
        umbilicus_func = lambda z: umbilicus_xz_at_y(self.graph.umbilicus_data, z)

        # get winding angles
        winding_angles = points[:, 3]

        min_wind = np.min(winding_angles)
        max_wind = np.max(winding_angles)
        print(f"Min and max winding angles: {min_wind}, {max_wind}")

        min_z = np.min(points[:, 1])
        max_z = np.max(points[:, 1])

        z_positions = np.arange(min_z, max_z, 2)

        ordered_pointsets = []

        # test
        test_angle = -2090
        extracted_points_indices = self.points_at_winding_angle(points, test_angle, max_angle_diff=180)
        points_test = points[extracted_points_indices]
        colors_test = points_test[:,3]
        points_test = points_test[:,:3]
        normals_test = normals[extracted_points_indices]
        print(f"extracted {len(points_test)} points at {test_angle} degrees")
        # save as ply
        ordered_pointsets_test = [(points_test, normals_test)]
        test_pointset_ply_path = os.path.join(self.save_path, "ordered_pointset_test.ply")
        self.pointcloud_from_ordered_pointset(ordered_pointsets_test, test_pointset_ply_path, color=colors_test)

        # TODO: multithread this loop
        # iterate through winding angles
        for winding_angle in tqdm(np.arange(min_wind, max_wind, 2), desc="Winding angles"):
            extracted_points_indices = self.points_at_winding_angle(points, winding_angle) # TODO: optimize this function by first sorting by the winding angle
            extracted_points = points[extracted_points_indices]
            extracted_normals = normals[extracted_points_indices]

            res = self.points_at_angle(extracted_points, extracted_normals, z_positions, winding_angle, max_eucledian_distance=4)
            ordered_pointsets.append(res)

        ordered_pointsets_final = []
        # Interpolate None ordered_pointset from good neighbours, do it by t's on same z values
        indices = [o for o in range(len(ordered_pointsets)) if ordered_pointsets[o] is not None and ordered_pointsets[o][0] is not None]
        for o in range(len(ordered_pointsets)):
            if ordered_pointsets[o][0] is not None: # Good ordered_pointset entry, finalize it
                ordered_pointsets_final.append((ordered_pointsets[o][0], ordered_pointsets[o][1]))
                continue
            ordered_pointset = []
            # interpolate eich height level
            angle_vector = ordered_pointsets[o][3]
            for i, z in enumerate(z_positions):
                t_vals = [ordered_pointsets[u][2][i] for u in range(len(ordered_pointsets)) if ordered_pointsets[u][0] is not None]
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

        # First, convert all pointsets into a single list of vertices
        vertices, normals = [], []
        for point, normal in ordered_pointsets:
            if point is not None:
                vertices.append(point)
                normals.append(normal)
        # Convert to Open3D compatible format
        vertices = np.concatenate(vertices, axis=0)
        normals = np.concatenate(normals, axis=0)


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
                triangles.append([idx + 1, idx + num_cols + 1, idx + num_cols])

        # Create the mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        return mesh
    
    def save_mesh(self, mesh, filename):
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

    def unroll(self):
        # get points
        points, normals, colors = self.build_points()

        # filter points
        # points, normals, colors = self.filter_points_multiple_occurrences(points, normals, colors)
        # points, normals, colors = self.filter_points_clustering(points, normals, colors)

        # self.save_4D_ply(points, normals, os.path.join(self.save_path, "4D_points.ply"))

        # get nodes
        ordered_pointsets = self.rolled_ordered_pointset(points, normals)
        pointset_ply_path = os.path.join(self.save_path, "ordered_pointset.ply")
        self.pointcloud_from_ordered_pointset(ordered_pointsets, pointset_ply_path)

        mesh = self.mesh_from_ordered_pointset(ordered_pointsets)
        mesh_path = os.path.join(self.save_path, "mesh.obj")
        self.save_mesh(mesh, mesh_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Unroll a graph to a sheet')
    parser.add_argument('--path', type=str, help='Path to the instances', required=True)
    parser.add_argument('--graph', type=str, help='Path to the graph file from --Path', required=True)
    args = parser.parse_args()

    graph_path = os.path.join(os.path.dirname(args.path), args.graph)
    graph = load_graph(graph_path)
    walk = WalkToSheet(graph, args.path)
    walk.unroll()