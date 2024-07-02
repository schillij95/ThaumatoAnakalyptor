### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2024

import open3d as o3d
import numpy as np
from collections import deque
from math import atan2, pi, sqrt
from .sheet_to_mesh import load_xyz_from_file, scale_points, shuffling_points_axis
from scipy.interpolate import interp1d
from copy import deepcopy
from tqdm import tqdm
import os

class MeshSplitter:
    def __init__(self, mesh_path, umbilicus_path):
        # Load mesh
        self.mesh_path = mesh_path
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.vertices_np = np.asarray(self.mesh.vertices).copy()  # Create a copy of the vertices as a NumPy array
        self.visited_vertices = set()

        axis_indices = [2, 0, 1]
        self.axis_indices = axis_indices

        # Load the umbilicus data
        umbilicus_data = load_xyz_from_file(umbilicus_path)
        # scale and swap axis
        umbilicus_data = scale_points(umbilicus_data, 1.0, axis_offset=-500)
        umbilicus_data, _ = shuffling_points_axis(umbilicus_data, umbilicus_data, axis_indices)
        # Separate the coordinates
        x, y, z = umbilicus_data.T
        # Create interpolation functions for x and z based on y
        self.fx = interp1d(z, x, kind='linear', fill_value="extrapolate")
        self.fy = interp1d(z, y, kind='linear', fill_value="extrapolate")
        self.interpolate_umbilicus = lambda z: self.umbilicus_xy_at_z(z)

    def umbilicus_xy_at_z(self, z_new):
        """
        Interpolate between points in the provided 2D array based on z values.

        :param z_new: A 1D numpy array of y-values.
        :return: A 2D numpy array with interpolated points for each z value.
        """

        # Calculate interpolated x and z values
        x_new = self.fx(z_new)
        y_new = self.fy(z_new)

        # Return the combined x, y, and z values as a 2D array
        return x_new, y_new, z_new

    def normalize_angle_diff(self, angle_diff):
        if angle_diff > pi:
            angle_diff -= 2 * pi
        elif angle_diff < -pi:
            angle_diff += 2 * pi
        return angle_diff*180/pi

    def angle_between_vertices(self, v1, v2):
        z1 = v1[2]
        umbilicus_xy1 = self.interpolate_umbilicus([z1])
        dx1 = v1[0] - umbilicus_xy1[0][0]
        dy1 = v1[1] - umbilicus_xy1[1][0]

        z2 = v2[2]
        umbilicus_xy2 = self.interpolate_umbilicus([z2])
        dx2 = v2[0] - umbilicus_xy2[0][0]
        dy2 = v2[1] - umbilicus_xy2[1][0]
        
        angle1 = atan2(dy1, dx1)
        angle2 = atan2(dy2, dx2)
        return self.normalize_angle_diff(angle2 - angle1)
    
    def compute_adjacency_list(self):
        """
        Compute the adjacency list for the mesh.
        """
        adjacent_dict = {}
        for triangle in tqdm(np.array(self.mesh.triangles)):
            for i in range(3):
                vertex = int(triangle[i])
                adjacent = int(triangle[(i + 1) % 3])
                if vertex not in adjacent_dict:
                    adjacent_dict[vertex] = set()
                adjacent_dict[vertex].add(adjacent)
                if adjacent not in adjacent_dict:
                    adjacent_dict[adjacent] = set()
                adjacent_dict[adjacent].add(vertex)

        vertices = np.array(self.mesh.vertices)
        for vertex in range(vertices.shape[0]):
            vertex = int(vertex)
            if vertex not in adjacent_dict:
                adjacent_dict[vertex] = []
            else:
                adjacent_dict[vertex] = list(adjacent_dict[vertex])

        self.adjacency_list = adjacent_dict

    def get_adjacent_vertices(self, vertex_idx):
        # Check if the adjacency list exists for the mesh
        if not hasattr(self, 'adjacency_list'):
            print("Generating adjacency list...")
            self.compute_adjacency_list()  # Compute the adjacency list if it doesn't exist
        # Get and return adjacent vertex indices directly
        adjacent = self.adjacency_list[int(vertex_idx)]
        return adjacent

    def compute_uv_with_bfs(self, start_vertex_idx):
        # check
        index_c, count_c, area_c = self.mesh.cluster_connected_triangles()
        print(f"Found {len(area_c)} clusters.")

        bfs_queue = deque()
        processing = set()
        uv_coordinates = {}

        # Set the start vertex and angle
        start_index_angle = self.angle_between_vertices(np.array([0.0, 0.0, 0.0]), self.vertices_np[start_vertex_idx])
        bfs_queue.append((start_vertex_idx, start_index_angle))

        i = 0
        # tqdm progress bar
        pbar = tqdm(total=self.vertices_np.shape[0], desc="BFS Angle Calculation Progress")
        while bfs_queue:
            pbar.update(1)
            i += 1
            vertex_idx, current_angle = bfs_queue.popleft()
            
            if vertex_idx in self.visited_vertices:
                continue

            self.visited_vertices.add(vertex_idx)
            z = self.vertices_np[vertex_idx, 2]
            umbilicus_xy = self.interpolate_umbilicus([z])
            dx = self.vertices_np[vertex_idx, 0] - umbilicus_xy[0][0]
            dy = self.vertices_np[vertex_idx, 1] - umbilicus_xy[1][0]
            distance = sqrt(dx * dx + dy * dy)

            uv_coordinates[vertex_idx] = (current_angle, distance)
            adjacent_vertex_indices = self.get_adjacent_vertices(vertex_idx)
            for next_vertex_idx in adjacent_vertex_indices:
                # if next_vertex_idx in self.visited_vertices:
                #     angle_diff = self.angle_between_vertices(self.vertices_np[vertex_idx], self.vertices_np[next_vertex_idx])
                #     next_angle, _ = uv_coordinates[next_vertex_idx]
                #     assert abs(next_angle - angle_diff - current_angle) < 1e-6, f"Angle difference is not correct: {next_angle} != {angle_diff + current_angle}, current angle: {current_angle}"

                if next_vertex_idx in processing:
                    continue
                angle_diff = self.angle_between_vertices(self.vertices_np[vertex_idx], self.vertices_np[next_vertex_idx])
                bfs_queue.append((next_vertex_idx, current_angle + angle_diff))
                processing.add(next_vertex_idx)

        pbar.close()
        
        for vertex_idx, (u, v) in tqdm(uv_coordinates.items(), desc="UV Coordinates Assignment Progress"):
            self.vertices_np[vertex_idx, 0] = u
            self.vertices_np[vertex_idx, 1] = v
        
        if not self.have_visited_all_vertices():
            # Red writing
            print(f"\033[91mWarning: Not all vertices were visited during BFS! {len(self.visited_vertices)} visited, {self.vertices_np.shape[0]} total.\033[0m")
        else:
            # Green writing
            print("\033[92mAll vertices were visited.\033[0m")

        self.reset_visited()

    def scale_uv_x(self):
        print("Scaling in x direction...")
        self.scale_uv_x_complete()

    # # Old function with less accurate flattening related to z height
    def scale_uv_x_complete(self):
        min_x = np.min(self.vertices_np[:, 0])
        max_x = np.max(self.vertices_np[:, 0])
        print(f"Min and max x: {min_x}, {max_x}")

        window_size = 0.02
        offset_window_average = 7.0
        addition_offset = 0
        processed_vertices = set()
        new_vertices = deepcopy(self.vertices_np)

        # Sort vertices by x-coordinate
        sorted_indices = np.argsort(self.vertices_np[:, 0])
        self.sorted_vertices_np = self.vertices_np[sorted_indices]

        x_additions = []
        
        # window_start = min_x
        # while window_start <= max_x:
        window_start_index = 0
        window_end_index = 0
        window_start_offset_index = 0
        window_end_offset_index = 0
        num_vertices = len(self.vertices_np)
        stop_x_ind = int(np.floor((max_x - min_x) / window_size)) + 1
        for window_start_ind in tqdm(range(stop_x_ind)):
            window_start = window_start_ind * window_size

            # Move window_start_index to the start of the window
            while window_start_index < num_vertices and self.vertices_np[window_start_index, 0] < window_start:
                window_start_index += 1

            # Move window_end_index to the end of the window
            window_end_index = window_start_index
            while window_end_index < num_vertices and self.vertices_np[window_end_index, 0] <= window_start + window_size:
                window_end_index += 1

            # Move window_start_offset_index to the start of the window with offset
            while window_start_offset_index < num_vertices and self.vertices_np[window_start_offset_index, 0] < window_start - offset_window_average:
                window_start_offset_index += 1
            
            # Move window_end_offset_index to the end of the window with offset
            window_end_offset_index = window_start_offset_index
            while window_end_offset_index < num_vertices and self.vertices_np[window_end_offset_index, 0] <= window_start + window_size + offset_window_average:
                window_end_offset_index += 1

            # vertices_mask = (self.vertices_np[:, 0] >= window_start) & (self.vertices_np[:, 0] < window_start + window_size)
            # vertices_mask_offset = (self.vertices_np[:, 0] >= window_start-offset_window_average) & (self.vertices_np[:, 0] < window_start + window_size+offset_window_average)
            # y_values_in_window = self.vertices_np[vertices_mask_offset, 1]

            # avg_y_distance = np.mean(y_values_in_window) if y_values_in_window.size else 0
            avg_y_distance = np.mean(self.vertices_np[window_start_offset_index:window_end_offset_index, 1]) if window_end_offset_index > window_start_offset_index else 0
            x_addition_scale = (window_size / 360.0) * avg_y_distance * 2 * pi
            x_additions.append(x_addition_scale)

            # indices_in_window = np.where(vertices_mask)
            # for idx in indices_in_window[0]:
            for idx in range(window_start_index, window_end_index):
                if idx not in processed_vertices:
                    new_vertices[idx, 0] = addition_offset + ((self.vertices_np[idx, 0] - window_start) / window_size) * x_addition_scale
                    new_vertices[idx, 1] -= avg_y_distance
                    processed_vertices.add(idx)

            addition_offset += x_addition_scale
            # window_start += window_size

        self.vertices_np = new_vertices

    def have_visited_all_vertices(self):
        """
        Check if all vertices have been visited.
        :return: True if all vertices are visited, False otherwise.
        """
        return len(self.visited_vertices) == self.vertices_np.shape[0]

    def reset_visited(self):
        """
        Reset the set of visited vertices.
        """
        self.visited_vertices.clear()

    def save_vertices(self):
        # numpy tp mesh path
        vertices_path = os.path.join(os.path.dirname(self.mesh_path), "vertices_flattened.npy")
        # Save as npz
        with open(vertices_path, 'wb') as f:
            np.savez(f, vertices=self.vertices_np)

    def load_vertices(self):
        vertices_path = os.path.join(os.path.dirname(self.mesh_path), "vertices_flattened.npy")
        # Open the npz file
        with open(vertices_path, 'rb') as f:
            npzfile = np.load(f)
            self.vertices_np = npzfile['vertices']


    def split_mesh(self, split_width):
        # Function to cut the mesh into pieces along the x-axis
        vertices = np.asarray(self.mesh.vertices)
        normals = np.asarray(self.mesh.vertex_normals)
        triangles = np.asarray(self.mesh.triangles)
        triangle_uvs = np.asarray(self.mesh.triangle_uvs)

        # split the mesh based on the vertices_np and split_width
        min_u = np.min(self.vertices_np[:, 0])
        max_u = np.max(self.vertices_np[:, 0])
        mesh_paths = []
        window_start = min_u
        while window_start < max_u:
            window_end = window_start + split_width
            window_indices = np.where((self.vertices_np[:, 0] >= window_start) & (self.vertices_np[:, 0] < window_end))
            qualifying_triangles = np.any(np.isin(triangles, window_indices), axis=1)
            qualifying_uvs = qualifying_triangles.repeat(3).reshape(-1)

            selected_triangles = triangles[qualifying_triangles]
            selected_uvs = triangle_uvs[qualifying_uvs]

             # Create a new mesh with the selected vertices and triangles
            cut_mesh = o3d.geometry.TriangleMesh()
            cut_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            cut_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
            cut_mesh.triangles = o3d.utility.Vector3iVector(selected_triangles)
            cut_mesh.triangle_uvs = o3d.utility.Vector2dVector(selected_uvs)
            cut_mesh = cut_mesh.remove_unreferenced_vertices()
            print(f"Nr triangles in cut mesh: {len(np.asarray(cut_mesh.triangles))}")
            print(f"Nr uvs in cut mesh: {len(np.asarray(cut_mesh.triangle_uvs))}")
            print(f"Nr vertices in cut mesh: {len(np.asarray(cut_mesh.vertices))}")
            
            path_window = os.path.join(os.path.dirname(self.mesh_path), "windowed_mesh", os.path.basename(self.mesh_path).replace(".obj", f"_window_{int(window_start)}_{int(window_end)}.obj"))
            mesh_paths.append(path_window)
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(path_window), exist_ok=True)
            # Save the mesh to an .obj file
            o3d.io.write_triangle_mesh(path_window, cut_mesh)
            window_start = window_end
        
        return mesh_paths
        
    def compute(self, split_width=50000, fresh_start=True):
        if fresh_start:
            self.compute_uv_with_bfs(0)
            self.scale_uv_x()
            self.save_vertices()
        else:
            self.load_vertices()
        return self.split_mesh(split_width)

if __name__ == '__main__':
    import argparse
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Split a mesh into pieces by normalized UVs')
    parser.add_argument('--umbilicus_path', type=str, help='Path of center umbilicus positions for the mesh scroll', required=True)
    parser.add_argument('--mesh', type=str, help='Path of .obj Mesh', required=True)
    parser.add_argument('--split_width', type=int, help='Width of the split windows', default=50000)

    # Take arguments back over
    args = parser.parse_args()
    print(f"Mesh Splitter arguments: {args}")

    umbilicus_path = args.umbilicus_path
    split_width = args.split_width
    
    splitter = MeshSplitter(args.mesh, umbilicus_path)
    splitter.compute(args.mesh, split_width)