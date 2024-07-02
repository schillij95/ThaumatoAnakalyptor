### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2024

import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from scipy.cluster.hierarchy import fcluster, linkage
import argparse
import os
from .split_mesh import MeshSplitter
import tempfile
from tqdm import tqdm

def load_point_cloud(ply_file):
    # Load the point cloud from a PLY file
    pcd = o3d.io.read_point_cloud(ply_file)
    return np.asarray(pcd.points)

def save_point_cloud(filename, points):
    # mkdir if not exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

def load_mesh_vertices(obj_file):
    # copy mesh to tempfile
    with tempfile.NamedTemporaryFile(suffix=".obj") as temp_file:
        # copy mesh to tempfile
        temp_path = temp_file.name
        # os copy
        os.system(f"cp {obj_file} {temp_path}")
        # load mesh
        mesh = o3d.io.read_triangle_mesh(temp_path, print_progress=True)

    # Coordinate transform to pointcloud coordinate system
    vertices = np.asarray(mesh.vertices)
    vertices += 500
    vertices = vertices[:,[1, 2, 0]]
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    triangles = np.asarray(mesh.triangles)

    print(f"Number triangles in mesh: {len(triangles)}")

    scene = setup_closest_triangles(mesh)
    return triangles, scene

def calculate_winding_angle(mesh_path, pointcloud_dir):
    umbilicus_path = os.path.join(os.path.dirname(pointcloud_dir), "umbilicus.txt")
    splitter =  MeshSplitter(mesh_path, umbilicus_path)
    splitter.compute_uv_with_bfs(0)
    winding_angles = splitter.vertices_np[:, 0]
    return winding_angles

def find_closest_vertices(points, tree):    
    # Find the closest vertex for each point
    distances, indices = tree.query(points)
    return indices, distances

def setup_closest_triangles(mesh):
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh
    return scene

def find_closest_triangles(points, scene):
    # Find the closest triangle and the distance for each point
    query_points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)

    # We compute the closest point on the surface for the point at position [0,0,0].
    ans = scene.compute_closest_points(query_points)
    points_on_triangle = ans['points'].numpy()
    distances = np.linalg.norm(points_on_triangle - points, axis=-1)
    triangles_id = ans['primitive_ids'].numpy()
    return triangles_id, distances

def cluster_points_by_winding_angle(winding_angles_points, threshold_angle=45):
    # Use linkage with 'single' method specifying the clustering criterion
    # Here 'euclidean' is used; ensure your angles are prepared for this metric (could use 'cityblock' if thinking in terms of cyclic angles)
    Z = linkage(winding_angles_points.reshape(-1, 1), method='single', metric='euclidean')

    # Form flat clusters with the specified maximum cophenetic distance (threshold)
    clusters = fcluster(Z, t=threshold_angle, criterion='distance')
    return clusters

def generate_sample(index, winding_angles, scene, triangles, ply_path, output_dir, max_distance, max_distance_valid):
    # Load data
    points = load_point_cloud(ply_path)

    # Find closest vertices
    closest_triangle, closest_distances = find_closest_triangles(points, scene)
    # Get the index of the first vertex in the closest triangle
    closest_indices = triangles[closest_triangle][:, 0]

    # Check that every point is close enough to the mesh
    valid = np.all(closest_distances < max_distance_valid)
    print("Valid point cloud:", valid)
    if not valid:
        return
    
    # Only keep points that are close enough to the mesh
    mask_distance = closest_distances < max_distance
    closest_indices = closest_indices[mask_distance]
    clostest_points = points[mask_distance]

    # Find the winding angle for each point
    winding_angles_points = winding_angles[closest_indices]

    # Check if there are any points
    if len(winding_angles_points) == 0:
        return
    
    if len(winding_angles_points) < 10000:
        print(f"Skipping cube {index} with {len(winding_angles_points)} points")
        return
    
    # Save GT PointCloud
    ply_name = os.path.basename(ply_path)[:-4]
    gt_file = os.path.join(output_dir, ply_name, f"{ply_name}.ply")
    save_point_cloud(gt_file, points)

    # Cluster points by winding angle
    clusters = cluster_points_by_winding_angle(winding_angles_points)
    unique_clusters = np.unique(clusters)

    # Calculate the mean winding angle per cluster
    mean_winding_angles = np.array([np.mean(winding_angles_points[clusters == cluster_id]) for cluster_id in unique_clusters])
    # Sort unique clusters by mean winding angle
    cluster_indices = np.argsort(mean_winding_angles)

    # Optionally, output clustered points
    for sorted_index, cluster_index in enumerate(cluster_indices):
        cluster_id = unique_clusters[cluster_index]
        mask = clusters == cluster_id
        cluster_points = clostest_points[mask]
        cluster_file = os.path.join(output_dir, ply_name, "surfaces", f"{sorted_index}.ply")
        # Save cluster to a PLY file
        save_point_cloud(cluster_file, cluster_points)

def compute(mesh_file, pointcloud_dir, max_distance, max_distance_valid):
    output_dir = pointcloud_dir + "_mask3d_labels"
    
    # Load the mesh vertices
    triangles, scene = load_mesh_vertices(mesh_file)

    # Calculate winding angles for each vertex
    winding_angles = calculate_winding_angle(mesh_file, pointcloud_dir)

    ply_files = [f for f in os.listdir(pointcloud_dir) if f.endswith('.ply')]
    for i, ply_file in enumerate(tqdm(ply_files, desc="Processing point clouds")):
        ply_path = os.path.join(pointcloud_dir, ply_file)
        generate_sample(i, winding_angles, scene, triangles, ply_path, output_dir, max_distance, max_distance_valid)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Convert a 3D mesh and a PointCloud to 3D PointCloud instance labels.")
    parser.add_argument("--mesh_file", type=str, help="Path to the 3D mesh file (e.g., .obj)")
    parser.add_argument("--pointcloud_dir", type=str, help="Path to the 3D point cloud directory (containing .ply)")
    parser.add_argument("--max_distance", type=float, help="Maximum distance for a point to be considered part of the mesh", default=25)
    parser.add_argument("--max_distance_valid", type=int, help="Maximum distance all points must be to be considered valid a valid pointcloud", default=100)

    args = parser.parse_args()

    # Compute the 3D mask with labels
    compute(args.mesh_file, args.pointcloud_dir, args.max_distance, args.max_distance_valid)


if __name__ == "__main__":
    main()

# Example command: python3 -m ThaumatoAnakalyptor.mesh_to_mask3d_labels --mesh_file /scroll.volpkg/merging_test_merged/20230929220926-20231005123336-20231007101619-20231210121321-20231012184424-20231022170901-20231221180251-20231106155351-20231031143852-20230702185753-20231016151002.obj --pointcloud_dir /scroll1_surface_points/point_cloud_colorized_verso/point_cloud_colorized_verso