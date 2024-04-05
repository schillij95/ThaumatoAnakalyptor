import open3d as o3d
import numpy as np

def apply_mls_smoothing(input_ply_path, output_ply_path, search_radius=10, polynomial_order=2):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(input_ply_path)
    
    # Convert Open3D.PointCloud to numpy array
    points = np.asarray(pcd.points)
    
    # Prepare an Open3D KDTree for neighborhood searches
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    # Container for smoothed points
    smoothed_points = []
    
    for i in range(len(points)):
        # For each point, find its neighbors within the specified radius
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], search_radius)
        
        if k < 3:
            # If not enough points for polynomial fitting, skip smoothing
            smoothed_points.append(points[i])
            continue
        
        # Extract the neighboring points
        neighbors = points[idx, :]
        
        # Compute weights (e.g., using a simple inverse distance weighting)
        weights = 1 / np.linalg.norm(neighbors - points[i], axis=1)
        
        # Fit a polynomial surface to the neighbors
        # Note: For simplicity, this example uses a basic form of weighting. For actual MLS, you'd need to solve a weighted least squares problem.
        # Here, we approximate this step by weighted average for demonstration.
        weighted_sum = np.sum(neighbors.T * weights, axis=1) / np.sum(weights)
        smoothed_points.append(weighted_sum)
    
    # Create a new point cloud with the smoothed points
    smoothed_pcd = o3d.geometry.PointCloud()
    smoothed_pcd.points = o3d.utility.Vector3dVector(np.array(smoothed_points))
    
    # Save the smoothed point cloud
    o3d.io.write_point_cloud(output_ply_path, smoothed_pcd)
    print(f"Smoothed point cloud saved to {output_ply_path}")

# Example usage
input_ply_path = 'ThaumatoAnakalyptor/mesh_test/input_point_cloud.ply'
output_ply_path = 'ThaumatoAnakalyptor/mesh_test/smoothed_point_cloud.ply'
apply_mls_smoothing(input_ply_path, output_ply_path)
