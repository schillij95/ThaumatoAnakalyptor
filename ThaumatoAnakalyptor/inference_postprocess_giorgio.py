import os
import py7zr
import open3d as o3d
import numpy as np
import hdbscan
import json
import tempfile
from tqdm import tqdm
from multiprocessing import Pool, Manager

# Paths to the input and output folders
input_folder = '/scroll_dest/scroll1_surface_points/point_cloud_colorized_recto_subvolume_blocks'
output_folder = '/scroll_dest/scroll1_surface_points/filtered_instances'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

already_computed = os.listdir(output_folder)

# Get a list of all .7z files to process
all_archives = [
    f for f in os.listdir(input_folder)
    if f.endswith('.7z') and "temp" not in f and f not in already_computed
]

def process_archive(args):
    filename, input_folder, output_folder = args

    try:
        # Initialize the set for assigned point IDs within the archive
        assigned_point_ids = set()
        input_archive_path = os.path.join(input_folder, filename)

        # Create a temporary directory to extract files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract all files from the archive to the temporary directory
            with py7zr.SevenZipFile(input_archive_path, 'r') as archive:
                archive.extractall(path=temp_dir)

            # Get lists of .ply and metadata files
            all_files = os.listdir(temp_dir)
            ply_files = [f for f in all_files if f.endswith('.ply')]
            metadata_files = [
                f for f in all_files if f.startswith('metadata_') and f.endswith('.json')
            ]

            # Build a mapping from instance name to score
            instance_scores = {}
            for metadata_file in metadata_files:
                metadata_path = os.path.join(temp_dir, metadata_file)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                # Extract the base name of the instance
                base_name = metadata_file.replace('metadata_', '').replace('.json', '')
                score = metadata.get('score', 0)
                instance_scores[base_name] = score

            # Sort instances by score in descending order
            sorted_instances = sorted(
                instance_scores.items(), key=lambda x: x[1], reverse=True
            )

            processed_instances = []

            # Prepare to collect processed files
            processed_files = []

            for instance_name, score in sorted_instances:
                ply_file = instance_name + '.ply'
                metadata_file = 'metadata_' + instance_name + '.json'

                if ply_file not in ply_files:
                    continue  # Skip if the .ply file is missing

                ply_path = os.path.join(temp_dir, ply_file)
                metadata_path = os.path.join(temp_dir, metadata_file)

                # Load the point cloud from the file
                try:
                    pcd = o3d.io.read_point_cloud(ply_path)
                except (OSError, ValueError) as e:
                    print(f"Failed to read point cloud {ply_file}: {e}")
                    continue

                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)

                # Check if any points are left after filtering
                if points.size < 10:
                    continue  # Skip this instance

                # Convert points to integer coordinates
                rounded_points_int = np.round(points * 100).astype(np.int32)

                # Create unique IDs for points
                unique_ids = (rounded_points_int[:, 0].astype(np.int64) << 42) | \
                             (rounded_points_int[:, 1].astype(np.int64) << 21) | \
                             rounded_points_int[:, 2].astype(np.int64)

                # Remove duplicates within the same instance
                unique_ids, unique_indices = np.unique(unique_ids, return_index=True)
                points = points[unique_indices]
                colors = colors[unique_indices]
                rounded_points_int = rounded_points_int[unique_indices]

                # Remove points that overlap with assigned_point_ids
                if assigned_point_ids:
                    mask = np.array([uid not in assigned_point_ids for uid in unique_ids], dtype=bool)
                else:
                    mask = np.ones(len(unique_ids), dtype=bool)
                points = points[mask]
                colors = colors[mask]
                rounded_points_int = rounded_points_int[mask]
                unique_ids = unique_ids[mask]

                # Check if any points are left after filtering
                if points.size < 10:
                    continue  # Skip this instance

                # Perform HDBSCAN clustering
                try:
                    db = hdbscan.HDBSCAN(
                        min_samples=10, allow_single_cluster=True, core_dist_n_jobs=1
                    ).fit(points)
                    cluster_labels = db.labels_
                    # Get the largest cluster label (excluding noise labeled as -1)
                    labels, counts = np.unique(cluster_labels, return_counts=True)
                    label_counts = dict(zip(labels, counts))
                    # Exclude noise label (-1)
                    label_counts.pop(-1, None)
                    if not label_counts:
                        continue  # Skip if no clusters
                    largest_cluster_label = max(label_counts, key=label_counts.get)
                    points_mask = cluster_labels == largest_cluster_label
                except Exception as e:
                    print(f"Clustering failed for {ply_file}: {e}")
                    continue

                # Check if the largest cluster meets the size criteria
                if np.sum(points_mask) >= 0.2 * len(points):  # Adjust as needed
                    # Keep only the points in the largest cluster
                    points = points[points_mask]
                    colors = colors[points_mask]
                    rounded_points_int = rounded_points_int[points_mask]
                    unique_ids = unique_ids[points_mask]
                else:
                    continue  # Skip this instance

                # Second Overlap Filtering After HDBSCAN
                if assigned_point_ids:
                    mask = np.array([uid not in assigned_point_ids for uid in unique_ids], dtype=bool)
                else:
                    mask = np.ones(len(unique_ids), dtype=bool)
                points = points[mask]
                colors = colors[mask]
                rounded_points_int = rounded_points_int[mask]
                unique_ids = unique_ids[mask]

                # Check if any points are left after filtering
                if points.size < 10:
                    continue  # Skip this instance

                # Update assigned_point_ids with the current instance's points
                assigned_point_ids.update(unique_ids)

                # Save the processed instance back to the temporary directory
                pcd_processed = o3d.geometry.PointCloud()
                pcd_processed.points = o3d.utility.Vector3dVector(points)
                pcd_processed.colors = o3d.utility.Vector3dVector(colors)
                processed_ply_path = os.path.join(temp_dir, ply_file)
                o3d.io.write_point_cloud(processed_ply_path, pcd_processed)

                # Collect processed files to add to the archive later
                processed_files.append((processed_ply_path, ply_file))
                processed_files.append((metadata_path, metadata_file))

                # Keep track of processed instances
                processed_instances.append(instance_name)

            if processed_instances:
                # Create the output archive and add processed files
                output_archive_path = os.path.join(output_folder, filename)
                with py7zr.SevenZipFile(output_archive_path, 'w') as output_archive:
                    for file_path, arcname in processed_files:
                        output_archive.write(file_path, arcname=arcname)
            #else:
            #    print(f"No instances left after processing {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False

    return True



if __name__ == '__main__':
    import multiprocessing

    # Set multiprocessing start method
    try:
        multiprocessing.set_start_method('spawn')  # Use 'spawn' or 'fork' depending on the OS
    except RuntimeError:
        pass  # Start method has already been set

    # Create a list of arguments for each process
    args_list = [(filename, input_folder, output_folder) for filename in all_archives]

    # Use multiprocessing Pool to parallelize the main loop
    num_workers = 200  # Adjust as needed
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_archive, args_list), total=len(args_list), desc='Processing Archives'))
