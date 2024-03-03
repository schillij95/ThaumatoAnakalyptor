### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

import json
import numpy as np
import open3d as o3d
from PIL import Image
# no max size for images
Image.MAX_IMAGE_PIXELS = None
import glob
import argparse
import os

def load_transform(transform_path):
    with open(transform_path, 'r') as file:
        data = json.load(file)
    return np.array(data["params"])

def invert_transform(transform_matrix):
    inv_transform = np.linalg.inv(transform_matrix)
    return inv_transform

def combine_transforms(transform_a, transform_b):
    return np.dot(transform_b, transform_a)

def apply_transform_to_mesh(mesh_path, transform_matrix):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) @ transform_matrix[:3, :3].T + transform_matrix[:3, 3])
    return mesh

def apply_transform_to_image(image_path, transform_matrix):
    image = Image.open(image_path)
    scale_x, scale_y = transform_matrix[0, 0], transform_matrix[1, 1]
    new_size = int(image.width * scale_x), int(image.height * scale_y)
    return image.resize(new_size, Image.ANTIALIAS)

def compute(transform_path, original_volume_id, target_volume_id, base_path):
    # Load transform from original to canonical
    if not (original_volume_id is None):
        transform_to_canonical_path = glob.glob(f"{transform_path}/*-to-{original_volume_id}.json")
        if len(transform_to_canonical_path) == 0:
            inverted_transform_to_canonical = np.eye(4)
        else:
            transform_to_canonical_path = transform_to_canonical_path[0]
            transform_to_canonical = load_transform(transform_to_canonical_path)

            # Invert it since we need to go from original to canonical
            inverted_transform_to_canonical = invert_transform(transform_to_canonical)
            assert np.allclose(np.dot(transform_to_canonical, inverted_transform_to_canonical), np.eye(4))
    else:
        inverted_transform_to_canonical = np.eye(4)
    
    # Load transform from canonical to target
    transform_to_target_path = glob.glob(f"{transform_path}/*-to-{target_volume_id}.json")[0]
    transform_to_target = load_transform(transform_to_target_path)
    
    # Combine the transformations
    combined_transform = combine_transforms(inverted_transform_to_canonical, transform_to_target)
    
    # Find .obj
    obj_name = sorted(glob.glob(f"{base_path}/*.obj"))[0][:-4]

    # Paths and application of transforms
    mesh_path = f"{base_path}/{obj_name}.obj"
    image_path = f"{base_path}/{obj_name}_0.png"
    mtl_path = f"{base_path}/{obj_name}.mtl"

    base_path_save = base_path + f"_{target_volume_id}"
    mesh_save_path = f"{base_path_save}/{obj_name}.obj"
    image_save_path = f"{base_path_save}/{obj_name}_0.png"
    mtl_save_path = f"{base_path_save}/{obj_name}.mtl"

    mesh = apply_transform_to_mesh(mesh_path, combined_transform)
    os.makedirs(os.path.dirname(mesh_save_path), exist_ok=True)
    o3d.io.write_triangle_mesh(mesh_save_path, mesh)

    transformed_image = apply_transform_to_image(image_path, combined_transform)
    transformed_image.save(image_save_path)

    with open(mtl_path, 'r') as file:
        mtl_data = file.read()
    with open(mtl_save_path, 'w') as file:
        file.write(mtl_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform mesh .obj with VC volume-volume transformation .json')
    parser.add_argument("--transform_path", type=str, required=True, help="Folder containing the transform .json files")
    parser.add_argument("--original_volume_id", type=str, required=True, help="Volume ID of the original scroll volume for the input mesh")
    parser.add_argument("--target_volume_id", type=str, required=True, help="Volume ID of the target scroll volume to transform the input mesh to")
    parser.add_argument("--base_path", type=str, required=True, help="Path to your .obj file")

    args = parser.parse_args()

    print(f"Processing from {args.original_volume_id} to {args.target_volume_id}")
    compute(args.transform_path, args.original_volume_id, args.target_volume_id, args.base_path)
