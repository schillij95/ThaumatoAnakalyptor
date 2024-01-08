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

def apply_transform_to_mesh(mesh_path, transform_matrix):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) @ transform_matrix[:3, :3].T + transform_matrix[:3, 3])
    return mesh

def apply_transform_to_image(image_path, transform_matrix):
    image = Image.open(image_path)
    scale_x, scale_y = transform_matrix[0, 0], transform_matrix[1, 1]
    new_size = int(image.width * scale_x), int(image.height * scale_y)
    return image.resize(new_size, Image.ANTIALIAS)

def main(transform_path, mesh_path, image_path, mtl_path, mesh_save_path, image_save_path, mtl_save_path):
    transform_matrix = load_transform(transform_path)
    mesh = apply_transform_to_mesh(mesh_path, transform_matrix)
    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(mesh_save_path), exist_ok=True)
    # Save or process the transformed mesh as needed
    o3d.io.write_triangle_mesh(mesh_save_path, mesh)

    transformed_image = apply_transform_to_image(image_path, transform_matrix)
    # Save or display the transformed image as needed
    transformed_image.save(image_save_path)

    # Save the mtl file
    with open(mtl_path, 'r') as file:
        mtl_data = file.read()
    with open(mtl_save_path, 'w') as file:
        file.write(mtl_data)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform mesh .obj with VC volume-volume transformation .json')
    parser.add_argument("--transform_path", type=str, default="path_to_your_json_file.json")
    parser.add_argument("--base_path", type=str, default="path_to_your_obj_file.obj")

    args = parser.parse_args()
    transform_path = args.transform_path
    base_path = args.base_path

    # Replace these paths with your actual file paths

    transform_path = f"{transform_path}/*.json"
    # Find the first json file in the directory
    transform_path = glob.glob(transform_path)[0]
    mesh_path = f"{base_path}/point_cloud_colorized_verso_subvolume_blocks_uv.obj"
    image_path = f"{base_path}/point_cloud_colorized_verso_subvolume_blocks_uv_0.png"
    mtl_path = f"{base_path}/point_cloud_colorized_verso_subvolume_blocks_uv.mtl"

    base_path_save = base_path + "_8um"
    mesh_save_path = f"{base_path_save}/point_cloud_colorized_verso_subvolume_blocks_uv.obj"
    image_save_path = f"{base_path_save}/point_cloud_colorized_verso_subvolume_blocks_uv_0.png"
    mtl_save_path = f"{base_path_save}/point_cloud_colorized_verso_subvolume_blocks_uv.mtl"

    print(f"Processing {transform_path}, {mesh_path}, {image_path}")
    print(f"Saving to {base_path_save}, {mesh_save_path}, {image_save_path}")

    main(transform_path, mesh_path, image_path, mtl_path, mesh_save_path, image_save_path, mtl_save_path)
