# loads a mesh, then takes the min and max of all vertices in every dimension and extracts all grid cells to a new folder
import open3d as o3d
import numpy as np
import os
import argparse
from tqdm import tqdm

# Load mesh
def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    return mesh

def get_min_max(mesh):
    vertices = np.asarray(mesh.vertices)
    min_x = np.min(vertices[:,0])
    max_x = np.max(vertices[:,0])
    min_y = np.min(vertices[:,1])
    max_y = np.max(vertices[:,1])
    min_z = np.min(vertices[:,2])
    max_z = np.max(vertices[:,2])
    return min_x, max_x, min_y, max_y, min_z, max_z

def copy_grid_cell(source, dest):
    try:
        os.system("cp " + source + " " + dest)
    except Exception as e:
        print("Error copying file: ", e, source, dest)

def extract_grid_cells(source_template, dest_template, min_x, max_x, min_y, max_y, min_z, max_z, grid_size):
    start_cell_x = int(min_x / grid_size)
    start_cell_y = int(min_y / grid_size)
    start_cell_z = int(min_z / grid_size)
    end_cell_x = int(max_x / grid_size)
    end_cell_y = int(max_y / grid_size)
    end_cell_z = int(max_z / grid_size)

    # make dir
    dest_dir = os.path.dirname(dest_template)
    os.makedirs(dest_dir, exist_ok=True)

    print("Extracting grid cells from x: {} to {}, y: {} to {}, z: {} to {}".format(start_cell_x, end_cell_x, start_cell_y, end_cell_y, start_cell_z, end_cell_z))
    for i in tqdm(range(start_cell_x, end_cell_x+1), desc="Extracting grid cells"):
        for j in range(start_cell_y, end_cell_y+1):
            for k in range(start_cell_z, end_cell_z+1):
                source = source_template.format(i, j, k)
                dest = dest_template.format(i, j, k)
                if os.path.exists(source):
                    copy_grid_cell(source, dest)

def main():
    parser = argparse.ArgumentParser("Extract grid cells from a mesh.")
    parser.add_argument("--mesh_dir", type=str, help="dir to mesh files")
    parser.add_argument("--source", type=str, help="source directory for grid cell files")
    parser.add_argument("--dest", type=str, help="destination directory for grid cell files")
    parser.add_argument("--grid_size", type=int, help="size of grid cell", default=500)
    parser.add_argument("--z_min", type=int, help="min z value", default=None)
    parser.add_argument("--z_max", type=int, help="max z value", default=None)
    args = parser.parse_args()

    mesh_dir = args.mesh_dir
    # list of all .obj in the directory
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith(".obj")]
    min_x, max_x, min_y, max_y, min_z, max_z = 1948, 5502, 1532, 5836, 4, 13399
    for mesh_file in mesh_files:
        mesh_path = os.path.join(mesh_dir, mesh_file)
        mesh = load_mesh(mesh_path)
        min_x, max_x, min_y, max_y, min_z, max_z = get_min_max(mesh)
        if min_x is None or min_x > min_x:
            min_x = min_x
        if max_x is None or max_x < max_x:
            max_x = max_x
        if min_y is None or min_y > min_y:
            min_y = min_y
        if max_y is None or max_y < max_y:
            max_y = max_y
        if min_z is None or min_z > min_z:
            min_z = min_z
        if max_z is None or max_z < max_z:
            max_z = max_z
        
    if args.z_min is not None:
        min_z = args.z_min
    if args.z_max is not None:
        max_z = args.z_max

    source_template = args.source + "/cell_yxz_{:03}_{:03}_{:03}.tif"
    dest_template = args.dest + "/cell_yxz_{:03}_{:03}_{:03}.tif"

    extract_grid_cells(source_template, dest_template, min_x, max_x, min_y, max_y, min_z, max_z, args.grid_size)

if __name__ == "__main__":
    main()