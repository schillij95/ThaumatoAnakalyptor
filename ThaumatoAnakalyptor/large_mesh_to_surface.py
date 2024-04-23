### Julian Schilliger - ThaumatoAnakalyptor - 2024

from .finalize_mesh import main as finalize_mesh_main
from .mesh_to_surface import ppm_and_texture

from tqdm import tqdm
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale and cut a mesh into pieces, normalize UVs, and handle textures")
    parser.add_argument("--input_mesh", type=str, required=True, help="Path to the input mesh file")
    parser.add_argument('--grid_cell', type=str, required=True, help='Path to the grid cells')
    parser.add_argument('--format', type=str, default='jpg')
    parser.add_argument("--scale_factor", type=float, help="Scaling factor for vertices", default=1.0)
    parser.add_argument("--cut_size", type=int, help="Size of each cut piece along the X axis", default=40000)
    parser.add_argument("--output_folder", type=str, help="Folder to save the cut meshes", default=None)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--r', type=int, default=32)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--nr_workers', type=int, default=None)
    parser.add_argument('--prefetch_factor', type=int, default=2)

    args = parser.parse_known_args()[0]
    print(f"Known args: {args}")
    obj_paths = finalize_mesh_main(args.output_folder, args.input_mesh, args.scale_factor, args.cut_size, False)
    for obj_path in tqdm(obj_paths, desc='Texturing meshes'):
        print(f"Texturing {obj_path}")
        ppm_and_texture(obj_path, gpus=args.gpus, grid_cell_path=args.grid_cell, output_path=None, r=args.r, format=args.format, display=args.display, nr_workers=args.nr_workers, prefetch_factor=args.prefetch_factor)
