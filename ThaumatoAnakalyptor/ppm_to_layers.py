from rendering_utils.interpolate_image_3d import extract_from_image_3d, insert_into_image_3d
from rendering_utils.ppmparser import PPMParser
from grid_to_pointcloud import load_grid
import argparse
from tqdm import tqdm

import os
import tifffile

import numpy as np
import torch

def load_ppm_cubes(path, cube_size=500):
    with PPMParser(path).open() as ppm:
        im_shape = ppm.im_shape()
        cubes = ppm.classify_entries_to_cubes(cube_size=cube_size)
    return cubes, im_shape

def cube_coords(cube_key, padding, cube_size):
    x, y, z = cube_key
    start_coords = np.array([x*cube_size - padding, y*cube_size - padding, z*cube_size - padding]) + cube_size # spelufo offset, 1 indexing
    grid_block_size = cube_size + 2*padding
    return start_coords, grid_block_size

def main(args):
    working_path = os.path.dirname(args.ppm_path)
    path_template = working_path + "/" + args.grid_volume_path + "/cell_yxz_{:03}_{:03}_{:03}.tif"

    # load ppm cubes
    cubes, im_shape = load_ppm_cubes(args.ppm_path, cube_size=args.cube_size)
    print(f"Loaded {len(cubes)} cubes from {args.ppm_path}")

    # pytorch array uint16 on cpu of size 2*r, im_shape
    layers = torch.zeros((2*args.r, im_shape[1], im_shape[0]), dtype=torch.float32, device='cpu')
    layers_path = working_path + "/layers/"

    print(f"All parameters: {args}, im_shape: {im_shape}, layers_path: {layers_path}, path_template: {path_template}")

    for cube in tqdm(cubes.keys()):
        print(cube)
        start_coords, grid_block_size = cube_coords(cube, args.r, args.cube_size)
        start_coords_offset = (start_coords//args.cube_size)*args.cube_size
        # start_coords_offset = start_coords_offset[::-1].copy()
        # load grid volume
        grid_volume = load_grid(path_template, tuple(start_coords), grid_block_size, args.cube_size, uint8=False).astype(np.float32)
        # to torch cuda
        grid_volume = torch.from_numpy(grid_volume).cuda()
        print(f"Min and max grid_volume: {grid_volume.min()}, {grid_volume.max()}")

        # construct volume indexing
        cube_ppm = cubes[cube]
        xyz = torch.tensor([c[2:5] for c in cube_ppm], dtype=torch.float32).cuda()
        normals = torch.tensor([c[5:] for c in cube_ppm], dtype=torch.float32).cuda()
        # construct all coordinate in positive and negative r
        coords = torch.cat([xyz + r * normals for r in range(-args.r, args.r+1)], dim=0)
        print(f"first Min and max coords of each dimension: x: {coords[:, 0].min()}, {coords[:, 0].max()}, y: {coords[:, 1].min()}, {coords[:, 1].max()}, z: {coords[:, 2].min()}, {coords[:, 2].max()}")
        # recalculate coords to zero on grid_volume
        coords = coords - torch.tensor(start_coords_offset, dtype=torch.float32).cuda()
        print(f"Min and max coords of each dimension: x: {coords[:, 0].min()}, {coords[:, 0].max()}, y: {coords[:, 1].min()}, {coords[:, 1].max()}, z: {coords[:, 2].min()}, {coords[:, 2].max()}")
        # extract from grid volume
        samples = extract_from_image_3d(grid_volume, coords)
        print(f"Min and max samples: {samples.min()}, {samples.max()}")

        # construct layers coords
        xy = torch.tensor([c[:2][::-1] for c in cube_ppm], dtype=torch.int32).cpu()  # x, y coordinates
        # construct z coordinates for each layer
        z_layers = torch.arange(0, 2*args.r+1, dtype=torch.int32).repeat(len(cube_ppm), 1).T.contiguous().view(-1).cpu()
        # repeat xy coordinates for each layer
        xy_repeated = xy.repeat(2*args.r+1, 1)
        # combine xy and z coordinates
        xyz_layers = torch.cat([z_layers[:, None], xy_repeated], dim=1).cpu()  # z, x, y order
        print(f"xyz_layers shape: {xyz_layers.shape}")
        print(f"Min and max xyz_layers of each dimension: x: {xyz_layers[:, 1].min()}, {xyz_layers[:, 1].max()}, y: {xyz_layers[:, 2].min()}, {xyz_layers[:, 2].max()}, z: {xyz_layers[:, 0].min()}, {xyz_layers[:, 0].max()}")

        # insert into layers
        layers = insert_into_image_3d(samples.cpu(), xyz_layers, layers)

    # save layers
    for i in range(layers.shape[0]):
        nr_zeros = len(str(2*args.r))
        layer = layers[i].cpu().numpy().astype(np.uint16)
        # save layer with leading 0's for 2*r layers
        layer_nr = str(i).zfill(nr_zeros)
        layer_path = layers_path + f"{layer_nr}.tif"
        tifffile.imsave(layer_path, layer)

if __name__ == '__main__':
    # parse ppm path, grid volume path, r=32, cube_size=500 default
    parser = argparse.ArgumentParser()
    parser.add_argument('ppm_path', type=str)
    parser.add_argument('grid_volume_path', type=str)
    parser.add_argument('--r', type=int, default=32)
    parser.add_argument('--cube_size', type=int, default=500)
    args = parser.parse_args()

    main(args)