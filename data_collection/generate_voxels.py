'''
Script that generate voxels from point clouds.
'''

import os
import open3d as o3d

input_directory = "human_demonstrations/point_clouds"
output_directory = "human_demonstrations/voxels"

# Function to create voxel grid from point cloud
def create_voxel_grid(input_file, output_file):
    pcd = o3d.io.read_point_cloud(input_file)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
    o3d.io.write_voxel_grid(output_file, voxel_grid)

# Recursive function to process directories and files
def process_directory(input_dir, output_dir):
    for item in os.listdir(input_dir):
        input_item = os.path.join(input_dir, item)
        output_item = os.path.join(output_dir, item)
        if os.path.isdir(input_item):
            # Create corresponding directory in output directory
            os.makedirs(output_item, exist_ok=True)
            # Recursively process subdirectory
            process_directory(input_item, output_item)
        elif input_item.endswith('.ply'):
            output_item_voxel = output_item.replace('point_clouds', 'voxels')#.replace('.ply', '.obj') # Determine filetype by chaging the extension here
            create_voxel_grid(input_item, output_item_voxel)


if __name__ == '__main__':
    os.makedirs(output_directory, exist_ok=True)
    process_directory(input_directory, output_directory)