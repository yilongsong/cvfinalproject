"""
Script to generate voxels as npy files of demonstrations from
a set of demonstrations stored in a hdf5 file.

Camera angles used: sideview, frontview, birdview, robot0_eye_in_hand

Arguments:
    --folder (str): Path to demonstrations

Example:
    $ python generate_voxels_npy.py --dataset .../mimicgen_environments/datasets/core/coffee_preparation_d0.hdf5 --output_name .../LanManip/LanManip/data/mimicgen_200_d2/place_coffee_pod_into_coffee_machine/data.npy

    $ python generate_voxels_npy.py --dataset .../mimicgen_environments/datasets/core/coffee_preparation_d0.hdf5 --output_name .../LanManip/LanManip/data/mimicgen_200_d2/place_coffee_pod_into_coffee_machine/data.npy
"""

import argparse
import json
import os
import random

import h5py
import numpy as np

import robosuite

import open3d as o3d

from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

'''
Copy the following code into robosuite/robosuite/utils/opencv_renderer.py in the class definition

    def quaternion_to_rotation_matrix(self, Q): # helper function that turns quaternion into rotation matrix
        """
        Copied from quaternion to rotation matrix example
        Covert a quaternion into a full three-dimensional rotation matrix.
    
        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
    
        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]
        
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
                                
        return rot_matrix
    
    def render_pointcloud(self):
        depth_im = self.sim.render(camera_name=self.camera_name, height=self.height, width=self.width, depth=True)[1]

        # if self.camera_name == 'sideview':
        depth_im = np.flip(depth_im, axis=0)

        depth_im = camera_utils.get_real_depth_map(self.sim, depth_im)

        intrinsics = camera_utils.get_camera_intrinsic_matrix(self.sim, self.camera_name, self.height, self.width)

        xyz_camera = ravens.get_pointcloud(depth_im, intrinsics)

        extrinsics = camera_utils.get_camera_extrinsic_matrix(self.sim, self.camera_name)

        xyz_world = ravens.transform_pointcloud(xyz_camera, extrinsics)

        xyz_world = xyz_world.reshape(-1,3)

        # remove points too far away from the origin
        mask = np.logical_and(xyz_world[:, 0] >= -1, xyz_world[:, 0] <= 0.75) & \
        np.logical_and(xyz_world[:, 1] >= -1, xyz_world[:, 1] <= 1) & \
        np.logical_and(xyz_world[:, 2] >= 0.5, xyz_world[:, 2] <= 2)
        xyz_world = xyz_world[mask]

        return xyz_world
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to your .hdf5 demonstration file, e.g.: "
        "'.../mimicgen_environments/datasets/core/coffee_preparation_d0.hdf5'",
    ),
    parser.add_argument(
        "--output_name",
        type=str,
        help="Path to which you wish to store the .npy files consisting voxels, e.g.: "
        "'.../LanManip/LanManip/data/mimicgen_200_d2/coffee_preparation_d0/'",
    ),
    parser.add_argument(
        "--camera",
        choices = ['single', 'multi'],
        default = 'multi',
        help="Choose from 'single' and 'multi' camera for 3D reconstruction"
    )
    args = parser.parse_args()

    hdf5_path = args.dataset
    f = h5py.File(hdf5_path, "r")
    # env_name = f["data"].attrs["env"]
    # env_info = json.loads(f["data"].attrs["env_info"])

    env1 = robosuite.make(
            env_name = "Lift", # This is potentially problematic
            robots = 'Panda',
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera='agentview',
            ignore_done=True,
            use_camera_obs=True,
            reward_shaping=True,
            control_freq=20,
            #camera_names='all-' + args.camera, camera_heights=80, camera_widths=80, camera_depths=True, camera_segmentations=None
            camera_depths=True
        )


    if args.camera == 'multi':
        env1 = robosuite.make(
            env_name = "Lift",
            robots = 'Panda',
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera='sideview',
            ignore_done=True,
            use_camera_obs=True,
            reward_shaping=True,
            control_freq=20,
            #camera_names='all-' + args.camera, camera_heights=80, camera_widths=80, camera_depths=True, camera_segmentations=None
            camera_depths=True
        )

        env2 = robosuite.make(
            env_name = "Lift",
            robots = 'Panda',
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera='frontview',
            ignore_done=True,
            use_camera_obs=True,
            reward_shaping=True,
            control_freq=20,
            #camera_names='all-' + args.camera, camera_heights=80, camera_widths=80, camera_depths=True, camera_segmentations=None
            camera_depths=True
        )

        env3 = robosuite.make(
            env_name = "Lift",
            robots = 'Panda',
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera='robot0_eye_in_hand',
            ignore_done=True,
            use_camera_obs=True,
            reward_shaping=True,
            control_freq=20,
            #camera_names='all-' + args.camera, camera_heights=80, camera_widths=80, camera_depths=True, camera_segmentations=None
            camera_depths=True
        )

        env4 = robosuite.make(
            env_name = "Lift",
            robots = 'Panda',
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera='birdview',
            ignore_done=True,
            use_camera_obs=True,
            reward_shaping=True,
            control_freq=20,
            #camera_names='all-' + args.camera, camera_heights=80, camera_widths=80, camera_depths=True, camera_segmentations=None
            camera_depths=True
        )




    # list of all demonstrations episodes
    demos = list(f["data"].keys())


    for j in range(len(demos)):
        print("Working on episode " + str(j))

        # go through each episode
        ep = demos[j]

        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]

        env1.reset()
        env2.reset()
        env3.reset()
        env4.reset()

        xml1 = env1.edit_model_xml(model_xml)
        xml2 = env2.edit_model_xml(model_xml)
        xml3 = env3.edit_model_xml(model_xml)
        xml4 = env3.edit_model_xml(model_xml)
        
        env1.reset_from_xml_string(xml1)
        env2.reset_from_xml_string(xml2)
        env3.reset_from_xml_string(xml3)
        env4.reset_from_xml_string(xml4)
        
        env1.sim.reset()
        env2.sim.reset()
        env3.sim.reset()
        env4.sim.reset()
        # env.viewer.set_camera(0)

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]

        count = 0

        if args.use_actions: # Not useful for now
            print('use_actions')

            # load the initial state
            env1.sim.set_state_from_flattened(states[0])
            env1.sim.forward()

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]

            for j, action in enumerate(actions):
                env1.step(action)
                env1.render()

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env1.sim.get_state().flatten()
                    if not np.all(np.equal(states[j + 1], state_playback)):
                        err = np.linalg.norm(states[j + 1] - state_playback)
                        print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

        else:

            # force the sequence of internal mujoco states one by one
            for i in range(len(states)):
                directory = args.output_name
                if not os.path.exists(directory):
                    os.makedirs(directory)
                print('State ' + str(i) + ' point cloud saved as ' + directory + str(i) + '.ply')
                state = states[i]
                env1.sim.set_state_from_flattened(state)
                env2.sim.set_state_from_flattened(state)
                env3.sim.set_state_from_flattened(state)
                env4.sim.set_state_from_flattened(state)
                env1.sim.forward()
                env2.sim.forward()
                env3.sim.forward()
                env4.sim.forward()

                env1_pointcloud = env1.viewer.render_pointcloud()
                env2_pointcloud = env2.viewer.render_pointcloud()
                env3_pointcloud = env3.viewer.render_pointcloud()
                env4_pointcloud = env4.viewer.render_pointcloud()

                xyz_world = np.vstack((env1_pointcloud, env2_pointcloud, env3_pointcloud, env4_pointcloud))

                pcd = o3d.geometry.PointCloud()

                pcd.points = o3d.utility.Vector3dVector(xyz_world)

                # if count % 200 == 0: # Visualization for debugging
                #     o3d.visualization.draw_geometries([pcd])

                o3d.io.write_point_cloud(directory + str(i) + '.ply', pcd, write_ascii=True)

                exit(0)

                env1.reset()
                env2.reset()
                env3.reset()
                env4.reset()

    f.close()


if __name__ == "__main__":
    main()
