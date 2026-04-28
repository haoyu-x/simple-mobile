# Author: Jimmy Wu
# Date: October 2024
#
# References:
# - https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/dataset_states_to_obs.py

import argparse
from pathlib import Path
import cv2 as cv
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from constants import POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT
from episode_storage import EpisodeReader

def main(input_dir, output_path):
    # Get list of episode dirs
    episode_dirs = sorted([child for child in Path(input_dir).iterdir() if child.is_dir()])

    # Convert to robomimic HDF5 format
    with h5py.File(output_path, 'w') as f:
        data_group = f.create_group('data')

        # Iterate through episodes
        for episode_idx, episode_dir in enumerate(tqdm(episode_dirs)):
            # Load episode data
            reader = EpisodeReader(episode_dir)

            # Extract observations
            observations = {}
            for obs in reader.observations:
                
                for k, v in obs.items():
                    if k == 'base_pose':
                        # skip
                        continue

                    if v.ndim == 3:
                        # Resize image
                        v = cv.resize(v, (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT))

                    # Append extracted observation
                    if k not in observations:
                        # print(f'Observation key: {k}')
                        observations[k] = []
                    observations[k].append(v)

            # Extract actions
            actions = [
                np.concatenate((
                    action['base_velocity'],
                    action['arm_pos'],
                    Rotation.from_quat(action['arm_quat']).as_rotvec(),  # Convert quat to axis-angle
                    action['gripper_pos'],
                )) for action in reader.actions
            ]


            actions = np.array(actions)
   
            # Write to HDF5
            episode_key = f'demo_{episode_idx}'
            episode_group = data_group.create_group(episode_key)
            for k, v in observations.items():
                episode_group.create_dataset(f'obs/{k}', data=np.array(v))
            episode_group.create_dataset('actions', data=np.array(actions))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/demos')
    parser.add_argument('--output-path', default='data/demos.hdf5')
    args = parser.parse_args()
    main(args.input_dir, args.output_path)



# Observation key: arm_pos
# Observation key: arm_quat
# Observation key: gripper_pos
# Observation key: base_image
# Observation key: wrist_image
# action dim: 10 (base_velocity(3), arm_pos(3), arm_quat(3), gripper_pos(1))