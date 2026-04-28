# Date: April 2026

import argparse
import math
import os
import sys
import queue
import threading
import time
from collections import deque

# Add diffusion_policy to path
DIFFUSION_POLICY_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'diffusion_policy')
sys.path.insert(0, os.path.abspath(DIFFUSION_POLICY_DIR))

import cv2 as cv
import dill
import hydra
import numpy as np
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from constants import POLICY_CONTROL_PERIOD
LATENCY_BUDGET = 0.1  # 100 ms including policy inference and communication
LATENCY_STEPS = math.ceil(LATENCY_BUDGET / POLICY_CONTROL_PERIOD)


class DiffusionPolicy:
    def __init__(self, ckpt_path):
        # Load checkpoint
        with open(ckpt_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload)

        # Load policy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        device = torch.device('cuda')
        policy.eval().to(device)

        # Store attributes
        self.policy = policy
        self.device = device
        self.obs_shape_meta = cfg.shape_meta['obs']
        self.rotation_transformer = RotationTransformer(from_rep='rotation_6d', to_rep='quaternion')
        self.warmed_up = False

    def reset(self):
        self.policy.reset()

    def step(self, obs_sequence):
        obs_dict = self._convert_obs(obs_sequence)
        with torch.no_grad():
            if not self.warmed_up:
                print('Warming up policy...')
                self.policy.predict_action(obs_dict)
                self.warmed_up = True
            result = self.policy.predict_action(obs_dict)
            action = result['action'][0].detach().to('cpu').numpy()
        act_sequence = self._convert_action(action)
        return act_sequence

    def _convert_obs(self, obs_sequence):
        obs_dict_np = {}
        for key, value in self.obs_shape_meta.items():
            if value.get('type') == 'rgb':
                images = np.stack([obs[key] for obs in obs_sequence], axis=0)
                assert images.dtype == np.uint8
                images = images.astype(np.float32) / 255.0
                images = np.transpose(images, (0, 3, 1, 2))
                assert images.shape[1:] == tuple(value['shape'])
                obs_dict_np[key] = images
            else:
                obs_dict_np[key] = np.stack([obs[key] for obs in obs_sequence], axis=0).astype(np.float32)
        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
        return obs_dict

    def _convert_action(self, action):
        act_sequence = []
        for act in action:
            action_dict = {
                'base_velocity': act[:3],
                'arm_pos': act[3:6],
                'arm_quat': self.rotation_transformer.forward(act[6:12])[[1, 2, 3, 0]],  # (w, x, y, z) -> (x, y, z, w)
                'gripper_pos': act[12:13],
            }
            act_sequence.append(action_dict)
        return act_sequence


class PolicyWrapper:
    def __init__(self, policy, n_obs_steps=2, n_action_steps=8):
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.obs_queue = queue.Queue()
        self.act_queue = queue.Queue()

        # Start inference loop
        threading.Thread(target=self.inference_loop, args=(policy,), daemon=True).start()

    def reset(self):
        self.obs_queue.put('reset')

    def step(self, obs):
        self.obs_queue.put(obs)
        action = None if self.act_queue.empty() else self.act_queue.get()
        if action is None:
            print('Warning: Unexpected idle action queue. Is the latency budget set too low?')
        return action

    def inference_loop(self, policy):
        obs_history = deque(maxlen=self.n_obs_steps)
        start_of_episode = True
        while True:
            try:
                # Check for new obs
                if not self.obs_queue.empty():
                    obs = self.obs_queue.get()

                    # Reset policy
                    if obs == 'reset':
                        policy.reset()
                        obs_history.clear()
                        start_of_episode = True
                        while not self.act_queue.empty():
                            self.act_queue.get()
                        continue

                    # Append obs to history
                    obs_history.append(obs)

                if self.act_queue.qsize() < LATENCY_STEPS and len(obs_history) == self.n_obs_steps:
                    obs_sequence = list(obs_history)
                    act_sequence = policy.step(obs_sequence)
                    if not self.act_queue.empty():
                        print('Warning: Unexpected action queue backlog. Is the latency budget set too high?')
                    if start_of_episode:
                        act_sequence = act_sequence[:self.n_action_steps - LATENCY_STEPS]
                        start_of_episode = False
                    else:
                        act_sequence = act_sequence[LATENCY_STEPS:self.n_action_steps]
                    for action in act_sequence:
                        self.act_queue.put(action)

                time.sleep(0.001)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'Error in inference loop: {e}')
                break


class PolicyLocal:
    """Runs the diffusion policy locally without a ZMQ server."""

    def __init__(self, ckpt_path):
        diffusion_policy = DiffusionPolicy(ckpt_path)

        # Get expected image sizes from the checkpoint's shape_meta
        self.image_sizes = {}
        for key, value in diffusion_policy.obs_shape_meta.items():
            if value.get('type') == 'rgb':
                # shape is [C, H, W]
                _, h, w = value['shape']
                self.image_sizes[key] = (w, h)

        self.policy = PolicyWrapper(diffusion_policy)
        print(f'Policy loaded from {ckpt_path}')

    def reset(self):
        self.policy.reset()

    def step(self, obs):
        # Remove base_pose from obs (policy operates in local frame)
        obs.pop('base_pose', None)

        # Resize images to the resolution expected by the policy checkpoint
        for k, v in obs.items():
            if k in self.image_sizes:
                obs[k] = cv.resize(v, self.image_sizes[k])

        action = self.policy.step(obs)
        return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', default='data/outputs/2024.10.08/23.42.04_train_diffusion_unet_hybrid_sim-v1/checkpoints/epoch=0500-train_loss=0.001.ckpt')
    args = parser.parse_args()
    policy = PolicyLocal(args.ckpt_path)