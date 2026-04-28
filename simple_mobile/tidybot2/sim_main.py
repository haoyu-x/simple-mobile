# Teleop in MuJoCo simulation
# Simplified from main.py — sim + teleop only

import argparse
import time
from itertools import count

from constants import POLICY_CONTROL_PERIOD
from episode_storage import EpisodeWriter
from mujoco_env import MujocoEnv
from policies import TeleopPolicy


def run_episode(env, policy, writer=None):
    print('Resetting env...')
    env.reset()
    print('Env has been reset')

    print('Press "Start episode" in the web app when ready to start new episode')
    policy.reset()
    print('Starting new episode')

    episode_ended = False
    start_time = time.time()
    for step_idx in count():
        # Enforce desired control freq
        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)

        obs = env.get_obs()
        action = policy.step(obs)

        if action is None:
            continue

        if isinstance(action, dict):
            env.step(action)
            if writer is not None and not episode_ended:
                writer.step(obs, action)

        elif not episode_ended and action == 'end_episode':
            episode_ended = True
            print('Episode ended')
            if writer is not None:
                if len(writer) == 0:
                    print('Discarding empty episode')
                else:
                    writer.flush_async()
            print('Teleop is now active. Press "Reset env" in the web app when ready to proceed.')

        elif episode_ended and action == 'reset_env':
            break

    if writer is not None:
        writer.wait_for_flush()


def main(args):
    env = MujocoEnv(show_images=True)
    policy = TeleopPolicy()

    try:
        while True:
            writer = EpisodeWriter(args.output_dir) if args.save else None
            run_episode(env, policy, writer)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Teleop in MuJoCo simulation')
    parser.add_argument('--save', action='store_true', help='Save episode data')
    parser.add_argument('--output-dir', default='data/demos', help='Directory to save episodes')
    main(parser.parse_args())
