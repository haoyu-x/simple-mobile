# Automated data generation script for policy rollouts (real bimanual YAM+Hex by default; --sim for MuJoCo).

import argparse
import multiprocessing as mp
import time
from tqdm import tqdm
from constants import POLICY_CONTROL_PERIOD
from episode_storage import EpisodeWriter

mp.set_start_method('spawn', force=True)


def rollout_episode(env, policy, writer=None, max_steps=150):
    """Run a single episode for automated rollout."""
    env.reset()
    policy.reset()
    print('Robot reset. Use the iPhone web app to tap Start episode when ready.')

    episode_ended = False
    start_time = time.time()
    step_idx = 0
    while True:
        # Cap pre-end_episode phase by max_steps; after end_episode, loop until iPhone reset.
        if not episode_ended and step_idx >= max_steps:
            episode_ended = True
            if writer is not None:
                if len(writer) == 0:
                    print('Discarding empty episode')
                else:
                    writer.flush_async()
            print('Max steps reached. Tap Reset on the iPhone web app to continue.')

        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)
        step_idx += 1

        obs = env.get_obs()
        action = policy.step(obs)

        if action is None:
            continue

        if isinstance(action, dict):
            env.step(action)
            if writer is not None and not episode_ended:
                writer.step(obs, action)
            continue

        if not episode_ended and action == 'end_episode':
            episode_ended = True
            if writer is not None:
                if len(writer) == 0:
                    print('Discarding empty episode')
                else:
                    writer.flush_async()
            continue

        if episode_ended and action == 'reset_env':
            break

    if writer is not None:
        if len(writer) > 0:
            writer.flush_async()
        writer.wait_for_flush()


def main(args):
    if args.sim:
        from mujoco_env import MujocoEnv
        env = MujocoEnv(show_viewer=not args.offscreen)

        from policy_local_mujoco import PolicyLocal as PolicyLocalCls
    else:
        from real_yam_bimanual_hex_env import RealYamBimanualHexEnv
        env = RealYamBimanualHexEnv(use_cameras=not args.no_cameras)

        from policy_local import PolicyLocal as PolicyLocalCls

    from policies import LocalPhoneBimanualBaseOnlyPolicy, LocalPhonePolicy

    if args.sim:
        policy = LocalPhonePolicy(args.ckpt_path, policy_local_cls=PolicyLocalCls)
    else:
        # Single phone: base xy+yaw teleop to drive the robot back after end_episode; arms held.
        policy = LocalPhoneBimanualBaseOnlyPolicy(args.ckpt_path, policy_local_cls=PolicyLocalCls)

    try:
        for _ in tqdm(range(args.num_episodes), desc='Episodes'):
            writer = EpisodeWriter(args.output_dir) if args.save else None
            rollout_episode(env, policy, writer, max_steps=args.max_steps)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automated policy rollouts (real bimanual YAM+Hex; --sim for MuJoCo)')
    parser.add_argument('--sim', action='store_true', help='Run in MuJoCo simulation instead of the real bimanual robot')
    parser.add_argument('--offscreen', action='store_true', help='MuJoCo offscreen mode (no viewer window); sim only')
    parser.add_argument('--no-cameras', action='store_true', help='Disable OAK cameras on real env (use zero images)')
    parser.add_argument('--save', action='store_true', help='Save rollout data')
    parser.add_argument('--output-dir', default='data/rollouts', help='Output directory for saved data')
    parser.add_argument('--num-episodes', type=int, default=20, help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=200, help='Max steps per episode')
    parser.add_argument('--ckpt-path', default='diffusion_policy/data/outputs/2024.10.08/23.42.04_train_diffusion_unet_hybrid_sim-v1/checkpoints/epoch=0500-train_loss=0.001.ckpt',
                        help='Path to policy checkpoint')
    main(parser.parse_args())
