# Real env: Hex base (RPC) + two YAM arms (two yam_server processes on YAM_RPC_PORT_LEFT / RIGHT).
# Observations use camera-shaped zero arrays only (no cameras). See policies.BimanualYamHexTeleopPolicy.

import threading

import numpy as np

from cameras import OAKCamera
from constants import BASE_RPC_HOST, BASE_RPC_PORT, RPC_AUTHKEY, YAM_RPC_HOST
from constants import YAM_RPC_PORT_LEFT, YAM_RPC_PORT_RIGHT
from constants import HEAD_CAMERA_SERIAL, LEFT_WRIST_CAMERA_SERIAL, RIGHT_WRIST_CAMERA_SERIAL
from hex_base_server import BaseManager
from yam_server import YamArmManager


class RealYamBimanualHexEnv:
    def __init__(self, use_cameras=True):
        base_manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            base_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception(
                "Could not connect to base RPC server, is hex_base_server.py running?"
            ) from e

        yam_l = YamArmManager(address=(YAM_RPC_HOST, YAM_RPC_PORT_LEFT), authkey=RPC_AUTHKEY)
        try:
            yam_l.connect()
        except ConnectionRefusedError as e:
            raise Exception(
                f"Could not connect to left YAM RPC on {YAM_RPC_HOST}:{YAM_RPC_PORT_LEFT}. "
                "Start e.g. `python yam_server.py --channel can_follower_l --rpc-port "
                f"{YAM_RPC_PORT_LEFT}`."
            ) from e

        yam_r = YamArmManager(address=(YAM_RPC_HOST, YAM_RPC_PORT_RIGHT), authkey=RPC_AUTHKEY)
        try:
            yam_r.connect()
        except ConnectionRefusedError as e:
            raise Exception(
                f"Could not connect to right YAM RPC on {YAM_RPC_HOST}:{YAM_RPC_PORT_RIGHT}. "
                "Start e.g. `python yam_server.py --channel can_follower_r --rpc-port "
                f"{YAM_RPC_PORT_RIGHT}`."
            ) from e

        self.base = base_manager.Base(max_vel=(0.5, 0.5, 1.57), max_accel=(0.15, 0.15, 0.5))
        self.arm_left = yam_l.YamArm()
        self.arm_right = yam_r.YamArm()

        # Cameras — opened in parallel to avoid 3× sequential firmware-upload delay.
        # NOTE: All 3 OAK cameras share a USB 2 hub (Bus 03, ~60 MB/s effective).
        # 3x 640x480 BGR @ 30fps = 83 MB/s → exceeds budget → device crashes.
        # fps=15 → 41 MB/s, fits safely. Move cameras to a USB 3 port to run at 30fps.
        if use_cameras:
            self.head_camera = None
            self.left_wrist_camera = None
            self.right_wrist_camera = None

            errors = {}
            def _open(attr, device_id):
                try:
                    setattr(self, attr, OAKCamera(device_id=device_id, fps=15))
                except Exception as e:
                    errors[attr] = e

            threads = [
                threading.Thread(target=_open, args=('head_camera', HEAD_CAMERA_SERIAL)),
                threading.Thread(target=_open, args=('left_wrist_camera', LEFT_WRIST_CAMERA_SERIAL)),
                threading.Thread(target=_open, args=('right_wrist_camera', RIGHT_WRIST_CAMERA_SERIAL)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            if errors:
                raise RuntimeError(f'Failed to open cameras: {errors}')
        else:
            self.head_camera = None
            self.left_wrist_camera = None
            self.right_wrist_camera = None

    def get_obs(self):
        obs = {}
        # obs["base_pose"] = self.base.get_state()["base_pose"]

        state_l = self.arm_left.get_state()
        obs["arm_pos_l"] = state_l["arm_pos"]
        obs["arm_quat_l"] = state_l["arm_quat"]
        obs["gripper_pos_l"] = state_l["gripper_pos"]
        state_r = self.arm_right.get_state()
        obs["arm_pos_r"] = state_r["arm_pos"]
        obs["arm_quat_r"] = state_r["arm_quat"]
        obs["gripper_pos_r"] = state_r["gripper_pos"]
    
        obs["head_image"] = self.head_camera.get_image() if self.head_camera is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        obs["left_wrist_image"] = self.left_wrist_camera.get_image() if self.left_wrist_camera is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        obs["right_wrist_image"] = self.right_wrist_camera.get_image() if self.right_wrist_camera is not None else np.zeros((480, 640, 3), dtype=np.uint8)
       
        return obs

    def reset(self):
        """Hex: soft zero-velocity hold (see ``hex_base_server``). Both YAM RPC arms: full ``YamArm.reset()`` homing."""
        print("Resetting Hex base + bimanual YAM...")
        self.base.reset()
        # Left / right are separate yam_server processes; each must receive reset() for TCP home pose.
        self.arm_left.reset()
        self.arm_right.reset()
        print("Robot has been reset")

    def step(self, action):
        if action is None:
            return
        self.base.execute_action({"base_velocity": action["base_velocity"]})
        self.arm_left.execute_action(action["arm_left"])
        self.arm_right.execute_action(action["arm_right"])

    def get_cameras(self):
        """Return a name→OAKCamera dict for visualization (empty if cameras disabled)."""
        cameras = {}
        if self.head_camera is not None:
            cameras['head'] = self.head_camera
        if self.left_wrist_camera is not None:
            cameras['left_wrist'] = self.left_wrist_camera
        if self.right_wrist_camera is not None:
            cameras['right_wrist'] = self.right_wrist_camera
        return cameras

    def close(self):
        self.base.close()
        self.arm_left.close()
        self.arm_right.close()
        for cam in (self.head_camera, self.left_wrist_camera, self.right_wrist_camera):
            if cam is not None:
                cam.close()
