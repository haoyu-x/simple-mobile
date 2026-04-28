# Author: Jimmy Wu and Haoyu Xiong
# Date: October 2024

import logging
import math
import socket
import threading
import time
from queue import Empty, Queue
import cv2 as cv
import numpy as np
import zmq
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from scipy.spatial.transform import Rotation as R
from constants import POLICY_SERVER_HOST, POLICY_SERVER_PORT, POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT
from constants import TELEOP_BASE_GAMEPAD_MAX_VEL, TELEOP_BASE_GAMEPAD_DEADZONE

class Policy:
    def reset(self):
        raise NotImplementedError

    def step(self, obs):
        raise NotImplementedError

class WebServer:
    def __init__(self, queue):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.queue = queue

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.socketio.on('message')
        def handle_message(data):
            # Send the timestamp back for RTT calculation (expected RTT on 5 GHz Wi-Fi is 7 ms)
            emit('echo', data['timestamp'])

            # Add data to queue for processing (client IP tags left vs right phone in bimanual mode)
            self.queue.put((data, request.remote_addr))

        # Reduce verbose Flask log output
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

    def run(self):
        # Get IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(('8.8.8.8', 1))
            address = s.getsockname()[0]
        except Exception:
            address = '127.0.0.1'
        finally:
            s.close()
        print(f'Starting server at {address}:5000')
        self.socketio.run(self.app, host='0.0.0.0')

DEVICE_CAMERA_OFFSET = np.array([0.0, 0.02, -0.04])  # iPhone 14 Pro

# Convert coordinate system from WebXR to robot
def convert_webxr_pose(pos, quat):
    # WebXR: +x right, +y up, +z back; Robot: +x forward, +y left, +z up
    pos = np.array([-pos['z'], -pos['x'], pos['y']], dtype=np.float64)
    rot = R.from_quat([-quat['z'], -quat['x'], quat['y'], quat['w']])

    # Apply offset so that rotations are around device center instead of device camera
    pos = pos + rot.apply(DEVICE_CAMERA_OFFSET)

    return pos, rot

TWO_PI = 2 * math.pi


def teleop_base_gamepad_deadzone(arr, deadzone_size=TELEOP_BASE_GAMEPAD_DEADZONE):
    return np.where(
        np.abs(arr) <= deadzone_size,
        0,
        np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size),
    )


class TeleopController:
    def __init__(self):
        # Teleop device IDs
        self.primary_device_id = None    # Primary device controls either the arm or the base
        self.secondary_device_id = None  # Optional secondary device controls the base
        self.enabled_counts = {}

        # Mobile base pose
        self.base_pose = None

        # Teleop targets
        self.targets_initialized = False
        self.base_target_pose = None
        self.arm_target_pos = None
        self.arm_target_rot = None
        self.gripper_target_pos = None

        # WebXR reference poses
        self.base_xr_ref_pos = None
        self.base_xr_ref_rot_inv = None
        self.arm_xr_ref_pos = None
        self.arm_xr_ref_rot_inv = None

        # Robot reference poses
        self.base_ref_pose = None
        self.arm_ref_pos = None
        self.arm_ref_rot = None
        self.arm_ref_base_pose = None  # For optional secondary control of base
        self.gripper_ref_pos = None

        # Bottom-screen virtual gamepad: normalized [-1,1] (local base frame)
        self._gp_lock = threading.Lock()
        self._gp_norm = np.zeros(3, dtype=np.float64)

    def _ingest_base_gamepad(self, data):
        bg = data['base_gamepad']
        with self._gp_lock:
            self._gp_norm[:] = [float(bg['x']), float(bg['y']), float(bg['yaw'])]

    def process_message(self, data, remote_addr=None):
        del remote_addr  # single-device teleop ignores client IP
        if 'base_gamepad' in data:
            self._ingest_base_gamepad(data)
            return

        if not self.targets_initialized:
            return

        # Use device ID to disambiguate between primary and secondary devices
        device_id = data['device_id']

        # Update enabled count for the device that sent this message
        self.enabled_counts[device_id] = self.enabled_counts.get(device_id, 0) + 1 if 'teleop_mode' in data else 0

        # Assign primary and secondary devices
        if self.enabled_counts[device_id] > 2:
            if self.primary_device_id is None and device_id != self.secondary_device_id:
                # Note: We skip the first 2 steps because WebXR pose updates have higher latency than touch events
                self.primary_device_id = device_id
            elif self.secondary_device_id is None and device_id != self.primary_device_id:
                self.secondary_device_id = device_id
        elif self.enabled_counts[device_id] == 0:
            if device_id == self.primary_device_id:
                self.primary_device_id = None  # Primary device no longer enabled
                self.base_xr_ref_pos = None
                self.arm_xr_ref_pos = None
            elif device_id == self.secondary_device_id:
                self.secondary_device_id = None
                self.base_xr_ref_pos = None

        # Teleop is enabled
        if self.primary_device_id is not None and 'teleop_mode' in data:
            pos, rot = convert_webxr_pose(data['position'], data['orientation'])

            # Base movement
            if data['teleop_mode'] == 'base' or device_id == self.secondary_device_id:  # Note: Secondary device can only control base
                # Store reference poses
                if self.base_xr_ref_pos is None:
                    self.base_ref_pose = self.base_pose.copy()
                    self.base_xr_ref_pos = pos[:2]
                    self.base_xr_ref_rot_inv = rot.inv()

                # Position
                self.base_target_pose[:2] = self.base_ref_pose[:2] + (pos[:2] - self.base_xr_ref_pos)

                # Orientation
                base_fwd_vec_rotated = (rot * self.base_xr_ref_rot_inv).apply([1.0, 0.0, 0.0])
                base_target_theta = self.base_ref_pose[2] + math.atan2(base_fwd_vec_rotated[1], base_fwd_vec_rotated[0])
                self.base_target_pose[2] += (base_target_theta - self.base_target_pose[2] + math.pi) % TWO_PI - math.pi  # Unwrapped

            # Arm movement
            elif data['teleop_mode'] == 'arm':
                # Store reference poses
                if self.arm_xr_ref_pos is None:
                    self.arm_xr_ref_pos = pos
                    self.arm_xr_ref_rot_inv = rot.inv()
                    self.arm_ref_pos = self.arm_target_pos.copy()
                    self.arm_ref_rot = self.arm_target_rot
                    self.arm_ref_base_pose = self.base_pose.copy()
                    self.gripper_ref_pos = self.gripper_target_pos

                # Rotations around z-axis to go between global frame (base) and local frame (arm)
                z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.base_pose[2])
                z_rot_inv = z_rot.inv()
                ref_z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.arm_ref_base_pose[2])

                # Position
                pos_diff = pos - self.arm_xr_ref_pos  # WebXR
                pos_diff += ref_z_rot.apply(self.arm_ref_pos) - z_rot.apply(self.arm_ref_pos)  # Secondary base control: Compensate for base rotation
                pos_diff[:2] += self.arm_ref_base_pose[:2] - self.base_pose[:2]  # Secondary base control: Compensate for base translation
                self.arm_target_pos = self.arm_ref_pos + z_rot_inv.apply(pos_diff)

                # Orientation
                self.arm_target_rot = (z_rot_inv * (rot * self.arm_xr_ref_rot_inv) * ref_z_rot) * self.arm_ref_rot

                # Gripper position
                self.gripper_target_pos = np.clip(self.gripper_ref_pos + data['gripper_delta'], 0.0, 1.0)

        # Teleop is disabled
        elif self.primary_device_id is None:
            # Update target pose in case base is pushed while teleop is disabled
            self.base_target_pose = self.base_pose

    def step(self, obs):
        # Update robot state
        self.base_pose = obs['base_pose']

        # Initialize targets
        if not self.targets_initialized:
            self.base_target_pose = obs['base_pose']
            self.arm_target_pos = obs['arm_pos']
            self.arm_target_rot = R.from_quat(obs['arm_quat'])
            self.gripper_target_pos = obs['gripper_pos']
            self.targets_initialized = True

        with self._gp_lock:
            gp_cmd = teleop_base_gamepad_deadzone(self._gp_norm.copy())
        gp_active = bool(np.max(np.abs(gp_cmd)) > 1e-5)

        # Allow Hex-style drive from gamepad before canvas touch; WebXR arm/base unchanged when gamepad idle
        if self.primary_device_id is None and not gp_active:
            return None

        # Get most recent teleop command
        arm_quat = self.arm_target_rot.as_quat()
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness (Note: Not strictly necessary since policy training uses 6D rotation representation)
            np.negative(arm_quat, out=arm_quat)
        action = {
            'arm_pos': self.arm_target_pos.copy(),
            'arm_quat': arm_quat,
            'gripper_pos': self.gripper_target_pos.copy(),
        }
        if gp_active:
            action['base_velocity'] = TELEOP_BASE_GAMEPAD_MAX_VEL * gp_cmd
        else:
            action['base_velocity'] = np.zeros(3, dtype=np.float64)

        return action

    def hold_action(self, obs):
        """Zero base velocity + current arm targets (WebXR ended; keep Hex command stream alive)."""
        self.base_pose = obs['base_pose']
        if not self.targets_initialized:
            self.base_target_pose = obs['base_pose'].copy()
            self.arm_target_pos = obs['arm_pos'].copy()
            self.arm_target_rot = R.from_quat(obs['arm_quat'])
            self.gripper_target_pos = obs['gripper_pos'].copy()
            self.targets_initialized = True
        arm_quat = self.arm_target_rot.as_quat()
        if arm_quat[3] < 0.0:
            np.negative(arm_quat, out=arm_quat)
        return {
            'arm_pos': self.arm_target_pos.copy(),
            'arm_quat': arm_quat,
            'gripper_pos': self.gripper_target_pos.copy(),
            'base_velocity': np.zeros(3, dtype=np.float64),
        }


class BimanualYamHexTeleopController:
    """Two iPhones: first to connect → left YAM + base (vx, vy); second → right YAM + base ω.

    Sides are assigned automatically in order of first message received (device UUID), so no IP
    configuration is needed. The web app is unchanged: each phone streams ``base_gamepad`` (we take
    x,y from left, yaw from right) and WebXR ``teleop_mode == 'arm'`` for the corresponding arm.
    """

    def __init__(self):
        self.targets_initialized = False

        # Device IDs assigned in order of first message received
        self.device_id_left = None
        self.device_id_right = None

        self.arm_target_pos_l = None
        self.arm_target_rot_l = None
        self.gripper_target_l = None
        self.arm_xr_ref_pos_l = None
        self.arm_xr_ref_rot_inv_l = None
        self.arm_ref_pos_l = None
        self.arm_ref_rot_l = None
        self.gripper_ref_l = None

        self.arm_target_pos_r = None
        self.arm_target_rot_r = None
        self.gripper_target_r = None
        self.arm_xr_ref_pos_r = None
        self.arm_xr_ref_rot_inv_r = None
        self.arm_ref_pos_r = None
        self.arm_ref_rot_r = None
        self.gripper_ref_r = None

        self._gp_lock = threading.Lock()
        self._gp_left_xy = np.zeros(2, dtype=np.float64)
        self._gp_right_yaw = 0.0
        self._left_arm_touch = False
        self._right_arm_touch = False

    def _assign_side(self, device_id):
        """Assign device_id to left or right in order of first appearance; return the side."""
        if device_id == self.device_id_left:
            return "left"
        if device_id == self.device_id_right:
            return "right"
        if self.device_id_left is None:
            self.device_id_left = device_id
            print(f'[BimanualYamHexTeleop] assigned LEFT  to device {device_id!r}')
            return "left"
        if self.device_id_right is None:
            self.device_id_right = device_id
            print(f'[BimanualYamHexTeleop] assigned RIGHT to device {device_id!r}')
            return "right"
        return None  # Third+ device; ignore

    def process_message(self, data, remote_addr=None):
        device_id = data.get('device_id')
        if device_id is None:
            return

        side = self._assign_side(device_id)
        if side is None:
            return

        if "base_gamepad" in data:
            with self._gp_lock:
                if side == "left":
                    bg = data["base_gamepad"]
                    self._gp_left_xy[:] = [float(bg["x"]), float(bg["y"])]
                elif side == "right":
                    bg = data["base_gamepad"]
                    self._gp_right_yaw = float(bg["yaw"])
            return

        if not self.targets_initialized:
            return

        if side == "left":
            self._left_arm_touch = "teleop_mode" in data and data.get("teleop_mode") == "arm"
            if not self._left_arm_touch:
                self.arm_xr_ref_pos_l = None
        else:
            self._right_arm_touch = "teleop_mode" in data and data.get("teleop_mode") == "arm"
            if not self._right_arm_touch:
                self.arm_xr_ref_pos_r = None

        if not ("teleop_mode" in data and data["teleop_mode"] == "arm"):
            return

        pos, rot = convert_webxr_pose(data["position"], data["orientation"])
        if side == "left":
            self._apply_arm_delta_l(data, pos, rot)
        else:
            self._apply_arm_delta_r(data, pos, rot)

    def _apply_arm_delta_l(self, data, pos, rot):
        if self.arm_xr_ref_pos_l is None:
            self.arm_xr_ref_pos_l = pos
            self.arm_xr_ref_rot_inv_l = rot.inv()
            self.arm_ref_pos_l = self.arm_target_pos_l.copy()
            self.arm_ref_rot_l = self.arm_target_rot_l
            self.gripper_ref_l = self.gripper_target_l

        self.arm_target_pos_l = self.arm_ref_pos_l + (pos - self.arm_xr_ref_pos_l)
        self.arm_target_rot_l = (rot * self.arm_xr_ref_rot_inv_l) * self.arm_ref_rot_l
        self.gripper_target_l = np.clip(self.gripper_ref_l + data["gripper_delta"], 0.0, 1.0)

    def _apply_arm_delta_r(self, data, pos, rot):
        if self.arm_xr_ref_pos_r is None:
            self.arm_xr_ref_pos_r = pos
            self.arm_xr_ref_rot_inv_r = rot.inv()
            self.arm_ref_pos_r = self.arm_target_pos_r.copy()
            self.arm_ref_rot_r = self.arm_target_rot_r
            self.gripper_ref_r = self.gripper_target_r

        self.arm_target_pos_r = self.arm_ref_pos_r + (pos - self.arm_xr_ref_pos_r)
        self.arm_target_rot_r = (rot * self.arm_xr_ref_rot_inv_r) * self.arm_ref_rot_r
        self.gripper_target_r = np.clip(self.gripper_ref_r + data["gripper_delta"], 0.0, 1.0)

    def step(self, obs):
        if not self.targets_initialized:
            self.arm_target_pos_l = obs["arm_pos_l"].copy()
            self.arm_target_rot_l = R.from_quat(obs["arm_quat_l"])
            self.gripper_target_l = obs["gripper_pos_l"].copy()
            self.arm_target_pos_r = obs["arm_pos_r"].copy()
            self.arm_target_rot_r = R.from_quat(obs["arm_quat_r"])
            self.gripper_target_r = obs["gripper_pos_r"].copy()
            self.targets_initialized = True

        with self._gp_lock:
            gp_vec = np.array(
                [self._gp_left_xy[0], self._gp_left_xy[1], self._gp_right_yaw],
                dtype=np.float64,
            )
        gp_cmd = teleop_base_gamepad_deadzone(gp_vec)
        gp_active = bool(np.max(np.abs(gp_cmd)) > 1e-5)

        if not gp_active and not self._left_arm_touch and not self._right_arm_touch:
            return None

        def quat_positive_w(rot: R):
            q = rot.as_quat()
            if q[3] < 0.0:
                np.negative(q, out=q)
            return q

        action = {
            "arm_left": {
                "arm_pos": self.arm_target_pos_l.copy(),
                "arm_quat": quat_positive_w(self.arm_target_rot_l),
                "gripper_pos": self.gripper_target_l.copy(),
            },
            "arm_right": {
                "arm_pos": self.arm_target_pos_r.copy(),
                "arm_quat": quat_positive_w(self.arm_target_rot_r),
                "gripper_pos": self.gripper_target_r.copy(),
            },
        }
        if gp_active:
            action["base_velocity"] = TELEOP_BASE_GAMEPAD_MAX_VEL * gp_cmd
        else:
            action["base_velocity"] = np.zeros(3, dtype=np.float64)

        return action

    def hold_action(self, obs):
        """Zero base velocity + hold both arm targets (WebXR ended; keep Hex alive)."""
        if not self.targets_initialized:
            self.arm_target_pos_l = obs["arm_pos_l"].copy()
            self.arm_target_rot_l = R.from_quat(obs["arm_quat_l"])
            self.gripper_target_l = obs["gripper_pos_l"].copy()
            self.arm_target_pos_r = obs["arm_pos_r"].copy()
            self.arm_target_rot_r = R.from_quat(obs["arm_quat_r"])
            self.gripper_target_r = obs["gripper_pos_r"].copy()
            self.targets_initialized = True

        def quat_positive_w(rot: R):
            q = rot.as_quat()
            if q[3] < 0.0:
                np.negative(q, out=q)
            return q

        return {
            "arm_left": {
                "arm_pos": self.arm_target_pos_l.copy(),
                "arm_quat": quat_positive_w(self.arm_target_rot_l),
                "gripper_pos": self.gripper_target_l.copy(),
            },
            "arm_right": {
                "arm_pos": self.arm_target_pos_r.copy(),
                "arm_quat": quat_positive_w(self.arm_target_rot_r),
                "gripper_pos": self.gripper_target_r.copy(),
            },
            "base_velocity": np.zeros(3, dtype=np.float64),
        }

class SinglePhoneBimanualBaseOnlyTeleopController:
    """Single iPhone -> base (vx, vy, yaw) only; both arms held at their current pose.

    Used after ``end_episode`` during automated rollouts: the operator teleops the mobile base
    back to the reset area without perturbing the arms (no WebXR arm input is consumed).
    Takes the first device that sends a message and ignores subsequent devices.
    """

    def __init__(self):
        self.targets_initialized = False
        self.device_id = None  # Locked to first device seen

        self.arm_target_pos_l = None
        self.arm_target_rot_l = None
        self.gripper_target_l = None
        self.arm_target_pos_r = None
        self.arm_target_rot_r = None
        self.gripper_target_r = None

        self._gp_lock = threading.Lock()
        self._gp_xyyaw = np.zeros(3, dtype=np.float64)

    def process_message(self, data, remote_addr=None):
        device_id = data.get('device_id')
        if device_id is None:
            return
        if self.device_id is None:
            self.device_id = device_id
            print(f'[SinglePhoneBimanualBaseOnlyTeleop] assigned BASE to device {device_id!r}')
        elif device_id != self.device_id:
            return  # Ignore extra phones

        if "base_gamepad" in data:
            bg = data["base_gamepad"]
            with self._gp_lock:
                self._gp_xyyaw[0] = float(bg["x"])
                self._gp_xyyaw[1] = float(bg["y"])
                self._gp_xyyaw[2] = float(bg["yaw"])

    def step(self, obs):
        if not self.targets_initialized:
            self.arm_target_pos_l = obs["arm_pos_l"].copy()
            self.arm_target_rot_l = R.from_quat(obs["arm_quat_l"])
            self.gripper_target_l = obs["gripper_pos_l"].copy()
            self.arm_target_pos_r = obs["arm_pos_r"].copy()
            self.arm_target_rot_r = R.from_quat(obs["arm_quat_r"])
            self.gripper_target_r = obs["gripper_pos_r"].copy()
            self.targets_initialized = True

        with self._gp_lock:
            gp_vec = self._gp_xyyaw.copy()
        gp_cmd = teleop_base_gamepad_deadzone(gp_vec)
        gp_active = bool(np.max(np.abs(gp_cmd)) > 1e-5)

        if not gp_active:
            return None

        def quat_positive_w(rot: R):
            q = rot.as_quat()
            if q[3] < 0.0:
                np.negative(q, out=q)
            return q

        return {
            "arm_left": {
                "arm_pos": self.arm_target_pos_l.copy(),
                "arm_quat": quat_positive_w(self.arm_target_rot_l),
                "gripper_pos": self.gripper_target_l.copy(),
            },
            "arm_right": {
                "arm_pos": self.arm_target_pos_r.copy(),
                "arm_quat": quat_positive_w(self.arm_target_rot_r),
                "gripper_pos": self.gripper_target_r.copy(),
            },
            "base_velocity": TELEOP_BASE_GAMEPAD_MAX_VEL * gp_cmd,
        }

    def hold_action(self, obs):
        """Zero base velocity + hold both arm targets (phone idle; keep Hex alive)."""
        if not self.targets_initialized:
            self.arm_target_pos_l = obs["arm_pos_l"].copy()
            self.arm_target_rot_l = R.from_quat(obs["arm_quat_l"])
            self.gripper_target_l = obs["gripper_pos_l"].copy()
            self.arm_target_pos_r = obs["arm_pos_r"].copy()
            self.arm_target_rot_r = R.from_quat(obs["arm_quat_r"])
            self.gripper_target_r = obs["gripper_pos_r"].copy()
            self.targets_initialized = True

        def quat_positive_w(rot: R):
            q = rot.as_quat()
            if q[3] < 0.0:
                np.negative(q, out=q)
            return q

        return {
            "arm_left": {
                "arm_pos": self.arm_target_pos_l.copy(),
                "arm_quat": quat_positive_w(self.arm_target_rot_l),
                "gripper_pos": self.gripper_target_l.copy(),
            },
            "arm_right": {
                "arm_pos": self.arm_target_pos_r.copy(),
                "arm_quat": quat_positive_w(self.arm_target_rot_r),
                "gripper_pos": self.gripper_target_r.copy(),
            },
            "base_velocity": np.zeros(3, dtype=np.float64),
        }


# Teleop using WebXR phone web app
class TeleopPolicy(Policy):
    def __init__(self):
        self.web_server_queue = Queue()
        self.teleop_controller = None
        self._state_lock = threading.Lock()
        self.teleop_state = None  # States: episode_started -> episode_ended -> reset_env
        self.episode_ended = False

        # Web server for serving the WebXR phone web app
        server = WebServer(self.web_server_queue)
        threading.Thread(target=server.run, daemon=True).start()

        # Listener thread to process messages from WebXR client
        threading.Thread(target=self.listener_loop, daemon=True).start()

    def _drain_web_queue(self):
        """Drop pending phone messages so a new session is not confused by the previous episode."""
        while True:
            try:
                self.web_server_queue.get_nowait()
            except Empty:
                break

    def reset(self):
        self._drain_web_queue()
        self.teleop_controller = TeleopController()
        with self._state_lock:
            self.episode_ended = False
            self.teleop_state = None

        # Wait for user to signal that episode has started
        while True:
            with self._state_lock:
                if self.teleop_state == 'episode_started':
                    break
            time.sleep(0.01)

    def step(self, obs):
        with self._state_lock:
            # Signal that user has ended episode
            if not self.episode_ended and self.teleop_state == 'episode_ended':
                self.episode_ended = True
                return 'end_episode'

            # "Reset env" should only end the outer loop after we have ended an episode; otherwise
            # a late reset_env from the phone can block all env.step (no base/arm commands).
            if self.teleop_state == 'reset_env':
                if self.episode_ended:
                    return 'reset_env'
                self.teleop_state = 'episode_started'

            episode_ended = self.episode_ended

        action = self._step(obs)
        # After "End episode", WebXR and gamepad stop → controller returns None, but Hex still needs
        # periodic execute_action(base_velocity=0) to avoid firmware / command-stream faults.
        if episode_ended and action is None and self.teleop_controller is not None:
            hold = self.teleop_controller.hold_action(obs)
            if hold is not None:
                action = hold
        return action

    def _step(self, obs):
        return self.teleop_controller.step(obs)

    def listener_loop(self):
        while True:
            try:
                item = self.web_server_queue.get_nowait()
            except Empty:
                time.sleep(0.001)
                continue

            if isinstance(item, tuple) and len(item) == 2:
                data, remote_addr = item
            else:
                data, remote_addr = item, None

            # Update state
            if 'state_update' in data:
                new_state = data['state_update']
                with self._state_lock:
                    # Ignore reset_env while we are not in "ended episode" mode (stale from last session).
                    if new_state == 'reset_env' and not self.episode_ended:
                        pass
                    else:
                        self.teleop_state = new_state

            # Process message if not stale
            elif 1000 * time.time() - data['timestamp'] < 250:  # 250 ms
                self._process_message(data, remote_addr)

    def _process_message(self, data, remote_addr=None):
        # Listener runs as soon as __init__ starts; messages can arrive before policy.reset()
        # (e.g. during env.reset()) when teleop_controller is still None.
        if self.teleop_controller is None:
            return
        self.teleop_controller.process_message(data, remote_addr)


class BimanualYamHexTeleopPolicy(TeleopPolicy):
    """Same WebXR app / Flask server as ``TeleopPolicy``; assigns sides by client IP (see ``constants``)."""

    def reset(self):
        self._drain_web_queue()
        self.teleop_controller = BimanualYamHexTeleopController()
        with self._state_lock:
            self.episode_ended = False
            self.teleop_state = None
        while True:
            with self._state_lock:
                if self.teleop_state == 'episode_started':
                    break
            time.sleep(0.01)

# Execute policy running on remote server
class RemotePolicy(TeleopPolicy):
    def __init__(self):
        super().__init__()

        # Use phone as enabling device during policy rollout
        self.enabled = False

        # Connection to policy server
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f'tcp://{POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}')
        print(f'Connected to policy server at {POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}')

    def reset(self):
        # Wait for user to signal that episode has started
        super().reset()  # Note: Comment out to run without phone

        # Check connection to policy server and reset policy
        default_timeout = self.socket.getsockopt(zmq.RCVTIMEO)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # Temporarily set 1000 ms timeout
        self.socket.send_pyobj({'reset': True})
        try:
            self.socket.recv_pyobj()  # Note: Not secure. Only unpickle data you trust.
        except zmq.error.Again as e:
            raise Exception('Could not communicate with policy server') from e
        self.socket.setsockopt(zmq.RCVTIMEO, default_timeout)  # Put default timeout back

        # Disable policy execution until user presses on screen
        self.enabled = False  # Note: Set to True to run without phone

    def _step(self, obs):
        # Return teleop command if episode has ended
        if self.episode_ended:
            return self.teleop_controller.step(obs)

        # Return no action if robot is not enabled
        if not self.enabled:
            return None

        # Encode images
        encoded_obs = {}
        for k, v in obs.items():
            if v.ndim == 3:
                # Resize image to resolution expected by policy server
                v = cv.resize(v, (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT))

                # Encode image as JPEG
                _, v = cv.imencode('.jpg', v)  # Note: Interprets RGB as BGR
                encoded_obs[k] = v
            else:
                encoded_obs[k] = v

        # Send obs to policy server
        req = {'obs': encoded_obs}
        self.socket.send_pyobj(req)

        # Get action from policy server
        rep = self.socket.recv_pyobj()  # Note: Not secure. Only unpickle data you trust.
        action = rep['action']

        return action

    def _process_message(self, data, remote_addr=None):
        if self.teleop_controller is None:
            return
        with self._state_lock:
            episode_ended = self.episode_ended
        if episode_ended:
            # Run teleop controller if episode has ended
            self.teleop_controller.process_message(data, remote_addr)
        else:
            # Enable policy execution if user is pressing on screen
            self.enabled = 'teleop_mode' in data


class LocalPhonePolicy(TeleopPolicy):
    """PolicyLocal with iPhone for episode control (start / end / reset) only — no touch-to-enable.

    After **Start episode**, the diffusion policy runs every step until **End episode**; then WebXR
    teleop controls the robot until **Reset env**.
    """

    def __init__(self, ckpt_path, policy_local_cls=None):
        super().__init__()
        if policy_local_cls is None:
            from policy_local import PolicyLocal as policy_local_cls
        self._local = policy_local_cls(ckpt_path)

    def reset(self):
        super().reset()
        self._local.reset()

    def _step(self, obs):
        if self.episode_ended:
            return self.teleop_controller.step(obs)
        # Shallow copy so PolicyLocal can pop/resize without mutating env obs dict
        return self._local.step(dict(obs))

    def _process_message(self, data, remote_addr=None):
        if self.teleop_controller is None:
            return
        with self._state_lock:
            episode_ended = self.episode_ended
        if episode_ended:
            self.teleop_controller.process_message(data, remote_addr)


class LocalPhoneBimanualPolicy(LocalPhonePolicy):
    """Same as LocalPhonePolicy but uses ``BimanualYamHexTeleopController`` after ``end_episode`` (real bimanual env)."""

    def reset(self):
        self._drain_web_queue()
        self.teleop_controller = BimanualYamHexTeleopController()
        with self._state_lock:
            self.episode_ended = False
            self.teleop_state = None
        while True:
            with self._state_lock:
                if self.teleop_state == 'episode_started':
                    break
            time.sleep(0.01)
        self._local.reset()


class LocalPhoneBimanualBaseOnlyPolicy(LocalPhonePolicy):
    """Like ``LocalPhoneBimanualPolicy`` but after ``end_episode`` a single phone teleops the base only.

    Intended for automated rollout reset: operator drives the mobile base back to start with one
    phone (xy + yaw); both YAM arms remain held at their current pose.
    """

    def reset(self):
        self._drain_web_queue()
        self.teleop_controller = SinglePhoneBimanualBaseOnlyTeleopController()
        with self._state_lock:
            self.episode_ended = False
            self.teleop_state = None
        while True:
            with self._state_lock:
                if self.teleop_state == 'episode_started':
                    break
            time.sleep(0.01)
        self._local.reset()


if __name__ == '__main__':
    # WebServer(Queue()).run(); time.sleep(1000)
    # WebXRListener(); time.sleep(1000)
    from constants import POLICY_CONTROL_PERIOD
    obs = {
        # 'base_pose': np.zeros(3),
        'arm_pos': np.zeros(3),
        'arm_quat': np.array([0.0, 0.0, 0.0, 1.0]),
        'gripper_pos': np.zeros(1),
        'base_image': np.zeros((640, 360, 3)),
        'wrist_image': np.zeros((640, 480, 3)),
    }
    policy = TeleopPolicy()
    # policy = RemotePolicy()
    while True:
        policy.reset()
        for _ in range(100):
            print(policy.step(obs))
            time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
