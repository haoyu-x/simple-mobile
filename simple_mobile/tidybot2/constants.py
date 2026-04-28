import numpy as np

################################################################################
# Mobile base

# Vehicle center to steer axis (m)
# h_x, h_y = 0.190150 * np.array([1.0, 1.0, -1.0, -1.0]), 0.170150 * np.array([-1.0, 1.0, 1.0, -1.0])  # Kinova / Franka
# h_x, h_y = 0.140150 * np.array([1.0, 1.0, -1.0, -1.0]), 0.120150 * np.array([-1.0, 1.0, 1.0, -1.0])  # ARX5
h_x, h_y = 0.140000 * np.array([1.0, 1.0, -1.0, -1.0]), 0.212000 * np.array([-1.0, 1.0, 1.0, -1.0])  # Hexfellow maverX4
# h_x, h_y = 0.160000 * np.array([1.0, 1.0, -1.0, -1.0]), 0.180000 * np.array([-1.0, 1.0, 1.0, -1.0])  # Hexfellow maverL4 default

# Encoder magnet offsets
ENCODER_MAGNET_OFFSETS = [0.0 / 4096, 0.0 / 4096, 0.0 / 4096, 0.0 / 4096]  # TODO

# Hexfellow Base
HEX_BASE_URL = 'ws://192.168.1.131:8439'
HEX_MOTOR_MAP = np.array(
    [
        (0, 7, -1),
        (1, 6, +1),
        (2, 1, -1),
        (3, 0, +1),
        (4, 3, -1),
        (5, 2, +1),
        (6, 5, -1),
        (7, 4, +1),
    ],
    dtype=[
        ('tidy_idx', 'i4'),
        ('motor_idx', 'i4'),
        ('reverse_factor', 'i4'),
    ],
)


################################################################################
# Teleop and imitation learning

# Base and arm RPC servers
BASE_RPC_HOST = 'localhost'
BASE_RPC_PORT = 50000
ARM_RPC_HOST = 'localhost'
ARM_RPC_PORT = 50001
RPC_AUTHKEY = b'secret password'

# Policy
POLICY_SERVER_HOST = 'localhost'
POLICY_SERVER_PORT = 5555
POLICY_CONTROL_FREQ = 10
POLICY_CONTROL_PERIOD = 1.0 / POLICY_CONTROL_FREQ

RAW_IMAGE_WIDTH = 640
RAW_IMAGE_HEIGHT = 480
POLICY_IMAGE_WIDTH = 128
POLICY_IMAGE_HEIGHT = 128

# iPhone bottom gamepad → Hex base local velocity (see templates/index.html, policies.TeleopController)
TELEOP_BASE_GAMEPAD_MAX_VEL = np.array([0.15, 0.15, 0.2])  # m/s, m/s, rad/s
TELEOP_BASE_GAMEPAD_DEADZONE = 0.05

# Two-iPhone bimanual YAM + Hex teleop (`--yam-bimanual-hex`): assign side by phone LAN IP (Wi‑Fi).
TELEOP_LEFT_IP = '10.29.233.176'
TELEOP_RIGHT_IP = '10.29.142.37'

HEAD_CAMERA_SERIAL = "1944301061907E4800"  # head / exogenous view
RIGHT_WRIST_CAMERA_SERIAL = "19443010B128714800"
LEFT_WRIST_CAMERA_SERIAL = "14442C10C1C5D3D200"

################################################################################
# YAM arm (i2rt MotorChainRobot + yam_server.py)

YAM_RPC_HOST = 'localhost'
YAM_RPC_PORT = 50002
# Second process for left arm (run e.g. `python yam_server.py --channel can_follower_l --rpc-port 50003`)
YAM_RPC_PORT_LEFT = 50003
YAM_RPC_PORT_RIGHT = 50002
YAM_CAN_CHANNEL_LEFT = 'can_follower_l'
YAM_CAN_CHANNEL_RIGHT = 'can_follower_r'
# SocketCAN interface name (see `ip link`). Errno 19 (ENODEV) = wrong name or link not up.
YAM_CAN_CHANNEL = 'can_follower_r'
YAM_ARM_TYPE_STR = 'yam'
YAM_GRIPPER_TYPE_STR = 'linear_4310'
# Fixed-rate arm servo (i2rt minimum_gello follower uses ~100 Hz joint commands).
YAM_CONTROL_HZ = 100.0
YAM_CONTROL_PERIOD = 1.0 / YAM_CONTROL_HZ
# 1.0 = command raw IK joints each tick; lower = extra low-pass on top of velocity IK (smoother, more lag).
YAM_JOINT_CMD_ALPHA = 1.0
YAM_SPACEMOUSE_PATH = '/dev/hidraw4'
YAM_USE_SIM = False
YAM_RESET_DURATION_S = 3.0
# MuJoCo joint order (same as ``MotorChainRobot.get_joint_pos``): joint1..joint6 (rad) + gripper [0, 1].
# ``yam_server.YamArm.reset()`` ramps here with ``command_joint_pos``; tune for your workspace.
YAM_INITIAL_JOINTS = np.array([0.0, 0.4, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
