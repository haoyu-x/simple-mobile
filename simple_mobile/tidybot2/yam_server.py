# RPC server for a single YAM arm (I2RT MotorChainRobot) with PyRoKi + J-PARSE IK.
#
# Dependencies (install in addition to tidybot2 base requirements):
#   pip install jax jaxlib jaxlie yourdfpy pyroki pyspacemouse
#   pip install -e ../i2rt
#
# Phone / policy teleop (tidybot2 main.py): run this server, then real_yam_env + --yam.
# The arm servos at YAM_CONTROL_HZ with latest Cartesian goal (i2rt-style streaming, not per-RPC IK bursts).
# SpaceMouse EE teleop (no RPC): python yam_server.py --spacemouse
#
# Use multiprocessing "spawn" (set below) so JAX is not initialized in a forked process.

from __future__ import annotations

import argparse
import sys
import threading
import time
from multiprocessing.managers import BaseManager as MPBaseManager
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy
from scipy.spatial.transform import Rotation as R

from constants import RPC_AUTHKEY
from constants import YAM_ARM_TYPE_STR
from constants import YAM_CAN_CHANNEL
from constants import YAM_CONTROL_PERIOD
from constants import YAM_GRIPPER_TYPE_STR
from constants import YAM_JOINT_CMD_ALPHA
from constants import YAM_INITIAL_JOINTS
from constants import YAM_RESET_DURATION_S
from constants import YAM_RPC_HOST
from constants import YAM_RPC_PORT
from constants import YAM_SPACEMOUSE_PATH
from constants import YAM_USE_SIM
from yam_jparse import jparse_step

# -----------------------------------------------------------------------------
# Repo paths: src/tidybot2 -> src/i2rt, src/pyroki
# -----------------------------------------------------------------------------
_SRC_ROOT = Path(__file__).resolve().parent.parent
_I2RT_ROOT = _SRC_ROOT / "i2rt"
if _I2RT_ROOT.is_dir() and str(_I2RT_ROOT) not in sys.path:
    sys.path.insert(0, str(_I2RT_ROOT))

from i2rt.robots.get_robot import get_yam_robot  # noqa: E402
from i2rt.robots.utils import ArmType, GripperType  # noqa: E402
from i2rt.robots.utils import ARM_YAM_XML_PATH  # noqa: E402

# Fixed transform link_6 -> TCP (same frame convention as TRI Raiden yam stack).
_T_LINK6_TO_TCP: np.ndarray = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_T_TCP_TO_LINK6: np.ndarray = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

_GRIPPER_SAFETY_THRESHOLD = 6.0 / 71.0  # linear gripper, ~0.5 mm in normalized units


def _smooth_move_to_joint_q(
    robot,
    q_goal: np.ndarray,
    duration_s: float,
    dt: float,
) -> None:
    """Ramp joint commands from current pose to ``q_goal`` (MuJoCo 7-vector) at rate ``dt``."""
    q_goal = np.asarray(q_goal, dtype=np.float64).reshape(7)
    q_start = np.asarray(robot.get_joint_pos(), dtype=np.float64).copy()
    steps = max(1, int(duration_s / dt))
    for i in range(steps):
        alpha = float(i + 1) / float(steps)
        q_cmd = (1.0 - alpha) * q_start + alpha * q_goal
        robot.command_joint_pos(q_cmd)
        time.sleep(dt)


def _mujoco_to_pyroki(q_mujoco6: np.ndarray) -> np.ndarray:
    return np.asarray(q_mujoco6, dtype=np.float64)[::-1].copy()


def _pyroki_to_mujoco(q_pk6: np.ndarray) -> np.ndarray:
    return np.asarray(q_pk6, dtype=np.float64)[::-1].copy()


def _load_yam_urdf() -> yourdfpy.URDF:
    urdf_path = Path(ARM_YAM_XML_PATH).with_suffix(".urdf")
    assets_dir = urdf_path.parent / "assets"

    def _pkg_handler(fname, dir=None):  # noqa: A002
        if isinstance(fname, str) and fname.startswith("package://assets/"):
            return str(assets_dir / fname.replace("package://assets/", ""))
        return fname

    return yourdfpy.URDF.load(
        str(urdf_path),
        filename_handler=_pkg_handler,
        load_meshes=False,
        load_collision_meshes=False,
    )


def _setup_pyroki_ik(dt: float):
    urdf = _load_yam_urdf()
    pk_robot = pk.Robot.from_urdf(urdf)
    link6_idx = list(pk_robot.links.names).index("link_6")
    step_jit = jax.jit(jparse_step, static_argnames=("method",))

    dummy = np.zeros(6, dtype=np.float64)
    result, _ = step_jit(
        robot=pk_robot,
        cfg=dummy,
        target_link_index=link6_idx,
        target_position=np.zeros(3, dtype=np.float64),
        target_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        method="jparse",
        dt=dt,
        home_cfg=dummy,
    )
    jax.block_until_ready(result)
    # Warm up forward_kinematics too — its first call JIT-compiles and can
    # block the command loop long enough to trip DM motor comm watchdogs.
    fk_out = pk_robot.forward_kinematics(jnp.asarray(dummy, dtype=jnp.float64))
    jax.block_until_ready(fk_out)
    return pk_robot, link6_idx, step_jit


def _pose_to_T_tcp(pos_xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(pos_xyz, dtype=np.float64).reshape(3)
    T[:3, :3] = R.from_quat(np.asarray(quat_xyzw, dtype=np.float64).reshape(4)).as_matrix()
    return T


def _T_tcp_to_ik_targets(T_tcp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T_link6 = T_tcp @ _T_TCP_TO_LINK6
    target_pos = T_link6[:3, 3].astype(np.float64)
    xyzw = R.from_matrix(T_link6[:3, :3]).as_quat()
    target_wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    return target_pos, target_wxyz


def _fk_T_tcp(pk_robot, link6_idx: int, q_pyroki: np.ndarray) -> np.ndarray:
    poses = pk_robot.forward_kinematics(jnp.asarray(q_pyroki, dtype=jnp.float64))
    T_link6 = np.array(jaxlie.SE3(poses[link6_idx]).as_matrix(), dtype=np.float64)
    return T_link6 @ _T_LINK6_TO_TCP


def _fk_tcp(pk_robot, link6_idx: int, q_pyroki: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T_tcp = _fk_T_tcp(pk_robot, link6_idx, q_pyroki)
    pos = T_tcp[:3, 3].copy()
    quat_xyzw = R.from_matrix(T_tcp[:3, :3]).as_quat()
    if quat_xyzw[3] < 0.0:
        quat_xyzw = -quat_xyzw
    return pos, quat_xyzw


def _ik_step_once(
    pk_robot,
    link6_idx: int,
    step_jit,
    q_pyroki: np.ndarray,
    T_target_tcp: np.ndarray,
    home_cfg: np.ndarray,
    dt: float,
) -> np.ndarray:
    """One J-PARSE velocity-IK integration step (matches one tick of Raiden-style EE teleop)."""
    target_pos, target_wxyz = _T_tcp_to_ik_targets(T_target_tcp)
    q_new, _ = step_jit(
        robot=pk_robot,
        cfg=np.asarray(q_pyroki, dtype=np.float64),
        target_link_index=link6_idx,
        target_position=target_pos,
        target_wxyz=target_wxyz,
        method="jparse",
        dt=dt,
        home_cfg=home_cfg,
    )
    return np.asarray(q_new, dtype=np.float64)


def spacemouse_to_target_pose(
    state,
    T_current: np.ndarray,
    vel_scale: float,
    rot_scale: float,
    invert_rotation: bool = False,
) -> np.ndarray:
    """Map SpaceMouse axes to a target EE pose (same mapping as Raiden)."""
    del invert_rotation  # API parity with Raiden; rotation sign handled in scales if needed
    dx = state.y * vel_scale
    dy = -state.x * vel_scale
    dz = state.z * vel_scale
    drx = state.roll * rot_scale
    dry = state.pitch * rot_scale
    drz = -state.yaw * rot_scale

    T_target = T_current.copy()
    T_target[:3, 3] += np.array([dx, dy, dz], dtype=np.float64)
    T_target[:3, :3] = (
        R.from_euler("xyz", [drx, dry, drz]).as_matrix() @ T_current[:3, :3]
    )
    return T_target


class YamArm:
    """YAM arm with the same RPC surface as tidybot2 Kinova `Arm` (execute_action / get_state / reset / close).

    For two-arm setups, set ``YamArm.server_channel`` before ``serve_forever`` (see ``__main__``), or rely
    on the default ``YAM_CAN_CHANNEL`` from ``constants``.

    Control follows i2rt-style joint streaming (cf. minimum_gello ~100 Hz): a fixed-period loop commands
    ``MotorChainRobot.command_joint_pos``. Cartesian targets from ``execute_action`` update the latest
    goal (no queue drops). IK is one J-PARSE step per tick with ``dt`` = control period, integrating a
    virtual joint state like Raiden SpaceMouse teleop—not a multi-step solve per RPC call.
    """

    #: CAN channel for this server process (set from ``--channel`` in RPC mode).
    server_channel: str | None = None

    def __init__(self):
        self._robot = None
        self._pk_robot = None
        self._link6_idx = None
        self._step_jit = None
        self._home_cfg = np.zeros(6, dtype=np.float64)
        self._target_lock = threading.Lock()
        self._target_T = np.eye(4, dtype=np.float64)
        self._target_gripper = 1.0
        self._q_pyroki: np.ndarray | None = None
        self._q_cmd: np.ndarray | None = None
        self._control_stop = threading.Event()
        self._control_pause = threading.Event()
        self._loop_tick_lock = threading.Lock()
        self._control_thread: threading.Thread | None = None

    def _ensure_robot(self):
        if self._robot is not None:
            return
        # JIT-compile pyroki BEFORE bringing up the motor chain. get_yam_robot()
        # spawns a CAN command thread that must tick at ~YAM_CONTROL_HZ; a JAX
        # compile running after it starves that thread and trips DM motor
        # comm-loss watchdogs.
        if self._pk_robot is None:
            self._pk_robot, self._link6_idx, self._step_jit = _setup_pyroki_ik(YAM_CONTROL_PERIOD)
        arm_type = ArmType.from_string_name(YAM_ARM_TYPE_STR)
        gripper_type = GripperType.from_string_name(YAM_GRIPPER_TYPE_STR)
        channel = self.server_channel or YAM_CAN_CHANNEL
        self._robot = get_yam_robot(
            channel=channel,
            arm_type=arm_type,
            gripper_type=gripper_type,
            zero_gravity_mode=False,
            sim=YAM_USE_SIM,
        )
        self._control_stop.clear()
        self._control_pause.clear()
        self._control_thread = threading.Thread(target=self._control_loop, name="yam-control", daemon=True)
        self._control_thread.start()

    def _control_loop(self):
        period = YAM_CONTROL_PERIOD
        dt = period
        while not self._control_stop.is_set():
            if self._control_pause.is_set():
                time.sleep(0.002)
                continue
            with self._loop_tick_lock:
                t0 = time.perf_counter()
                try:
                    if self._q_pyroki is None:
                        q_full = self._robot.get_joint_pos()
                        self._q_pyroki = _mujoco_to_pyroki(q_full[:6])
                        self._q_cmd = q_full.astype(np.float64).copy()
                        with self._target_lock:
                            self._target_T = _fk_T_tcp(self._pk_robot, self._link6_idx, self._q_pyroki)
                            self._target_gripper = float(self._q_cmd[6])
                    with self._target_lock:
                        T_tgt = self._target_T.copy()
                        g_tgt = self._target_gripper
                    self._q_pyroki = _ik_step_once(
                        self._pk_robot,
                        self._link6_idx,
                        self._step_jit,
                        self._q_pyroki,
                        T_tgt,
                        self._home_cfg,
                        dt,
                    )
                    q_arm = _pyroki_to_mujoco(self._q_pyroki)
                    a = float(YAM_JOINT_CMD_ALPHA)
                    if a >= 1.0:
                        self._q_cmd[:6] = q_arm
                    else:
                        self._q_cmd[:6] = (1.0 - a) * self._q_cmd[:6] + a * q_arm
                    self._q_cmd[6] = g_tgt
                    self._robot.command_joint_pos(self._q_cmd.copy())
                except Exception as e:
                    print(f"[yam_server] control loop: {e}")
                elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))

    def reset(self):
        self._ensure_robot()
        self._control_pause.set()
        # Hold _loop_tick_lock for the entire reset: waits for any in-progress tick
        # and blocks any rogue tick that slipped past the _control_pause check before
        # it was set (race: loop passed pause check → reset sets pause → loop acquires
        # lock → concurrent CAN access with _smooth_move_to_joint_q).
        with self._loop_tick_lock:
            _smooth_move_to_joint_q(
                self._robot,
                YAM_INITIAL_JOINTS,
                YAM_RESET_DURATION_S,
                YAM_CONTROL_PERIOD,
            )
            self._q_pyroki = None
            self._q_cmd = None
            q_full = self._robot.get_joint_pos()
            q_pk = _mujoco_to_pyroki(q_full[:6])
            with self._target_lock:
                self._target_T = _fk_T_tcp(self._pk_robot, self._link6_idx, q_pk)
                self._target_gripper = float(q_full[6])
        self._control_pause.clear()

    def execute_action(self, action):
        self._ensure_robot()
        T = _pose_to_T_tcp(action["arm_pos"], action["arm_quat"])
        grip = float(np.asarray(action["gripper_pos"]).reshape(-1)[0])
        with self._target_lock:
            self._target_T[:] = T
            self._target_gripper = grip

    def get_state(self):
        self._ensure_robot()
        q_full = self._robot.get_joint_pos()
        q_pyroki = _mujoco_to_pyroki(q_full[:6])
        arm_pos, arm_quat = _fk_tcp(self._pk_robot, self._link6_idx, q_pyroki)
        return {
            "arm_pos": arm_pos,
            "arm_quat": arm_quat,
            "gripper_pos": np.array([float(q_full[6])]),
        }

    def close(self):
        self._control_stop.set()
        if self._control_thread is not None:
            self._control_thread.join(timeout=3.0)
            self._control_thread = None
        if self._robot is not None:
            self._robot.close()
            self._robot = None
        self._q_pyroki = None
        self._q_cmd = None


class YamArmManager(MPBaseManager):
    pass


YamArmManager.register("YamArm", YamArm)


def _run_spacemouse_teleop(
    *,
    channel: str,
    arm: str,
    gripper: str,
    sim: bool,
    path: str,
    vel_scale: float,
    rot_scale: float,
    dt: float,
):
    arm_type = ArmType.from_string_name(arm)
    gripper_type = GripperType.from_string_name(gripper)
    # JIT-compile pyroki BEFORE bringing up the motor chain (see _ensure_robot).
    pk_robot, link6_idx, step_jit = _setup_pyroki_ik(dt)
    robot = get_yam_robot(
        channel=channel,
        arm_type=arm_type,
        gripper_type=gripper_type,
        zero_gravity_mode=False,
        sim=sim,
    )
    alpha = float(YAM_JOINT_CMD_ALPHA)
    home_cfg = np.zeros(6, dtype=np.float64)

    import pyspacemouse

    dev = pyspacemouse.open_by_path(path)
    if dev is None:
        raise RuntimeError(f"Could not open SpaceMouse at {path}")

    q_full = robot.get_joint_pos()
    q_arm = _mujoco_to_pyroki(q_full[:6])

    state_lock = threading.Lock()
    latest_state = {"s": None}
    shutdown = threading.Event()

    def _reader():
        while not shutdown.is_set():
            s = dev.read()
            if s is not None:
                with state_lock:
                    latest_state["s"] = s

    reader_t = threading.Thread(target=_reader, name="spacemouse-read", daemon=True)
    reader_t.start()

    q_cmd: np.ndarray | None = None
    print("SpaceMouse teleop: move puck to translate, twist to rotate. Ctrl+C to exit.")
    try:
        while True:
            t0 = time.monotonic()
            with state_lock:
                state = latest_state["s"]
            if state is None:
                time.sleep(dt)
                continue

            T_curr = _fk_T_tcp(pk_robot, link6_idx, q_arm)
            T_target = spacemouse_to_target_pose(state, T_curr, vel_scale, rot_scale)
            q_arm = _ik_step_once(pk_robot, link6_idx, step_jit, q_arm, T_target, home_cfg, dt=dt)

            gripper_actual = float(robot.get_joint_pos()[6])
            gripper_q = gripper_actual
            buttons = getattr(state, "buttons", [])
            if len(buttons) > 0 and buttons[0]:
                gripper_q = max(0.0, gripper_actual - 0.9 * _GRIPPER_SAFETY_THRESHOLD)
            elif len(buttons) > 1 and buttons[1]:
                gripper_q = min(1.0, gripper_actual + 0.9 * _GRIPPER_SAFETY_THRESHOLD)

            q_muj = _pyroki_to_mujoco(q_arm)
            if q_cmd is None:
                q_cmd = np.append(q_muj, gripper_q)
            else:
                if alpha >= 1.0:
                    q_cmd[:6] = q_muj
                else:
                    q_cmd[:6] = (1.0 - alpha) * q_cmd[:6] + alpha * q_muj
                q_cmd[6] = gripper_q
            robot.command_joint_pos(q_cmd.copy())

            elapsed = time.monotonic() - t0
            rem = dt - elapsed
            if rem > 0:
                time.sleep(rem)
    finally:
        shutdown.set()
        reader_t.join(timeout=1.0)
        robot.close()


if __name__ == "__main__":
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="YAM arm server (tidybot2 RPC) or SpaceMouse teleop")
    parser.add_argument(
        "--spacemouse",
        action="store_true",
        help="Run direct SpaceMouse EE teleop (no RPC).",
    )
    parser.add_argument("--channel", type=str, default=YAM_CAN_CHANNEL)
    parser.add_argument("--rpc-host", type=str, default=YAM_RPC_HOST, help="ROS-style RPC bind address.")
    parser.add_argument(
        "--rpc-port",
        type=int,
        default=YAM_RPC_PORT,
        help="RPC port (use a second port + --channel for the other arm).",
    )
    parser.add_argument("--arm", type=str, default=YAM_ARM_TYPE_STR)
    parser.add_argument("--gripper", type=str, default=YAM_GRIPPER_TYPE_STR)
    parser.add_argument("--sim", action="store_true", help="I2RT sim robot (no CAN).")
    parser.add_argument("--spacemouse-path", type=str, default=YAM_SPACEMOUSE_PATH)
    parser.add_argument("--vel-scale", type=float, default=0.12)
    parser.add_argument("--rot-scale", type=float, default=3.0)
    parser.add_argument(
        "--dt",
        type=float,
        default=YAM_CONTROL_PERIOD,
        help="IK integration dt (match control rate; default = YAM_CONTROL_PERIOD).",
    )
    args = parser.parse_args()

    if args.spacemouse:
        _run_spacemouse_teleop(
            channel=args.channel,
            arm=args.arm,
            gripper=args.gripper,
            sim=args.sim,
            path=args.spacemouse_path,
            vel_scale=args.vel_scale,
            rot_scale=args.rot_scale,
            dt=args.dt,
        )
    else:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        YamArm.server_channel = args.channel
        manager = YamArmManager(address=(args.rpc_host, args.rpc_port), authkey=RPC_AUTHKEY)
        server = manager.get_server()
        print(
            f"YAM arm manager at {args.rpc_host}:{args.rpc_port} "
            f"(channel={args.channel!r}; spawn; JAX-safe)"
        )
        server.serve_forever()
