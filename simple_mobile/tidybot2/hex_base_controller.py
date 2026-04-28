#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-07-30
################################################################

import os
os.environ['CTR_TARGET'] = 'Hardware'  # pylint: disable=wrong-import-position

import math
import queue
import threading
import time
from enum import Enum
import numpy as np
# import phoenix6
# from phoenix6 import configs, controls, hardware
from hex_device import HexDeviceApi
from hex_device.motor_base import CommandType as HexCommandType
from ruckig import InputParameter, OutputParameter, Result, Ruckig, ControlInterface
from threadpoolctl import threadpool_limits
from constants import h_x, h_y, ENCODER_MAGNET_OFFSETS
from constants import HEX_BASE_URL, HEX_MOTOR_MAP
from constants import POLICY_CONTROL_PERIOD
from utils import create_pid_file

# Vehicle
CONTROL_FREQ = 250                   # 250 Hz
CONTROL_PERIOD = 1.0 / CONTROL_FREQ  # 4 ms
NUM_CASTERS = 4

# Caster
b_x = 0.020                    # Caster offset (m)
b_y = 0.0                      # Lateral caster offset (m)
r = 0.0625                     # Wheel radius (m)
N_s = 1.0                      # Steer gear ratio
N_r1 = 1.0                     # Drive gear ratio (1st stage)
N_r2 = 1.0                     # Drive gear ratio (2nd stage)
N_w = 1.0                      # Wheel gear ratio
N_r1_r2_w = N_r1 * N_r2 * N_w
N_s_r2_w = N_s * N_r2 * N_w
TWO_PI = 2 * math.pi

class HexMotorInterface:

    def __init__(
        self,
        ws_url=HEX_BASE_URL,
        control_hz=CONTROL_FREQ,
        control_mode="speed",
        motor_map=HEX_MOTOR_MAP,
        encoder_offsets=ENCODER_MAGNET_OFFSETS,
    ):
        self.__vehicle_api = HexDeviceApi(
            ws_url=ws_url,
            control_hz=control_hz,
        )
        # init hex vehicle
        self.__vehicle = None
        try:
            print('Waiting for vehicle...')
            while self.__vehicle is None:
                self.__vehicle = self.__vehicle_api.find_device_by_robot_type(20)
                time.sleep(0.001)
            print('Vehicle found')
        except KeyboardInterrupt:
            print('Keyboard interrupt, exiting...')
            exit(1)
        self.__vehicle.start()

        self.__motor_map = motor_map
        self.__encoder_offsets = np.zeros(8)
        self.__encoder_offsets[::2] = np.array(encoder_offsets)

    def get_positions(self):
        motor_positions = np.array(self.__vehicle.get_motor_positions())
        motor_positions *= self.__motor_map['reverse_factor']
        tidy_positions = np.zeros_like(motor_positions)
        tidy_positions[self.__motor_map['tidy_idx']] = motor_positions[
            self.__motor_map['motor_idx']]
        tidy_positions += self.__encoder_offsets
        return tidy_positions

    def get_velocities(self):
        motor_velocities = np.array(self.__vehicle.get_motor_velocities(pop = False))
        motor_velocities *= self.__motor_map['reverse_factor']
        tidy_velocities = np.zeros_like(motor_velocities)
        tidy_velocities[self.__motor_map['tidy_idx']] = motor_velocities[
            self.__motor_map['motor_idx']]
        return tidy_velocities

    def set_velocities(self, tidy_velocities: np.ndarray):
        self.__vehicle.enable()
        motor_velocities = np.zeros_like(tidy_velocities)
        motor_velocities[self.__motor_map['motor_idx']] = tidy_velocities[
            self.__motor_map['tidy_idx']]
        motor_velocities *= self.__motor_map['reverse_factor']
        self.__vehicle.motor_command(HexCommandType.SPEED, motor_velocities.tolist())

    def set_neutral(self):
        self.__vehicle.disable()

    def stop(self):
        self.__vehicle.stop()
        time.sleep(0.5)

class CommandType(Enum):
    POSITION = 'position'
    VELOCITY = 'velocity'

# Currently only used for velocity commands
class FrameType(Enum):
    GLOBAL = 'global'
    LOCAL = 'local'

class Vehicle:

    def __init__(
        self,
        max_vel=(0.5, 0.5, 1.57),
        max_accel=(0.15, 0.15, 0.5),
        ws_url=HEX_BASE_URL,
        control_hz=CONTROL_FREQ,
        control_mode="speed",
        motor_map=HEX_MOTOR_MAP,
        encoder_offsets=ENCODER_MAGNET_OFFSETS,
        close_accel=(0.1, 0.1, 0.35),
        error_threshold=(0.1, 0.2),
    ):
        self.max_vel = np.array(max_vel)
        self.max_accel = np.array(max_accel)
        self.close_accel = np.array(close_accel)
        self.error_threshold = np.array(error_threshold)

        # Use PID file to enforce single instance
        create_pid_file('tidybot2-base-controller')

        # Initialize casters
        # self.casters = [Caster(num) for num in range(1, NUM_CASTERS + 1)]
        self.__hex_motor_interface = HexMotorInterface(
            ws_url=ws_url,
            control_hz=control_hz,
            control_mode=control_mode,
            motor_map=motor_map,
            encoder_offsets=encoder_offsets,
        )

        # # CAN bus update frequency
        # self.status_signals = [signal for caster in self.casters for signal in caster.status_signals]
        # phoenix6.BaseStatusSignal.set_update_frequency_for_all(CONTROL_FREQ, self.status_signals)

        # Joint space
        num_motors = 2 * NUM_CASTERS
        self.q = np.zeros(num_motors)
        self.dq = np.zeros(num_motors)
        self.tau = np.zeros(num_motors)

        # Operational space (global frame)
        num_dofs = 3  # (x, y, theta)
        self.x = np.zeros(num_dofs)
        self._x_lock = threading.Lock()
        self.dx = np.zeros(num_dofs)

        # C matrix relating operational space velocities to joint velocities
        self.C = np.zeros((num_motors, num_dofs))
        self.C_steer = self.C[::2]
        self.C_drive = self.C[1::2]

        # C_p matrix relating operational space velocities to wheel velocities at the contact points
        self.C_p = np.zeros((num_motors, num_dofs))
        self.C_p_steer = self.C_p[::2]
        self.C_p_drive = self.C_p[1::2]
        self.C_p_steer[:, :2] = [1.0, 0.0]
        self.C_p_drive[:, :2] = [0.0, 1.0]

        # C_qp^# matrix relating joint velocities to operational space velocities
        self.C_pinv = np.zeros((num_motors, num_dofs))
        self.CpT_Cqinv = np.zeros((num_dofs, num_motors))
        self.CpT_Cqinv_steer = self.CpT_Cqinv[:, ::2]
        self.CpT_Cqinv_drive = self.CpT_Cqinv[:, 1::2]

        # OTG (online trajectory generation)
        # Note: It would be better to couple x and y using polar coordinates
        self.otg = Ruckig(num_dofs, CONTROL_PERIOD)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_res = Result.Working
        self.otg_inp.max_velocity = self.max_vel
        self.otg_inp.max_acceleration = self.max_accel

        # Control loop
        self.command_queue = queue.Queue(1)
        self.control_loop_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_loop_running = False

        # Debugging
        # self.data = []
        # import redis
        # self.redis_client = redis.Redis()

    def update_state(self) -> None:
        # # Update all status signals (sensor values)
        # phoenix6.BaseStatusSignal.refresh_all(self.status_signals)  # Note: Signal latency is roughly 4 ms

        # Joint positions and velocities
        # for i, caster in enumerate(self.casters):
        #     self.q[2*i : 2*i + 2] = caster.get_positions()
        #     self.dq[2*i : 2*i + 2] = caster.get_velocities()
        self.q = self.__hex_motor_interface.get_positions()
        self.dq = self.__hex_motor_interface.get_velocities()

        q_steer = self.q[::2]
        s = np.sin(q_steer)
        c = np.cos(q_steer)

        # C matrix
        self.C_steer[:, 0] = s / b_x
        self.C_steer[:, 1] = -c / b_x
        self.C_steer[:, 2] = (-h_x*c - h_y*s) / b_x - 1.0
        self.C_drive[:, 0] = c/r - b_y*s / (b_x*r)
        self.C_drive[:, 1] = s/r + b_y*c / (b_x*r)
        self.C_drive[:, 2] = (h_x*s - h_y*c) / r + b_y * (h_x*c + h_y*s) / (b_x*r)

        # C_p matrix
        self.C_p_steer[:, 2] = -b_x*s - b_y*c - h_y
        self.C_p_drive[:, 2] = b_x*c - b_y*s + h_x

        # C_qp^# matrix
        self.CpT_Cqinv_steer[0] = b_x*s + b_y*c
        self.CpT_Cqinv_steer[1] = -b_x*c + b_y*s
        self.CpT_Cqinv_steer[2] = b_x * (-h_x*c - h_y*s - b_x) + b_y * (h_x*s - h_y*c - b_y)
        self.CpT_Cqinv_drive[0] = r * c
        self.CpT_Cqinv_drive[1] = r * s
        self.CpT_Cqinv_drive[2] = r * (h_x*s - h_y*c - b_y)
        with threadpool_limits(limits=1, user_api='blas'):  # Prevent excessive CPU usage
            self.C_pinv = np.linalg.solve(self.C_p.T @ self.C_p, self.CpT_Cqinv)

        # Odometry
        dx_local = self.C_pinv @ self.dq
        theta_avg = self.x[2] + 0.5 * dx_local[2] * CONTROL_PERIOD
        R = np.array([
            [math.cos(theta_avg), -math.sin(theta_avg), 0.0],
            [math.sin(theta_avg), math.cos(theta_avg), 0.0],
            [0.0, 0.0, 1.0]
        ])
        dx_global = R @ dx_local
        self.dx = dx_global
        with self._x_lock:
            self.x += dx_global * CONTROL_PERIOD

    def get_x(self):
        with self._x_lock:
            return self.x.copy()

    def start_control(self):
        if self.control_loop_thread is None:
            print('To initiate a new control loop, please create a new instance of Vehicle.')
            return
        self.control_loop_running = True
        self.control_loop_thread.start()

    def stop_control(self):
        # safe stop hex vehicle
        self.__hex_motor_interface.stop()
        self.control_loop_running = False
        self.control_loop_thread.join()
        self.control_loop_thread = None

    def control_loop(self):
        # Set real-time scheduling policy
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO)))
        except PermissionError:
            print('Failed to set real-time scheduling policy, please edit /etc/security/limits.d/99-realtime.conf')

        disable_motors = True
        last_command_time = time.time()
        last_step_time = time.time()
        while self.control_loop_running:
            # Maintain the desired control frequency
            while time.time() - last_step_time < CONTROL_PERIOD:
                time.sleep(0.0001)
            curr_time = time.time()
            step_time = curr_time - last_step_time
            last_step_time = curr_time
            if step_time > 0.005:  # 5 ms
                print(f'Warning: Step time {1000 * step_time:.3f} ms in {self.__class__.__name__} control_loop')

            # Update state
            self.update_state()

            # Global to local frame conversion
            theta = self.x[2]
            R = np.array([
                [math.cos(theta), math.sin(theta), 0.0],
                [-math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0]
            ])

            # Check for new command
            if not self.command_queue.empty():
                command = self.command_queue.get()
                last_command_time = time.time()
                target = command['target']

                # Velocity command
                if command['type'] == CommandType.VELOCITY:
                    if command['frame'] == FrameType.LOCAL:
                        target = R.T @ target
                    self.otg_inp.max_acceleration = self.max_accel
                    self.otg_inp.control_interface = ControlInterface.Velocity
                    self.otg_inp.target_velocity = np.clip(target, -self.max_vel, self.max_vel)

                # Position command
                elif command['type'] == CommandType.POSITION:
                    pos_error = np.linalg.norm(self.x[:2] - target[:2])
                    angle_error = abs(self.x[2] - target[2])
                    if pos_error < self.error_threshold[0] and angle_error < self.error_threshold[1]:
                        self.otg_inp.max_acceleration = self.close_accel
                    else:
                        self.otg_inp.max_acceleration = self.max_accel
                    self.otg_inp.control_interface = ControlInterface.Position
                    self.otg_inp.target_position = target
                    self.otg_inp.target_velocity = np.zeros_like(self.dx)

                self.otg_res = Result.Working
                disable_motors = False

            # Maintain current pose if command stream is disrupted
            if time.time() - last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
                self.otg_inp.target_position = self.otg_out.new_position
                self.otg_inp.target_velocity = np.zeros_like(self.dx)
                self.otg_inp.current_velocity = self.dx  # Set this to prevent lurch when command stream resumes
                self.otg_res = Result.Working
                disable_motors = True

            # Slow down base during caster flip
            # Note: At low speeds, this section can be disabled for smoother movement
            if np.max(np.abs(self.dq[::2])) > 12.56:  # Steer joint speed > 720 deg/s
                if self.otg_inp.control_interface == ControlInterface.Position:
                    self.otg_inp.target_position = self.otg_out.new_position
                elif self.otg_inp.control_interface == ControlInterface.Velocity:
                    self.otg_inp.target_velocity = np.zeros_like(self.dx)

            # Update OTG
            if self.otg_res == Result.Working:
                self.otg_inp.current_position = self.x
                self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
                self.otg_out.pass_to_input(self.otg_inp)

            if disable_motors:
                # Hex firmware expects continuous algorithm control (enable + SPEED commands).
                # set_neutral() calls disable() and triggers "AlgorithmControl timeout" / e-stop
                # if held idle (see gamepad_teleop.py: stream zeros at policy rate instead).
                dq_hold = np.zeros_like(self.q)
                self.__hex_motor_interface.set_velocities(dq_hold)

            else:
                # # Send enable signal to devices
                # phoenix6.unmanaged.feed_enable(0.1)
                
                # Operational space velocity
                dx_d = self.otg_out.new_velocity
                dx_d_local = R @ dx_d

                # Joint velocities
                dq_d = self.C @ dx_d_local

                # Send motor velocity commands
                # for i in range(NUM_CASTERS):
                #     self.casters[i].set_velocities(dq_d[2*i], dq_d[2*i + 1])
                self.__hex_motor_interface.set_velocities(dq_d)

            # Debugging
            # self.data.append({
            #     'timestamp': time.time(),
            #     'q': self.q.tolist(),
            #     'dq': self.dq.tolist(),
            #     'x': self.x.tolist(),
            #     'dx': self.dx.tolist(),
            # })
            # self.redis_client.set(f'x', f'{self.x[0]} {self.x[1]} {self.x[2]}')
            # self.redis_client.set(f'dx', f'{self.dx[0]} {self.dx[1]} {self.dx[2]}')

    def _enqueue_command(self, command_type, target, frame=None):
        if self.command_queue.full():
            print('Warning: Command queue is full. Is control loop running?')
        else:
            command = {'type': command_type, 'target': target}
            if frame is not None:
                command['frame'] = FrameType(frame)
            self.command_queue.put(command, block=False)
        
    def set_target_velocity(self, velocity, frame='global'):
        self._enqueue_command(CommandType.VELOCITY, velocity, frame)

    def set_target_position(self, position):
        self._enqueue_command(CommandType.POSITION, position)

    # def get_encoder_offsets(self):
    #     offsets = []
    #     for caster in self.casters:
    #         caster.cancoder.configurator.refresh(caster.cancoder_cfg)  # Read current config
    #         curr_offset = caster.cancoder_cfg.magnet_sensor.magnet_offset
    #         caster.steer_position_signal.wait_for_update(0.1)
    #         curr_position = caster.cancoder.get_absolute_position().value
    #         offsets.append(f'{round(4096 * (curr_offset - curr_position))}.0 / 4096')
    #     print(f'ENCODER_MAGNET_OFFSETS = [{", ".join(offsets)}]')

if __name__ == '__main__':
    vehicle = Vehicle(max_vel=(0.25, 0.25, 0.79))
    # vehicle.get_encoder_offsets(); exit()
    vehicle.start_control()
    try:
        for _ in range(50):
            # vehicle.set_target_velocity(np.array([0.0, 0.0, 0.39]))
            vehicle.set_target_velocity(np.array([0.25, 0.0, 0.0]))
            # vehicle.set_target_position(np.array([0.5, 0.0, 0.0]))
            print(f'Vehicle - x: {vehicle.x} dx: {vehicle.dx}')
            time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    finally:
        vehicle.stop_control()
        # import pickle
        # output_path = 'controller-states.pkl'
        # with open(output_path, 'wb') as f:
        #     pickle.dump(vehicle.data, f)
        # print(f'Data saved to {output_path}')
