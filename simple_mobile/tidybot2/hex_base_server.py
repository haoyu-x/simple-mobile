#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-07-30
################################################################
#
# RPC wrapper for hex_base_controller.Vehicle. The low-level loop streams
# zero joint velocities when idle (not disable()) so the chassis stays in
# algorithm control and avoids Hex "AlgorithmControl timeout" / e-stop.
# Teleop clients should still call execute_action at ~POLICY_CONTROL_PERIOD
# when driving so position/velocity targets stay fresh.
################################################################

import time

import numpy as np
from multiprocessing.managers import BaseManager as MPBaseManager

from constants import BASE_RPC_HOST, BASE_RPC_PORT, POLICY_CONTROL_PERIOD, RPC_AUTHKEY
from hex_base_controller import Vehicle

class Base:
    def __init__(self, max_vel=(0.5, 0.5, 1.57), max_accel=(0.15, 0.15, 0.5)):
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.vehicle = None

    def reset(self):
        """Bring base to a safe telop-ready state without tearing down the Hex control loop.

        First client call: create Vehicle and start the real-time loop. Later calls: keep the
        same loop and only stream zero local velocity (avoids disable/reinit that faults the base).
        """
        if self.vehicle is None:
            self.vehicle = Vehicle(max_vel=self.max_vel, max_accel=self.max_accel)
            self.vehicle.start_control()
            while not self.vehicle.control_loop_running:
                time.sleep(0.01)
            return

        z = np.zeros(3, dtype=np.float64)
        for _ in range(3):
            self.execute_action({"base_velocity": z})
            time.sleep(POLICY_CONTROL_PERIOD)

    def execute_action(self, action):
        if 'base_velocity' in action:
            vel = np.asarray(action['base_velocity'], dtype=np.float64).reshape(3)
            self.vehicle.set_target_velocity(vel, frame='local')
        else:
            self.vehicle.set_target_position(action['base_pose'])

    def get_state(self):
        state = {'base_pose': self.vehicle.get_x()}
        return state

    def close(self):
        if self.vehicle is not None:
            if self.vehicle.control_loop_running:
                self.vehicle.stop_control()

class BaseManager(MPBaseManager):
    pass

BaseManager.register('Base', Base)

if __name__ == '__main__':
    manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f'Base manager server started at {BASE_RPC_HOST}:{BASE_RPC_PORT}')
    server.serve_forever()
    # import numpy as np
    # from constants import POLICY_CONTROL_PERIOD
    # manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
    # manager.connect()
    # base = manager.Base()
    # try:
    #     base.reset()
    #     for i in range(50):
    #         base.execute_action({'base_pose': np.array([(i / 50) * 0.5, 0.0, 0.0])})
    #         print(base.get_state())
    #         time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    # finally:
    #     base.close()
