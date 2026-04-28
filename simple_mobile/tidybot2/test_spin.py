#!/usr/bin/env python3
"""Test script: make the robot base spin in place to verify the hex base API."""

import sys
import time

sys.path.insert(0, sys.path[0] + "/..")

import numpy as np
from hex_base_controller import Vehicle

POLICY_CONTROL_PERIOD = 0.1


def main():
    vehicle = Vehicle(max_vel=(0.12, 0.12, 0.157))
    vehicle.start_control()
    try:
        print("Spinning in place (angular velocity = 0.39 rad/s) for 3 seconds...")
        steps = int(40.0 / POLICY_CONTROL_PERIOD)
        for i in range(steps):
            vehicle.set_target_velocity(np.array([0.0, 0.0, 0.157]))
            print(f"[{i}/{steps}] x: {vehicle.x}  dx: {vehicle.dx}")
            time.sleep(POLICY_CONTROL_PERIOD)

        print("Stopping...")
        for _ in range(10):
            vehicle.set_target_velocity(np.array([0.0, 0.0, 0.0]))
            time.sleep(POLICY_CONTROL_PERIOD)

        print("Done.")
    finally:
        vehicle.stop_control()


if __name__ == "__main__":
    main()
