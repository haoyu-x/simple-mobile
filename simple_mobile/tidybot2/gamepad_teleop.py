# Author: Jimmy Wu
# Date: October 2024
#
# Note: This code is only intended for debugging the base controller.
# In external code, avoid directly importing Vehicle as done here.
# Instead, please use the RPC server in base_server.py, which runs the
# low-level controller in a dedicated, real-time process to help minimize
# unintended latency spikes caused by external code.

import os
import signal
import time
import numpy as np
import pygame
from pygame.joystick import Joystick
from constants import POLICY_CONTROL_PERIOD
from hex_base_controller import Vehicle

pygame.init()

def apply_deadzone(arr, deadzone_size=0.05):
    return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))


def print_user_guide():
    print(
        """
================================================================================
  HEX BASE — GAMEPAD TELEOP (Logitech F710 / similar)
================================================================================
  Start           Begin base control (connects to vehicle, starts controller)
  Back            Stop base control and disconnect

  Hold LB or RB   Drive enabled (commands sent at policy rate; release = zero vel)
    LB held       Velocity in ROBOT frame  (forward = robot +X)
    RB only       Velocity in WORLD frame (forward = world +X)

  Left stick      Forward / back and strafe left / right
  Right stick X   Rotate in place (yaw)

  Tip: Hold a bumper whenever you want to move; without bumpers the base only
  receives zero-velocity commands (keeps the hex link from timing out).

================================================================================
"""
    )


class GamepadTeleop:
    def __init__(self):
        self.joy = Joystick(0)  # Logitech F710
        self.vehicle = None

    def run(self):
        last_enabled = False
        print_user_guide()
        print('Press Start on the gamepad to begin base control.\n')
        while True:
            pygame.event.pump()

            # Start control
            if not self.vehicle and self.joy.get_button(7):  # 7 is the "Start" button
                # Match test_spin.py limits; stream commands at POLICY_CONTROL_PERIOD to avoid
                # hex_device "AlgorithmControl timeout" when the policy loop goes quiet.
                self.vehicle = Vehicle(max_vel=(0.25, 0.25, 0.79))
                self.vehicle.start_control()
                last_enabled = False
                frame = 'local'
                print('Control started')

            # Stop control
            if self.vehicle and self.joy.get_button(6):  # 6 is the "Back" button
                self.vehicle.stop_control()
                self.vehicle = None
                print('Control stopped')

            if self.vehicle:
                # Hold bumpers to drive; always stream velocity at POLICY_CONTROL_PERIOD
                # (like test_spin.py) so the hex stack keeps receiving commands.
                left_bumper = self.joy.get_button(4)
                right_bumper = self.joy.get_button(5)
                frame = 'local' if left_bumper else 'global'
                if left_bumper or right_bumper:
                    if not last_enabled:
                        print(f'Robot enabled ({frame} frame)')
                        last_enabled = True
                    # Stick signs matched to hex base +X forward, +Y left, +theta CCW
                    x = self.joy.get_axis(1)
                    y = self.joy.get_axis(0)
                    th = self.joy.get_axis(3)
                    target_velocity = np.array([x, y, th], dtype=float)
                    target_velocity = apply_deadzone(target_velocity)
                    target_velocity = self.vehicle.max_vel * target_velocity
                else:
                    if last_enabled:
                        print('Robot disabled (sending zero velocity at policy rate)')
                        last_enabled = False
                    target_velocity = np.zeros(3, dtype=float)

                self.vehicle.set_target_velocity(target_velocity, frame=frame)
                time.sleep(POLICY_CONTROL_PERIOD)
            else:
                time.sleep(0.01)

# Handle SIGTERM
def handler(signum, frame):
    os.kill(os.getpid(), signal.SIGINT)
signal.signal(signal.SIGTERM, handler)

if __name__ == '__main__':
    teleop = GamepadTeleop()
    try:
        teleop.run()
    finally:
        if teleop.vehicle:
            teleop.vehicle.stop_control()
