# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=line-too-long
"""Xbox 360 Gamepad class that uses pygame under the hood.

Adapted from motion_imitation for Xbox 360 wireless controllers.
"""
import threading
import time

import pygame
import numpy as np


def _interpolate(value, old_max, new_scale, deadzone=0.1):
  ret = value * new_scale / old_max
  if abs(ret) < deadzone:
    return 0.0
  return ret


class Gamepad:
  """Gamepad class that reads from an Xbox 360 wireless controller."""

  def __init__(
      self,
      controller_index=0,  # Use js0 by default
      vel_scale_x=0.4,
      vel_scale_y=0.4,
      vel_scale_rot=1.0,
  ):
    pygame.init()
    pygame.joystick.init()
    
    self._controller_index = controller_index
    self._vel_scale_x = vel_scale_x
    self._vel_scale_y = vel_scale_y
    self._vel_scale_rot = vel_scale_rot

    self.vx = 0.0
    self.vy = 0.0
    self.wz = 0.0
    self.is_running = True

    self._device = None

    self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
    self.read_thread.start()

  def _connect_device(self):
    try:
      if pygame.joystick.get_count() == 0:
        print("No joystick/gamepad detected")
        return False
      
      if self._controller_index >= pygame.joystick.get_count():
        print(f"Controller index {self._controller_index} not available")
        return False
      
      self._device = pygame.joystick.Joystick(self._controller_index)
      self._device.init()
      print(f"Connected to: {self._device.get_name()}")
      return True
    except Exception as e:
      print(f"Error connecting to device: {e}")
      return False

  def read_loop(self):
    if not self._connect_device():
      self.is_running = False
      return

    while self.is_running:
      try:
        pygame.event.pump()  # Process pygame events
        self.update_command()
        time.sleep(0.01)  # Small delay to prevent excessive CPU usage
      except Exception as e:
        print(f"Error reading from device: {e}")
        break

    if self._device:
      self._device.quit()

  def update_command(self):
    if not self._device:
      return
      
    # Xbox 360 controller mapping:
    # Axis 0: Left stick X (left/right)
    # Axis 1: Left stick Y (up/down)
    # Axis 3: Right stick X (left/right) 
    
    left_x = self._device.get_axis(0)   # Left stick horizontal
    left_y = -self._device.get_axis(1)  # Left stick vertical (inverted)
    right_x = self._device.get_axis(3)  # Right stick horizontal

    self.vx = _interpolate(left_y, 1.0, self._vel_scale_x)
    self.vy = _interpolate(left_x, 1.0, self._vel_scale_y)
    self.wz = _interpolate(right_x, 1.0, self._vel_scale_rot)

  def get_command(self):
    return np.array([self.vx, self.vy, self.wz])

  def stop(self):
    self.is_running = False
    if self._device:
      self._device.quit()
    pygame.quit()


if __name__ == "__main__":
  try:
    gamepad = Gamepad()
    while True:
      print(gamepad.get_command())
      time.sleep(0.1)
  except KeyboardInterrupt:
    gamepad.stop()