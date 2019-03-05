""" asdc.position: pythonnet .NET interface to VersaSTAT motion controller """

import os
import sys
import time
import numpy as np
from contextlib import contextmanager

@contextmanager
def controller(ip='192.168.10.11', speed=1e-4):
    """ context manager that wraps position controller class Position. """
    pos = Position(ip=ip, speed=speed)
    try:
        yield pos
    except Exception as exc:
        print('unwinding position controller due to exception.')
        raise exc
    finally:
        pass

class Position():
    """ Interface to the VersaSTAT motion controller library """

    def __init__(self, ip='192.168.10.11', speed=0.0001):
        """ instantiate a Position controller context manager """
        self._ip = ip
        self._speed = speed

        # Set up and connect to the position controller
        self.controller = None
        self.settings = None

        self.axis = [0, 0, 0]

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed):
        self._speed = speed

    def home(block_interval=1):
        """ execute the homing operation, blocking for `block_interval` seconds.

        Warning: this will cause the motion stage to return to it's origin.
        This happens to be the maximum height for the stage...
        """
        time.sleep(1)

    def print_status(self):
        """ print motion controller status for each axis. """
        print('ok')

    def current_position(self):
        """ return the current coordinates as a list

        axis.Values holds (position, speed, error)
        """
        return self.axis

    def at_setpoint(self, verbose=False):
        """ check that each axis of the position controller is at its setpoint """
        return True

    def update_single_axis(self, axis=0, delta=0.001, verbose=False, poll_interval=0.1):
        """ update position setpoint and busy-wait until the motion controller has finished.

        poll_interval: busy-waiting polling interval (seconds)
        """
        self.axis[axis] += delta

    def update_x(self, delta=0.001, verbose=False, poll_interval=0.1):
        return self.update_single_axis(axis=0, delta=delta, verbose=verbose, poll_interval=poll_interval)

    def update_y(self, delta=0.001, verbose=False, poll_interval=0.1):
        return self.update_single_axis(axis=1, delta=delta, verbose=verbose, poll_interval=poll_interval)

    def update_z(self, delta=0.001, verbose=False, poll_interval=0.1):
        return self.update_single_axis(axis=2, delta=delta, verbose=verbose, poll_interval=poll_interval)

    def update(self, delta=[0.001, 0.001, 0.0], step_height=None, compress=None, verbose=False, poll_interval=0.1, max_wait_time=25):
        """ update position setpoint and busy-wait until the motion controller has finished.

        delta: position update [dx, dy, dz]
        step_height: ease off vertically before updating position
        poll_interval: busy-waiting polling interval (seconds)
        """

        if step_height is not None and step_height > 0:
            step_height = abs(step_height)
            self.update_z(delta=step_height, verbose=verbose)

        for d, ax in zip(delta, self.controller.Parameters):

            if verbose:
                print(ax.Quantity)

            if d != 0.0:
                ax.SetPoint = ax.Values[0] + d

        # busy-wait while the motion controller moves the stage
        time_elapsed = 0
        while not self.at_setpoint(verbose=verbose):

            time.sleep(poll_interval)
            time_elapsed += poll_interval

            if time_elapsed > max_wait_time:
                raise TimeoutError('Max position update time of {}s exceeded'.format(max_wait_time))

        if step_height is not None and step_height > 0:
            self.update_z(delta=-step_height, verbose=verbose)

        if compress is not None and abs(compress) > 0:
            compress = np.clip(abs(compress), 0, 5e-5)

            self.update_z(delta=-compress)
            self.update_z(delta=compress)