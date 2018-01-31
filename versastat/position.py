""" versastat.position: pythonnet .NET interface to VersaSTAT motion controller """

import os
import clr
import sys
import time
from contextlib import contextmanager

# pythonnet checks PYTHONPATH for assemblies to load...
# so add the VeraScan libraries to sys.path
versascan_path = "C:/Program Files (x86)/Princeton Applied Research/VersaSCAN"
sys.path.append(versascan_path)
sys.path.append(os.path.join(versascan_path, "Devices"))

dlls = [
    'CommsLibrary',
    'DeviceInterface',
    'ScanDevices',
    'NanomotionXCD'
]
for dll in dlls:
    clr.AddReference(dll)

clr.AddReference('System')
clr.AddReference('System.Net')

from System.Net import IPAddress
from SolartronAnalytical.DeviceInterface.NanomotionXCD import XCD, XcdSettings

@contextmanager
def controller(ip='192.168.10.11', speed=1e-4):
    """ context manager that wraps position controller class Position. """
    pos = Position(ip=ip, speed=speed)
    try:
        pos.controller.Connect()
        yield pos
    finally:
        pos.controller.Disconnect()

class Position():
    """ Interface to the VersaSTAT motion controller library """
    
    def __init__(self, ip='192.168.10.11', speed=0.0001):
        """ instantiate a Position controller context manager """
        self._ip = ip
        self._speed = speed

        # Set up and connect to the position controller
        self.controller = XCD()
        self.settings = XcdSettings()

        self.settings.Speed = self._speed
        self.settings.IPAddress = IPAddress.Parse(self._ip)

        self.controller.Connect()

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed):
        self._speed = speed
        self.settings.Speed = self._speed

    def home(block_interval=1):
        """ execute the homing operation, blocking for `block_interval` seconds.

        Warning: this will cause the motion stage to return to it's origin.
        This happens to be the maximum height for the stage...
        """
        if not self.controller.IsHomingDone:
            self.controller.DoHoming()
            time.sleep(block_interval)

    def print_status(self):
        """ print motion controller status for each axis. """
        for axis in self.controller.Parameters:
            print('{} setpoint = {} {}'.format(axis.Quantity, axis.SetPoint, axis.Units))

            for idx in range(axis.ValueNames.Length):
                print(axis.ValueNames[idx], axis.Values[idx], axis.Units)
                print()

    def current_position(self):
        """ return the current coordinates as a list

        axis.Values holds (position, speed, error)
        """
        return [axis.Values[0] for axis in self.controller.Parameters]

    def at_setpoint(self, verbose=False):
        """ check that each axis of the position controller is at its setpoint """

        for ax in self.controller.Parameters:

            if verbose:
                print(ax.Values[0], ax.Units)

            if not ax.IsAtSetPoint:
                return False

        return True

    def update_x(self, delta=0.001, verbose=False, poll_interval=0.1):
        """ update position setpoint and busy-wait until the motion controller has finished.

        poll_interval: busy-waiting polling interval (seconds)
        """

        # update the setpoint for the x axis
        for idx, ax in enumerate(self.controller.Parameters):

            if verbose:
                print(ax.Quantity)

            ax.SetPoint = ax.Values[0] + delta

            break

        # busy-wait while the motion controller moves the stage
        while not self.at_setpoint(verbose=verbose):
            time.sleep(poll_interval)

        return

    def update(self, delta=[0.001, 0.001, 0.0], verbose=False, poll_interval=0.1, max_wait_time=25):
        """ update position setpoint and busy-wait until the motion controller has finished.

        delta: position update [dx, dy, dz]
        poll_interval: busy-waiting polling interval (seconds)
        """

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

