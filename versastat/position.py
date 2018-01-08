""" versastat.position: pythonnet .NET interface to VersaSTAT motion controller """

import os
import clr
import sys
import time

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

class Position():
    """ Interface to the VersaSTAT motion controller library """
    
    def __init__(self, ip='192.168.10.11', speed=0.0001):
        """ instantiate a Position controller context manager """
        self._ip = ip
        self._speed = speed

    def __enter__(self):
        """ Set up and connect to the position controller """
        self.controller = XCD()
        self.settings = XcdSettings()

        self.settings.Speed = self._speed
        self.settings.IPAddress = IPAddress.Parse(self._ip)

        self.controller.Connect()

    def __exit__(self):
        """ gracefully shut down the position controller interface """
        self.controller.Disconnect()

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
        for axis in self.positioner.Parameters:
            print('{} setpoint = {} {}'.format(axis.Quantity, axis.SetPoint, axis.Units))

            for idx in range(axis.ValueNames.Length):
                print(axis.ValueNames[idx], axis.Values[idx], axis.Units)
                print()

    def update_x(self, delta=0.001, verbose=False, poll_interval=0.1):
        """ update position setpoint and busy-wait until the motion controller has finished.

        poll_interval: busy-waiting polling interval (seconds)
        """

        # update the setpoint for the x axis
        for idx, ax in enumerate(self.positioner.Parameters):

            if verbose:
                print(ax.Quantity)

            ax.SetPoint = ax.Values[0] + delta

            break

        # busy-wait while the motion controller moves the stage
        while not ax.IsAtSetPoint:

            if verbose:
                print(ax.Values[0], ax.Units)

            # wait 100ms
            time.sleep(poll_interval)

        return
