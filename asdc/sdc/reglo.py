""" interface for the Reglo peristaltic pump """

import typing
from typing import Dict
from enum import IntEnum
from collections import Iterable

import regloicclib

class Channel(IntEnum):
    """ index organization for the pump channels

    needle defaults counterclockwise (-)
    dump (-)
    loop (+)
    source (+)
    """

    NEEDLE = 1
    DUMP = 2
    LOOP = 3
    SOURCE = 4

# 12 mL/min

class Reglo(regloicclib.Pump):
    """ thin wrapper around the pump interface from regloicc

    TODO: rewrite the serial interface...
    """
    def __init__(self, address=None, tubing_inner_diameter=1.52):
        super().__init__()

        self.pump = regloicclib.Pump(address=address)
        self.tubing_inner_diameter = tubing_inner_diameter

        for channel in range(1,5):
            self.setTubingInnerDiameter(self.tubing_inner_diameter, channel=channel)

    def set_rates(self, setpoints: Dict[Channel, float]):

        for channel, rate in setpoints.items():
            if rate == 0:
                self.stop(channel=channel.value)
            else:
                self.continuousFlow(rate, channel=channel.value)

        return


    def continuousFlow(self, rate, channel=None):
        if type(channel) is Channel:
            channel = channel.value
        super().continuousFlow(rate, channel)

    def stop(self, channel=None):

        if channel is None or type(channel) is int:
            super().stop(channel=channel)

        elif type(channel) is Channel:
            super().stop(channel=channel.value)

        elif isinstance(channel, Iterable):
            for c in channel:
                super().stop(channel=c.value)

        return
