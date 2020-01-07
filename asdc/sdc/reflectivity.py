import json
import time
import serial

from asdc.sdc.microcontroller import MicrocontrollerInterface

class Reflectometer(MicrocontrollerInterface):
    """ interface for the ThorLabs PDA36A2/Adafruit

    TODO: refactor this class and the peristaltic pump interface to share code
    """

    def collect(self):
        """ start pumping """
        return self.eval({"op": "laser"})
