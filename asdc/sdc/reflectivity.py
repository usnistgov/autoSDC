import json
import time
import serial

from asdc.sdc.microcontroller import MicrocontrollerInterface

class Reflectometer(MicrocontrollerInterface):
    """ interface for the ThorLabs PDA36A2/Adafruit

    TODO: refactor this class and the peristaltic pump interface to share code
    """

    def collect(self):
        """ collect reading from laser reflectance setup """
        response = self.eval({"op": "laser"}, timeout=0.5)

        # TODO: check response content / add response status info
        reflectance_data = json.loads(response[1])

        return reflectance_data
