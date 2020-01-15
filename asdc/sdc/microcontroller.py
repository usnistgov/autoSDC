import json
import time
import serial

from asdc.sdc.utils import encode, decode
from asdc.sdc.utils import flow_to_proportion, proportion_to_flow

class MicrocontrollerInterface():
    """ interface for the equipment hooked up through a microcontroller board """

    def __init__(self, port='COM9', baudrate=115200, timeout=0.5):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

    def eval(self, command, timeout=None, ser=None):

        if timeout is None:
            timeout = self.timeout

        with serial.Serial(port=self.port, baudrate=self.baudrate, timeout=timeout) as ser:

            # block until the whole command is echoed
            ser.write(encode(json.dumps(command, separators=(',', ':'))))
            ack = ser.readline()
            print(decode(ack))

            response = ser.readline()
            return decode(response)

    def read(self):
        with serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout) as ser:
            response = ser.readlines()
            return decode(response)

class Reflectometer(MicrocontrollerInterface):
    """ interface for the ThorLabs PDA36A2/Adafruit """

    def collect(self, timeout=None):

        if timeout is None:
            timout = self.timeout

        """ collect reading from laser reflectance setup """
        response = self.eval({"op": "laser"}, timeout=timeout)

        # TODO: check response content / add response status info
        # reflectance_data = json.loads(response[1])
        reflectance = float(response)

        return reflectance

class PeristalticPump(MicrocontrollerInterface):
    """ interface for the ISMATEC/Adafruit """

    def start(self):
        """ start pumping """
        return self.eval({"op": "start"})

    def stop(self):
        """ start pumping """
        return self.eval({"op": "stop"})

    def set_flow(self, rate):
        """ set pumping rate to counterbalance a nominal target flow rate in ml/min """

        ismatec_proportion = flow_to_proportion(rate)
        print(f'ismatec_proportion: {ismatec_proportion}')
        self.eval({"op": "set_flow", "rate": ismatec_proportion})

    def set_flow_proportion(self, proportion):
        """ set proportional flow rate """
        self.eval({"op": "set_flow", "rate": proportion})
