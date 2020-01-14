import json
import time
import serial

from asdc.sdc.utils import encode, decode

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

            response = ser.readline()
            return decode(response)

    def read(self):
        with serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout) as ser:
            response = ser.readlines()
            return decode(response)
