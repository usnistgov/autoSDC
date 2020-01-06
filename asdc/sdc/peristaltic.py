import json
import time
import serial

from asdc.sdc.utils import encode, decode

def hello():
    print('hi')

class PeristalticPump():
    """ interface for the ISMATEC/Adafruit """

    def __init__(self, port='COM9', baudrate=115200, timeout=0.01):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

    def eval(self, command, timeout=0.05, ser=None):

        with serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout) as ser:
            ser.write(encode(json.dumps(command)))
            time.sleep(timeout)
            response = ser.readlines()
            return decode(response)

    def read(self):
        with serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout) as ser:
            response = ser.readlines()
            return decode(response)
