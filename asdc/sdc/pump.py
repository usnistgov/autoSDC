from __future__ import print_function

import serial

class PumpArray():
    """ KDS Legato pump array interface """
    def __init__(self, port='COM7', baud=115200, timeout=1):
        self.port = port
        self.baud = baud
        self.timeout = 1

    def print_config(self):
        with serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout) as ser:
            print('connected')
            ser.write('config\r\n'.encode())
            s = ser.read(100)
            print(s)

    def address(self):
        with serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout) as ser:
            ser.write('address\r\n'.encode())
            s = ser.read(100)
            print(s)
