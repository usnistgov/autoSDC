from __future__ import print_function

import serial

class PumpArray():
    def __init__(port='COM7', baud=115200, timeout=1):
        self.port = port
        self.baud = baud
        self.timeout = 1

    def print_config(selfport='COM7', baud=115200, timeout=1):
        with serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout) as ser:
            print('connected')
            ser.write('config\r\n'.encode())
            s = ser.read(100)
            print(s)
