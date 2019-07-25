from __future__ import print_function

import serial

def encode(message):
    message = message + '\r\n'
    return message.encode()

class PumpArray():
    """ KDS Legato pump array interface """
    def __init__(self, port='COM7', baud=115200, timeout=1, output_buffer=100, fast=False):
        self.port = port
        self.baud = baud
        self.timeout = 1
        self.buffer_size = output_buffer
        self.fast = fast

    def print_config(self):
        with serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout) as ser:
            print('connected')
            ser.write(encode('config'))
            s = ser.read(100)
            print(s)

    def eval(self, command, override=False):
        if not override:
            if self.fast:
                command = '@{}'.format(command)

        with serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout) as ser:
            ser.write(encode(command))
            s = ser.read(self.buffer_size)
            print(s)

    def refresh_ui(self):
        with serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout) as ser:
            ser.write(encode(''))

    def run(self):
        self.eval('run')

    def stop(self):
        self.eval('stop')

    def version(self, verbose=False):

        if verbose:
            self.eval('version')
        else:
            self.eval('ver')

    def address(self, idx=None):

        if idx is not None:
            command = 'address {}'.format(idx)
        else:
            command = 'address'

        self.eval(command)

    def crate(self):
        self.eval('crate')

    def diameter(self, setpoint=None):

        if setpoint is not None:
            command = 'diameter {}'.format(setpoint)
        else:
            command = 'diameter'

        self.eval(command)

    def infusion_rate(self, rate=None, units='ml/min'):

        if rate is not None:
            command = 'irate {} {}'.format(rate, units)
        else:
            command = 'irate'

        self.eval(command)
