from __future__ import print_function

import serial

def encode(message):
    message = message + '\r\n'
    return message.encode()

# placeholder config for development
CONFIG = {
    0: {
        'solution': 'HCl',
        'concentration': '1 molar',
    },
    1: {
        'solution': 'HCl',
        'concentration': '0.1 molar',
    }
}

class PumpArray():
    """ KDS Legato pump array interface """
    def __init__(self, config=CONFIG, port='COM7', baud=115200, timeout=1, output_buffer=100, fast=False):
        """ pump array.
        What is needed? concentrations and flow rates.
        Low level interface: set individual flow rates
        High level interface: set total flow rate and composition
        """
        self.config = config

        # serial interface things
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

    def eval(self, command, pump_id=0, override=False):

        command = '{} {}'.format(pump_id, command)

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

    def run(self, pump_id=0):
        self.eval('run', pump_id=pump_id)

    def run_all(self):
        for pump_id in self.config.keys():
            self.run(pump_id=pump_id)

    def stop(self, pump_id=0):
        self.eval('stop', pump_id=pump_id)

    def stop_all(self):
        for pump_id in self.config.keys():
            self.stop(pump_id=pump_id)

    def version(self, pump_id=0, verbose=False):

        if verbose:
            self.eval('version', pump_id=pump_id)
        else:
            self.eval('ver', pump_id=0)

    def address(self, pump_id=0, new_id=None):

        if new_id is not None:
            command = 'address {}'.format(new_id)
        else:
            command = 'address'

        self.eval(command, pump_id=pump_id)

    def crate(self, pump_id=0):
        self.eval('crate', pump_id=pump_id)

    def diameter(self, pump_id=0, setpoint=None):

        if setpoint is not None:
            command = 'diameter {}'.format(setpoint)
        else:
            command = 'diameter'

        self.eval(command, pump_id=pump_id)

    def infusion_rate(self, pump_id=0, rate=None, units='ml/min'):

        if rate is not None:
            command = 'irate {} {}'.format(rate, units)
        else:
            command = 'irate'

        self.eval(command, pump_id=pump_id)
