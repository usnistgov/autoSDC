from __future__ import print_function

import chempy
import serial
import numpy as np
from scipy import optimize
from chempy import equilibria
from collections import defaultdict

def encode(message):
    message = message + '\r\n'
    return message.encode()

# placeholder config for development
SOLUTIONS = {
    0: {'H2SO4': 0.1},
    1: {'Na2SO4': 0.1},
    2: {'CuSO4': 0.1}
}

def mix(solutions, fraction):
    """ compute nominal compositions when mixing multiple solutions """

    solution = defaultdict(float)
    for sol, x in zip(solutions.values(), fraction):
        for species, conc in sol.items():
            solution[species] += x*conc
    return solution

def sulfuric_eq_pH(solution, verbose=False):

    eqsys = equilibria.EqSystem.from_string(
        """
        HSO4- = H+ + SO4-2; 10**-2
        H2SO4 = H+ + HSO4-; 2.4e6
        H2O = H+ + OH-; 10**-14/55.4
        """
    )

    nominal_sulfates = solution['CuSO4'] + solution['Na2SO4']
    arr, info, sane = eqsys.root(defaultdict(float, {'H2O': 55.4, 'H2SO4': solution['H2SO4'], 'SO4-2': nominal_sulfates}))
    conc = dict(zip(eqsys.substances, arr))

    pH = -np.log10(conc['H+'])

    if verbose:
        print("pH: %.2f" % pH)
        print()
        pprint(conc)

    return -np.log10(conc['H+'])

def pH_error(target_pH, stock=SOLUTIONS):

    def f(x):
        """ perform linear mixing between just two solutions """
        s = mix(stock, [x, 1-x, 0])
        pH = sulfuric_eq_pH(s, verbose=False)
        return pH

    return lambda x: f(x) - target_pH

class PumpArray():
    """ KDS Legato pump array interface """
    def __init__(self, solutions=SOLUTIONS, port='COM7', baud=115200, timeout=1, output_buffer=100, fast=False, flow_rate=0.5, flow_units='ml/min'):
        """ pump array.
        What is needed? concentrations and flow rates.
        Low level interface: set individual flow rates
        High level interface: set total flow rate and composition

        TODO: look into using serial.tools.list_ports.comports to identify the correct COM port to connect to...
        the id string should be something like 'USB serial port for Syringe Pump (COM*)'
        """
        self.solutions = solutions

        # serial interface things
        self.port = port
        self.baud = baud
        self.timeout = 1
        self.buffer_size = output_buffer
        self.fast = fast
        self.flow_rate = flow_rate
        self.flow_units = flow_units
        self.flow_setpoint = {pump_id: 0.0 for pump_id in self.solutions.keys()}

    def print_config(self):
        with serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout) as ser:
            print('connected')
            ser.write(encode('config'))
            s = ser.read(100)
            print(s.strip())

    def eval(self, command, pump_id=0, override=False, check_response=False):
        """ evaluate a PumpChain command.
        Currently establishes a new serial connection for every command.
        TODO: consider batching commands together...
        """
        if not override:
            if self.fast:
                command = '@{}'.format(command)

        command = '{} {}'.format(pump_id, command)

        with serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout) as ser:
            ser.write(encode(command))
            if check_response:
                s = ser.read(self.buffer_size)
                print(s)

    def refresh_ui(self):
        with serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout) as ser:
            ser.write(encode(''))

    def run(self, pump_id=0):
        self.eval('run', pump_id=pump_id)

    def run_all(self):
        for pump_id in self.solutions.keys():
            if self.flow_setpoint[pump_id] > 0:
                self.run(pump_id=pump_id)

    def stop(self, pump_id=0):
        self.eval('stop', pump_id=pump_id)

    def stop_all(self):
        for pump_id in self.solutions.keys():
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

    def set_pH(self, setpoint=3.0):
        """ control pH -- limited to two pumps for now. """

        if setpoint == 7.0:
            print('forcing Na2SO4-only run')
            x = 0.0
        else:
            x, r = optimize.brentq(pH_error(setpoint, stock=self.solutions), 0, 1, full_output=True)

        print(x)

        self.infusion_rate(pump_id=0, rate=x*self.flow_rate, units=self.flow_units)
        self.infusion_rate(pump_id=1, rate=(1-x)*self.flow_rate, units=self.flow_units)

        self.flow_setpoint = {0: x*self.flow_rate, 1: (1-x)*self.flow_rate}

    def get_pump_id(self, q):
        for key, value in self.solutions.items():
            if q in value:
                return key

    def set_rates(self, setpoints, units='ml/min'):
        """ directly set relative flow rates """

        # reset rates to 0
        for pump_id in self.flow_setpoint.keys():
            self.flow_setpoint[pump_id] = 0.0
            self.infusion_rate(pump_id=pump_id, rate=0.0, units=units)

        print(setpoints)
        for species, setpoint in setpoints.items():
            print(species, setpoint)
            pump_id = self.get_pump_id(species)
            print(pump_id)
            self.flow_setpoint[pump_id] = setpoint * self.flow_rate
            self.infusion_rate(pump_id=pump_id, rate=setpoint*self.flow_rate, units=units)
