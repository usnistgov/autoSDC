from __future__ import print_function

import time
import chempy
import serial
import numpy as np
from scipy import optimize
from chempy import equilibria
from collections import defaultdict

from asdc.sdc.utils import encode
from asdc.sdc.microcontroller import PeristalticPump

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
    def __init__(
            self,
            solutions=SOLUTIONS,
            port='COM7',
            baud=115200,
            timeout=1,
            output_buffer=100,
            fast=False,
            flow_rate=0.5,
            flow_units='ml/min',
            counterpump_port=None,
            counterpump_ratio=0.95):
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
        self.timeout = timeout
        self.buffer_size = output_buffer
        self.fast = fast
        self.flow_rate = flow_rate
        self.flow_units = flow_units
        self.flow_setpoint = {pump_id: 0.0 for pump_id in self.solutions.keys()}

        self.counterpump = PeristalticPump(port=counterpump_port, timeout=self.timeout)
        self.counterpump_ratio = counterpump_ratio

    def relative_rates(self):
        total_rate = sum(self.flow_setpoint.values())
        return {key: rate / total_rate for key, rate in self.flow_setpoint.items()}

    def eval(self, command, pump_id=0, ser=None, check_response=False, fast=False):
        """ evaluate a PumpChain command.
        consider batches commands together using connection `ser`
        """

        if fast or self.fast:
            command = '@{}'.format(command)

        command = '{} {}'.format(pump_id, command)

        if ser is not None:
            ser.write(encode(command))

            if check_response:
                s = ser.read(self.buffer_size)
                print(s)
        else:
            with serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout) as ser:
                ser.write(encode(command))
                if check_response:
                    s = ser.read(self.buffer_size)
                    print(s)

    def refresh_ui(self, pump_id=0):
        """ for whatever reason, 'ver' refreshes the pump UI when other commands do not """
        self.eval('ver', pump_id=pump_id)

    def run(self, pump_id=0):
        print(f'asking pump {pump_id} to run')
        self.eval('run', pump_id=pump_id)

    def run_all(self):
        with serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout) as ser:
            for pump_id in self.solutions.keys():
                if self.flow_setpoint[pump_id] > 0:
                    self.eval('run', pump_id=pump_id, ser=ser)
                    time.sleep(0.05)
                else:
                    self.eval('stop', pump_id=pump_id, ser=ser)
                    time.sleep(0.05)

    def refresh_all(self):
        with serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout) as ser:
            for pump_id in self.solutions.keys():
                self.eval('ver', pump_id=pump_id, ser=ser)
                time.sleep(0.05)

    def stop(self, pump_id=0):
        self.eval('stop', pump_id=pump_id)

    def stop_all(self, counterbalance='off'):
        with serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout) as ser:
            for pump_id in self.solutions.keys():
                self.eval('stop', pump_id=pump_id, ser=ser)

        if counterbalance == 'full':
            # set counterbalance pumping rate
            self.counterpump.set_flow(1.0)
            self.counterpump.start()

        elif counterbalance == 'off':
            self.counterpump.stop()

        else:
            self.counterpump.stop()

    def diameter(self, pump_id=0, setpoint=None):

        if setpoint is not None:
            command = 'diameter {}'.format(setpoint)
        else:
            command = 'diameter'

        self.eval(command, pump_id=pump_id)

    def infusion_rate(self, ser=None, pump_id=0, rate=None, units='ml/min', fast=False):

        if rate is not None:
            command = 'irate {} {}'.format(rate, units)
        else:
            command = 'irate'

        self.eval(command, pump_id=pump_id, ser=ser, fast=fast)

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

    def set_rates(self, setpoints, units='ml/min', counterpump_ratio=None, start=False):
        """ directly set absolute flow rates

        flow_setpoint is a dict containing absolute flow rates for each syringe
        TODO: incorporate peristaltic pump here and set rates appropriately? need to set rates separately sometimes.
        """

        total_setpoint = sum(setpoints.values())

        if counterpump_ratio is None or counterpump_ratio == 'default':

            # use default counterpump ratio unless total setpoint is zero
            if total_setpoint == 0:
                counterpump_ratio = 1.0 # max
            else:
                counterpump_ratio = self.counterpump_ratio

            counterpump_ratio = max(0, counterpump_ratio)
            # counterpump_ratio = min(counterpump_ratio, 1.0)
            counterbalance_setpoint = counterpump_ratio * total_setpoint

        elif counterpump_ratio == 'max':
            # set to 1 mL/min
            counterbalance_setpoint = 1.0

        elif counterpump_ratio == 'off':
            counterbalance_setpoint = 0.0

        else:
            counterpump_ratio = max(0, counterpump_ratio)
            # counterpump_ratio = min(counterpump_ratio, 1.0)
            counterbalance_setpoint = counterpump_ratio * total_setpoint

        # reset rates to 0
        for pump_id in self.flow_setpoint.keys():
            self.flow_setpoint[pump_id] = 0.0

        # set flowrates for the syringe pump array
        with serial.Serial(port=self.port, baudrate=self.baud, timeout=self.timeout) as ser:
            print(setpoints)
            time.sleep(0.05)
            for species, setpoint in setpoints.items():
                print(species, setpoint)
                pump_id = self.get_pump_id(species)
                print(pump_id)
                if setpoint > 0:
                    self.flow_setpoint[pump_id] = setpoint
                    self.infusion_rate(pump_id=pump_id, ser=ser, rate=setpoint, units=units)
                    time.sleep(0.05)

        print(self.flow_setpoint)
        print(f'counter: {counterbalance_setpoint}')

        if start:
            self.run_all()

        # set counterbalance pumping rate
        self.counterpump.set_flow(counterbalance_setpoint)

        if start and (counterbalance_setpoint > 0):
            self.counterpump.start()
        elif counterbalance_setpoint == 0:
            self.counterpump.stop()
