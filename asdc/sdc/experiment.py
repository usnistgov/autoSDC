import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

from asdc import _slack

from .experiment_defaults import *

# if sys.platform == 'win32':
#     from . import potentiostat
# else:
#     # except ModuleNotFoundError:
#     from .shims import potentiostat

MIN_SAMPLING_FREQUENCY = 1.0e-5

potentiostat_ops = {
    'cv': CyclicVoltammetry,
    'lsv': LSV,
    'lpr': LPR,
    'tafel': Tafel,
    'corrosion_oc': CorrosionOpenCircuit,
    'open_circuit': OpenCircuit,
    'potentiostatic': Potentiostatic
}

def from_command(instruction):
    """ {"op": "lpr", "initial_potential": -0.5, "final_potential": 0.5, "step_size": 0.1, "step_time": 0.5} """

    opname = instruction.get('op')
    op = potentiostat_ops.get(opname)

    if op is None:
        return None

    Expt = potentiostat_ops[op]

    return Expt(**instruction)

@dataclass
class LPR(LPRArgs):
    """ linear polarization resistance

    initial_potential (float:volt)
    final_potential (float:volt)
    step_size: (float: V)
    step_time: (float:second)

    {"op": "lpr", "initial_potential": -0.5, "final_potential": 0.5, "step_size": 0.1, "step_time": 0.5}

    """
    versus: str = 'VS OC'
    setup_func: str = 'AddLinearPolarizationResistance'

    def getargs(self):

        # override any default arguments...
        args = self.__dict__
        args['versus_initial'] = args['versus_final'] = args['versus']

        args = LPRArgs.from_dict(args)
        return args.format()

@dataclass
class Potentiostatic(PotentiostaticArgs):
    """ potentiostatic: hold at constant potential

    potential (float:volt)
    duration (float:second)

    {"op": "potentiostatic", "potential": Number(volts), "duration": Time(seconds)}
    """
    n_points: int = 3000
    duration: int = 10
    versus: str = 'VS REF'
    setup_func: str = 'AddPotentiostatic'

    def getargs(self):

        time_per_point = np.maximum(self.duration / self.n_points, MIN_SAMPLING_FREQUENCY)

        # override any default arguments...
        args = self.__dict__
        args['time_per_point'] = time_per_point
        args['versus_initial'] = args['versus']

        args = PotentiostaticArgs.from_dict(args)
        return args.format()


@dataclass
class LSV(LSVArgs):
    """ linear scan voltammetry

    initial_potential (float:volt)
    final_potential (float:volt)
    scan_rate (float:volt/second)

    {"op": "lsv", "initial_potential": 0.0, "final_potential": 1.0, "scan_rate": 0.075}

    """
    versus: str = 'VS REF'
    setup_func: str = 'AddLinearScanVoltammetry'

    def getargs(self):

        # override any default arguments...
        args = self.__dict__
        args['versus_initial'] = args['versus_final'] = args['versus']

        args = LSVArgs.from_dict(args)
        return args.format()

@dataclass
class Tafel(TafelArgs):
    """ Tafel analysis

    {"op": "tafel", "initial_potential": V, "final_potential": V, "step_height": V, "step_time": s}

    """
    versus: str = 'VS OC'
    setup_func: str = 'AddTafel'

    def getargs(self):

        # override any default arguments...
        args = self.__dict__
        args['versus_initial'] = args['versus_final'] = args['versus']

        args = TafelArgs.from_dict(args)
        return args.format()

@dataclass
class OpenCircuit(OpenCircuitArgs):
    """ Open circuit hold

    duration (float:second)

    {"op": "open_circuit", "duration": Time}
    """
    setup_func: str = 'AddOpenCircuit'

    def getargs(self):

        # override any default arguments...
        args = self.__dict__
        args = OpenCircuitArgs.from_dict(args)
        return args.format()

@dataclass
class CorrosionOpenCircuit(CorrosionOpenCircuitArgs):
    """ Corrosion open circuit hold

    duration (float:second)

    {"op": "corrosion_oc", "duration": Time}
    """
    setup_func: str = 'AddCorrosionOpenCircuit'

    def getargs(self):

        # override any default arguments...
        args = self.__dict__
        args = CorrosionOpenCircuitArgs.from_dict(args)
        return args.format()

@dataclass
class CyclicVoltammetry(CyclicVoltammetryArgs):
    """ set up a CV experiment

    initial_potential (float:volt)
    vertex_potential_1 (float:volt)
    vertex_potential_2 (float:volt)
    final_potential (float:volt)
    scan_rate (float)
    cycles (int)

    {
        "op": "cv",
        "initial_potential": 0.0,
        "vertex_potential_1": -1.0,
        "vertex_potential_2": 1.2,
        "final_potential": 0.0,
        "scan_rate": 0.075,
        "cycles": 2
    }
    """

    versus: str = 'VS REF'
    setup_func: str = 'AddCyclicVoltammetry'

    def getargs(self):

        # override any default arguments...
        args = self.__dict__

        for key in ('versus_initial', 'versus_vertex_1', 'versus_vertex_2', 'versus_final'):
            args[key] = args['versus']

        args = CyclicVoltammetryArgs.from_dict(args)
        return args.format()


def run(instructions, cell='INTERNAL', verbose=False):

    with potentiostat.controller(start_idx=potentiostat_id) as pstat:

        if type(instructions) is dict and instructions.get('op'):
            # single experiment -- just run it
            instructions = [instructions]

        _params = []
        for instruction in instructions:
            params = None
            opname = instruction.get('op')
            op = potentiostat_ops.get(opname)

            if op is not None:
                status, params = op(pstat, instruction, cell=cell)

                if params:
                    _params.append(params)

        # _slack.post_message(f'starting experiment sequence')
        scan_data, metadata = run_experiment_sequence(pstat)

    metadata['measurement'] = json.dumps([instruction.get('op') for instruction in instructions])
    metadata['parameters'] = json.dumps(_params)

    return scan_data, metadata
