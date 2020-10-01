import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Sequence
from datetime import datetime

from asdc import _slack
from asdc import analysis

from .experiment_defaults import *

# if sys.platform == 'win32':
#     from . import potentiostat
# else:
#     # except ModuleNotFoundError:
#     from .shims import potentiostat

MIN_SAMPLING_FREQUENCY = 1.0e-5

def from_command(instruction):
    """ {"op": "lpr", "initial_potential": -0.5, "final_potential": 0.5, "step_size": 0.1, "step_time": 0.5} """

    # don't mangle the original dictionary at all
    instruction_data = instruction.copy()

    opname = instruction_data.get('op')

    Expt = potentiostat_ops.get(opname)

    if Expt is None:
        return None

    del instruction_data['op']

    return Expt(**instruction_data)

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

    def marshal(self, echem_data: Dict[str, Sequence[float]]):
        return analysis.LPRData(echem_data)

@dataclass
class StaircaseLSV(StaircaseLSVArgs):
    """ staircase linear scan voltammetry

    initial_potential (float:volt)
    final_potential (float:volt)
    step_height: (float:volt)
    step_time: (float:second)
    scan_rate (float:volt/second)

    {"op": "staircase_lsv", "initial_potential": 0.0, "final_potential": 1.0, "step_height": 0.001, "step_time": 0.8}
    """
    versus: str = 'VS REF'
    setup_func: str = 'AddStaircaseLinearScanVoltammetry'
    filter: Optional[str] = None

    def getargs(self):

        # override any default arguments...
        args = self.__dict__
        args['versus_initial'] = args['versus_final'] = args['versus']
        if args['filter'] is not None:
            args['e_filter'] = args['i_filter'] = args['filter']

        args = StaircaseLSVArgs.from_dict(args)
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
    filter: Optional[str] = None

    def getargs(self):

        # override any default arguments...
        args = self.__dict__
        args['versus_initial'] = args['versus_final'] = args['versus']
        if args['filter'] is not None:
            args['e_filter'] = args['i_filter'] = args['filter']

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

    def marshal(self, echem_data: Dict[str, Sequence[float]]):
        return analysis.TafelData(echem_data)


@dataclass
class OpenCircuit(OpenCircuitArgs):
    """ Open circuit hold

    duration (float:second)

    {"op": "open_circuit", "duration": Time, "time_per_point": Time}
    """
    setup_func: str = 'AddOpenCircuit'

    def getargs(self):

        # override any default arguments...
        args = self.__dict__

        args = OpenCircuitArgs.from_dict(args)
        return args.format()

    def marshal(self, echem_data: Dict[str, Sequence[float]]):
        return analysis.OCPData(echem_data)

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
    setup_func: str = 'AddMultiCyclicVoltammetry'

    def getargs(self):

        # override any default arguments...
        args = self.__dict__

        for key in ('initial', 'vertex_1', 'vertex_2', 'final'):
            args[f'versus_{key}'] = args['versus']

        args = CyclicVoltammetryArgs.from_dict(args)
        return args.format()

    def marshal(self, echem_data: Dict[str, Sequence[float]]):
        return analysis.CVData(echem_data)


potentiostat_ops = {
    'cv': CyclicVoltammetry,
    'lsv': LSV,
    'lpr': LPR,
    'tafel': Tafel,
    'corrosion_oc': CorrosionOpenCircuit,
    'open_circuit': OpenCircuit,
    'potentiostatic': Potentiostatic,
    'staircase_lsv': StaircaseLSV
}
