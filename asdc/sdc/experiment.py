import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Sequence
from datetime import datetime

import streamz
import streamz.dataframe

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
    """ {"op": "lpr", "initial_potential": -0.5, "final_potential": 0.5, "step_height": 0.1, "step_time": 0.5} """

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

    Attributes:
        initial_potential (float): starting potential (V)
        final_potential (float): ending potential (V)
        step_height (float): scan step size (V)
        step_time (float): scan point duration (s)

    Example:
        ```json
        {"op": "lpr", "initial_potential": -0.5, "final_potential": 0.5, "step_height": 0.1, "step_time": 0.5}
        ```

    """
    versus: str = 'VS OC'
    stop_execution: bool = False
    setup_func: str = 'AddLinearPolarizationResistance'

    def register_early_stopping(self, sdf: streamz.dataframe.DataFrame):
        return None

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

    Attributes:
        initial_potential (float): starting potential (V)
        final_potential (float): ending potential (V)
        step_height (float): scan step size (V)
        step_time (float): scan point duration (s)


    Example:
        ```json
        {"op": "staircase_lsv", "initial_potential": 0.0, "final_potential": 1.0, "step_height": 0.001, "step_time": 0.8}
        ```
    """
    versus: str = 'VS REF'
    stop_execution: bool = False
    setup_func: str = 'AddStaircaseLinearScanVoltammetry'
    filter: Optional[str] = None

    def register_early_stopping(self, sdf: streamz.dataframe.DataFrame):
        return None

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

    Attributes:
        potential (float): (V)
        duration (float) : (s)

    Example:
        ```json
        {"op": "potentiostatic", "potential": Number(volts), "duration": Time(seconds)}
        ```
    """
    n_points: int = 3000
    duration: int = 10
    versus: str = 'VS REF'
    stop_execution: bool = False
    setup_func: str = 'AddPotentiostatic'

    def register_early_stopping(self, sdf: streamz.dataframe.DataFrame):
        return None

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
    """linear scan voltammetry

    Attributes:
        initial_potential (float): starting potential (V)
        final_potential (float): ending potential (V)
        scan_rate (float): scan rate (V/s)
        current_range (str): current range setting to use

    Example:
        ```json
        {"op": "lsv", "initial_potential": 0.0, "final_potential": 1.0, "scan_rate": 0.075}
        ```

    """
    versus: str = 'VS REF'
    stop_execution: bool = False
    setup_func: str = 'AddLinearScanVoltammetry'
    filter: Optional[str] = None

    def register_early_stopping(self, sdf: streamz.dataframe.DataFrame):
        return None

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

    Attributes:
        initial_potential (float): starting potential (V)
        final_potential (float): ending potential (V)
        step_height (float): scan step size (V)
        step_time (float): scan point duration (s)

    Example:
        ```json
        {"op": "tafel", "initial_potential": V, "final_potential": V, "step_height": V, "step_time": s}
        ```

    """
    versus: str = 'VS OC'
    stop_execution: bool = False
    setup_func: str = 'AddTafel'

    def register_early_stopping(self, sdf: streamz.dataframe.DataFrame):
        return None

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

    If a `stabilization_window` is specified, allow the OCP hold to terminate early
    if the OCP fluctuation is less than `stabilization_range` volts over the window.

    Attributes:
        duration (float) : maximum OCP hold duration (s)
        stabilization_range (float): maximum allowed fluctuation for OCP stabilization (V)
        stabilization_window (float): OCP stabilization time period (s)

    Example:
        json:
        ```json
        {"op": "open_circuit", "duration": 60, "time_per_point": 0.5}
        ```

        python:
        ```python
        expt = OpenCircuit(duration=60, time_per_point=0.5)
        ```
    """
    stabilization_range: float = 0.01
    stabilization_window: float = 0
    stop_execution: bool = False
    setup_func: str = 'AddOpenCircuit'

    def getargs(self):

        # override any default arguments...
        args = self.__dict__

        args = OpenCircuitArgs.from_dict(args)
        return args.format()

    def marshal(self, echem_data: Dict[str, Sequence[float]]):
        return analysis.OCPData(echem_data)

    def signal_stop(self, value):
        if value < self.stabilization_range:
            self.stop_execution = True

    def register_early_stopping(self, sdf: streamz.dataframe.DataFrame):
        """ streaming dataframe -> early stopping criterion

        the potentiostat interface will check `experiment.stop_execution`
        """
        if self.stabilization_window <= 0:
            # by default, do not register early stopping at all
            # if the stabilization window is set to some positive time interval, proceed
            return None

        # set up streams to compute windowed potential range and trigger early stopping.
        # is there a more composable way to do this?
        # maybe the experiment object can have a function that accepts a streaming results dataframe
        # and builds and returns additional streams?
        # this might also be a decent way to register online error checkers

        def _min(old, new):
            chunk_min = min(new.values)
            return min(old, chunk_min)

        # compute rolling window range on potential
        potential_max = sdf.rolling(self.stabilization_window).potential.max()
        potential_min = sdf.rolling(self.stabilization_window).potential.min()

        # compute minimum rolling window range in each chunk
        potential_range = (potential_max - potential_min).stream.accumulate(_min, start=np.inf)
        return potential_range.sink(self.signal_stop)

@dataclass
class CorrosionOpenCircuit(CorrosionOpenCircuitArgs):
    """ Corrosion open circuit hold

    Attributes:
        duration (float) : (s)

    Example:
        ```json
        {"op": "corrosion_oc", "duration": Time, "time_per_point": Time}
        ```
    """
    stop_execution: bool = False
    setup_func: str = 'AddCorrosionOpenCircuit'

    def register_early_stopping(self, sdf: streamz.dataframe.DataFrame):
        return None

    def getargs(self):

        # override any default arguments...
        args = self.__dict__
        args = CorrosionOpenCircuitArgs.from_dict(args)
        return args.format()

@dataclass
class CyclicVoltammetry(CyclicVoltammetryArgs):
    """ set up a CV experiment

    Attributes:
        initial_potential (float): (V)
        vertex_potential_1 (float): (V)
        vertex_potential_2 (float): (V)
        final_potential (float) : (V)
        scan_rate (float): scan rate in (V/s)
        cycles (int): number of cycles

    Example:
        ```json
        {
            "op": "cv",
            "initial_potential": 0.0,
            "vertex_potential_1": -1.0,
            "vertex_potential_2": 1.2,
            "final_potential": 0.0,
            "scan_rate": 0.075,
            "cycles": 2
        }
        ```
    """

    versus: str = 'VS REF'
    stop_execution: bool = False
    setup_func: str = 'AddMultiCyclicVoltammetry'

    def register_early_stopping(self, sdf: streamz.dataframe.DataFrame):
        return None

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
