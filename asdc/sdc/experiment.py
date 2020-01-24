import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

from asdc import _slack

if sys.platform == 'win32':
    from . import potentiostat
else:
    # except ModuleNotFoundError:
    from .shims import potentiostat

# the 3F potentiostat
potentiostat_id = 17109013
POLL_INTERVAL = 1
MIN_SAMPLING_FREQUENCY = 1.0e-5

def run_experiment_sequence(pstat, poll_interval=POLL_INTERVAL):
    """ run an SDC experiment sequence -- busy wait until it's finished """

    pstat.start()

    error_codes = set()
    metadata = {'timestamp_start': datetime.now()}

    while pstat.sequence_running():
        time.sleep(poll_interval)
        pstat.update_status()
        overload_status = pstat.overload_status()
        if overload_status != 0:
            print('OVERLOAD:', overload_status)
            error_codes.add(overload_status)

    metadata['timestamp'] = datetime.now()
    metadata['error_codes'] = json.dumps(list(map(int, error_codes)))

    results = pd.DataFrame(
        {
            'current': pstat.current(),
            'potential': pstat.potential(),
            'elapsed_time': pstat.elapsed_time(),
            'applied_potential': pstat.applied_potential(),
            'current_range': pstat.current_range_history(),
            'segment': pstat.segment()
        }
    )

    return results, metadata

def setup_potentiostatic(pstat, data, cell='INTERNAL'):
    """ run a constant potential

    potential (float:volt)
    duration (float:second)

    {"op": "potentiostatic", "potential": Number(volts), "duration": Time(seconds)}
    """

    n_points = 3000
    duration = data.get('duration', 10)
    time_per_point = np.maximum(duration / n_points, MIN_SAMPLING_FREQUENCY)

    status, params = pstat.potentiostatic(
        initial_potential=data.get('potential'),
        time_per_point=time_per_point,
        duration=duration,
        current_range=data.get('current_range', 'AUTO'),
        e_filter='1HZ',
        i_filter='1HZ',
        cell_to_use=cell
    )

    return status, params

def setup_lsv(pstat, data, cell='INTERNAL'):
    """ linear scan voltammetry

    initial_potential (float:volt)
    final_potential (float:volt)
    scan_rate (float:volt/second)

    {
        "op": "lsv",
        "initial_potential": 0.0,
        "final_potential": 1.0,
        "scan_rate": 0.075
    }
    """

    status, params = pstat.linear_scan_voltammetry(
        initial_potential=data.get('initial_potential'),
        final_potential=data.get('final_potential'),
        scan_rate=data.get('scan_rate'),
        current_range=data.get('current_range', 'AUTO'),
        e_filter='1HZ',
        i_filter='1HZ',
        cell_to_use=cell
    )

    return status, params

def setup_lpr(pstat, data, cell='INTERNAL'):
    """ linear polarization resistance

    initial_potential (float:volt)
    final_potential (float:volt)
    step_size: (float: V)
    step_time: (float:second)
    cycles (int)
    vs (str)

    {
        "op": "lpr",
        "initial_potential": 0.0,
        "final_potential": 1.0,
        "step_size": 0.1,
        "step_time": 0.1,
        "vs": "VS OC"
    }
    """

    vs = data.get("vs", "VS OC")

    status, params = pstat.linear_polarization_resistance(
        initial_potential=data.get('initial_potential'),
        versus_initial=vs,
        final_potential=data.get('final_potential'),
        versus_final=vs,
        step_size=data.get('step_size'),
        step_time=data.get('step_time'),
        current_range=data.get('current_range', 'AUTO'),
        e_filter='1HZ',
        i_filter='1HZ',
        cell_to_use=cell
    )

    return status, params

def setup_corrosion_oc(pstat, data, cell='INTERNAL'):
    """ set up corrosion open circuit

    duration (float:second)

    {"op": "corrosion_oc", "duration": Time}
    """
    time_per_point = 1
    duration = data.get('duration', 10)

    # run an open-circuit followed by a CV experiment
    status, params = pstat.corrosion_open_circuit(
        time_per_point=time_per_point,
        duration=duration,
        current_range=data.get('current_range', 'AUTO'),
        e_filter='1HZ',
        i_filter='1HZ',
        cell_to_use=cell
    )

    return status, params

def setup_cv(pstat, data, cell='INTERNAL'):
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

    status, params = pstat.multi_cyclic_voltammetry(
        initial_potential=data.get('initial_potential'),
        vertex_potential_1=data.get('vertex_potential_1'),
        vertex_potential_2=data.get('vertex_potential_2'),
        final_potential=data.get('final_potential'),
        scan_rate=data.get('scan_rate'),
        cycles=data.get('cycles'),
        e_filter='1HZ',
        i_filter='1HZ',
        cell_to_use=cell,
        current_range=data.get('current_range', 'AUTO')
    )

    return status, params

potentiostat_ops = {
    'potentiostatic': setup_potentiostatic,
    'lsv': setup_lsv,
    'lpr': setup_lpr,
    'cv': setup_cv,
    'corrosion_oc': setup_corrosion_oc
}

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
