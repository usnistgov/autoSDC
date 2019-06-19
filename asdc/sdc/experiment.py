import time
import numpy as np
from datetime import datetime

from . import potentiostat

# the 3F potentiostat
potentiostat_id = 17109013

POLL_INTERVAL = 1
MIN_SAMPLING_FREQUENCY = 1.0e-5

def run_experiment_sequence(pstat, poll_interval=POLL_INTERVAL):
    """ run an SDC experiment -- busy wait until it's finished """

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

    # collect results
    scan_data = {
        'current': pstat.current(),
        'potential': pstat.potential(),
        'elapsed_time': pstat.elapsed_time(),
        'error_codes': list(map(int, error_codes)),
        'applied_potential': pstat.applied_potential(),
        'current_range': pstat.current_range_history(),
        'segment': pstat.segment()
    }

    return scan_data, metadata

def setup_potentiostatic(pstat, expt, cell='INTERNAL'):
    """ run a constant potential

    potential (float:volt)
    duration (float:second)

    {"op": "potentiostatic", "potential": Number(volts), "duration": Time(seconds)}
    """

    n_points = 1000
    duration = expt.get('duration', 10)
    time_per_point = np.maximum(duration / n_points, MIN_SAMPLING_FREQUENCY)

    status, params = pstat.potentiostatic(
        initial_potential=expt.get('potential'),
        time_per_point=time_per_point,
        duration=duration,
        current_range='AUTO',
        e_filter='1HZ',
        i_filter='1HZ',
        cell_to_use=cell
    )

    return status, params

def setup_corrosion_oc(pstat, expt, cell='INTERNAL'):
    """ set up corrosion open circuit

    duration (float:second)

    {"op": "corrosion_oc", "duration": Time}
    """
    time_per_point = 1
    duration = expt.get('duration', 10)

    # run an open-circuit followed by a CV experiment
    status, params = pstat.corrosion_open_circuit(
        time_per_point=time_per_point,
        duration=duration,
        current_range='AUTO',
        e_filter='1HZ',
        i_filter='1HZ',
        cell_to_use=cell
    )

    return status, params

def setup_cv(pstat, expt, cell='INTERNAL'):
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
        initial_potential=expt.get('initial_potential'),
        vertex_potential_1=expt.get('vertex_potential_1'),
        vertex_potential_2=expt.get('vertex_potential_2'),
        final_potential=expt.get('final_potential'),
        scan_rate=expt.get('scan_rate'),
        cycles=expt.get('cycles'),
        e_filter='1HZ',
        i_filter='1HZ',
        cell_to_use=cell
    )

    return status, params


def run(experiment_json, cell='INTERNAL', verbose=False):

    experiment_data = json.loads(experiment_json)

    with potentiostat.controller(start_idx=potentiostat_id) as pstat:

        if experiment_data.get('op'):
            # single experiment -- just run it
            experiment_data = [experiment_data]

        _params = []
        for expt in experiment_data:
            if expt.get('op') == 'potentiostatic':
                status, params = setup_potentiostatic(pstat, expt, cell=cell)

            elif expt.get('op') == 'cv':
                status, params = setup_cv(pstat, expt, cell=cell)

            elif expt.get('op') == 'corrosion_oc':
                status, params = setup_corrosion_oc(pstat, expt, cell=cell)

            _params.append(params)

        scan_data, metadata = run_experiment_sequence(pstat)

    metadata['measurement'] = [expt.get('op') for expt in experiment_data]
    metadata['parameters'] = _params

    return scan_data, metadata
