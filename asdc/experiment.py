import time
import numpy as np
from datetime import datetime

import asdc.control

def run_experiment(pstat):

    # check on experiment status periodically:
    poll_interval = 1

    timestamp_start = datetime.now().isoformat()
    pstat.start()

    error_codes = set()
    while pstat.sequence_running():
        time.sleep(poll_interval)
        pstat.update_status()
        overload_status = pstat.overload_status()
        if overload_status != 0:
            print('OVERLOAD:', overload_status)
            error_codes.add(overload_status)

    # collect and log data
    scan_data = {
        'timestamp_start': timestamp_start,
        'timestamp': datetime.now().isoformat(),
        'current': pstat.current(),
        'potential': pstat.potential(),
        'elapsed_time': pstat.elapsed_time(),
        'error_codes': list(map(int, error_codes)),
        'applied_potential': pstat.applied_potential(),
        'current_range': pstat.current_range_history(),
        'segment': pstat.segment()
    }

    return scan_data

def run_potentiostatic(potential=0.0, duration=10, cell='INTERNAL', verbose=False, initial_delay=0):
    """ run a constant potential
    potential (V)
    duration (s)
    """

    if verbose:
        print('initial delay', initial_delay)

    if initial_delay > 0:
        time.sleep(initial_delay)

    with asdc.control.controller(start_idx=17109013) as pstat:
        n_points = 1000
        time_per_point = np.maximum(duration / n_points, 1.0e-5)
        # run an open-circuit followed by a CV experiment
        status, params = pstat.potentiostatic(
            initial_potential=potential, time_per_point=time_per_point, duration=duration, current_range='AUTO', e_filter='1Hz', i_filter='1Hz',
            cell_to_use=cell
        )

        scan_data = run_experiment(pstat)
        scan_data['parameters'] = params
        scan_data['measurement'] = 'potentiostatic'

    return scan_data

def run_cv_scan(cell='INTERNAL', verbose=False, initial_delay=30):
    """ run a CV scan for each point """

    if verbose:
        print('initial delay', initial_delay)
    time.sleep(initial_delay)

    with asdc.control.controller(start_idx=17109013) as pstat:

        # # run an open-circuit followed by a CV experiment
        # status, oc_params = pstat.corrosion_open_circuit(
        #     time_per_point=1, duration=120, current_range='AUTO', e_filter='1Hz', i_filter='1Hz', cell_to_use=cell
        # )

        status, params = pstat.multi_cyclic_voltammetry(
            initial_potential=-0.5, vertex_potential_1=-1.2, vertex_potential_2=-0.5, final_potential=-0.5, scan_rate=0.02,
            cell_to_use=cell, e_filter='1Hz', i_filter='1Hz', cycles=2
        )

        if verbose:
            # print('OC added:', oc_params)
            print('CV added:', params)
            print(status)

        scan_data = run_experiment(pstat)
        scan_data['parameters'] = params
        scan_data['measurement'] = 'cyclic_voltammetry'

    return scan_data
