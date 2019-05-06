import time
import numpy as np
from datetime import datetime

from . import potentiostat

def run_experiment(pstat, poll_interval=1):
    """ simulate an SDC experiment -- busy wait until it's finished """

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

def run_potentiostatic(potential=0.0, duration=10, n_points=1000, cell='INTERNAL', verbose=False):
    """ run a constant potential
    potential (V)
    duration (s)
    """

    with potentiostat.controller(start_idx=17109013) as pstat:

        time_per_point = np.maximum(duration / n_points, 1.0e-5)

        status, params = pstat.potentiostatic(
            initial_potential=potential, time_per_point=time_per_point, duration=duration,
            current_range='AUTO', e_filter='1HZ', i_filter='1HZ', cell_to_use=cell
        )

        scan_data, metadata = run_experiment(pstat)
        metadata['parameters'] = params
        metadata['measurement'] = 'potentiostatic'

    return scan_data, metadata

def run_cv_scan(
        initial_potential=-0.5,
        vertex_potential_1=-1.2,
        vertex_potential_2=-0.5,
        final_potential=-0.5,
        scan_rate=0.02,
        cycles=2,
        cell='INTERNAL',
        verbose=False):
    """ run a CV scan for each point """

    with potentiostat.controller(start_idx=17109013) as pstat:

        # # run an open-circuit followed by a CV experiment
        # status, oc_params = pstat.corrosion_open_circuit(
        #     time_per_point=1, duration=120, current_range='AUTO', e_filter='1HZ', i_filter='1HZ', cell_to_use=cell
        # )

        status, params = pstat.multi_cyclic_voltammetry(
            initial_potential=initial_potential, vertex_potential_1=vertex_potential_1,
            vertex_potential_2=vertex_potential_2, final_potential=final_potential,
            scan_rate=scan_rate, cell_to_use=cell, e_filter='1HZ', i_filter='1HZ', cycles=cycles
        )

        if verbose:
            # print('OC added:', oc_params)
            print('CV added:', params)
            print(status)

        scan_data, metadata = run_experiment(pstat)
        metadata['parameters'] = params
        metadata['measurement'] = 'cyclic_voltammetry'

    return scan_data, metadata
