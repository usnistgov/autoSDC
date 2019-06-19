import time
import numpy as np
from datetime import datetime

from . import potentiostat

# the 3F potentiostat
potentiostat_id = 17109013

def run_experiment(pstat, poll_interval=1):
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

def run_potentiostatic(
        potential=0.0, duration=10, n_points=1000,
        pre_potential=-0.5, pre_duration=None,
        cell='INTERNAL', verbose=False):
    """ run a constant potential
    potential (V)
    duration (s)
    """

    with potentiostat.controller(start_idx=potentiostat_id) as pstat:

        time_per_point = np.maximum(duration / n_points, 1.0e-5)

        if pre_duration is not None:
            # add a preconditioning step
            # -0.5V might be a good potential to use for preconditioning
            # this should make the initial conditions more consistent across spots
            # because open circuit might not be the same for every spot
            status, params = pstat.potentiostatic(
                initial_potential=pre_potential, time_per_point=time_per_point*10, duration=pre_duration,
                current_range='AUTO', e_filter='1HZ', i_filter='1HZ', cell_to_use=cell
            )

        status, params = pstat.potentiostatic(
            initial_potential=potential, time_per_point=time_per_point, duration=duration,
            current_range='AUTO', e_filter='1HZ', i_filter='1HZ', cell_to_use=cell
        )

        scan_data, metadata = run_experiment(pstat)
        metadata['measurement'] = 'potentiostatic'
        metadata['parameters'] = params

    return scan_data, metadata

def run_cv_scan(
        initial_potential=-0.5,
        vertex_potential_1=-1.2,
        vertex_potential_2=-0.5,
        final_potential=-0.5,
        scan_rate=0.02,
        cycles=2,
        cell='INTERNAL',
        precondition_potential=None,
        precondition_duration=None,
        precondition_points=1000,
        verbose=False):
    """ run a CV scan for each point """

    with potentiostat.controller(start_idx=potentiostat_id) as pstat:

        # run an open-circuit followed by a CV experiment
        status, oc_params = pstat.corrosion_open_circuit(
            time_per_point=1, duration=120, current_range='AUTO', e_filter='1HZ', i_filter='1HZ', cell_to_use=cell
        )

        if precondition_duration is not None and precondition_potential is not None:
            # add a preconditioning step
            # -0.5V might be a good potential to use for preconditioning
            # this should make the initial conditions more consistent across spots
            # because open circuit might not be the same for every spot
            time_per_point = np.maximum(duration / n_points, 1.0e-5)

            status, params = pstat.potentiostatic(
                initial_potential=precondition_potential, time_per_point=time_per_point, duration=precondition_duration,
                current_range='AUTO', e_filter='1HZ', i_filter='1HZ', cell_to_use=cell
            )

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
        metadata['measurement'] = 'cyclic_voltammetry'
        metadata['parameters'] = params

    return scan_data, metadata
