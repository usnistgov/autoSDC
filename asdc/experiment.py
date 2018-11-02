import time
from datetime import datetime

import asdc.control

def run_cv_scan(cell='INTERNAL', verbose=False, initial_delay=30):
    """ run a CV scan for each point """

    # check on experiment status periodically:
    poll_interval = 1
    if verbose:
        print('initial delay', initial_delay)
    time.sleep(initial_delay)

    with asdc.control.controller(start_idx=17109013) as pstat:
        pstat.stop()
        pstat.clear()

        # run an open-circuit followed by a CV experiment
        status, oc_params = pstat.open_circuit(
            time_per_point=1, duration=120, current_range='AUTO', e_filter='1Hz', i_filter='1Hz'
        )

        if verbose:
            print('OC added.')
            print(status)
            print(oc_params)

        status, params = pstat.multi_cyclic_voltammetry(
            initial_potential=0.0, vertex_potential_1=-1.0, vertex_potential_2=1.2, final_potential=0.0, scan_rate=0.075,
            cell_to_use=cell, e_filter='1Hz', i_filter='1Hz', cycles=1
        )

        if verbose:
            print('CV added.')
            print(status)
            print(params)

        status, lsv_params = pstat.linear_scan_voltammetry(
            initial_potential=1.2, final_potential=0.0, scan_rate=0.075,
            cell_to_use=cell, e_filter='1Hz', i_filter='1Hz'
        )

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
            'measurement': 'cyclic_voltammetry',
            'parameters': params,
            'timestamp_start': timestamp_start,
            'timestamp': datetime.now().isoformat(),
            'current': pstat.current(),
            'potential': pstat.potential(),
            'elapsed_time': pstat.elapsed_time(),
            'error_codes': list(map(int, error_codes))
        }

        pstat.clear()

    return scan_data
