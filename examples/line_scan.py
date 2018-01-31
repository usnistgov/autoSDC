#!/usr/bin/env python

import json
import time
from datetime import datetime

import versastat.position
import versastat.control

def line_scan(speed=1e-5, poll_interval=0.5):
    """ perform a line scan with CV experiments, recording position, current, potential, and parameters in json log files
    Position units are METERS!
    """

    delta = [1e-4, 1e-4, 0.0]
    initial_delta = [0.0, 0.0, -2.60e-4]
    n_steps = 10

    with versastat.position.Position(ip='192.168.10.11', speed=speed) as pos:

        pos.print_status()
        pos.update(delta=initial_delta, verbose=True)
        pos.print_status()

        ctl = versastat.control.Control(start_idx=17109013)

        for idx in range(n_steps):
            # scan, log, take a position step

            # run a CV experiment
            status, params = ctl.cyclic_voltammetry()
            ctl.start()

            while ctl.sequence_running():
                time.sleep(poll_interval)

            # collect and log data
            scan_data = {
                'measurement': 'cyclic_voltammetry',
                'parameters': params,
                'index_in_sequence': idx,
                'timestamp': datetime.now().isoformat(),
                'current': ctl.current(),
                'potential': ctl.potential(),
                'position': pos.current_position()
            }

            logfile = 'line_scan_{}.json'.format(idx)
            with open(logfile, 'w') as f:
                json.dump(scan_data, f)

            ctl.clear()

            # update position
            pos.update(delta=delta, verbose=True)
            pos.print_status()

if __name__ == '__main__':
    line_scan()
