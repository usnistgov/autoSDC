import os
import sys
import json
import time
import serial
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)
from asdc import sdc

PUMP_PORT = 'COM12'
ORION_PORT = 'COM17'

solutions = {
    0: {'acid': 1.0},
    1: {'base': 1.0},
    2: {'neutral': 1.0}
}

# what interface would be nice?
# current:
# rates = {'acid': 0.4, 'base': 0.6}
# pump_array.set_rates(scale(rates, 2))
#
# pump_array.set_rates(rates, 2)
# pump_array.set_rates([0.4, 0.6, 0.0], 2)

S1 = {
    "mixfileVersion": 0.01,
    "name": "syringe1",
    "contents": [
        {
            "name": "Na2SO4",
            "formula": "Na2SO4",
            "quantity": 0.1,
            "units": "mol/L"
        },
        {
            "name": "water",
            "molfile": "\n\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\nM  END",
            "inchi": "InChI=1S/H2O/h1H2",
            "inchiKey": "InChIKey=XLYOFNOQVPJJNP-UHFFFAOYSA-N"
        }
    ]
}

def test_flow_mixing(data_dir, relative_rates, total_rate=11, dashboard=False):

    data_dir = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    phmeter = sdc.orion.PHMeter(ORION_PORT, zmq_pub=dashboard)
    pump_array = sdc.pump.PumpArray(solutions, port=PUMP_PORT, timeout=1)

    meta = []
    for idx, x in enumerate(relative_rates):
        setpoint = {'acid': x * total_rate, 'base': (1-x) * total_rate}

        logfile = f'pH-log-{idx}-x{x}.csv'
        with phmeter.monitor(interval=1, logfile=data_dir/logfile):
            pump_array.set_rates(setpoint, start=True, fast=True)
            meta.append({'logfile': logfile, 'setpoint': setpoint, 'ts': datetime.now().isoformat()})
            time.sleep(5*60)

    with open(data_dir/'metadata.json') as f:
        json.dump(meta, f)

def dryrun(data_dir, relative_rates, total_rate=11):
    data_dir = Path(data_dir)

    meta = []
    for idx, x in enumerate(relative_rates):
        setpoint = {'acid': x * total_rate, 'base': (1-x) * total_rate}
        logfile = f'pH-log-{idx}-x{x}.csv'
        print(data_dir / logfile, setpoint)
        meta.append({'logfile': logfile, 'setpoint': setpoint, 'ts': datetime.now().isoformat()})

        time.sleep(3)

    print(json.dumps(meta))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='flow mixing test harness')
    parser.add_argument('datadir', type=str, help='data and log directory')
    parser.add_argument('--dashboard', action='store_true', help='set up ZMQ publisher for dashboard')
    parser.add_argument('--dry-run', action='store_true', help='generate test output')
    args = parser.parse_args()

    total_rate = 11
    relative_rates = [0, 0.001, 0.003, 0.1, 0.3, 1]

    if args.dry_run:
        dryrun(args.datadir, relative_rates, total_rate=total_rate)
    else:
        test_flow_mixing(args.datadir, relative_rates, total_rate, dashboard=dashboard)
