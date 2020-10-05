import os
import sys
import json
import time
import numpy as np
from pathlib import Path

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

phmeter = sdc.orion.PHMeter(ORION_PORT)
pump_array = sdc.pump.PumpArray(solutions, port=PUMP_PORT, timeout=1)

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

data_dir = Path('pH-meter-test')
os.makedirs(data_dir, exist_ok=True)

total_rate = 11 # mL/min
relative_rates = [0.001, 0.003, 0.1, 0.3, 1]

for idx, x in enumerate(relative_rates):
    setpoint = {'acid': x * total_rate, 'base': (1-x) * total_rate}

    logfile = data_dir / f'pH-log-{idx}-x{x}.csv'
    with self.phmeter.monitor(interval=1, logfile=logfile):
        pump_arry.set_rates(setpoint, start=True, fast=True)
        time.sleep(5*60)
