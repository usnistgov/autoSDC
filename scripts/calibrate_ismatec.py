import sys
sys.path.append('.')

import json
import time
import numpy as np
import pandas as pd

from asdc.sdc import utils
from asdc.sdc import microcontroller

pump = microcontroller.PeristalticPump()

proportion = np.linspace(0, 0.6, 0.05)
volts_in, volts_out = [], []

pump.start()

for p in proportion:
    volts_in.append(p * 3.3)
    pump.set_flow_proportion(p)
    time.sleep(3)
    volts_out.append(pump.get_flow())

pump.stop()

df = pd.DataFrame(
    {
        'proportion': proportion,
        'volts_in': volts_in,
        'volts_out': volts_out
    }
)

df.to_csv('ismatec_calib.csv')
