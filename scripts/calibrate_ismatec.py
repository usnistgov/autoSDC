import sys
sys.path.append('.')

import json
import time
import numpy as np

from asdc.sdc import utils
from asdc.sdc import microcontroller

pump = microcontroller.PeristalticPump()


pump.set_flow_proportion(0.5)
pump.start()
time.sleep(3)
print(pump.get_flow())

pump.stop()
