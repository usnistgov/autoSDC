import json
import time
from asdc.sdc import utils
from asdc.sdc import microcontroller

pump = microcontroller.PeristalticPump()

pump.set_flow_proportion(0.5)
pump.start()

time.sleep(5)
pump.stop()
