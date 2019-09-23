import sys
import time
import numpy as np

sys.path.append('.')
from asdc import sdc

solutions = {0: {'H2SO4': 1.0}, 1: {'Na2SO4': 1.0}, 2: {'KOH': 1.0}}

def test_pump_array():
    print('connecting to pumps...')
    p = sdc.pump.PumpArray(solutions, port='COM6')
    p.print_config()

    # p.set_pH(setpoint=2.0)
    p.flow_rate = 0.1
    # p.set_rates({'KOH': 2, 'H2SO4': 3})
    print('set...')

    p.eval('@irate 0.25 ml/min')
    p.eval('1 @irate 0.25 ml/min')
    p.eval('@irun')
    #time.sleep(1)
    # p.eval('status', check_response=True)
    p.eval('1 @irun')

    p.eval('status')
    time.sleep(1)
    p.eval('1 address', check_response=True)
    # p.run_all(fast=False)


if __name__ == '__main__':
    test_pump_array()
