import sys
import numpy as np

sys.path.append('.')
from asdc import sdc

def test_pump_array():
    print('connecting to pumps...')
    p = sdc.pump.PumpArray(port='COM6')
    p.print_config()

    p.run_all()

    for setpoint in [2, 3, 4, 5]:
        print('setpoint pH:', setpoint)
        p.set_pH(setpoint=2.0)

        input("Press Enter to continue...")

    p.stop_all()

if __name__ == '__main__':
    test_pump_array()
