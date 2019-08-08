import sys
import numpy as np

sys.path.append('.')
from asdc import sdc

def test_pump_array():
    p = sdc.pump.PumpArray(port='COM6')
    p.print_config()

if __name__ == '__main__':
    print()
