import json
import time
import serial

from asdc.sdc.utils import encode, decode

def hello():
    print('hi')

def ismatec_to_flow(pct_rate):
    """ calibration curve from ismatec output fraction to flow in mL/min (0-100%)"""
    mL_per_min_rate = 0.0144 * pct_rate
    return mL_per_min_rate

def flow_to_ismatec(mL_per_min_rate):
    """ calibration curve from flow in mL/min to ismatec output fraction (0-100%)"""
    pct_rate = mL_per_min_rate / 0.0144
    return pct_rate

def proportion_to_flow(rate):
    """ calibration curve from ismatec output proportion (0,1) to flow in mL/min """
    mL_per_min_rate = 1.44 * rate
    return mL_per_min_rate

def flow_to_proportion(mL_per_min_rate):
    """ calibration curve from flow in mL/min to ismatec output proportion (0,1) """
    pct_rate = mL_per_min_rate / 1.44
    return pct_rate


class PeristalticPump():
    """ interface for the ISMATEC/Adafruit """

    def __init__(self, port='COM9', baudrate=115200, timeout=0.01):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

    def eval(self, command, timeout=0.05, ser=None):

        with serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout) as ser:
            ser.write(encode(json.dumps(command)))
            time.sleep(timeout)
            response = ser.readlines()
            return decode(response)

    def read(self):
        with serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout) as ser:
            response = ser.readlines()
            return decode(response)

    def start(self):
        """ start pumping """
        return self.eval({"op": "start"})

    def stop(self):
        """ start pumping """
        return self.eval({"op": "stop"})

    def set_flow(rate):
        """ set pumping rate to counterbalance a nominal target flow rate in ml/min """

        ismatec_proportion = flow_to_proportion(rate)
        self.eval({"op": "set_flow", "rate": ismatec_proportion})
