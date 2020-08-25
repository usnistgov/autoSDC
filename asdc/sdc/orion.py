import time
import serial
import typing
import argparse
import threading
from contextlib import contextmanager

def encode(message: str):
    message = message + '\r'
    return message.encode()

class PHMeter():

    supported_modes = {'pH', 'mV'}

    def __init__(self, address, baud=19200, timeout=0.5, mode='pH'):

        self.timeout = timeout

        self.ser = serial.Serial(
            port=address,
            baudrate=baud,
            parity=serial.PARITY_NONE,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout
        )

        self.mode = mode
        self._blocking = False
        self.blocking = False

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode in self.supported_modes:
            self._mode = mode
        else:
            raise ValueError(f'meter mode must be one of {self.supported_modes}')

        self.write(f'SETMODE {self.mode}')
        return self.check_response()

    @property
    def blocking(self):
        return self._blocking

    @blocking.setter
    def blocking(self, blocking: bool):
        assert isinstance(blocking, bool)

        if self._blocking == blocking:
            pass
        elif blocking:
            self.ser.timeout = None
        elif not blocking:
            self.ser.timeout = self.timeout

        self._blocking = blocking

    @contextmanager
    def sync(self):
        try:
            self.blocking = True
            yield
        finally:
            self.blocking = False

    def check_response(self):
        response = self.ser.read(size=2).decode()
        print(f'response: {response}')
        if response == '> ':
            return True
        else:
            return False

    def set_csv(self):
        self.write('SETCSV')
        return self.check_response()

    def write(self, msg):
        self.ser.write(encode(msg))

    def _process_pH(self):
        """
        Meter Model, Serial Number, Software Revision, User ID, Date & Time, Sample ID, Channel, Mode, pH Value, pH Unit, mV Value, mV Unit, Temperature Value, Temperature Unit, Slope Value, Slope Unit, Method #, Log #
        A211 pH, X01036, 3.04, ABCDE, 01/03/15 16:05:41, SAMPLE, CH-1, pH, 7.000, pH, 0.0, mV, 25.0, C, 98.1,%, M100, #1 <CR>
        """

        # get timestamp, sample_id, channel, mode, pH_value, pH_unit, mV_value, mV_unit, temperature_value, temperature_unit, slope_value, slope_unit

        pass

    def read(self, count: int = 1):

        if count == 1:
            self.write('GETMEAS')
            line = self.ser.read_until(b'\r')
            status = self.check_response()
            print(line)
            return line

        elif count > 1:
            self.write(f'GETMEAS {count}')
            lines = [self.ser.read_until(b'\r') for line in range(count)]
            status = self.check_response()
            print(lines)

    def readloop(self, interval=30, logfile=None):

        with self.sync():
            while True:
                target_ts = time.time() + interval

                line = self.read()

                delta = target_ts - time.time()
                time.sleep(max(0, delta))

    @contextmanager
    def monitor(self, interval=30, logfile=None):
        """ use this like

        ```
        with meter.monitor(interval=10):
            time.sleep(60)
        ```
        """
        io_worker = threading.Thread(target=self.readloop, args=(interval, logfile))
        try:
            io_worker.start()
            yield
        finally:
            io_worker.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pH meter client')
    parser.add_argument('--port',  default='COM21', help='COM port for the pH meter')
    parser.add_argument('--verbose', action='store_true', help='include extra debugging output')
    args = parser.parse_args()

    meter = PHMeter(args.port)

    # with meter.monitor(interval=10):
    #     time.sleep(60)
