import re
import time
import serial
import typing
import argparse
import threading
from contextlib import contextmanager

MODEL_NUMBER = 'A221'

def encode(message: str):
    message = message + '\r'
    return message.encode()

class PHMeter():

    supported_modes = {'pH', 'mV'}

    def __init__(self, address, baud=19200, timeout=0.5, mode='pH', model_number=MODEL_NUMBER):

        self.model_number = MODEL_NUMBER
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

    def _process_pH(self, response: str):
        """
        Meter Model, Serial Number, Software Revision, User ID, Date & Time, Sample ID, Channel, Mode, pH Value, pH Unit, mV Value, mV Unit, Temperature Value, Temperature Unit, Slope Value, Slope Unit, Method #, Log #

        b'GETMEAS         \r\n\r\n\rA221 pH,K10231,3.04,ABCDE,08/26/20 00:20:59,---,CH-1,pH,6.92,pH,-3.8, mV,23.3,C,99.6,%,M100,#162\n\r\r>'

        TODO: get timestamp, sample_id, channel, mode, pH_value, pH_unit, mV_value, mV_unit, temperature_value, temperature_unit, slope_value, slope_unit
        """

        # index into the response by searching for the model number
        model_match = re.search(self.model_number, response)
        data_idx = model_match.start()

        # remove any >, \r, and \n characters
        data = re.sub('[\>\\r\\n]', '', response[data_idx:])

        # strip any whitespace adjacent to the comma delimiters
        # and reconstruct the CSV string
        values = data.split(',')
        data = ','.join(map(str.strip, values))

        return data

    def read(self, count: int = 1):

        if count == 1:
            self.write('GETMEAS')
            response = self.ser.read_until(b'>')
            data = self._process_pH(response)
            return data

        elif count > 1:
            # FIXME: the multi-reading `GETMEAS` command will only return
            # a single `>` at the end of all the response lines...
            self.write(f'GETMEAS {count}')
            responses = [self.ser.read_until(b'>') for response in range(count)]
            data = [self._process_pH(response) for response in responses]
            return data

    def readloop(self, stop_event=None, interval=30, logfile='pHmeter_test.csv'):

        with self.sync():
            with open(logfile, 'a') as f:

                # main measurement loop to run at interval
                while True:

                    target_ts = time.time() + interval

                    reading = self.read()
                    print(reading, file=f)

                    # wait out the rest of the interval
                    # but return immediately if signalled
                    delta = target_ts - time.time()
                    stop_event.wait(timeout=max(0, delta))

                    if stop_event.is_set():
                        return

    @contextmanager
    def monitor(self, interval=30, logfile='pHmeter_test.csv'):
        """ use this like

        ```
        with meter.monitor(interval=10):
            time.sleep(60)
        ```
        """
        stop_event = threading.Event()

        io_worker = threading.Thread(target=self.readloop, args=(stop_event, interval, logfile))
        try:
            io_worker.start()
            yield
        finally:
            stop_event.set()
            io_worker.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pH meter client')
    parser.add_argument('--port',  default='COM21', help='COM port for the pH meter')
    parser.add_argument('--verbose', action='store_true', help='include extra debugging output')
    args = parser.parse_args()

    meter = PHMeter(args.port)

    with meter.monitor(interval=10):
        time.sleep(60)
