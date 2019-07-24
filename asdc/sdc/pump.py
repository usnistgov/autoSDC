from __future__ import print_function

import serial

def legato_check(port='COM7', baud=115200, timeout=1):
    with serial.Serial(port=port, baudrate=baud, timeout=timeout) as ser:
        print('connected')
        ser.write('config\r\n'.encode())
        s = ser.read(100)
        print(s)
    return
