#!/usr/bin/env python

import time

from solartron import Instrument, Immediate, Experiment
from solartron import XCD, XcdSettings, IPAddress

def print_position_status(pos):
    
    for axis in pos.Parameters:
        print('{} setpoint = {} {}'.format(axis.Quantity, axis.SetPoint, axis.Units))
    
        for idx in range(axis.ValueNames.Length):
            print(axis.ValueNames[idx], axis.Values[idx], axis.Units)
        print()

    return

def update_x_position(delta=0.001, verbose=False):

    # busy-wait on pos.m_commsLock instead?
    for idx, ax in enumerate(pos.Parameters):
        
        if verbose:
            print(ax.Quantity)
            
        ax.SetPoint = ax.Values[0] + delta


        while not ax.IsAtSetPoint:
        
            if verbose:
                print(ax.Values[0], ax.Units)
            
                # wait 100ms
                time.sleep(0.1)
                
        break
    
    return

def step():
    pos = XCD()
    settings = XcdSettings()

    settings.Speed = 0.0001
    settings.IPAddress = IPAddress.Parse('192.168.10.11')

    pos.Connect()

    if not pos.IsHomingDone:
        pos.DoHoming()
        time.sleep(1)

    print_position_status()

    update_x_position(delta=0.001, verbose=True)

    pos.Disconnect()
    
if __name__ == '__main__':
    step()
