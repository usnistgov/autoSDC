#!/usr/bin/env python

import os
import clr
import sys
import time

# pythonnet checks PYTHONPATH for assemblies to load...
# so add the VeraScan libraries to sys.path
vpath = "C:/Program Files (x86)/Princeton Applied Research/VersaSCAN"
vdkpath = "C:/Program Files (x86)/Princeton Applied Research/VersaSTAT Development Kit"
sys.path.append(vpath)
sys.path.append(os.path.join(vpath, "Devices"))
sys.path.append(vdkpath)

# load instrument control library...
clr.AddReference("VersaSTATControl")
from VersaSTATControl import Instrument 

# load motion controller library
dlls = [
    'CommsLibrary',
    'DeviceInterface',
    'ScanDevices',
    'NanomotionXCD'
]
for dll in dlls:
    clr.AddReference(dll)

from SolartronAnalytical.DeviceInterface.NanomotionXCD import XCD, XcdSettings

clr.AddReference('System')
clr.AddReference('System.Net')
from System.Net import IPAddress

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
    
if __name__ == __main__():
    step()
