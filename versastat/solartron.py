""" solartron: set up pythonnet .NET interface to VersaSTAT/VersaSCAN libraries """
import os
import clr
import sys

# pythonnet checks PYTHONPATH for assemblies to load...
# so add the VeraScan libraries to sys.path
vpath = "C:/Program Files (x86)/Princeton Applied Research/VersaSCAN"
vdkpath = "C:/Program Files (x86)/Princeton Applied Research/VersaSTAT Development Kit"
sys.path.append(vpath)
sys.path.append(os.path.join(vpath, "Devices"))
sys.path.append(vdkpath)

# load instrument control library...
clr.AddReference("VersaSTATControl")
from VersaSTATControl import Instrument, Immediate, Experiment

# load motion controller library
dlls = [
    'CommsLibrary',
    'DeviceInterface',
    'ScanDevices',
    'NanomotionXCD'
]
for dll in dlls:
    clr.AddReference(dll)

clr.AddReference('System')
clr.AddReference('System.Net')
from System.Net import IPAddress
from SolartronAnalytical.DeviceInterface.NanomotionXCD import XCD, XcdSettings


