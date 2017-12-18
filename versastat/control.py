""" solartron: set up pythonnet .NET interface to VersaSTAT/VersaSCAN libraries """
import os
import clr
import sys

# pythonnet checks PYTHONPATH for assemblies to load...
# so add the VeraScan libraries to sys.path
vdkpath = "C:/Program Files (x86)/Princeton Applied Research/VersaSTAT Development Kit"
sys.path.append(vdkpath)

# load instrument control library...
clr.AddReference("VersaSTATControl")
from VersaSTATControl import Instrument, Immediate, Experiment



