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
from VersaSTATControl import Instrument

class VersaStatError(Exception):
    pass

class Control():
    """ Interface to the VersaSTAT SDK library for instrument control 

    methods are broken out into `Immediate` (direct instrument control) and `Experiment`.
    """
    def __init__(self, start_idx=0, initial_mode='potentiostat'):
        self.instrument = Instrument()        
        self.connect()

        self.serial_number = self.instrument.GetSerialNumber()
        self.model = self.instrument.GetModel()
        self.options = self.instrument.GetOptions()
        self.low_current_interface = self.instrument.GetIsLowCurrentInterfacePresent()

        self.mode = initial_mode
        self.current_range = None
        return
    
    def connect(self):
        self.index = Instrument.FindNext(self.start_idx)
        self.connected = Instrument.Connect(idx)
        return

    def disconnect(self):
        self.instrument.Close()

    # Immediate methods -- direct instrument control

    def set_cell(self, status='on'):
        """ turn the cell on or off """
        
        if status not in ('on', 'off'):
            raise ArgumentError('specify valid cell status in {on, off}')

        if status == 'on':
            self.instrument.Immediate.SetCellOn()
            
        else:
            self.instrument.Immediate.SetCellOff()

    def choose_cell(self, choice='external'):
        """ choose between the internal and external cells. """
        
        if choice not in ('internal', 'external'):
            raise ArgumentError('specify valid cell in {internal, external}')

        if choice == 'external':
            self.instrument.Immediate.SetCellExternal()
            
        elif choice == 'internal':
            self.instrument.Immediate.SetCellExternal()
            
    def set_mode(self, mode):
        """ choose between potentiostat and galvanostat modes. """
        
        if mode not in ('potentiostat', 'galvanostat'):
            raise ArgumentError('set mode = {potentiostat, galvanostat}')

        if mode == 'potentiostat':
            self.instrument.Immediate.SetModePotentiostat()
            
        elif mode == 'galvanostat':
            self.instrument.Immediate.SetModeGalvanostat()
        

    def set_current_range(self, current_range):
        
        valid_current_ranges = ['2A', '200mA', '20mA', '2mA', '200uA', '20uA', '2uA', '200nA', '20nA', '2nA']
        
        if current_range not in valid_current_ranges:
            raise ArgumentError('specify valid current range ({})'.format(valid_current_ranges))

        self.current_range = current_range
        
        # dispatch the right SetIRange_* function....
        current = 'SetIRange_{}'.format(current_string)
        set_current = getattr(self.instrument.Immediate, current)
        set_current()

    def set_dc_potential(self, potential):
        """ Set the output DC potential (in Volts). This voltage must be within the instruments capability."""
        self.instrument.Immediate.SetDCPotential(potential)
    
    def set_dc_current(self, current):
        """ Set the output DC current (in Amps). This current must be within the instruments capability.

        Calling this method also changes to Galvanostat mode and sets the current range to the correct value. 
        WARNING: Once cell is enabled after setting the DC current, do not change to potentiostatic mode or change the current range.
        These will affect the value being applied to the cell. 
        """
        self.instrument.Immediate.SetDCCurrent(current)

    def set_ac_frequency(self, frequency):
        """ Sets the output AC Frequency (in Hz). This frequency must be within the instruments capability."""
        self.instrument.Immediate.SetACFrequency(frequency)

    def set_ac_amplitude(self, amplitude):
        """ Sets the output AC Amplitude (in RMS Volts). This amplitude must be within the instruments capabilities."""
        self.instrument.Immediate.SetACAmplitude(amplitude)

    def set_ac_waveform(self, mode='on'):
        waveform_modes = ['on', 'off']
        
        if mode not in waveform_modes:
            raise ArgumentError('specify valid AC waveform mode {on, off}.')

        if mode == 'on':
            self.instrument.Immediate.SetACWaveformOn()
        elif mode == 'off':
            self.instrument.Immediate.SetACWaveformOff()

    def update_status(self):
        """ Retrieve the status information from the instrument.
        Also auto-ranges the current if an experiment sequence is not in progress.

        Call this prior to calling the status methods below.
        """

        self.instrument.Immediate.UpdateStatus()    

    def potential(self):
        """ get the latest stored E value. """
        return self.instrument.Immediate.GetE()
    
    def current(self):
        """ get the latest stored I value. """
        return self.instrument.Immediate.GetI()

    def overload_status(self):
        """ check for overloading.
        0 indicates no overload, 1 indicates I (current) Overload, 2
indicates E, Power Amp or Thermal Overload has occurred.
        """
        overload_cause = {
            1: 'I (current) overload',
            2: 'E, Power Amp, or Thermal overload'
        }
        
        overload_code = self.instrument.Immediate.GetOverload()

        if overload:
            msg = 'A ' + overload_cause[overload_code] + ' has occurred.'
            raise VersaStatError(msg)
        
    def booster_enabled(self):
        """ check status of the booster switch. """
        return self.instrument.Immediate.GetBoosterEnabled()
    
    def cell_enabled(self):
        """ check status of the cell. """
        return self.instrument.Immediate.GetCellEnabled()

    def autorange_current(self, auto):
        """ Enable or disable (default is enabled) automatic current ranging while an experiment is not running.
        Disabling auto-ranging is useful when wanting to apply a DC current in immediate mode.
        """
        if auto:
            self.instrument.Immediate.SetAutoIRangeOn()
        else:
            self.instrument.Immediate.SetAutoIRangeOff()

    
    # Experiment methods
    # Experiment actions apparently can be run asynchronously
    
    def actions(self):
        """ get the current experiment action queue. """
        # Returns a list of comma delimited action names that are supported by the instrument that is currently connected
        action_list = self.instrument.Experiment.GetActionList()
        return action_list.split(',')

    def clear(self):
        """ clear the experiment action queue. """
        self.instrument.Experiment.Clear()
        

    def start(self):
        """ Starts the sequence of actions in the instrument that is currently connected. """
        self.instrument.Experiment.Start()

    def stop(self):
        """ Stops the sequence of actions that is currently running in the instrument that is currently connected. """
        self.instrument.Experiment.Stop()
        
    def skip(self):
        """ Skips the currently running action and immediately starts the next action.
        If there is no more actions to run, the sequence is simply stopped.
        """
        self.instrument.Experiment.Skip()

    def sequence_running():
        """ Returns true if a sequence is currently running on the connected instrument, false if not. """
        return self.instrument.Experiment.IsSequenceRunning()

    def points_available():
        """  Returns the number of points that have been stored by the instrument after a sequence of actions has begun.
        Returns -1 when all data has been retrieved from the instrument.
        """
        return self.instrument.Experiment.GetNumPointsAvailable()

    def last_open_circuit():
        """ Returns the last measured Open Circuit value.
        This value is stored at the beginning of the sequence (and updated anytime the “AddMeasureOpenCircuit” action is called) """
        return self.instrument.Experiment.GetLastMeasuredOC()


    # The following Action Methods can be called in order to create a sequence of Actions.
    # A single string argument encodes multiple parameters as comma-separated lists...
    # For example, AddOpenCircuit( string ) could be called, then AddEISPotentiostatic( string ) called.
    # This would create a sequence of two actions, when started, the open circuit experiment would run, then the impedance experiment.

    # TODO: write a class interface for different experimental actions to streamline logging and serialization?

    # TODO: code-generation for GetData* interface?
    
    def add_open_circuit(params):
        status = self.instrument.Experiment.AddOpenCircuit(params)
        return status

    def linear_scan_voltammetry(params):
        status = self.instrument.Experiment.AddLinearScanVoltammetry(params)
        return status

    def cyclic_voltammetry(params):
        status = self.instrument.Experiment.AddCyclycVoltammetry(params)
        return status
    
