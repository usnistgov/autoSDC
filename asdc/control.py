""" asdc.control: pythonnet .NET interface to VersaSTAT/VersaSCAN libraries """

import os
import clr
import sys
import time
import inspect
from contextlib import contextmanager

# pythonnet checks PYTHONPATH for assemblies to load...
# so add the VersaSCAN libraries to sys.path
vdkpath = "C:/Program Files (x86)/Princeton Applied Research/VersaSTAT Development Kit"
sys.path.append(vdkpath)

# load instrument control library...
clr.AddReference("VersaSTATControl")
from VersaSTATControl import Instrument

class VersaStatError(Exception):
    pass

@contextmanager
def controller(start_idx=17109013, initial_mode='potentiostat'):
    """ context manager that wraps potentiostat controller class Control. """
    ctl = Control(start_idx=start_idx, initial_mode=initial_mode)
    try:
        yield ctl
    except Exception as exc:
        print(exc)
        print('Exception: unwind potentiostat controller...')
        ctl.stop()
        ctl.clear()
        ctl.disconnect()
        raise
    finally:
        print('disconnect from potentiostat controller.')
        ctl.stop()
        ctl.clear()
        ctl.disconnect()

class Control():
    """ Interface to the VersaSTAT SDK library for instrument control

    methods are broken out into `Immediate` (direct instrument control) and `Experiment`.
    """
    def __init__(self, start_idx=0, initial_mode='potentiostat'):

        self.instrument = Instrument()
        self.start_idx = start_idx
        self.connect()

        self.serial_number = self.instrument.GetSerialNumber()
        self.model = self.instrument.GetModel()
        self.options = self.instrument.GetOptions()
        self.low_current_interface = self.instrument.GetIsLowCurrentInterfacePresent()

        self.mode = initial_mode
        self.current_range = None

        return

    def connect(self):
        self.index = self.instrument.FindNext(self.start_idx)
        self.connected = self.instrument.Connect(self.index)
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
        current = 'SetIRange_{}'.format(current_range)
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

    def latest_potential(self):
        """ get the latest stored E value. """
        return self.instrument.Immediate.GetE()

    def latest_current(self):
        """ get the latest stored I value. """
        return self.instrument.Immediate.GetI()

    def overload_status(self, raise_exception=False):
        """ check for overloading.
        0 indicates no overload, 1 indicates I (current) Overload, 2
indicates E, Power Amp or Thermal Overload has occurred.
        """
        overload_cause = {
            1: 'I (current) overload',
            2: 'E, Power Amp, or Thermal overload'
        }

        overload_code = self.instrument.Immediate.GetOverload()

        if overload_code and raise_exception:
            msg = 'A ' + overload_cause[overload_code] + ' has occurred.'
            raise VersaStatError(msg)

        return overload_code

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

    def start(self, max_wait_time=30, poll_interval=2):
        """ Starts the sequence of actions in the instrument that is currently connected.
        Wait until the instrument starts the action to return control flow. """

        self.instrument.Experiment.Start()

        # Note: ctl.start() can return before the sequence actually starts running,
        # so it's possible to skip right past the data collection spin-waiting loop
        # which writes a data-less log file and pushes the next experiment onto the queue
        # while the instrument is still going on with the current one.
        # it appears that this is not safe....
        elapsed = 0

        while not self.sequence_running():
            time.sleep(poll_interval)
            elapsed += poll_interval

            if elapsed > max_wait_time:
                raise VersaStatError("could not start experiment")
                raise KeyboardInterrupt("could not start.")
                break

        print('started experiment sequence successfully.')

        return

    def stop(self):
        """ Stops the sequence of actions that is currently running in the instrument that is currently connected. """
        self.instrument.Experiment.Stop()

    def skip(self):
        """ Skips the currently running action and immediately starts the next action.
        If there is no more actions to run, the sequence is simply stopped.
        """
        self.instrument.Experiment.Skip()

    def sequence_running(self):
        """ Returns true if a sequence is currently running on the connected instrument, false if not. """
        return self.instrument.Experiment.IsSequenceRunning()

    def points_available(self):
        """  Returns the number of points that have been stored by the instrument after a sequence of actions has begun.
        Returns -1 when all data has been retrieved from the instrument.
        """
        return self.instrument.Experiment.GetNumPointsAvailable()

    def last_open_circuit(self):
        """ Returns the last measured Open Circuit value.
        This value is stored at the beginning of the sequence (and updated anytime the “AddMeasureOpenCircuit” action is called) """
        return self.instrument.Experiment.GetLastMeasuredOC()


    # The following Action Methods can be called in order to create a sequence of Actions.
    # A single string argument encodes multiple parameters as comma-separated lists...
    # For example, AddOpenCircuit( string ) could be called, then AddEISPotentiostatic( string ) called.
    # This would create a sequence of two actions, when started, the open circuit experiment would run, then the impedance experiment.

    # TODO: write a class interface for different experimental actions to streamline logging and serialization?

    # TODO: code-generation for GetData* interface?

    def potential(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()

        values = self.instrument.Experiment.GetDataPotential(start, num_points)

        if as_list:
            return [value for value in values]

        return values

    def current(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()

        values = self.instrument.Experiment.GetDataCurrent(start, num_points)

        if as_list:
            return [value for value in values]

        return values

    def elapsed_time(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()

        values = self.instrument.Experiment.GetDataElapsedTime(start, num_points)

        if as_list:
            return [value for value in values]

        return values

    def applied_potential(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()

        values = self.instrument.Experiment.GetDataAppliedPotential(start, num_points)

        if as_list:
            return [value for value in values]

        return values

    def segment(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()

        values = self.instrument.Experiment.GetDataSegment(start, num_points)

        if as_list:
            return [value for value in values]

        return values

    def current_range_history(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()

        values = self.instrument.Experiment.GetDataCurrentRange(start, num_points)

        if as_list:
            return [value for value in values]

        return values

    def add_open_circuit(self, params):
        default_params = "1,10,NONE,<,0,NONE,<,0,2MA,AUTO,AUTO,AUTO,INTERNAL,AUTO,AUTO,AUTO"
        status = self.instrument.Experiment.AddOpenCircuit(default_params)
        return status, default_params

    def linear_scan_voltammetry(self,
        initial_potential=0.0,
        versus_initial='VS REF',
        final_potential=0.65,
        versus_final='VS REF',
        scan_rate=1.0,
        limit_1_type=None,
        limit_1_direction='<',
        limit_1_value=0,
        limit_2_type=None,
        limit_2_direction='<',
        limit_2_value=0,
        current_range='AUTO',
        electrometer='AUTO',
        e_filter='AUTO',
        i_filter='AUTO',
        leave_cell_on='NO',
        cell_to_use='INTERNAL',
        enable_ir_compensation='DISABLED',
        user_defined_the_amount_of_ir_comp=1,
        use_previously_determined_ir_comp='YES',
        bandwidth='AUTO',
        low_current_interface_bandwidth='AUTO'):
        """ linear_scan_voltammetry
        IP Vs FP Vs SR L1T L1D L1V L2T L2D L2V IR EM EF IF LCO CTU iRC UD UP BW LBW
        """
        # concatenate argument values in function signature order
        args = inspect.getfullargspec(self.linear_scan_voltammetry).args

        # get rid of self
        args = args[1:]
        vals = locals()

        params = ','.join([str(vals[arg]).upper() for arg in args])
        status = self.instrument.Experiment.AddLinearScanVoltammetry(params)
        return status, params

    def open_circuit(self,
            time_per_point=1,
            duration=10,
            limit_1_type='NONE',
            limit_1_direction='<',
            limit_1_value=0,
            limit_2_type=None,
            limit_2_direction='<',
            limit_2_value=0,
            current_range='2MA',
            electrometer='AUTO',
            e_filter='AUTO',
            i_filter='AUTO',
            cell_to_use='INTERNAL',
            bandwidth='AUTO',
            low_current_interface_bandwidth='AUTO',
            e_resolution='AUTO'):
        """ open_circuit
        limit_1_type [Limit 1 Type] {NONE, CURRENT, POTENTIAL or CHARGE}
        limit_1_direction [Limit 1 Direction] {< or >}
        limit_1_value [Limit 1 Value] {User value}
        limit_2_type [Limit 2 Type] {NONE, CURRENT, POTENTIAL or CHARGE}
        limit_2_direction [Limit 2 Direction] {< or >}
        limit_2_value [Limit 2 Value] {User value}
        current_range [Current Range] (*) {AUTO, 2A, 200MA, 20MA, 2MA,200UA,20UA,2UA,200NA, 20NA or 4N}
        electrometer [Electrometer] {AUTO, SINGLE ENDED or DIFFERENTIAL}
        e_filter [E Filter] (**) {AUTO, NONE, 200KHZ, 1KHZ, 1KHZ, 100HZ 10HZ, 1HZ}
        i_filter [I Filter] (**) {AUTO, NONE, 200KHZ, 1KHZ, 1KHZ, 100HZ, 10HZ, 1HZ}
        cell_to_use [Cell To Use] {INTERNAL or EXTERNAL}

        default_params = "1,10,NONE,<,0,NONE,<,0,2MA,AUTO,AUTO,AUTO,INTERNAL,AUTO,AUTO,AUTO"
        """

        # concatenate argument values in function signature order
        args = inspect.getfullargspec(self.open_circuit).args

        # get rid of self
        args = args[1:]
        vals = locals()

        params = ','.join([str(vals[arg]).upper() for arg in args])
        status = self.instrument.Experiment.AddOpenCircuit(params)
        return status, params

    def corrosion_open_circuit(self,
            time_per_point=1,
            duration=10,
            limit_1_type='NONE',
            limit_1_direction='<',
            limit_1_value=0,
            limit_2_type=None,
            limit_2_direction='<',
            limit_2_value=0,
            current_range='2MA',
            electrometer='AUTO',
            e_filter='AUTO',
            i_filter='AUTO',
            cell_to_use='INTERNAL',
            bandwidth='AUTO',
            low_current_interface_bandwidth='AUTO',
            e_resolution='AUTO'):
        """ corrosion_open_circuit
        limit_1_type [Limit 1 Type] {NONE, CURRENT, POTENTIAL or CHARGE}
        limit_1_direction [Limit 1 Direction] {< or >}
        limit_1_value [Limit 1 Value] {User value}
        limit_2_type [Limit 2 Type] {NONE, CURRENT, POTENTIAL or CHARGE}
        limit_2_direction [Limit 2 Direction] {< or >}
        limit_2_value [Limit 2 Value] {User value}
        current_range [Current Range] (*) {AUTO, 2A, 200MA, 20MA, 2MA,200UA,20UA,2UA,200NA, 20NA or 4N}
        electrometer [Electrometer] {AUTO, SINGLE ENDED or DIFFERENTIAL}
        e_filter [E Filter] (**) {AUTO, NONE, 200KHZ, 1KHZ, 1KHZ, 100HZ 10HZ, 1HZ}
        i_filter [I Filter] (**) {AUTO, NONE, 200KHZ, 1KHZ, 1KHZ, 100HZ, 10HZ, 1HZ}
        cell_to_use [Cell To Use] {INTERNAL or EXTERNAL}

        """

        # concatenate argument values in function signature order
        args = inspect.getfullargspec(self.corrosion_open_circuit).args

        # get rid of self
        args = args[1:]
        vals = locals()

        params = ','.join([str(vals[arg]).upper() for arg in args])
        status = self.instrument.Experiment.AddCorrosionOpenCircuit(params)
        return status, params

    # NOTE: use enum for options?
    def cyclic_voltammetry(self,
            initial_potential=0.0,
            versus_initial='VS REF',
            vertex_potential=0.65,
            versus_vertex='VS REF',
            vertex_hold=0,
            acquire_data_during_vertex_hold=True,
            final_potential=0.25,
            versus_final='VS REF',
            scan_rate=0.1,
            limit_1_type=None,
            limit_1_direction='<',
            limit_1_value=0,
            limit_2_type=None,
            limit_2_direction='<',
            limit_2_value=0,
            current_range='AUTO',
            electrometer='AUTO',
            e_filter='AUTO',
            i_filter='AUTO',
            leave_cell_on='NO',
            cell_to_use='INTERNAL',
            enable_ir_compensation='DISABLED',
            user_defined_the_amount_of_ir_comp=1,
            use_previously_determined_ir_comp='YES',
            bandwidth='AUTO',
            low_current_interface_bandwidth='AUTO'):
        """ cyclic_voltammetry

        initial_potential [Initial Potential] (V) {User value -10 to 10 (could be “NOT USED” for Multi-Cycle CV)}
        versus [Versus] {VS OC, VS REF or VS PREVIOUS}
        vertex_potential [Vertex Potential] (V) {User value -10 to 10 (could be “NOT USED” for Multi-Vertex Scan)}
        versus [Versus] {VS OC, VS REF or VS PREVIOUS}
        vertex_hold [Vertex Hold] (s) {User value}
        acquire_data_during_vertex_hold [Acquire data during Vertex Hold] {YES or NO}
        final_potential [Final Potential] (V) {User value -10 to 10 (could be “NOT USED” for Multi-Cycle CV)}
        versus [Versus] {VS OC, VS REF or VS PREVIOUS}
        scan_rate [Scan Rate] (V/s) {User value}
        limit_1_type [Limit 1 Type] {NONE, CURRENT, POTENTIAL or CHARGE}
        limit_1_direction [Limit 1 Direction] {< or >}
        limit_1_value [Limit 1 Value] {User value}
        limit_2_type [Limit 2 Type] {NONE, CURRENT, POTENTIAL or CHARGE}
        limit_2_direction [Limit 2 Direction] {< or >}
        limit_2_value [Limit 2 Value] {User value}
        current_range [Current Range] (*) {AUTO, 2A, 200MA, 20MA, 2MA,200UA,20UA,2UA,200NA, 20NA or 4N}
        electrometer [Electrometer] {AUTO, SINGLE ENDED or DIFFERENTIAL}
        e_filter [E Filter] (**) {AUTO, NONE, 200KHZ, 1KHZ, 1KHZ, 100HZ 10HZ, 1HZ}
        i_filter [I Filter] (**) {AUTO, NONE, 200KHZ, 1KHZ, 1KHZ, 100HZ, 10HZ, 1HZ}
        leave_cell_on [Leave Cell On] {YES or NO}
        cell_to_use [Cell To Use] {INTERNAL or EXTERNAL}
        enable_ir_compensation [enable iR Compensation] {ENABLED or DISABLED}
        user_defined_the_amount_of_ir_comp [User defined the amount of iR Comp] (ohms) {User value}
        use_previously_determined_ir_comp [Use previously determined iR Comp] {YES or NO}
        bandwidth [Bandwidth] (***) {AUTO, HIGH STABILITY, 1MHZ, 100KHZ, 1KHZ}
        low_current_interface_bandwidth [Low Current Interface Bandwidth] (****) {AUTO, NORMAL, SLOW, VERY SLOW}
        """

        # concatenate argument values in function signature order
        args = inspect.getfullargspec(self.cyclic_voltammetry).args

        # get rid of self
        args = args[1:]

        vals = locals()

        params = ','.join([str(vals[arg]).upper() for arg in args])
        status = self.instrument.Experiment.AddCyclicVoltammetry(params)

        return status, params


    def multi_cyclic_voltammetry(self,
            initial_potential=0.0,
            versus_initial='VS REF',
            vertex_potential_1=1.0,
            versus_vertex_1='VS REF',
            vertex_hold_1=0,
            acquire_data_during_vertex_hold_1='NO',
            vertex_potential_2=-1.0,
            versus_vertex_2='VS REF',
            vertex_hold_2=0,
            acquire_data_during_vertex_hold_2='NO',
            scan_rate=0.1,
            cycles=3,
            limit_1_type='NONE',
            limit_1_direction='<',
            limit_1_value=0,
            limit_2_type='NONE',
            limit_2_direction='<',
            limit_2_value=0,
            current_range='AUTO',
            electrometer='AUTO',
            e_filter='AUTO',
            i_filter='AUTO',
            leave_cell_on='NO',
            cell_to_use='INTERNAL',
            enable_ir_compensation='DISABLED',
            user_defined_the_amount_of_ir_comp=1,
            use_previously_determined_ir_comp='YES',
            bandwidth='AUTO',
            final_potential=0.0,
            versus_final='VS REF',
            low_current_interface_bandwidth='AUTO'):
        """ multi_cyclic_voltammetry

        initial_potential [Initial Potential] (V) {User value -10 to 10 (could be “NOT USED” for Multi-Cycle CV)}
        versus [Versus] {VS OC, VS REF or VS PREVIOUS}
        vertex_potential [Vertex Potential] (V) {User value -10 to 10 (could be “NOT USED” for Multi-Vertex Scan)}
        versus [Versus] {VS OC, VS REF or VS PREVIOUS}
        vertex_hold [Vertex Hold] (s) {User value}
        acquire_data_during_vertex_hold [Acquire data during Vertex Hold] {YES or NO}
        vertex_potential [Vertex Potential] (V) {User value -10 to 10 (could be “NOT USED” for Multi-Vertex Scan)}
        versus [Versus] {VS OC, VS REF or VS PREVIOUS}
        vertex_hold [Vertex Hold] (s) {User value}
        acquire_data_during_vertex_hold [Acquire data during Vertex Hold] {YES or NO}
        scan_rate [Scan Rate] (V/s) {User value}
        cycles [Cycles] (#) {User value}
        limit_1_type [Limit 1 Type] {NONE, CURRENT, POTENTIAL or CHARGE}
        limit_1_direction [Limit 1 Direction] {< or >}
        limit_1_value [Limit 1 Value] {User value}
        limit_2_type [Limit 2 Type] {NONE, CURRENT, POTENTIAL or CHARGE}
        limit_2_direction [Limit 2 Direction] {< or >}
        limit_2_value [Limit 2 Value] {User value}
        current_range [Current Range] (*) {AUTO, 2A, 200MA, 20MA, 2MA,200UA,20UA,2UA,200NA, 20NA or 4N}
        electrometer [Electrometer] {AUTO, SINGLE ENDED or DIFFERENTIAL}
        e_filter [E Filter] (**) {AUTO, NONE, 200KHZ, 1KHZ, 1KHZ, 100HZ 10HZ, 1HZ}
        i_filter [I Filter] (**) {AUTO, NONE, 200KHZ, 1KHZ, 1KHZ, 100HZ, 10HZ, 1HZ}
        leave_cell_on [Leave Cell On] {YES or NO}
        cell_to_use [Cell To Use] {INTERNAL or EXTERNAL}
        enable_ir_compensation [enable iR Compensation] {ENABLED or DISABLED}
        user_defined_the_amount_of_ir_comp [User defined the amount of iR Comp] (ohms) {User value}
        use_previously_determined_ir_comp [Use previously determined iR Comp] {YES or NO}
        bandwidth [Bandwidth] (***) {AUTO, HIGH STABILITY, 1MHZ, 100KHZ, 1KHZ}
        final_potential [Final Potential] (V) {User value -10 to 10 (could be “NOT USED” for Multi-Cycle CV)}
        versus [Versus] {VS OC, VS REF or VS PREVIOUS}
        low_current_interface_bandwidth [Low Current Interface Bandwidth] (****) {AUTO, NORMAL, SLOW, VERY SLOW}
        """

        # concatenate argument values in function signature order
        args = inspect.getfullargspec(self.multi_cyclic_voltammetry).args

        # get rid of 'self' argument
        args = args[1:]

        vals = locals()

        params = ','.join([str(vals[arg]) for arg in args])
        status = self.instrument.Experiment.AddMultiCyclicVoltammetry(params)
        return status, params


    def potentiostatic(self,
            initial_potential=0.0,
            versus_initial='VS REF',
            time_per_point=0.00001,
            duration=10,
            limit_1_type=None,
            limit_1_direction='<',
            limit_1_value=0,
            limit_2_type=None,
            limit_2_direction='<',
            limit_2_value=0,
            current_range='AUTO',
            acquisition_mode='AUTO',
            electrometer='AUTO',
            e_filter='AUTO',
            i_filter='AUTO',
            leave_cell_on='NO',
            cell_to_use='INTERNAL',
            enable_ir_compensation='DISABLED',
            bandwidth='AUTO',
            low_current_interface_bandwidth='AUTO'):
        """ potentiostatic

        initial_potential [Initial Potential] (V) {User value -10 to 10 (could be “NOT USED” for Multi-Cycle CV)}
        versus [Versus] {VS OC, VS REF or VS PREVIOUS}
        time_per_point [TPP] (s) {float 0.00001  to ...}
        duration [Dur] (s) {float 0.00001 to ...}
        limit_1_type [Limit 1 Type] {NONE, CURRENT, POTENTIAL or CHARGE}
        limit_1_direction [Limit 1 Direction] {< or >}
        limit_1_value [Limit 1 Value] {User value}
        limit_2_type [Limit 2 Type] {NONE, CURRENT, POTENTIAL or CHARGE}
        limit_2_direction [Limit 2 Direction] {< or >}
        limit_2_value [Limit 2 Value] {User value}
        current_range [Current Range] (*) {AUTO, 2A, 200MA, 20MA, 2MA,200UA,20UA,2UA,200NA, 20NA or 4N}
        acquisition_mode [AM] {AUTO, NONE, 4/4, AVERAGE}
        electrometer [Electrometer] {AUTO, SINGLE ENDED or DIFFERENTIAL}
        e_filter [E Filter] (**) {AUTO, NONE, 200KHZ, 1KHZ, 1KHZ, 100HZ 10HZ, 1HZ}
        i_filter [I Filter] (**) {AUTO, NONE, 200KHZ, 1KHZ, 1KHZ, 100HZ, 10HZ, 1HZ}
        leave_cell_on [Leave Cell On] {YES or NO}
        cell_to_use [Cell To Use] {INTERNAL or EXTERNAL}
        enable_ir_compensation [enable iR Compensation] {ENABLED or DISABLED}
        bandwidth [Bandwidth] (***) {AUTO, HIGH STABILITY, 1MHZ, 100KHZ, 1KHZ}
        low_current_interface_bandwidth [Low Current Interface Bandwidth] (****) {AUTO, NORMAL, SLOW, VERY SLOW}
        """

        # concatenate argument values in function signature order
        args = inspect.getfullargspec(self.potentiostatic).args

        # get rid of self
        args = args[1:]

        vals = locals()

        params = ','.join([str(vals[arg]).upper() for arg in args])
        status = self.instrument.Experiment.AddPotentiostatic(params)

        return status, params
