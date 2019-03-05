""" asdc.control: pythonnet .NET interface to VersaSTAT/VersaSCAN libraries """

import os
import sys
import time
import inspect
import numpy as np
from contextlib import contextmanager

n_data = 5

class VersaStatError(Exception):
    pass

@contextmanager
def controller(start_idx=17109013, initial_mode='potentiostat'):
    """ context manager that wraps potentiostat controller class Control. """
    ctl = Potentiostat(start_idx=start_idx, initial_mode=initial_mode)
    try:
        ctl.stop()
        ctl.clear()
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

class Potentiostat():
    """ Interface to the VersaSTAT SDK library for instrument control

    methods are broken out into `Immediate` (direct instrument control) and `Experiment`.
    """
    def __init__(self, start_idx=0, initial_mode='potentiostat'):

        self.instrument = None
        self.start_idx = start_idx
        self.connect()

        self.serial_number = None
        self.model = None
        self.options = None
        self.low_current_interface = None

        self.mode = initial_mode
        self.current_range = None

        # action buffer for shim
        self.action_queue = []

    def connect(self):
        self.index = self.start_idx
        self.connected = True

    def disconnect(self):
        self.connected = False

    # Immediate methods -- direct instrument control

    def set_cell(self, status='on'):
        """ turn the cell on or off """

        if status not in ('on', 'off'):
            raise ArgumentError('specify valid cell status in {on, off}')

    def choose_cell(self, choice='external'):
        """ choose between the internal and external cells. """

        if choice not in ('internal', 'external'):
            raise ArgumentError('specify valid cell in {internal, external}')

    def set_mode(self, mode):
        """ choose between potentiostat and galvanostat modes. """

        if mode not in ('potentiostat', 'galvanostat'):
            raise ArgumentError('set mode = {potentiostat, galvanostat}')

    def set_current_range(self, current_range):

        valid_current_ranges = ['2A', '200mA', '20mA', '2mA', '200uA', '20uA', '2uA', '200nA', '20nA', '2nA']

        if current_range not in valid_current_ranges:
            raise ArgumentError('specify valid current range ({})'.format(valid_current_ranges))

        self.current_range = current_range

    def set_dc_potential(self, potential):
        """ Set the output DC potential (in Volts). This voltage must be within the instruments capability."""
        pass

    def set_dc_current(self, current):
        """ Set the output DC current (in Amps). This current must be within the instruments capability.

        Calling this method also changes to Galvanostat mode and sets the current range to the correct value.
        WARNING: Once cell is enabled after setting the DC current, do not change to potentiostatic mode or change the current range.
        These will affect the value being applied to the cell.
        """
        pass

    def set_ac_frequency(self, frequency):
        """ Sets the output AC Frequency (in Hz). This frequency must be within the instruments capability."""
        pass

    def set_ac_amplitude(self, amplitude):
        """ Sets the output AC Amplitude (in RMS Volts). This amplitude must be within the instruments capabilities."""
        pass

    def set_ac_waveform(self, mode='on'):
        waveform_modes = ['on', 'off']

        if mode not in waveform_modes:
            raise ArgumentError('specify valid AC waveform mode {on, off}.')

    def update_status(self):
        """ Retrieve the status information from the instrument.
        Also auto-ranges the current if an experiment sequence is not in progress.

        Call this prior to calling the status methods below.
        """

        pass

    def latest_potential(self):
        """ get the latest stored E value. """
        return None

    def latest_current(self):
        """ get the latest stored I value. """
        return None

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

        return None

    def booster_enabled(self):
        """ check status of the booster switch. """
        return None

    def cell_enabled(self):
        """ check status of the cell. """
        return None

    def autorange_current(self, auto):
        """ Enable or disable (default is enabled) automatic current ranging while an experiment is not running.
        Disabling auto-ranging is useful when wanting to apply a DC current in immediate mode.
        """
        pass

    # Experiment methods
    # Experiment actions apparently can be run asynchronously

    def actions(self):
        """ get the current experiment action queue. """
        # Returns a list of comma delimited action names that are supported by the instrument that is currently connected
        return None

    def clear(self):
        """ clear the experiment action queue. """
        self.action_queue = []

    def start(self, max_wait_time=30, poll_interval=2):
        """ Starts the sequence of actions in the instrument that is currently connected.
        Wait until the instrument starts the action to return control flow. """
        print('started experiment sequence successfully.')

        return

    def stop(self):
        """ Stops the sequence of actions that is currently running in the instrument that is currently connected. """
        pass

    def skip(self):
        """ Skips the currently running action and immediately starts the next action.
        If there is no more actions to run, the sequence is simply stopped.
        """
        self.action_queue.pop(0)

    def sequence_running(self):
        """ Returns true if a sequence is currently running on the connected instrument, false if not. """
        pass

    def points_available(self):
        """  Returns the number of points that have been stored by the instrument after a sequence of actions has begun.
        Returns -1 when all data has been retrieved from the instrument.
        """
        return None

    def last_open_circuit(self):
        """ Returns the last measured Open Circuit value.
        This value is stored at the beginning of the sequence (and updated anytime the “AddMeasureOpenCircuit” action is called) """
        return None


    # The following Action Methods can be called in order to create a sequence of Actions.
    # A single string argument encodes multiple parameters as comma-separated lists...
    # For example, AddOpenCircuit( string ) could be called, then AddEISPotentiostatic( string ) called.
    # This would create a sequence of two actions, when started, the open circuit experiment would run, then the impedance experiment.

    # TODO: write a class interface for different experimental actions to streamline logging and serialization?

    # TODO: code-generation for GetData* interface?

    def potential(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()

        values = np.random.random(n_data)

        if as_list:
            return [value for value in values]

        return values

    def current(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()

        values = np.random.random(n_data)

        if as_list:
            return [value for value in values]

        return values

    def elapsed_time(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()

        values = np.random.random(n_data)

        if as_list:
            return [value for value in values]

        return values

    def applied_potential(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()

        values = np.random.random(n_data)

        if as_list:
            return [value for value in values]

        return values

    def segment(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()

        values = np.random.random(n_data)

        if as_list:
            return [value for value in values]

        return values

    def current_range_history(self, start=0, num_points=None, as_list=True):

        if num_points is None:
            num_points = self.points_available()

        values = np.random.random(n_data)

        if as_list:
            return [value for value in values]

        return values

    def hardcoded_open_circuit(self, params):
        default_params = "1,10,NONE,<,0,NONE,<,0,2MA,AUTO,AUTO,AUTO,INTERNAL,AUTO,AUTO,AUTO"
        print(default_params)
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
        parameters = locals().copy()

        # concatenate argument values in function signature order
        args = inspect.getfullargspec(self.linear_scan_voltammetry).args

        # remove reference to controller object
        args = args[1:]
        del parameters['self']

        paramstring = ','.join([str(parameters[arg]).upper() for arg in args])
        status = 'ok'
        self.action_queue.append(parameters)
        return status, parameters

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
        parameters = locals().copy()

        # concatenate argument values in function signature order
        args = inspect.getfullargspec(self.open_circuit).args

        # remove reference to controller object
        args = args[1:]
        del parameters['self']

        paramstring = ','.join([str(parameters[arg]).upper() for arg in args])
        status = 'ok'
        self.action_queue.append(parameters)

        return status, parameters

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
        parameters = locals().copy()

        # concatenate argument values in function signature order
        args = inspect.getfullargspec(self.corrosion_open_circuit).args

        # remove reference to controller object
        args = args[1:]
        del parameters['self']

        paramstring = ','.join([str(parameters[arg]).upper() for arg in args])
        status = 'ok'
        self.action_queue.append(parameters)

        return status, parameters

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
        parameters = locals().copy()

        # concatenate argument values in function signature order
        args = inspect.getfullargspec(self.cyclic_voltammetry).args

        # remove reference to controller object
        args = args[1:]
        del parameters['self']

        paramstring = ','.join([str(parameters[arg]).upper() for arg in args])
        status = 'ok'
        self.action_queue.append(parameters)

        return status, parameters

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
        parameters = locals().copy()

        # concatenate argument values in function signature order
        args = inspect.getfullargspec(self.multi_cyclic_voltammetry).args

        # remove reference to controller object
        args = args[1:]
        del parameters['self']

        paramstring = ','.join([str(parameters[arg]).upper() for arg in args])
        status = 'ok'
        self.action_queue.append(parameters)

        return status, parameters

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
        parameters = locals().copy()

        # concatenate argument values in function signature order
        args = inspect.getfullargspec(self.potentiostatic).args

        # remove reference to controller object
        args = args[1:]
        del parameters['self']

        paramstring = ','.join([str(parameters[arg]).upper() for arg in args])
        status = 'ok'
        self.action_queue.append(parameters)

        return status, parameters