import inspect
from typing import Optional
from dataclasses import dataclass

class SDCArgs:
    """ base class for default experiment arguments

    relies on a bit of introspection magic to generate param strings for the Ametek backend
    the backend wants a comma-delimited argument string

    subclass SDCargs with a dataclass to specify argument defaults (and ordering)

    use a thin subclass of these dataclasses to make the interface nicer
    e.g. the interface subclass can take `versus` as a single argument
    and pass that value to `versus_initial` and `versus_final`

    """
    @classmethod
    def from_dict(cls, args):
        """ override default dataclass arguments, but skip any keys that aren't attributes of the dataclass """
        return cls(**{
            k: v for k, v in args.items()
            if k in inspect.signature(cls).parameters
        })

    def format(self):
        """ format a comma-delimited argument string in the order expected by Ametek backend """
        paramstring = ','.join(
            [str(arg).upper() for name, arg in self.__dict__.items()]
        )
        return paramstring

    def as_dict(self):
        return self.__dict__

@dataclass
class LPRArgs(SDCArgs):
    initial_potential: float = 0.0
    versus_initial: str = 'VS REF'
    final_potential: float = 1.0
    versus_final: str = 'VS REF'
    step_size: float = 0.1
    step_time: float = 0.1
    limit_1_type: Optional[str] = None
    limit_1_direction: str = '<'
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = '<'
    limit_2_value: float = 0
    current_range: str = 'AUTO'
    acquisition_mode: str = 'AUTO'
    electrometer: str = 'AUTO'
    e_filter: str = 'AUTO'
    i_filter: str = 'AUTO'
    leave_cell_on: str = 'YES'
    cell: str = 'INTERNAL'
    enable_ir_compensation: str = 'DISABLED'
    bandwidth: str = 'AUTO'
    low_current_interface_bandwidth: str = 'AUTO'


@dataclass
class PotentiostaticArgs(SDCArgs):
    initial_potential: float = 0.0
    versus_initial: str = 'VS REF'
    time_per_point: float = 0.00001
    duration: float = 10
    limit_1_type: Optional[str] = None
    limit_1_direction: str = '<'
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = '<'
    limit_2_value: float = 0
    current_range: str = 'AUTO'
    acquisition_mode: str = 'AUTO'
    electrometer: str = 'AUTO'
    e_filter: str = 'AUTO'
    i_filter: str = 'AUTO'
    leave_cell_on: str = 'YES'
    cell: str = 'INTERNAL'
    enable_ir_compensation: str = 'DISABLED'
    bandwidth: str = 'AUTO'
    low_current_interface_bandwidth: str = 'AUTO'


@dataclass
class LSVArgs(SDCArgs):
    initial_potential: float = 0.0
    versus_initial: str = 'VS REF'
    final_potential: float = 0.65
    versus_final: str = 'VS REF'
    scan_rate: float = 1.0
    limit_1_type: Optional[str] = None
    limit_1_direction: str = '<'
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = '<'
    limit_2_value: float = 0
    current_range: str = 'AUTO'
    electrometer: str = 'AUTO'
    e_filter: str = 'AUTO'
    i_filter: str = 'AUTO'
    leave_cell_on: str = 'YES'
    cell: str = 'INTERNAL'
    enable_ir_compensation: str = 'DISABLED'
    user_defined_the_amount_of_ir_comp: float = 1
    use_previously_determined_ir_comp: str = 'YES'
    bandwidth: str = 'AUTO'
    low_current_interface_bandwidth: str = 'AUTO'


@dataclass
class TafelArgs(SDCArgs):
    initial_potential: float = -0.25
    versus_initial: str = 'VS OC'
    final_potential: float = 0.25
    versus_final: str = 'VS OC'
    step_height: float = 0.001
    step_time: float = 0.5
    limit_1_type: Optional[str] = None
    limit_1_direction: str = '<'
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = '<'
    limit_2_value: float = 0
    current_range: str = 'AUTO'
    acquisition_mode: str = 'AUTO'
    electrometer: str = 'AUTO'
    e_filter: str = 'AUTO'
    i_filter: str = 'AUTO'
    leave_cell_on: str = 'YES'
    cell: str = 'INTERNAL'
    enable_ir_compensation: str = 'DISABLED'
    bandwidth: str = 'AUTO'
    low_current_interface_bandwidth: str = 'AUTO'

@dataclass
class OpenCircuitArgs(SDCArgs):
    time_per_point: float = 1
    duration: float = 10
    limit_1_type: Optional[str] = None
    limit_1_direction: str = '<'
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = '<'
    limit_2_value: float = 0
    current_range: str = '2MA'
    electrometer: str = 'AUTO'
    e_filter: str = 'AUTO'
    i_filter: str = 'AUTO'
    cell: str = 'INTERNAL'
    bandwidth: str = 'AUTO'
    low_current_interface_bandwidth: str = 'AUTO'
    e_resolution: str = 'AUTO'

@dataclass
class CorrosionOpenCircuitArgs(SDCArgs):
    time_per_point: float = 1
    duration: float = 10
    limit_1_type: Optional[str] = None
    limit_1_direction: str = '<'
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = '<'
    limit_2_value: float = 0
    current_range: str = '2MA'
    electrometer: str = 'AUTO'
    e_filter: str = 'AUTO'
    i_filter: str = 'AUTO'
    cell: str = 'INTERNAL'
    bandwidth: str = 'AUTO'
    low_current_interface_bandwidth: str = 'AUTO'
    e_resolution: str = 'AUTO'

class CyclicVoltammetryArgs(SDCArgs):
    initial_potential: float = 0.0
    versus_initial: str = 'VS REF'
    vertex_potential_1: float = 1.0
    versus_vertex_1: str = 'VS REF'
    vertex_hold_1: float = 0
    acquire_data_during_vertex_hold_1: str = 'NO'
    vertex_potential_2: float = -1.0
    versus_vertex_2: str = 'VS REF'
    vertex_hold_2: float = 0
    acquire_data_during_vertex_hold_2: str = 'NO'
    scan_rate: float = 0.1
    cycles: int = 3
    limit_1_type: Optional[str] = None
    limit_1_direction: str = '<'
    limit_1_value: float = 0
    limit_2_type: Optional[str] = None
    limit_2_direction: str = '<'
    limit_2_value: float = 0
    current_range: str = 'AUTO'
    electrometer: str = 'AUTO'
    e_filter: str = 'AUTO'
    i_filter: str = 'AUTO'
    leave_cell_on: str = 'YES'
    cell: str = 'INTERNAL'
    enable_ir_compensation: str = 'DISABLED'
    user_defined_the_amount_of_ir_comp: float = 1
    use_previously_determined_ir_comp: str = 'YES'
    bandwidth: str = 'AUTO'
    final_potential: float = 0.0
    versus_final: str = 'VS REF'
    low_current_interface_bandwidth: str = 'AUTO'