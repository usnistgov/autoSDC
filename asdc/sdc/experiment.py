import json
import time
import asyncio
import websockets
import numpy as np
import pandas as pd
from datetime import datetime

from asdc import slack
# from .pump import PumpArray

try:
    from . import potentiostat
except ModuleNotFoundError:
    from .shims import potentiostat

# the 3F potentiostat
potentiostat_id = 17109013
POLL_INTERVAL = 1
MIN_SAMPLING_FREQUENCY = 1.0e-5

async def run_experiment_sequence(pstat, poll_interval=POLL_INTERVAL, streaming_ws="ws://localhost:8765"):
    """ run an SDC experiment sequence -- busy wait until it's finished """

    pstat.start()

    error_codes = set()
    metadata = {'timestamp_start': datetime.now()}

    if streaming_ws is None:
        while pstat.sequence_running():
            time.sleep(poll_interval)
            pstat.update_status()
            overload_status = pstat.overload_status()
            if overload_status != 0:
                print('OVERLOAD:', overload_status)
                error_codes.add(overload_status)
    else:
        async with websockets.connect(streaming_ws) as ws:
            current_idx = 0
            while pstat.sequence_running():
                time.sleep(poll_interval)

                pstat.update_status()
                overload_status = pstat.overload_status()
                if overload_status != 0:
                    print('OVERLOAD:', overload_status)
                    error_codes.add(overload_status)

                n_avail = pstat.points_available()
                chunk_size = n_avail - current_idx
                results_chunk = {
                    'elapsed_time': pstat.potential(start=current_idx, num_points=chunk_size),
                    'potential': pstat.potential(start=current_idx, num_points=chunk_size)
                }
                current_idx = n_avail
                await ws.send(json.dumps(results_chunk))
            await ws.send(json.dumps({'type': 'goodbye'}))

    metadata['timestamp'] = datetime.now()
    metadata['error_codes'] = json.dumps(list(map(int, error_codes)))

    results = pd.DataFrame(
        {
            'current': pstat.current(),
            'potential': pstat.potential(),
            'elapsed_time': pstat.elapsed_time(),
            'applied_potential': pstat.applied_potential(),
            'current_range': pstat.current_range_history(),
            'segment': pstat.segment()
        }
    )

    return results, metadata

def setup_potentiostatic(pstat, data, cell='INTERNAL'):
    """ run a constant potential

    potential (float:volt)
    duration (float:second)

    {"op": "potentiostatic", "potential": Number(volts), "duration": Time(seconds)}
    """

    n_points = 3000
    duration = data.get('duration', 10)
    time_per_point = np.maximum(duration / n_points, MIN_SAMPLING_FREQUENCY)

    status, params = pstat.potentiostatic(
        initial_potential=data.get('potential'),
        time_per_point=time_per_point,
        duration=duration,
        current_range=data.get('current_range', 'AUTO'),
        e_filter='1HZ',
        i_filter='1HZ',
        cell_to_use=cell
    )

    return status, params

def setup_corrosion_oc(pstat, data, cell='INTERNAL'):
    """ set up corrosion open circuit

    duration (float:second)

    {"op": "corrosion_oc", "duration": Time}
    """
    time_per_point = 1
    duration = data.get('duration', 10)

    # run an open-circuit followed by a CV experiment
    status, params = pstat.corrosion_open_circuit(
        time_per_point=time_per_point,
        duration=duration,
        current_range=data.get('current_range', 'AUTO'),
        e_filter='1HZ',
        i_filter='1HZ',
        cell_to_use=cell
    )

    return status, params

def setup_cv(pstat, data, cell='INTERNAL'):
    """ set up a CV experiment

    initial_potential (float:volt)
    vertex_potential_1 (float:volt)
    vertex_potential_2 (float:volt)
    final_potential (float:volt)
    scan_rate (float)
    cycles (int)

    {
        "op": "cv",
        "initial_potential": 0.0,
        "vertex_potential_1": -1.0,
        "vertex_potential_2": 1.2,
        "final_potential": 0.0,
        "scan_rate": 0.075,
        "cycles": 2
    }
    """

    status, params = pstat.multi_cyclic_voltammetry(
        initial_potential=data.get('initial_potential'),
        vertex_potential_1=data.get('vertex_potential_1'),
        vertex_potential_2=data.get('vertex_potential_2'),
        final_potential=data.get('final_potential'),
        scan_rate=data.get('scan_rate'),
        cycles=data.get('cycles'),
        e_filter='1HZ',
        i_filter='1HZ',
        cell_to_use=cell,
        current_range=data.get('current_range', 'AUTO')
    )

    return status, params


async def run(instructions, cell='INTERNAL', verbose=False, streaming_ws=None):

    with potentiostat.controller(start_idx=potentiostat_id) as pstat:

        if type(instructions) is dict and instructions.get('op'):
            # single experiment -- just run it
            instructions = [instructions]

        _params = []
        for instruction in instructions:
            params = None
            if instruction.get('op') == 'potentiostatic':
                status, params = setup_potentiostatic(pstat, instruction, cell=cell)

            elif instruction.get('op') == 'cv':
                status, params = setup_cv(pstat, instruction, cell=cell)

            elif instruction.get('op') == 'corrosion_oc':
                status, params = setup_corrosion_oc(pstat, instruction, cell=cell)

            if params:
                _params.append(params)

            # elif instruction.get('op') == 'set_pH':
            #     print('setting the pH!')
            #     params = f"pH={instruction.get('pH')}"
            #     pump_array.set_pH(setpoint=instruction.get('pH'))
            #     pump_array.run_all()
            #     hold_time = instruction.get('hold_time', 0)
            #     print(f'waiting {hold_time} (s) for solution composition to reach steady state')
            #     time.sleep(hold_time)

        slack.post_message(f'starting experiment sequence')
        scan_data, metadata = await run_experiment_sequence(pstat, streaming_ws=streaming_ws)
        # if pump_array:
        #     pump_array.stop_all()
        #     time.sleep(60)

    metadata['measurement'] = json.dumps([instruction.get('op') for instruction in instructions])
    metadata['parameters'] = json.dumps(_params)

    return scan_data, metadata
