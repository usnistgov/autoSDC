import sys
import asyncio
import numpy as np

import regloicclib

sys.path.append('.')
from asdc import sdc

# needle defaults counterclockwise (-)
# dump (-)
# loop (+)
# source (+)

from enum import Enum

class Channel(Enum):
    NEEDLE = 1
    DUMP = 2
    LOOP = 3
    SOURCE = 4

# 12 mL/min

class Reglo():
    def __init__(self, address=None, tubing_inner_diameter=1.52):
        self.pump = regloicclib.Pump(address=address)
        self.tubing_inner_diameter = tubing_inner_diameter

        for channel in range(1,5):
            self.pump.setTubingInnerDiameter(self.tubing_inner_diameter, channel=channel)

    def droplet(
            self,
            prep_height=0.004,
            wetting_height=0.0011,
            fill_rate = 1.0,
            fill_counter_ratio=0.75,
            fill_time = None,
            shrink_ratio = 1.1,
            shrink_time = None,
            flow_rate = 0.5,
            target_rate = 0.05,
            cleanup_duration = 0,
            cleanup_pulse_duration = 0,
            stage_speed = 0.001,
    ):
        """ slack bot command for prototyping droplet contact routine

        #### json arguments

        | Name             | Type  | Description                                         | Default |
        |------------------|-------|-----------------------------------------------------|---------|
        | `prep_height`    | float | z setting to grow the droplet                       |     4mm |
        | `wetting_height` | float | z setting to wet the droplet to the surface         |   1.1mm |
        | `fill_rate`      | float | pumping rate during droplet growth                  | 1 mL/min |
        | `fill_counter_ratio` | float | counterpumping ratio during droplet growth          |    0.75 |
        | `fill_time`      | float | droplet growth duration (s)                         |    None |
        | `shrink_counter_ratio`    | float | counterpumping ratio during droplet wetting phase   |     1.1 |
        | `shrink_time`    | float | droplet wetting duration (s)                        |    None |
        | `flow_rate`      | float | total flow rate during droplet formation (mL/min)   |     0.5 |
        | `target_rate`    | float | final flow rate after droplet formation  (mL/min)   |    0.05 |
        | `cleanup`        | float | duration of pre-droplet-formation cleanup siphoning |       0 |
        | `stage_speed`    | float | stage velocity during droplet formation op          |   0.001 |

        """

        # stage speed is specified in m/s
        stage_speed = min(stage_speed, 1e-3)
        stage_speed = max(stage_speed, 1e-5)

        # start at zero
        with sdc.position.sync_z_step(height=wetting_height, speed=stage_speed):

            if cleanup_duration > 0:
                # TODO: turn on the needle
                # make an option to pulse loop and dump simultaneously, same rate opposite directions?
                print('cleaning up...')
                self.pump.continuousFlow(-10.0, channel=Channel.NEEDLE)

                if cleanup_pulse_duration > 0:
                    pulse_flowrate = -1.0
                    self.pump.continuousFlow(pulse_flowrate, channel=Channel.LOOP)
                    self.pump.continuousFlow(pulse_flowrate, channel=Channel.DUMP)

                    time.sleep(cleanup_pulse_duration)

                self.pump.stop(channel=Channel.SOURCE)
                self.pump.stop(channel=Channel.LOOP)
                self.pump.stop(channel=Channel.DUMP)

                time.sleep(cleanup_duration)

            height_difference = prep_height - wetting_height
            height_difference = max(0, height_difference)
            with sdc.position.sync_z_step(height=height_difference, speed=stage_speed):

                # counterpump slower to fill the droplet
                print('filling droplet')
                counter_flowrate = fill_rate * fill_counter_ratio
                self.pump.continuousFlow(fill_rate, channel=Channel.SOURCE)
                self.pump.continuousFlow(-counter_flowrate, channel=Channel.LOOP)
                self.pump.continuousFlow(-counter_flowrate, channel=Channel.DUMP)

                fill_start = time.time()
                if fill_time is None:
                    input('*filling droplet*: press enter to continue...')
                else:
                    time.sleep(fill_time)
                fill_time = time.time() - fill_start

            # drop down to wetting height
            # counterpump faster to shrink the droplet
            print('shrinking droplet')
            shrink_flowrate = fill_rate * shrink_counter_ratio
            self.pump.continuousFlow(-shrink_flowrate, channel=Channel.LOOP)
            self.pump.continuousFlow(-shrink_flowrate, channel=Channel.DUMP)


            shrink_start = time.time()
            if shrink_time is None:
                input('*shrinking droplet*: press enter to continue...')
            else:
                time.sleep(shrink_time)
            shrink_time = time.time() - shrink_start

            print('equalizing differential pumping rate')
            self.pump.stop(channel=Channel.SOURCE)
            self.pump.stop(channel=Channel.DUMP)
            self.pump.continuousFlow(fill_rate, channel=Channel.LOOP)


        # drop down to contact height
        instructions['fill_time'] = fill_time
        instructions['shrink_time'] = shrink_time

        time.sleep(3)

        print(f'stepping flow rates to {rates}')
        self.pump.continuousFlow(target_rate, channel=Channel.LOOP)

        message = f"contact routine with {json.dumps(instructions)}"
        print(message)

        return
