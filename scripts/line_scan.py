#!/usr/bin/env python

import versascan.position
import versascan.control

def line_scan(speed=0.0001, poll_interval=0.5):

    delta = [0.001, 0.001, 0.0]
    with versascan.position.Position(ip='192.168.10.11', speed=speed) as pos:
        
        pos.print_status()

        ctl = versascan.control.Control(start_idx=1300)

        for idx in range(10):

            # run a CV experiment
            status, params = ctl.cyclic_voltammetry()
            ctl.start()

            while ctl.sequence_running():
                sleep(poll_interval)
                
            # collect data
            I = ctl.current()
            V = ctl.potential()

            ctl.clear()
            
            # update position
            pos.update(delta=delta, verbose=True)
            pos.print_status()
    

if __name__ == '__main__':
    line_scan()
