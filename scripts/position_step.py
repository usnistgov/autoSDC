#!/usr/bin/env python

import versascan.position

def position_step(speed=0.0001):

    with versascan.position.Position(ip='192.168.10.11', speed=speed) as pos:
        
        pos.print_status()
        pos.update_x(delta=0.001, verbose=True)
        pos.print_status()
    

if __name__ == '__main__':
    position_step()
