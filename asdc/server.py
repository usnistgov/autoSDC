#!/usr/bin/env python

import json
import time
import asyncio
import websockets
from datetime import datetime

INSTRUMENTS = set()

cmds = [
    {'id': 1, 'type': 'potentiostatic', 'params': {'x': 1, 'y': 2, 'potential': -0.9, 'duration': 30, 'cell': 'EXTERNAL'}},
    {'id': 2, 'type': 'cv', 'params': {'x': 1, 'y': 2, 'cell': 'EXTERNAL'}},
    {'id': 3, 'type': 'potentiostatic', 'params': {'x': 3, 'y': 3, 'potential': -0.95, 'duration': 35, 'cell': 'EXTERNAL'}},
    {'id': 4, 'type': 'cv', 'params': {'x': 3, 'y': 3, 'cell': 'EXTERNAL'}},
    {'id': 5, 'type': 'stop', 'params': {}},
]

commands = asyncio.Queue()
for cmd in cmds:
    commands.put_nowait(cmd)

async def instrument_transaction(websocket, command):
    """ send command, await ack, await data, send ack """

    # send command
    await websocket.send(json.dumps(command))

    # get ack
    ack = json.loads(await websocket.recv())
    print(ack)

    # await data
    data = json.loads(await websocket.recv())
    print(data)
    print()

    # send ack
    response = {
        'ok': True,
        'reply_to': data['id'],
        'ts': datetime.now().isoformat(),
        'message': 'results recieved.'
    }
    await websocket.send(json.dumps(response))

async def instrument_handler(websocket):
    """ configure an instrument from an incoming connection

    dispatch command transactions to the instrument from a command queue
    """

    # register instrument connection
    INSTRUMENTS.add(websocket)
    inst = await websocket.recv()
    print(inst)

    message = 'acknowledged, {}'.format(inst)
    await websocket.send(message)

    try:
        while True:
            command = await commands.get()
            await instrument_transaction(websocket, command)

    finally:
        INSTRUMENTS.remove(websocket)

async def client_handler(websocket):
    print('hello client.')
    message = '{} instruments available'.format(len(INSTRUMENTS))
    await websocket.send(message)

    request = await websocket.recv()
    print(request)

    await commands.put(
        {'id': 5, 'type': 'diffraction', 'params': {'x': 1, 'y': 2, 'duration': 30}}
    )

async def handler(websocket, path):

    if path == '/instrument':
        await instrument_handler(websocket)
    elif path == '/client':
        await client_handler(websocket)

start_server = websockets.serve(handler, 'localhost', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

print('ok')
