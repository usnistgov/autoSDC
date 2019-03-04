""" websockets + json api for requesting experiments from the sdc """
#!/usr/bin/env python

import json
import asyncio
import websockets
from datetime import datetime

from asdc import sdc

def potentiostatic(params):
    del params['x']
    del params['y']
    results = sdc.experiment.run_potentiostatic(**params)
    return results

def cv(params):
    del params['x']
    del params['y']
    results = sdc.experiment.run_cv_scan(**params)
    return results


async def sdc_transaction(websocket):
    """ await command, send ack, send data, await ack """

    # await command
    command = json.loads(await websocket.recv())
    print(command)

    # send ack
    response = {
        'ok': True,
        'reply_to': command['id'],
        'ts': datetime.now().isoformat(),
        'message': 'experiment started.'
    }
    await websocket.send(json.dumps(response))

    if command['type'] == 'potentiostatic':
        results = potentiostatic(command['params'])
    elif command['type'] == 'cv':
        results = cv(command['params'])
    elif command['type'] == 'stop':
        results = {}

    # send data
    data = {
        'type': 'results',
        'id': command['id'],
        'results': results,
        'ts': datetime.now().isoformat()
    }
    await websocket.send(json.dumps(data))

    # await ack
    ack = json.loads(await websocket.recv())
    if ack['ok']:
        return

async def sdc_client(endpoint='ws://localhost:8765/instrument'):
    """ establish a websockets connection and wait for experiment transactions to fulfill """

    async with websockets.connect(endpoint) as websocket:
        name = input("Instrument name? ")

        await websocket.send(name)
        print(">", name)

        greeting = await websocket.recv()
        print("<", greeting)

        while True:
            await sdc_transaction(websocket)

if __name__ == '__main__':

    instrument_endpoint = 'ws://localhost:8765/instrument'
    asyncio.get_event_loop().run_until_complete(sdc_client(instrument_endpoint))
