""" websockets + json api for requesting experiments from the sdc """
#!/usr/bin/env python

import json
import asyncio
import websockets
from datetime import datetime

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

    # send data
    data = {
        'type': 'results',
        'id': command['id'],
        'api': '0.1',
        'results': [1,2,3],
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
