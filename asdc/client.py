#!/usr/bin/env python

# WS client example

import asyncio
import websockets

async def client():
    async with websockets.connect('ws://localhost:8765/client') as websocket:

        message = await websocket.recv()
        print("<", message)

        request = 'run an XRD experiment.'
        await websocket.send(request)
        print("<", request)

asyncio.get_event_loop().run_until_complete(client())
