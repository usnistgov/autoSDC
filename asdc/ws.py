import json
import asyncio
import websockets

async def echo(websocket, path):
    while True:
        chunk = await websocket.recv()

        data = json.loads(chunk)

        if data.get('type') == 'goodbye':
            break

        print(f"< {chunk}")

start_server = websockets.serve(echo, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
