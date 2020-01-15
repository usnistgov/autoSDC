import json
import asyncio
import websockets

PROVIDERS = set()
SUBSCRIBERS = set()

async def echo(websocket, path):
    while True:
        chunk = await websocket.recv()

        data = json.loads(chunk)

        if data.get('type') == 'goodbye':
            break

        print(f"< {chunk}")

async def sdc_handler(websocket, command):
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

async def subscriber_handler(websocket):
    print('hello client.')
    SUBSCRIBERS.add(websocket)

    try:
        message = {'type': 'hello'}
        await websocket.send(json.dumps(message))

        async for message in websocket:
            print(message)
            for subscriber in SUBSCRIBERS:
                await client.send(message)

    except websockets.exceptions.ConnectionClosed:
        print('connection closed')

    finally:
        CLIENTS.remove(websocket)

async def handler(websocket, path):

    if path == '/sdc':
        await sdc_handler(websocket)

    elif path == '/subscriber':
        await subscriber_handler(websocket)

start_server = websockets.serve(handler, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
