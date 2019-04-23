import os
import sys
import json

sys.path.append('../scirc')
import scirc

class SDC(scirc.Client):
    """ autonomous scanning droplet cell client """

    command = scirc.CommandRegistry()

    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        self.command.update(super().command)

    @command
    async def random(self, ws, msgdata, args):
        r = '42'
        response = {'id': 2, 'type': 'message', 'channel': msgdata['channel'], 'text': r}
        print(response)
        await ws.send_str(json.dumps(response))

    @command
    async def dm(self, ws, msgdata, args):
        """ echo random string to DM channel """
        dm_channel = 'DHNHM74TU'
        r = ' '.join(args)
        response = {'id': 2, 'type': 'message', 'channel': dm_channel, 'text': r}
        await ws.send_str(json.dumps(response))

if __name__ == '__main__':

    sdc = SDC()
    sdc.run()
