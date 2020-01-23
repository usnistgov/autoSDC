import re
import slack
import time
import asyncio
import concurrent
from datetime import datetime

BOT_TOKEN = open('slacktoken.txt', 'r').read().strip()

bot_patterns = {
    'sdc': '<@UHT11TM6F>',
    'ctl': '<@UHNHM7198>'
}

class SlackBot(object):

    def __init__(self, name='sdc'):
        self.name = name
        # self._pattern = f'@{self.name}'
        self._pattern = bot_patterns.get(name) # '<@UHT11TM6F>'
        self.command = {}

        slack.RTMClient.run_on(event='message')(self.handle_message)

    async def handle_message(self, **payload):
        print(payload)
        data = payload['data']
        text = data.get('text', '')
        print(text)

        bot_reference = re.match(self._pattern, text)
        if bot_reference:
            print('reference to bot!')

            # handle bot commands
            _, end = bot_reference.span()
            m = text[end:]

            try:
                command, args = m.strip().split(None, 1)
            except ValueError:
                command, args = m.strip(), None

            print(command)

            if command not in self.command:
                user = data['user']
                r = f"sorry <@{user}>, I didn't understand what you meant by `{m.strip()}`..."
                print(r)
                web_client = payload['web_client']

                web_client.chat_postMessage(
                    channel=data['channel_id'],
                    text=r,
                    thread_ts=data['ts']
                )

            else:
                msgdata = {'username': data['user'], 'channel': message['channel_id']}
                self.log(text)

                # dispatch command method by name
                # await getattr(self, command)(ws, msgdata, args)
                await getattr(self, command)(web_client, args)


    async def main(self):

        self.loop = asyncio.get_event_loop()
        self.client = slack.RTMClient(token=BOT_TOKEN, run_async=True, loop=self.loop)
        # rtm_client = slack.RTMClient(token=BOT_TOKEN, run_async=True, loop=self.loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        await asyncio.ensure_future(self.client.start())

if __name__ == "__main__":
    bot = SlackBot()
    asyncio.run(bot.main())
