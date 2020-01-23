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

class CommandRegistry(set):
    """ wrap a set with a decorator to register commands """
    def register(self, method):
        self.add(method.__name__)
        return method

    def __call__(self, method):
        """ overload __call__ instead of having an explicit register method """
        return self.register(method)

class SlackBot(object):
    command = CommandRegistry()

    def __init__(self, name='sdc'):
        self.name = name
        # self._pattern = f'@{self.name}'
        self._pattern = bot_patterns.get(name) # '<@UHT11TM6F>'
        # self.command = {}

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

            print('cmd: ', command)
            print('args: ', args)

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
                # msgdata = {'username': data['user'], 'channel': data['channel_id']}
                # self.log(text)

                # dispatch command method by name
                # await getattr(self, command)(ws, msgdata, args)
                web_client = payload['web_client']
                getattr(self, command)(web_client, args, data)

    @command
    def echo(self, web_client, text, data):
        r = f"recieved command {text}"
        channel = data.get('channel')
        print(r)
        web_client.chat_postMessage(
            channel=channel,
            text=r,
            thread_ts=data['ts']
        )

    def main(self):

        self.client = slack.RTMClient(token=BOT_TOKEN)
        self.client.start()

if __name__ == "__main__":
    bot = SlackBot()
    bot.main()
