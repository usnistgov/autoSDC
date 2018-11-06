import os
import slackclient

with open('slacktoken.txt', 'r') as f:
    SLACK_TOKEN = f.read()

sc = slackclient.SlackClient(SLACK_TOKEN)

def post_message(message):
    """ post text to #asdc """
    sc.api_call(
        "chat.postMessage",
        channel="#asdc",
        text=message
    )

def post_image(image_path, title='an image...'):
    """ post a figure to #asdc """

    with open(image_path, 'rb') as file_content:
        status = sc.api_call(
            "files.upload",
            channels='#asdc',
            file=file_content,
            title=title,
        )
