import os
import time
import slackclient

with open('slacktoken.txt', 'r') as f:
    SLACK_TOKEN = f.read()

sc = slackclient.SlackClient(SLACK_TOKEN)

def post_message(message, sleep=1):
    """ post text to #asdc
    by default, sleep for 1s to respect slack's API limits
    """
    sc.api_call(
        "chat.postMessage",
        channel="#asdc",
        text=message
    )
    time.sleep(sleep)

def post_image(image_path, title='an image...', sleep=1):
    """ post a figure to #asdc
    by default, sleep for 1s to respect slack's API limits
    """

    with open(image_path, 'rb') as file_content:
        status = sc.api_call(
            "files.upload",
            channels='#asdc',
            file=file_content,
            title=title,
        )

    time.sleep(sleep)
