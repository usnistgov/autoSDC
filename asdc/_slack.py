import os
import json
import time
import slack
import requests

with open('slacktoken.txt', 'r') as f:
    SLACK_TOKEN = f.read()

sc = slack.WebClient(SLACK_TOKEN)

def post_message(message, sleep=1):
    """ post text to #asdc
    by default, sleep for 1s to respect slack's API limits
    """
    sc.chat_postMessage(
        channel='#asdc',
        text=message,
    )

    time.sleep(sleep)

def post_file_to_slack(text, file_name, file_bytes, file_type=None, title=None):
    return requests.post(
      'https://slack.com/api/files.upload',
      {
        'token': SLACK_TOKEN,
        'filename': file_name,
        'channels': '#asdc',
        'filetype': file_type,
        'initial_comment': text,
        'title': title
      },
      files = { 'file': file_bytes }).json()

def post_image(web_client, image_path, title='an image...', sleep=1):
    """ post a figure to #asdc
    by default, sleep for 1s to respect slack's API limits
    """

    with open(image_path, 'rb') as file_content:
        status = post_file_to_slack(title, f'{title}.png', file_content)

    time.sleep(sleep)
