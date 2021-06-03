import pyupbit
import numpy as np
import pandas as pd
import datetime
import timeit
import time
import random
import sys
from json.decoder import JSONDecodeError
import requests
import json


def post_message(*values):
    """슬랙 메시지 전송"""
    myToken = "xoxb-2120785924737-2120807096337-Omv2JW7ryhY6QBAuR0zJszGZ"
    texts = []
    for item in values:
        texts.append(str(item))
    str_text = ' '.join(texts)
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+myToken},
        data={"channel": '#online_kyt',"text": str_text}
    )
    print(str_text)

number =2
post_message("hey", number)