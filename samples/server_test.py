# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/12/29 19:36
@File: server_test.py
@Desc: 
"""
import json

import requests

url = 'http://192.168.6.7:8081'
file_path = '1003.jpg'

files = {
    'file': (file_path, open(file_path, 'rb'), 'image/png'),
}

r = requests.post(url, files=files)
print(r.text)

