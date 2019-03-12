# -*- coding: utf-8 -*-
import requests
import time
import hashlib
import base64
import json

URL = "http://api.xfyun.cn/v1/service/v1/iat"
APPID = "5c74a129"
API_KEY = "5490a6843afeee325b40d8b7195c545d"


def getHeader(aue, engineType):
    curTime = str(int(time.time()))
    # curTime = '1526542623'
    param = "{\"aue\":\"" + aue + "\"" + ",\"engine_type\":\"" + engineType + "\"}"
    print("param:{}".format(param))
    print(param)
    paramBase64 = str(base64.b64encode(param.encode('utf-8')), 'utf-8')
    print("x_param:{}".format(paramBase64))

    m2 = hashlib.md5()
    m2.update((API_KEY + curTime + paramBase64).encode('utf-8'))
    checkSum = m2.hexdigest()
    # print('checkSum:{}'.format(checkSum))
    header = {
        'X-CurTime': curTime,
        'X-Param': paramBase64,
        'X-Appid': APPID,
        'X-CheckSum': checkSum,
        'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
    }
    print(header)
    return header


def getBody(filepath):
    binfile = open(filepath, 'rb')
    data = {'audio': base64.b64encode(binfile.read())}
    # print('data:{}'.format(type(data['audio'])))
    # print("type(data['audio']):{}".format(type(data['audio'])))
    return data


aue = "raw"
engineType = "sms16k"
audioFilePath = r".\lfasr.wav"

r = requests.post(URL, headers=getHeader(aue, engineType), data=getBody(audioFilePath))
content = r.content.decode('utf-8')
if len(content) > 0:
    content = json.loads(content)
    content = content["data"]
    if len(content) > 0 and content[-1] == "。":
        content = content[:-1]
    print(content)
else:
    print("Error 404")

