# coding=utf-8
import sys
import json
import base64
import time

from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode
timer = time.perf_counter

API_KEY = '4brKeEDaGTzSn9AbmXR0B6gs'
SECRET_KEY = 'FAoMX8GofXvdiA0XXBAOE7GE3yKftIyI'

class SpeechRecognizer(object):
    def __init__(self, api_key, secret_key, rate, audioformat):
        self.api_key = api_key
        self.secret_ket = secret_key
        self.rate = rate
        self.format = audioformat
        self.dev_pid = 1536  # 1537 表示识别普通话，使用输入法模型。1536表示识别普通话，使用搜索模型
        self.cuid = '123456PYTHON'
        self.asr_url = 'http://vop.baidu.com/server_api'
        self.token_url = 'http://openapi.baidu.com/oauth/2.0/token'
        self.scope = 'audio_voice_assistant_get'  # 若授权认证返回值中没有此字符串，那么表示用户应用中没有开通asr功能，需要到网页端开通
        self.token = self.fetch_token()

    def fetch_token(self):
        params = {'grant_type': 'client_credentials',
                  'client_id': self.api_key,
                  'client_secret': self.secret_ket}
        post_data = urlencode(params)
        post_data = post_data.encode('utf-8')
        req = Request(self.token_url, post_data)
        try:
            f = urlopen(req)
            result_str = f.read()
        except URLError as err:
            print('token http response http code : ' + str(err.code))
            result_str = err.read()
        result_str = result_str.decode()
        result = json.loads(result_str)
        if 'access_token' in result.keys() and 'scope' in result.keys():
            if self.scope not in result['scope'].split(' '):
                print('scope is not correct')
                return False
            print('SUCCESS WITH TOKEN: %s ; EXPIRES IN SECONDS: %s' % (result['access_token'], result['expires_in']))
            return result['access_token']
        else:
            print('MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not found in token response')
            return False

    def recognize(self, filepath):
        with open(filepath, 'rb') as speech_file:
            speech_data = speech_file.read()
        length = len(speech_data)
        if length == 0:
            print('file %s length read 0 bytes' % filepath)
            return [0, 'None']
        speech = base64.b64encode(speech_data)
        speech = str(speech, 'utf-8')
        params = {'dev_pid': self.dev_pid,
                  'format': self.format,
                  'rate': self.rate,
                  'token': self.token,
                  'cuid': self.cuid,
                  'channel': 1,
                  'speech': speech,
                  'len': length
                  }
        post_data = json.dumps(params, sort_keys=False)
        # print post_data
        req = Request(self.asr_url, post_data.encode('utf-8'))
        req.add_header('Content-Type', 'application/json')
        try:
            # begin = timer()
            f = urlopen(req)
            result_str = f.read()
            # print("Request time cost %f" % (timer() - begin))
        except URLError as err:
            print('asr http response http code : ' + str(err.code))
            result_str = err.read()
            return [0, result_str]
        result_str = str(result_str, 'utf-8')
        result_str = ((json.loads(result_str))["result"])[0]
        return [1, result_str]


if __name__ == '__main__':
    recognizer = SpeechRecognizer(API_KEY, SECRET_KEY, 16000, 'wav')
    result = recognizer.recognize('./pcm/lfasr.wav')
    print(result[1])

