#!/usr/bin/env python
import audioop
import json
import math
import multiprocessing
import os
import subprocess
import wave
import base64
import time
import shutil

from progressbar import ProgressBar, Percentage, Bar, ETA
from formatters import FORMATTERS
from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode
timer = time.perf_counter

API_KEY = '4brKeEDaGTzSn9AbmXR0B6gs'
SECRET_KEY = 'FAoMX8GofXvdiA0XXBAOE7GE3yKftIyI'


def which(program):
    """
    Return the path for a given executable.
    """
    def is_exe(file_path):
        """
        Checks whether a file is executable.
        """
        return os.path.isfile(file_path) and os.access(file_path, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def extract_audio(filepath, channels=1, rate=16000):
    if not os.path.isfile(filepath):
        print("The given file does not exist: {}".format(filepath))
        raise Exception("Invalid filepath: {}".format(filepath))
    if not which("ffmpeg"):
        print("ffmpeg: Executable not found on machine.")
        raise Exception("Dependency not found: ffmpeg")
    os.mkdir('temp')
    filename = (os.path.split(filepath)[-1])[:-3] + str('wav')
    tempname = os.path.join(os.getcwd(), 'temp', filename)
    command = ["ffmpeg", "-y", "-i", filepath, "-ac", str(channels), "-ar", str(rate), "-loglevel", "error", tempname]
    use_shell = True if os.name == "nt" else False
    subprocess.check_output(command, stdin=open(os.devnull), shell=use_shell)
    return tempname, rate


def percentile(arr, percent):
    arr = sorted(arr)
    k = (len(arr) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c: return arr[int(k)]
    d0 = arr[int(f)] * (c - k)
    d1 = arr[int(c)] * (k - f)
    return d0 + d1


def find_speech_regions(filename, frame_width=4096, min_region_size=0.5, max_region_size=6):
    reader = wave.open(filename)
    sample_width = reader.getsampwidth()
    rate = reader.getframerate()
    n_channels = reader.getnchannels()
    chunk_duration = float(frame_width) / rate
    n_chunks = int(math.ceil(reader.getnframes()*1.0 / frame_width))
    energies = []
    for i in range(n_chunks):
        chunk = reader.readframes(frame_width)
        energies.append(audioop.rms(chunk, sample_width * n_channels))
    threshold = percentile(energies, 0.2)
    elapsed_time = 0
    regions = []
    region_start = None
    num = 0
    for energy in energies:
        is_silence = energy <= threshold
        max_exceeded = region_start and elapsed_time - region_start >= max_region_size
        if (max_exceeded or is_silence) and region_start:
            if elapsed_time - region_start >= min_region_size:
                num = num + 1
                regions.append((region_start, elapsed_time, num))
                region_start = None
        elif (not region_start) and (not is_silence):
            region_start = elapsed_time
        elapsed_time += chunk_duration
    return regions


class WAVConverter(object):
    def __init__(self, source_path, include_before=0.25, include_after=0.25):
        self.source_path = source_path
        self.include_before = include_before
        self.include_after = include_after

    def __call__(self, region):
        try:
            start, end, num = region
            start = max(0, start - self.include_before)
            end += self.include_after
            tempname = os.path.join(os.getcwd(), 'temp', 'temp' + str(num) + '.wav')
            command = ["ffmpeg", "-ss", str(start), "-t", str(end - start),
                       "-y", "-i", self.source_path,
                       "-loglevel", "error", tempname]
            use_shell = True if os.name == "nt" else False
            subprocess.check_output(command, stdin=open(os.devnull), shell=use_shell)
            return tempname
        except KeyboardInterrupt:
            return 1


class SpeechRecognizer(object):
    def __init__(self, api_key, secret_key, rate, audioformat, retries=3):
        self.api_key = api_key
        self.secret_ket = secret_key
        self.rate = rate
        self.format = audioformat
        self.dev_pid = 1536  # 1537 表示识别普通话，使用输入法模型。1536表示识别普通话，使用搜索模型
        self.cuid = '123456PYTHON'
        self.asr_url = 'http://vop.baidu.com/server_api'
        self.token_url = 'http://openapi.baidu.com/oauth/2.0/token'
        self.scope = 'audio_voice_assistant_get'  # 若授权认证返回值中没有此字符串，那么表示用户应用中没有开通asr功能，需要到网页端开通
        self.retries = retries
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
            result_str = err.read()
            print('token http response http code : ' + str(err.code) + str(result_str))
            return 1

        result_str = result_str.decode()
        result = json.loads(result_str)
        if 'access_token' in result.keys() and 'scope' in result.keys():
            if self.scope not in result['scope'].split(' '):
                print('scope is not correct')
                return 0
            # print('SUCCESS WITH TOKEN: %s ; EXPIRES IN SECONDS: %s' % (result['access_token'], result['expires_in']))
            print('API Handshake success')
            return result['access_token']
        else:
            print('MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not found in token response')
            return 0

    def __call__(self, filepath):
        with open(filepath, 'rb') as speech_file:
            speech_data = speech_file.read()
        length = len(speech_data)
        if length == 0:
            print('file %s length read 0 bytes' % filepath)
            return 1
        for i in range(self.retries):
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
                result_str = str(result_str, 'utf-8')
                if ((json.loads(result_str))["err_no"]) == 0:
                    result_str = ((json.loads(result_str))["result"])[0]
                    return result_str
                elif ((json.loads(result_str))["err_no"]) == 3301:
                    # print('Poor audio quality, processed as blank voice!')
                    return ''
                elif ((json.loads(result_str))["err_no"]) == 3302:
                    self.token = self.fetch_token()
                    continue
                else:
                    error_no = ((json.loads(result_str))["err_no"])
                    print('Error % s' % error_no)
                    continue
            except URLError as err:
                print('asr http response http code : ' + str(err.code) + str(err.read()))
                continue

        print("Retry failed !")
        return 'Conversion failed'


def generate_subtitles(source_path, output, concurrency, subtitle_file_format, api_key, secret_key):

    audio_filename, audio_rate = extract_audio(source_path)
    regions = find_speech_regions(audio_filename)
    pool = multiprocessing.Pool(concurrency)
    converter = WAVConverter(source_path=audio_filename)
    recognizer = SpeechRecognizer(api_key, secret_key, audio_rate, audio_filename[-3:])
    transcripts = []
    if regions:
        try:
            widgets = ["Converting speech regions to WAV files: ", Percentage(), ' ', Bar(), ' ', ETA()]
            pbar = ProgressBar(widgets=widgets, maxval=len(regions)).start()
            extracted_regions = []
            for i, extracted_region in enumerate(pool.imap(converter, regions)):
                extracted_regions.append(extracted_region)
                pbar.update(i)
            pbar.finish()

            widgets = ["Performing speech recognition: ", Percentage(), ' ', Bar(), ' ', ETA()]
            pbar = ProgressBar(widgets=widgets, maxval=len(regions)).start()
            for i, transcript in enumerate(pool.imap(recognizer, extracted_regions)):
                if transcript == 1:
                    return 0
                else:
                    transcripts.append(transcript)
                pbar.update(i)
            pbar.finish()
        except KeyboardInterrupt:
            pbar.finish()
            pool.terminate()
            pool.join()
            print("Cancelling transcription")
            return 0
    timed_subtitles = [(r, t) for r, t in zip(regions, transcripts) if t]
    formatter = FORMATTERS.get(subtitle_file_format)
    formatted_subtitles = formatter(timed_subtitles)
    dest = output
    if not dest:
        base, ext = os.path.splitext(source_path)
        dest = "{base}.{format}".format(base=base, format=subtitle_file_format)

    with open(dest, 'wb') as f:
        f.write(formatted_subtitles.encode("utf-8"))
    shutil.rmtree('temp')
    return dest


def start(videopath, outputpath=None, concurrency=10, subformat='srt', api_key=API_KEY, secret_key=SECRET_KEY):
    # concurrency:Number of concurrent API requests to make
    # output:Output path for subtitles (by default, subtitles are saved in \
    #         the same directory and name as the source path
    # format:Destination subtitle format
    # api-key:The Baidu Translate API key to be used. (Required for subtitle)
    # secret-key:The Baidu Translate Secret key to be used. (Required for subtitle)
    try:
        subtitle_file_path = generate_subtitles(
            source_path=videopath,
            output=outputpath,
            concurrency=concurrency,
            subtitle_file_format=subformat,
            api_key=api_key,
            secret_key=secret_key
        )
    finally:
        if os.path.exists('temp'):
            shutil.rmtree('temp')
            return 'Conversion failed'
    return subtitle_file_path


