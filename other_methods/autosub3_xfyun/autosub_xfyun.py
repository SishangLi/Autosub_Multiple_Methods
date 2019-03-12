#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import audioop
import math
import multiprocessing
import os
import requests
import subprocess
import sys
import wave

import time
import hashlib
import base64
import shutil
import json

from progressbar import ProgressBar, Percentage, Bar, ETA

from constants import LANGUAGE_CODES, \
    XFYUN_SPEECH_API_KEY, XFYUN_SPEECH_API_ID, XFYUN_SPEECH_API_URL
from formatters import FORMATTERS


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


def extract_audio(filepath, channels=1, rate=16000):  # rate为采样率
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
    if f == c:
        return arr[int(k)]
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


class FLACConverter(object):
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
            return


class SpeechRecognizer(object):
    def __init__(self, aue="raw", engineType="sms16k", retries=3,
                 api_key=XFYUN_SPEECH_API_KEY, api_id=XFYUN_SPEECH_API_ID):
        self.retries = retries
        self.aue = aue
        self.engineType = engineType
        self.api_key = api_key
        self.api_id = api_id
        self.api_url = XFYUN_SPEECH_API_URL

    def getheader(self, aue, engineType):
        curTime = str(int(time.time()))
        param = "{\"aue\":\"" + aue + "\"" + ",\"engine_type\":\"" + engineType + "\"}"
        paramBase64 = str(base64.b64encode(param.encode('utf-8')), 'utf-8')
        m2 = hashlib.md5()
        m2.update((self.api_key + curTime + paramBase64).encode('utf-8'))
        checkSum = m2.hexdigest()
        header = {
            'X-CurTime': curTime,
            'X-Param': paramBase64,
            'X-Appid': self.api_id,
            'X-CheckSum': checkSum,
            'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
        }
        return header

    @staticmethod
    def getbody(filepath):
        binfile = open(filepath, 'rb')
        data = {'audio': base64.b64encode(binfile.read())}
        # print('data:{}'.format(type(data['audio'])))
        # print("type(data['audio']):{}".format(type(data['audio'])))
        return data

    def __call__(self, datapath):
        try:
            for i in range(self.retries):
                url = self.api_url
                headers = self.getheader(self.aue, self.engineType)
                data = self.getbody(datapath)
                try:
                    resp = requests.post(url, headers=headers, data=data)
                except requests.exceptions.ConnectionError:
                    continue
                content = resp.content.decode('utf-8')
                if len(content) > 0:
                    content = json.loads(content)
                    content = content["data"]
                    if len(content) > 0 and content[-1] == "。":
                        content = content[:-1]
                    return content
                else:
                    return "Error 404"

        except KeyboardInterrupt:
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_path', default="./CCTV_News.mp4", help="Path to the video or audio\
                            file to subtitle", nargs='?')
    parser.add_argument('-C', '--concurrency', help="Number of concurrent API requests to make", type=int, default=10)
    parser.add_argument('-o', '--output',
                        help="Output path for subtitles (by default, subtitles are saved in \
                        the same directory and name as the source path)")
    parser.add_argument('-F', '--format', help="Destination subtitle format", default="srt")
    parser.add_argument('-S', '--src-language', help="Language spoken in source file", default="zh-CN")
    parser.add_argument('-D', '--dst-language', help="Desired language for the subtitles", default="zh-CN")
    parser.add_argument('-K', '--api-key',
                        help="The Xfyun API key to be used. (Required for subtitle translation)")
    parser.add_argument('-ID', '--api-id',
                        help="The Xfyun API key ID to be used. (Required for subtitle translation)")
    parser.add_argument('--list-formats', help="List all available subtitle formats", action='store_true')
    parser.add_argument('--list-languages', help="List all available source/destination languages", action='store_true')

    args = parser.parse_args()

    if args.list_formats:
        print("List of formats:")
        for subtitle_format in list(FORMATTERS.keys()):
            print(("{format}".format(format=subtitle_format)))
        return 0

    if args.list_languages:
        print("List of all languages:")
        for code, language in sorted(LANGUAGE_CODES.items()):
            print(("{code}\t{language}".format(code=code, language=language)))
        return 0

    if args.format not in list(FORMATTERS.keys()):
        print("Subtitle format not supported. Run with --list-formats to see all supported formats.")
        return 1

    if args.src_language not in list(LANGUAGE_CODES.keys()):
        print("Source language not supported. Run with --list-languages to see all supported languages.")
        return 1

    if args.dst_language not in list(LANGUAGE_CODES.keys()):
        print(
            "Destination language not supported. Run with --list-languages to see all supported languages.")
        return 1

    if not args.source_path:
        print("Error: You need to specify a source path.")
        return 1

    audio_filename, audio_rate = extract_audio(args.source_path)
    regions = find_speech_regions(audio_filename)
    pool = multiprocessing.Pool(args.concurrency)
    converter = FLACConverter(source_path=audio_filename)
    recognizer = SpeechRecognizer()
    # 要切换自己的ID和KEY请使用下面这条语句
    # recognizer = SpeechRecognizer(api_key=args.api_key, api_id=args.api_id)
    transcripts = []
    if regions:
        try:
            widgets = ["Converting speech regions to FLAC files: ", Percentage(), ' ', Bar(), ' ', ETA()]
            pbar = ProgressBar(widgets=widgets, maxval=len(regions)).start()
            extracted_regions = []
            for i, extracted_region in enumerate(pool.imap(converter, regions)):
                extracted_regions.append(extracted_region)
                pbar.update(i)
            pbar.finish()

            widgets = ["Performing speech recognition: ", Percentage(), ' ', Bar(), ' ', ETA()]
            pbar = ProgressBar(widgets=widgets, maxval=len(regions)).start()
            # 这里是由于API收费限制，每天只能转换500次， 取352个片段，若没有限制可以去掉，
            extracted_regions = extracted_regions[0:352]
            for i, transcript in enumerate(pool.imap(recognizer, extracted_regions)):
                transcripts.append(transcript)
                pbar.update(i)
            pbar.finish()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            print("Cancelling transcription")
            return 1

    timed_subtitles = [(r, t) for r, t in zip(regions, transcripts) if t]
    formatter = FORMATTERS.get(args.format)
    formatted_subtitles = formatter(timed_subtitles)
    dest = args.output
    if not dest:
        base, ext = os.path.splitext(args.source_path)
        dest = "{base}.{format}".format(base=base, format=args.format)
    with open(dest, 'wb') as f:
        f.write(formatted_subtitles.encode("utf-8"))
    print("Subtitles file created at {}".format(dest))
    shutil.rmtree('temp')
    return 0


if __name__ == '__main__':
    sys.exit(main())


