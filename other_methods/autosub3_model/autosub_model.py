import argparse
import audioop
import warnings
import math
import multiprocessing
import os
import subprocess
import sys
import wave
import shutil

from progressbar import ProgressBar, Percentage, Bar, ETA
from constants import LANGUAGE_CODES
from formatters import FORMATTERS

# 模型需要的模块
import tensorflow as tf
import numpy as np
from utils import decode_ctc
from utils import data_process
from tqdm import tqdm


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


warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)


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
                num = num+1
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
            command = ["ffmpeg","-ss", str(start), "-t", str(end - start),
                       "-y", "-i", self.source_path,
                       "-loglevel", "error", tempname]
            use_shell = True if os.name == "nt" else False
            subprocess.check_output(command, stdin=open(os.devnull), shell=use_shell)
            return tempname
        except KeyboardInterrupt:
            return


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


def vocab_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        data_path='./vocabulary/',
        pny_file='pny.txt',
        han_file='han.txt')
    return params


class get_vocab:
    def __init__(self, args):
        self.data_path = args.data_path
        self.pny_vocfile = self.data_path + args.pny_file
        self.han_vocfile = self.data_path + args.han_file
        self.pny_vocab = []
        self.han_vocab = []
        self.source_init()

    def source_init(self):
        pny_lst = []
        han_lst = []
        print('load pny vocabulary...')
        sub_file = self.pny_vocfile
        with open(sub_file, 'r', encoding='utf8') as f:
            data = f.readlines()
        for line in tqdm(data):
            pny_lst.append([i for i in line.split(' ') if (i != '' and i != '\n')])
        print('\n')
        print('load han vocabulary...')
        sub_file = self.han_vocfile
        with open(sub_file, 'r', encoding='utf8') as f:
            data = f.readlines()
        for line in tqdm(data):
            han_lst.append([i for i in line.split(' ') if (i != '' and i != '\n')])

        print('\n')
        print('make am pinyin vocab...')
        self.ampny_vocab = self.mk_am_pny_vocab(pny_lst)
        print('\n')
        print('make lm pinyin vocab...')
        self.pny_vocab = self.mk_lm_pny_vocab(pny_lst)
        print('\n')
        print('make lm hanzi vocab and save...')
        self.han_vocab = self.mk_lm_han_vocab(han_lst)

    @staticmethod
    def mk_am_pny_vocab(data):
        vocab = []
        for line in tqdm(data):
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        vocab.append('_')
        return vocab

    @staticmethod
    def mk_lm_pny_vocab(data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        return vocab

    @staticmethod
    def mk_lm_han_vocab(data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            for han in line:
                if han not in vocab:
                    vocab.append(han)
        return vocab


class SpeechRecognizer(object):
    def __init__(self, language="en", rate=44100):
        self.language = language
        self.rate = rate
        self.vocab_args = vocab_hparams()
        self.vocab = get_vocab(self.vocab_args)
        self.data_processer = data_process()

        # 1.声学模型-----------------------------------
        from model_prepare.model_speech.cnn_ctc import Am, am_hparams
        self.am_args = am_hparams()
        self.am_args.vocab_size = len(self.vocab.ampny_vocab)
        self.am = Am(self.am_args)
        print('loading acoustic model...')
        self.am.ctc_model.load_weights('logs_am/model.h5')

        from model_prepare.model_language.transformer import Lm, lm_hparams
        self.lm_args = lm_hparams()
        self.lm_args.input_vocab_size = len(self.vocab.pny_vocab)
        self.lm_args.label_vocab_size = len(self.vocab.han_vocab)
        self.lm_args.dropout_rate = 0.
        print('loading language model...')
        self.lm = Lm(self.lm_args)
        self.sess = tf.Session(graph=self.lm.graph)
        with self.lm.graph.as_default():
            self.saver = tf.train.Saver()
        with self.sess.as_default():
            self.latest = tf.train.latest_checkpoint('logs_lm')
            self.saver.restore(self.sess, self.latest)

    def destroy(self):
        self.sess.close()

    def speech(self, filename):
        try:
            inputs, _ = self.data_processer.am_data_process(filename)
            audio_con = inputs['the_inputs']
            result = self.am.model.predict(audio_con, steps=1)
            _, text = decode_ctc(result, self.vocab.ampny_vocab)
            text = ' '.join(text)
            with self.sess.as_default():
                text = text.strip('\n').split(' ')
                x = np.array([self.vocab.pny_vocab.index(pny) for pny in text if pny != ''])
                if len(x) == 0:
                    x = np.array([120, 79, 1, 53])  # 为了不出现警告，临时添加的没用，加几都行
                x = x.reshape(1, -1)
                preds = self.sess.run(self.lm.preds, {self.lm.x: x})
                ultimate_result = ''.join(self.vocab.han_vocab[idx] for idx in preds[0])
            return ultimate_result
        except KeyboardInterrupt:
            return


class Translator(object):
    def __init__(self, language, api_key, src, dst):
        self.language = language
        self.api_key = api_key
        self.src = src
        self.dst = dst

    def __call__(self, sentence):
        try:
            if not sentence: return
            result = self.service.translations().list(
                source=self.src,
                target=self.dst,
                q=[sentence]
            ).execute()
            if 'translations' in result and len(result['translations']) and 'translatedText' in result['translations'][0]:
                return result['translations'][0]['translatedText']
            return ""
        except KeyboardInterrupt:
            return


def validate(args):
    """
    Check that the CLI arguments passed to autosub are valid.
    """
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

    return True


def generate_subtitles(source_path, concurrency, src_language, dst_language, subtitle_file_format, output=None):
    audio_filename, audio_rate = extract_audio(source_path)
    regions = find_speech_regions(audio_filename)
    pool = multiprocessing.Pool(concurrency)
    converter = WAVConverter(source_path=audio_filename)
    recognizer = SpeechRecognizer(src_language, rate=audio_rate)
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

            for i, file in enumerate(extracted_regions):
                transcript = recognizer.speech(file)
                transcripts.append(transcript)
                pbar.update(i)
            recognizer.destroy()
            pbar.finish()

        except KeyboardInterrupt:
            pbar.finish()
            pool.terminate()
            pool.join()
            print("Cancelling transcription")
            return 1

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_path', default=".\\CCTV_News.mp4", help="Path to\
                        the video or audio file to subtitle", nargs='?')
    parser.add_argument('-C', '--concurrency', help="Number of concurrent API requests to make", type=int, default=10)
    parser.add_argument('-o', '--output', help="Output path for subtitles (by default, subtitles are saved in \
                        the same directory and name as the source path)")
    parser.add_argument('-F', '--format', help="Destination subtitle format", default="srt")
    parser.add_argument('-S', '--src-language', help="Language spoken in source file", default="zh-CN")
    parser.add_argument('-D', '--dst-language', help="Desired language for the subtitles", default="zh-CN")
    parser.add_argument('-K', '--api-key',
                        help="The Google Translate API key to be used. (Required for subtitle translation)")
    parser.add_argument('--list-formats', help="List all available subtitle formats", action='store_true')
    parser.add_argument('--list-languages', help="List all available source/destination languages", action='store_true')
    args = parser.parse_args()

    if not validate(args):
        return 1

    try:
        subtitle_file_path = generate_subtitles(
            source_path=args.source_path,
            concurrency=args.concurrency,
            src_language=args.src_language,
            dst_language=args.dst_language,
            subtitle_file_format=args.format,
            output=args.output,
        )
        print("Subtitles file created at {}".format(subtitle_file_path))
    except KeyboardInterrupt:
        return 1


if __name__ == '__main__':
    sys.exit(main())

