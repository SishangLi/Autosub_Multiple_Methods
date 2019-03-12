#coding=utf-8
import os
import difflib
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from tqdm import tqdm
from scipy.fftpack import fft
from python_speech_features import mfcc
from random import shuffle
from keras import backend as K

class data_process():
	def __init__(self):
		print("数据处理接口已创建...")

	def am_data_process(self, filename):
		wav_data_lst = []
		fbank = compute_fbank(filename)
		pad_fbank = np.zeros((fbank.shape[0]//8*8+8, fbank.shape[1]))
		pad_fbank[:fbank.shape[0], :] = fbank
		# if pad_fbank.shape[0]//8 >= label_ctc_len:
		wav_data_lst.append(pad_fbank)
		pad_wav_data, input_length = self.wav_padding(wav_data_lst)
		inputs = {'the_inputs': pad_wav_data, 'input_length': input_length}
		outputs = {'ctc': np.zeros(pad_wav_data.shape[0],)}
		return inputs, outputs

	def get_lm_batch(self):
		batch_num = len(self.pny_lst) // self.batch_size
		for k in range(batch_num):
			begin = k * self.batch_size
			end = begin + self.batch_size
			input_batch = self.pny_lst[begin:end]
			label_batch = self.han_lst[begin:end]
			max_len = max([len(line) for line in input_batch])
			input_batch = np.array([self.pny2id(line, self.pny_vocab) + [0] * (max_len - len(line)) for line in input_batch])
			label_batch = np.array([self.han2id(line, self.han_vocab) + [0] * (max_len - len(line)) for line in label_batch])
			yield input_batch, label_batch

	def han2id(self, line, vocab):
		return [vocab.index(han) for han in line]

	def wav_padding(self, wav_data_lst):
		wav_lens = [len(data) for data in wav_data_lst]
		wav_max_len = max(wav_lens)
		wav_lens = np.array([leng//8 for leng in wav_lens])
		new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
		for i in range(len(wav_data_lst)):
			new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
		return new_wav_data_lst, wav_lens

# 对音频文件提取mfcc特征
def compute_mfcc(file):
	fs, audio = wav.read(file)
	mfcc_feat = mfcc(audio, samplerate=fs, numcep=26)
	mfcc_feat = mfcc_feat[::3]
	mfcc_feat = np.transpose(mfcc_feat)
	return mfcc_feat

# 获取信号的时频图
def compute_fbank(file):
	x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
	w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) ) # 汉明窗
	fs, wavsignal = wav.read(file)
	# wav波形 加时间窗以及时移10ms
	time_window = 25 # 单位ms
	wav_arr = np.array(wavsignal)
	range0_end = int(len(wavsignal)/fs*1000 - time_window) // 10 # 计算循环终止的位置，也就是最终生成的窗数
	data_input = np.zeros((range0_end, 200), dtype = np.float) # 用于存放最终的频率特征数据
	data_line = np.zeros((1, 400), dtype = np.float)
	for i in range(0, range0_end):
		p_start = i * 160
		p_end = p_start + 400
		data_line = wav_arr[p_start:p_end]
		data_line = data_line * w # 加窗
		data_line = np.abs(fft(data_line))
		data_input[i]=data_line[0:200] # 设置为400除以2的值（即200）是取一半数据，因为是对称的
	data_input = np.log(data_input + 1)
	#data_input = data_input[::]
	return data_input

# 定义解码器------------------------------------
def decode_ctc(num_result, num2word):
	result = num_result[:, :, :]
	in_len = np.zeros((1), dtype = np.int32)
	in_len[0] = result.shape[1]
	r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
	r1 = K.get_value(r[0][0])
	r1 = r1[0]
	text = []
	for i in r1:
		text.append(num2word[i])
	return r1, text
