import tensorflow as tf
from tqdm import tqdm
import sys
import os

def data_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        data_path = './updatefile/')
    return params

class get_data():
	def __init__(self, args):
		self.data_path = args.data_path
		self.source_init()

	def source_init(self):
		print('get source list...')
		read_files = []
		'''
		# 这段代码是自动读取目录下的所有txt文件，但是由于词库需要和训练时候的词库一一对应，一次需要指定读取顺序
		for root, dirs, files in os.walk(self.data_path):
			for file in files:
				if os.path.splitext(file)[1] == '.txt':
					read_files.append(file)
		'''
		# 此处的内容用于更新词库，顺序要和训练时的一致！
		read_files.append('thchs_train.txt')
		#read_files.append('thchs_test.txt')
		#read_files.append('thchs_dev.txt')
		read_files.append('aishell_train.txt')
		#read_files.append('aishell_test.txt')
		#read_files.append('aishell_dev.txt')
		read_files.append('prime.txt')
		read_files.append('stcmd.txt')

		for file in read_files:
			print(file)
		self.pny_lst = []
		self.han_lst = []
		for file in read_files:
			print('load ', file, ' data...')
			sub_file = self.data_path + file
			with open(sub_file, 'r', encoding='utf8') as f:
				data = f.readlines()
			for line in tqdm(data):
				_ , pny, han = line.split('\t')
				self.pny_lst.append([i for i in pny.split(' ') if i != ''])
				self.han_lst.append(''.join(han.strip('\n').split(' ')))
		print('make pinyin vocab and save...')
		self.pny_vocab = self.mk_lm_pny_vocab(self.pny_lst)
		self.sv_lm_pny_vocab(self.pny_vocab)
		print('make hanzi vocab and save...')
		self.han_vocab = self.mk_lm_han_vocab(self.han_lst)
		self.sv_lm_han_vocab(self.han_vocab)
	
	def mk_lm_pny_vocab(self, data):
		vocab = []
		for line in tqdm(data):
			for pny in line:
				if pny not in vocab:
					vocab.append(pny)
		return vocab
	
	def mk_lm_han_vocab(self, data):
		vocab = []
		for line in tqdm(data):
			for han in line:
				if han not in vocab:
					vocab.append(han)
		return vocab
	
	def sv_lm_pny_vocab(self, data):
		voc_pny = '.\\pny.txt'
		with open(voc_pny, 'wb') as f:
			for i, con_out in enumerate(data):
				f.write(('{} '.format(con_out)).encode("utf-8"))
				if (i+1) % 50 == 0:
					f.write(('\n').encode("utf-8"))
		f.close()
		
	def sv_lm_han_vocab(self, data):
		voc_han = '.\\han.txt'
		with open(voc_han, 'wb') as f:
			for i, con_out in enumerate(data):
				f.write(('{} '.format(con_out)).encode("utf-8"))
				if (i+1) % 50 == 0:
					f.write(('\n').encode("utf-8"))
		f.close()
		
def main():
	data_args = data_hparams()
	data = get_data(data_args)
	
if __name__ == '__main__':
    sys.exit(main())

	
	
	
	
	
	
	
	
	
	
	
	