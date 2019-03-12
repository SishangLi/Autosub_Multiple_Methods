import tensorflow as tf
from tqdm import tqdm
import sys


def vocab_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        data_path='./',
        pny_file = 'pny.txt',
        han_file = 'han.txt',
		ampny_file = 'ampny.txt')
    return params


class get_vocab():
    def __init__(self, args):
        self.data_path = args.data_path
        self.pny_vocfile = self.data_path + args.pny_file
        self.han_vocfile = self.data_path + args.han_file
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

        print('load han vocabulary...')
        sub_file = self.han_vocfile
        with open(sub_file, 'r', encoding='utf8') as f:
            data = f.readlines()
        for line in tqdm(data):
            han_lst.append([i for i in line.split(' ') if (i != '' and i != '\n')])

        print('make am pinyin vocab...')
        self.ampny_vocab = self.mk_am_pny_vocab(pny_lst)
        print('make lm pinyin vocab...')
        self.pny_vocab = self.mk_lm_pny_vocab(pny_lst)
        print('make lm hanzi vocab and save...')
        self.han_vocab = self.mk_lm_han_vocab(han_lst)

    def mk_am_pny_vocab(self, data):
        vocab = []
        for line in tqdm(data):
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        vocab.append('_')
        return vocab
		
    def mk_lm_pny_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        return vocab

    def mk_lm_han_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            for han in line:
                if han not in vocab:
                    vocab.append(han)
        return vocab

def main():
    vocab_args = vocab_hparams()
    vocab = get_vocab(vocab_args)
    # 构建模型需要和词库长度匹配，这是测试截取词库的代码
    vocab.ampny_vocab = vocab.ampny_vocab[:10]
    vocab.pny_vocab = vocab.pny_vocab[:10]
    vocab.han_vocab = vocab.han_vocab[:10]
    print('OK!')
    print(vocab.ampny_vocab)
    print(vocab.pny_vocab)
    print(vocab.han_vocab)



if __name__ == '__main__':
    sys.exit(main())












