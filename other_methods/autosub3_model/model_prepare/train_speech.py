# coding=utf-8
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from trainutils import get_data, data_hparams

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

# 0.准备训练所需数据------------------------------
data_args = data_hparams()
data_args.shuffle = True
train_data = get_data(data_args)


# 1.声学模型训练-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams
am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args)

if os.path.exists('../logs_am/model.h5'):
    print('load acoustic model...')
    am.ctc_model.load_weights('../logs_am/model.h5')

epochs = 30
batch_num = len(train_data.wav_lst) // train_data.batch_size

for k in range(epochs):
    print('this is the', k+1, 'th epochs trainning !!!')
    batch = train_data.get_am_batch()
    am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=1)
    am.ctc_model.save_weights('../logs_am/model_'+str(k+1)+'.h5')
    print('the', k+1, 'th epochs weight has been saved in log_am/model_'+str(k+1)+'.h5 !!!')

