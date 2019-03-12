# coding=utf-8
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from trainutils import get_data, data_hparams
from progressbar import ProgressBar, Percentage, Bar, ETA

config = tf.ConfigProto()
config.gpu_options.allow_growth = True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

# 0.准备训练所需数据------------------------------
data_args = data_hparams()
data_args.shuffle = True
train_data = get_data(data_args)

# 2.语言模型训练-------------------------------------------
from model_language.transformer  import Lm, lm_hparams
lm_args = lm_hparams()
lm_args.input_vocab_size = len(train_data.pny_vocab)
lm_args.label_vocab_size = len(train_data.han_vocab)
lm = Lm(lm_args)

epochs = 30
batch_num = len(train_data.wav_lst) // train_data.batch_size
with lm.graph.as_default():
    saver = tf.train.Saver()
with tf.Session(graph=lm.graph) as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    add_num = 0
    if os.path.exists('../logs_lm/checkpoint'):
        print('loading language model...')
        latest = tf.train.latest_checkpoint('logs_lm')
        add_num = int(latest.split('_')[-1])
        saver.restore(sess, latest)
    writer = tf.summary.FileWriter('../logs_lm/tensorboard', tf.get_default_graph())
    for k in range(epochs):
        total_loss = 0
        batch = train_data.get_lm_batch()
        widgets = ['this is the ' + str(k+1) + 'th epochs tinning !!!', Percentage(), ' ', Bar(), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=batch_num).start()
        for i in range(batch_num):
            input_batch, label_batch = next(batch)
            feed = {lm.x: input_batch, lm.y: label_batch}
            cost, _ = sess.run([lm.mean_loss, lm.train_op], feed_dict=feed)
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                rs = sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, k * batch_num + i)
            pbar.update(i)
        pbar.finish()
        print('epochs', k+1, ': average loss = ', total_loss/batch_num)
        saver.save(sess, '../logs_lm/model_%d' % (epochs + add_num))
        print('the', k + 1, 'th epochs weight has been saved in log_lm !!!')
    writer.close()
