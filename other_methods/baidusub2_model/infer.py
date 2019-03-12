"""Inferer for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import functools
import paddle.v2 as paddle
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from utils.utility import add_arguments, print_arguments
from progressbar import ProgressBar, Percentage, Bar, ETA

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,    5,      "# of samples to infer every batch.")
add_arg('trainer_count',    int,    1,      "# of Trainers (CPUs or GPUs).")
add_arg('beam_size',        int,    300,    "Beam search width.")
add_arg('num_proc_bsearch', int,    8,      "# of CPUs for beam search.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    1024,   "# of recurrent cells per layer.")
add_arg('alpha',            float,  2.6,    "Coef of LM for beam search.")
add_arg('beta',             float,  5.0,    "Coef of WC for beam search.")
add_arg('cutoff_prob',      float,  0.99,    "Cutoff probability for pruning.")
add_arg('cutoff_top_n',     int,    40,     "Cutoff number for pruning.")
add_arg('use_gru',           bool,   True,  "Use GRUs instead of simple RNNs.")
add_arg('use_gpu',           bool,   False,   "Use GPU or not.")
add_arg('share_rnn_weights', bool,   False,   "Share input-hidden weights across "
                                            "bi-directional RNNs. Not for GRU.")
add_arg('infer_manifest', str,  'temp/wavlist.txt', "Filepath of manifest to infer.")
add_arg('mean_std_path', str, 'models/am/mean_std.npz', "Filepath of normalizer's mean & std.")
add_arg('vocab_path', str, 'models/am/vocab.txt', "Filepath of vocabulary.")
add_arg('lang_model_path', str, 'models/lm/zh_giga.no_cna_cmn.prune01244.klm', "Filepath for language model.")
add_arg('model_path', str, 'models/am/params.tar.gz', "If None, the training starts from scratch, "
        "otherwise, it resumes from the pre-trained model.")
add_arg('decoding_method', str, 'ctc_beam_search', "Decoding method. Options: ctc_beam_search, ctc_greedy",
        choices=['ctc_beam_search', 'ctc_greedy'])
add_arg('error_rate_type', str, 'cer', "Error rate type for evaluation.", choices=['wer', 'cer'])
add_arg('specgram_type', str, 'linear', "Audio feature type. Options: linear, mfcc.", choices=['linear', 'mfcc'])
args = parser.parse_args()


def infer(filenum):
    """Inference for DeepSpeech2."""
    data_generator = DataGenerator(
        vocab_filepath=args.vocab_path,
        mean_std_filepath=args.mean_std_path,
        augmentation_config='{}',
        specgram_type=args.specgram_type,
        num_threads=1,
        keep_transcription_text=True)
    batch_reader = data_generator.batch_reader_creator(
        manifest_path=args.infer_manifest,
        batch_size=args.batch_size,
        min_batch_size=1,
        sortagrad=False,
        shuffle_method=None)
    ds2_model = DeepSpeech2Model(
        vocab_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        use_gru=args.use_gru,
        pretrained_model_path=args.model_path,
        share_rnn_weights=args.share_rnn_weights)
    # decoders only accept string encoded in utf-8
    vocab_list = [chars.encode("utf-8") for chars in data_generator.vocab_list]
    if args.decoding_method == "ctc_beam_search":
        ds2_model.init_ext_scorer(args.alpha, args.beta, args.lang_model_path, vocab_list)
    ds2_model.logger.info("start inference ...")
    transcript = []
    widgets = ["Start inference ...: ", Percentage(), ' ', Bar(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=filenum/args.batch_size).start()
    for i, infer_data in enumerate(batch_reader()):
        if args.decoding_method == "ctc_greedy":
            probs_split = ds2_model.infer_batch_probs(infer_data=infer_data, feeding_dict=data_generator.feeding)
            result_transcripts = ds2_model.decode_batch_greedy(probs_split=probs_split, vocab_list=vocab_list)
        else:
            probs_split = ds2_model.infer_batch_probs(infer_data=infer_data, feeding_dict=data_generator.feeding)
            result_transcripts = ds2_model.decode_batch_beam_search(
                probs_split=probs_split,
                beam_alpha=args.alpha,
                beam_beta=args.beta,
                beam_size=args.beam_size,
                cutoff_prob=args.cutoff_prob,
                cutoff_top_n=args.cutoff_top_n,
                vocab_list=vocab_list,
                num_processes=args.num_proc_bsearch)
        transcript = transcript + result_transcripts
        pbar.update(i)
    pbar.finish()
    print("finish inference")
    return transcript


def infer_interface(audiolistfile, filenum):
    args.infer_manifest = audiolistfile
    print_arguments(args)
    paddle.init(use_gpu=args.use_gpu,
                rnn_use_batch=True,
                trainer_count=args.trainer_count)

    result = infer(filenum)

    return result



