# -*- coding: utf-8 -*-
import torch
import numpy as np
from data.vocabulary import Vocabulary, LabelVocabulary
from data.dataset import CWSDataset, shuffle, cws_shuffle
from data.data_iterator import DataIterator
from utils.hyper_param import HyperParam
from utils.common_utils import *
from metric.poscws_metric import pos_evaluate_word_PRF
from driver.model_helper import CWSHelper
from driver.domain_helper import DomainCWSHelper
from model.base import BaseModel
from model.adapter import CWSPOSModel
from model.share_adapter import ShareModel
from model.domain_adapter import DomainModel
from optim import Optimizer
from optim.lr_scheduler import ReduceOnPlateauScheduler, NoamScheduler

import subprocess
import argparse
import random
import ntpath
import time
import os
import re


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True


def train(restore=False,
          mode='transformer',
          gpu_use=-1,
          name='base',
          model_name='base'):
    hp = HyperParam(mode=mode)
    hp._print_items()

    gpu = torch.cuda.is_available()
    print('begin with mode {}'.format(mode))
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    global_step = 0

    use_cuda = False
    if gpu and gpu_use >= 0:
        use_cuda = True
        torch.cuda.set_device(gpu_use)
        print("GPU ID: ", gpu_use)

    set_seed(1234)

    label_vocab = LabelVocabulary(hp.vocabulary_type, hp.label_vocab)

    if model_name == 'base':
        model = BaseModel(hp.bert_path, label_vocab, d_model=hp.bert_size)
    elif model_name == 'adapter':
        model = CWSPOSModel(hp.bert_path, label_vocab, d_model=hp.bert_size, trainsets_len=hp.trainsets_len)
    elif model_name == 'share':
        model = ShareModel(hp.bert_path, label_vocab, d_model=hp.bert_size, trainsets_len=hp.trainsets_len)
    elif model_name == 'domain':
        model = DomainModel(hp.bert_path, label_vocab, d_model=hp.bert_size, trainsets_len=hp.trainsets_len)
    else:
        raise NameError(f'no model named {model_name}')

    model.init_model(param_path=hp.pos_path, device=gpu_use)

    if model_name == 'domain':
        cws = DomainCWSHelper(model, label_vocab, hp, use_cuda=use_cuda, shuffle=cws_shuffle if hp.shuffle is True else None)
    else:
        cws = CWSHelper(model, label_vocab, hp, use_cuda=use_cuda, shuffle=shuffle if hp.shuffle is True else None)

    print('begin training:')

    if not os.path.exists('./save/' + name):
        os.mkdir('./save/' + name)

    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(
        os.path.join('./save/' + name, name)),
                             num_max_keeping=10)

    if restore is True:
        checkpoint_saver.load_latest(
            device=gpu_use,
            model=cws.model
        )
        print('restore successful')
        model._save_pos_params(f'./save/pos_ckpt/{name}.ckpt')
        # exit()

    (wP, wR, wF), (pP, pR, pF) = evaluate(cws, hp, global_step, name,
                                          hp.sport, is_test=True)

    print('cws f1:(%.2f, %.2f, %.2f), pos f1:(%.2f, %.2f, %.2f)' %
          (100 * wP, 100 * wR, 100 * wF, 100 * pP, 100 * pR, 100 * pF))


def evaluate(cws: CWSHelper, hp: HyperParam, global_step, name, path, is_test=True):
    batch_size = 32
    dev_dataset = CWSDataset(data_paths=[path])

    dev_iterator = DataIterator(dataset=dev_dataset,
                                batch_size=32,
                                use_bucket=True,
                                buffer_size=100,
                                batching_func=hp.batching_key,
                                numbering=True)

    cws.model.eval()

    numbers = []
    trans = []
    all_preds = []
    all_labels = []

    dev_iter = dev_iterator.build_generator(batch_size=batch_size)
    for i, batch in enumerate(dev_iter):
        seq_nums, seqs, labels, pos = batch
        numbers += seq_nums

        sub_trans, sub_preds, sub_labels = cws.translate_batch(
            seqs, labels, pos, is_test=is_test)

        trans += sub_trans
        all_preds += sub_preds
        all_labels += sub_labels

    origin_order = np.argsort(numbers).tolist()
    trans = [trans[ii] for ii in origin_order]

    head, tail = ntpath.split(path)
    hyp_path = os.path.join('./save/evaluation/' + name + '-' + tail + "." +
                            str(global_step))

    with open(hyp_path, 'w', encoding='utf-8') as f:
        for line in trans:
            f.write('%s\n' % re.sub('@@ ', '', line))

    (wP, wR, wF), (pP, pR, pF) = pos_evaluate_word_PRF(all_preds, all_labels)

    return (wP, wR, wF), (pP, pR, pF)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore',
                        default='False',
                        action='store_true',
                        help="to restore the last ckpt.")
    parser.add_argument('--mode',
                        default='base',
                        type=str,
                        help='the flow of multi_seq2seq')
    parser.add_argument('--GPU',
                        '-g',
                        default=0,
                        type=int,
                        help='choose the gpu to use')
    parser.add_argument('--name',
                        '-n',
                        default='defalut',
                        type=str,
                        help='the name of model')
    parser.add_argument('--model',
                        '-m',
                        default='wrong-transfer',
                        type=str,
                        help='choose the model to use')

    args = parser.parse_args()

    train(restore=args.restore,
          mode=args.mode,
          gpu_use=args.GPU,
          name=args.name,
          model_name=args.model)
