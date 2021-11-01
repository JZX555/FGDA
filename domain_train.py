# -*- coding: utf-8 -*-
import torch
import numpy as np
from data.vocabulary import Vocabulary, LabelVocabulary
from data.dataset import CWSDataset, shuffle, cws_shuffle
from data.data_iterator import DataIterator
from utils.hyper_param import HyperParam
from utils.common_utils import *
from metric.poscws_metric import pos_evaluate_word_PRF
from driver.domain_helper import DomainCWSHelper, Statistics
from model.base import BaseModel
from model.adapter import CWSPOSModel
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
        model = DomainModel(hp.bert_path, label_vocab, d_model=hp.bert_size, trainsets_len=hp.trainsets_len)
    else:
        raise NameError(f'no model named {model_name}')
    # model.init_model(param_path=hp.pos_path, device=gpu_use)

    if hp.shuffle is True:
        cws = DomainCWSHelper(model,
                        label_vocab,
                        hp,
                        use_cuda=use_cuda,
                        shuffle=cws_shuffle)
    else:
        cws = DomainCWSHelper(model,
                        label_vocab,
                        hp,
                        use_cuda=use_cuda,
                        shuffle=None)

    optim = Optimizer(name=hp.optim,
                      model=model,
                      lr=hp.lr,
                      grad_clip=-1.0,
                      optim_args=None)

    if hp.schedule_method == 'noam':
        scheduler = NoamScheduler(optimizer=optim,
                                  d_model=512,
                                  warmup_steps=hp.warmup_steps)
    else:
        scheduler = None

    print('begin training:')

    if not os.path.exists('./save/' + name):
        os.mkdir('./save/' + name)

    best_pF = -1
    best_wF = -1
    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(
        os.path.join('./save/' + name, name)),
                             num_max_keeping=20)

    if restore is True:
        checkpoint_saver.load_latest(device=gpu_use, model=model)
        print('restore successful')

    if hp.general_path is not None:
        model._load_param(hp.general_path, gpu_use)

    for epoch in range(100):
        total_stats = Statistics()
        training_iter = cws.training_iterator.build_generator()
        batch_iter, total_iters = 0, len(cws.training_iterator)

        for batch in training_iter:
            global_step += 1
            if hp.schedule_method is not None \
                    and hp.schedule_method != "loss":
                scheduler.step(global_step=global_step)

            seqs, label, pos = batch
            n_samples_t = len(seqs)
            batch_iter += n_samples_t
            n_words_t = sum(len(s) for s in seqs)

            lrate = list(optim.get_lrate())[0]
            optim.zero_grad()

            try:
                for seqs_txt_t, seqs_label_t, pos_t in split_shard(
                        seqs, label, pos, split_size=hp.update_cycle):
                    stat = cws.train_batch(seqs_txt_t,
                                           seqs_label_t,
                                           pos_t,
                                           n_samples_t,
                                           global_step=global_step,
                                           finetune=hp.finetune)
                    total_stats.update(stat)

                total_stats.print_out(global_step - 1, epoch, batch_iter,
                                      total_iters, lrate, n_words_t, best_wF, best_pF)
                optim.step()
            except RuntimeError as e:
                print('seqs_txt_t is:{}\nshape is: {}'.format(
                    seqs_txt_t, np.shape(seqs_txt_t)))
                print('seqs_label_t is:{}\nshape is: {}'.format(
                    seqs_label_t, np.shape(seqs_label_t)))
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    optim.zero_grad()
                elif 'cuda runtime error' in str(e):
                    print('| WARNING: unknow cuda error:{}, skipping batch'.
                          format(str(e)))
                    optim.zero_grad()
                elif 'CUDA error' in str(e):
                    print('| WARNING: unknow cuda error:{}, skipping batch'.
                          format(str(e)))
                    optim.zero_grad()
                else:
                    raise e

            if global_step % hp.valid_freq == 0:
                dev_start_time = time.time()
                (wP, wR, wF), (pP, pR, pF) = evaluate(cws, hp, global_step, name)
                during_time = float(time.time() - dev_start_time)
                print("step %d, epoch %d: ctb6 dev cws p: %.4f, r: %.4f, f1: %.4f| pos p: %.4f, r: %.4f, f1: %.4f, time %.2f"
                      % (global_step, epoch, wP, wR, wF, pP, pR, pF, during_time))

                if pF > best_pF - 0.0005:
                    print("exceed best ctb6 f1: history = %.2f, current = %.2f, lr_ratio = %.6f"
                          % (best_pF, pF, lrate))
                    best_pF = pF
                    best_wF = wF
                    checkpoint_saver.save(
                        global_step=global_step,
                        model=cws.model,
                        optim=optim,
                        lr_scheduler=scheduler)


def evaluate(cws: DomainCWSHelper, hp: HyperParam, global_step, name, is_test=False):
    batch_size = hp.batch_size

    dev_dataset = CWSDataset(data_paths=[hp.dev_data])

    dev_iterator = DataIterator(dataset=dev_dataset,
                                batch_size=10,
                                use_bucket=True,
                                buffer_size=100,
                                numbering=True)

    cws.model.eval()

    numbers = []
    trans = []
    all_preds = []
    all_labels = []

    dev_iter = dev_iterator.build_generator(batch_size=batch_size)
    for batch in dev_iter:
        seq_nums, seqs, labels, pos = batch
        numbers += seq_nums

        sub_trans, sub_preds, sub_labels = cws.translate_batch(seqs, labels, pos, is_test=is_test)
        trans += sub_trans
        all_preds += sub_preds
        all_labels += sub_labels

    origin_order = np.argsort(numbers).tolist()
    trans = [trans[ii] for ii in origin_order]

    if is_test is False:
        head, tail = ntpath.split(hp.dev_data)
        hyp_path = os.path.join('./save/' + name + '/' +
                                tail + "." + str(global_step))
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
                        default='transformer',
                        type=str,
                        help='hyperparams mode')
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
                        default='base',
                        type=str,
                        help='choose the model to use')

    args = parser.parse_args()

    train(restore=args.restore,
          mode=args.mode,
          gpu_use=args.GPU,
          name=args.name,
          model_name=args.model)
