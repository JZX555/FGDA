from torch import nn
from data.data_iterator import DataIterator
from data.dataset import CWSDataset
import torch
from torch.nn import functional as F
from transformers import BertTokenizer
import numpy as np
import time
import math
import sys
import json


def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


def kl_anneal_function(step, k=0.0025, x0=1000, min_lambda=0, start=-1):
    if step >= start:
        return max(float(1 / (1 + np.exp(-k * (step - x0)))), min_lambda)
    else:
        return 0


class Statistics(object):
    """
    Train/validate loss statistics.
    """
    def __init__(self, loss=0, vae_loss=0, n_words=0, n_correct=0, classifier_loss=0):
        self.loss = loss
        self.vae_loss = vae_loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.classifier_loss = classifier_loss

        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.vae_loss += stat.vae_loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.classifier_loss += stat.classifier_loss

    def ppl(self):
        return safe_exp(self.loss / self.n_words)

    def vae_ppl(self):
        return safe_exp(self.vae_loss / self.n_words)

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def elapsed_time(self):
        return time.time() - self.start_time

    def print_out(self, step, epoch, batch, n_batches, lr, batch_size, wf, pf):
        t = self.elapsed_time()

        out_info = ("Step %d, Epoch %d, %d/%d| lr: %.6f| words: %d| acc: %.2f| "
                    "ppl: %.2f, cla loss: %.2f, vae ppl: %.2f| %.1f tgt tok/s| %.2f s elapsed| best wf: %.2f, pf: %.2f") % \
                   (step, epoch, batch, n_batches, lr, int(batch_size), self.accuracy(), self.ppl(), self.classifier_loss / batch,
                    self.vae_ppl(), self.n_words / (t + 1e-5), time.time() - self.start_time, 100 * wf, 100 * pf)

        print(out_info)
        sys.stdout.flush()

    def print_valid(self, step):
        t = self.elapsed_time()
        out_info = ("Valid at step %d: acc %.2f, ppl: %.2f, %.1f tgt tok/s, %.2f s elapsed, label accuracy: %.2f") % \
                   (step, self.accuracy(), self.ppl(), self.n_words / (t + 1e-5),
                    time.time() - self.start_time, self.l_accuracy())
        print(out_info)
        sys.stdout.flush()


class DomainCWSHelper(object):
    def __init__(self, model, label_vocab, hp, use_cuda, shuffle=None):
        self.model = model

        self.tokenizer = BertTokenizer.from_pretrained(hp.bert_path)
        self.CLS = self.tokenizer.cls_token_id
        self.SEP = self.tokenizer.sep_token_id
        self.UNK = self.tokenizer.unk_token_id
        self.PAD = self.tokenizer.pad_token_id

        self.label_vocab = label_vocab
        self.pos_O = self.label_vocab.get_O_id()
        self.pos_PAD = self.label_vocab.pad()
        self.pos_UNK = self.label_vocab.unk()

        self.classifer_critic = nn.NLLLoss(reduction='none')
        self.vae_critic = nn.MSELoss(reduction='none')

        self.use_cuda = use_cuda

        if self.use_cuda:
            self.model = self.model.cuda()

        p = next(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.device = p.get_device() if self.use_cuda else None

        train_bitext_dataset = CWSDataset(data_paths=hp.train_data,
                                          max_len=hp.max_len,
                                          shuffle=True)

        train_batch_size = hp.batch_size * max(1, hp.update_cycle)
        train_buffer_size = hp.buffer_size * max(1, hp.update_cycle)

        self.training_iterator = DataIterator(dataset=train_bitext_dataset,
                                              batch_size=train_batch_size,
                                              use_bucket=hp.use_bucket,
                                              buffer_size=train_buffer_size,
                                              batching_func=hp.batching_key,
                                              shuffle=shuffle)

    def txt_data_id(self, src_input):
        result = self.tokenizer.convert_tokens_to_ids(src_input)
        return [self.CLS] + result + [self.SEP]

    def label_data_id(self, label_input):
        res = [self.label_vocab.token2id(label) for label in label_input]
        return [self.pos_O] + res + [self.pos_O]

    def prepare_eval_data(self, src_inputs):
        eval_data = []
        for src_input in src_inputs:
            eval_data.append((self.src_data_id(src_input), src_input))

        return eval_data

    def pair_data_variable(self, seqs_txt_t, seqs_label_t):
        batch_size = len(seqs_txt_t)

        txt_lengths = [len(seqs_txt_t[i]) for i in range(batch_size)]
        max_length = int(np.max(txt_lengths))

        txt_words = torch.zeros([batch_size, max_length],
                                dtype=torch.int64,
                                requires_grad=False)
        labels = torch.zeros([batch_size, max_length],
                             dtype=torch.int64,
                             requires_grad=False)
        crf_mask = torch.zeros([batch_size, max_length],
                           dtype=torch.int64,
                           requires_grad=False)
        bert_mask = torch.zeros([batch_size, max_length],
                           dtype=torch.int64,
                           requires_grad=False)

        txt_words = txt_words.fill_(self.PAD)
        labels = labels.fill_(self.pos_PAD)

        for b in range(batch_size):
            for index, word in enumerate(seqs_txt_t[b]):
                txt_words[b, index] = word

            for index, word in enumerate(seqs_label_t[b]):
                labels[b, index] = word

            for index, word in enumerate(seqs_txt_t[b]):
                bert_mask[b, index] = 1

            for index in range(1, len(seqs_txt_t[b]) - 1):
                if seqs_label_t[b][index] == self.pos_O:
                    crf_mask[b, index] = 0
                else:
                    crf_mask[b, index] = 1

        if self.use_cuda:
            txt_words = txt_words.cuda(self.device)
            labels = labels.cuda(self.device)
            crf_mask = crf_mask.cuda(self.device).bool()
            bert_mask = bert_mask.cuda(self.device).bool()

        return txt_words, labels, crf_mask, bert_mask

    def source_data_variable(self, seqs_x_t):
        batch_size = len(seqs_x_t)

        src_lengths = [len(seqs_x_t[i]) for i in range(batch_size)]
        max_src_length = int(np.max(src_lengths))

        src_words = torch.zeros([batch_size, max_src_length],
                                dtype=torch.int64,
                                requires_grad=False)
        src_words = src_words.fill_(self.src_pad)

        for b in range(batch_size):
            for index, word in enumerate(seqs_x_t[b]):
                src_words[b, index] = word

        if self.use_cuda:
            src_words = src_words.cuda(self.device)

        return src_words

    def compute_forward(self,
                        seqs,
                        labels,
                        pos,
                        crf_mask,
                        bert_mask,
                        norm,
                        global_step=None,
                        lambda_edit=1,
                        false_kl_lamda=0.00,
                        teach_forcing_radio=0.5,
                        finetune=False):
        batch_size, _ = seqs.size()
        domain_label = torch.tensor(pos, dtype=torch.long, requires_grad=False).cuda()

        self.model.train()
        self.classifer_critic.train()
        self.vae_critic.train()

        # For training
        with torch.enable_grad():
            general_res_dic, domain_res_dic, domain_logits = self.model(seqs, crf_mask, bert_mask, labels, finetune=finetune, pos=pos, inference=False)
            g_scores, g_tags, g_loss, g_hiddens = general_res_dic['scores'], general_res_dic[
                'predicted_tags'], general_res_dic['loss'], general_res_dic['adapter_out']
            d_scores, d_tags, d_loss, d_hiddens = domain_res_dic['scores'], domain_res_dic[
                'predicted_tags'], domain_res_dic['loss'], domain_res_dic['adapter_out']

            classifier_loss = self.classifer_critic(domain_logits, domain_label).sum()
            vae_loss = self.vae_critic(g_hiddens, d_hiddens).sum() + self.vae_critic(d_hiddens, g_hiddens).sum()

            if global_step is not None:
                classifier_loss = classifier_loss * (kl_anneal_function(global_step, x0=2000, start=1000))
                # g_loss = g_loss * (kl_anneal_function(global_step, start=500)**2)
                vae_loss = vae_loss * (kl_anneal_function(global_step, x0=5000, start=0))
            # print(d_loss, g_loss)
            loss = d_loss.sum() + g_loss.sum() + vae_loss + classifier_loss
            # loss = d_loss.sum() + vae_loss.sum()

        # torch.autograd.backward(y_loss)
        torch.autograd.backward(loss)
        # optim.step()

        pred = torch.ones_like(labels).long().fill_(self.pos_PAD)
        for b in range(batch_size):
            for index, word in enumerate(g_tags[b]):
                pred[b, index + 1] = word

        num_correct = labels.detach().eq(pred).float().masked_select(
            crf_mask).sum()
        num_total = crf_mask.sum().float()

        with torch.no_grad():
            lossvalue = (d_loss.sum() + g_loss.sum()).item()
            vaevalue = vae_loss.item()
            classifiervalue = classifier_loss.item()

        stats = Statistics(lossvalue, vaevalue, num_total, num_correct, classifiervalue)

        return stats

    def train_batch(self,
                    seqs_txt_t,
                    seqs_label_t,
                    pos_t,
                    norm,
                    global_step=None,
                    lambda_edit=1,
                    finetune=False):
        self.model.train()
        seqs_txt = [self.txt_data_id(txt_t) for txt_t in seqs_txt_t]
        seqs_label = [self.label_data_id(label_t) for label_t in seqs_label_t]

        txt_words, labels, crf_mask, bert_mask = self.pair_data_variable(seqs_txt, seqs_label)

        stat = self.compute_forward(txt_words,
                                    labels,
                                    pos_t,
                                    crf_mask,
                                    bert_mask,
                                    norm,
                                    global_step,
                                    lambda_edit=1,
                                    finetune=finetune)

        return stat

    def eval_batch(self, seqs_x_t, level_label, norm):
        self.model.train()
        seqs_x = [self.src_data_id(x_t) for x_t in seqs_x_t]

        src_words, level_label = self.pair_data_variable(seqs_x, level_label)

        batch_size, _ = src_words.size()
        x_input = src_words[:, :-1].contiguous()
        x_label = src_words[:, 1:].contiguous()

        self.model.eval()
        self.critic.eval()

        # For training
        with torch.enable_grad():
            log_probs, level = self.model(x_input)
            log_loss = self.critic(inputs=log_probs,
                                   labels=x_label,
                                   reduce=False,
                                   normalization=norm)
            level_loss = self.classifer_critic(level, level_label)

            loss = log_loss.sum() + level_loss.sum()

        mask = x_label.detach().ne(self.src_pad)
        pred = log_probs.detach().max(2)[1]  # [batch_size, seq_len]
        level_pred = level.detach().max(1)[1]  # [batch_size, seq_len]

        num_correct = x_label.detach().eq(pred).float().masked_select(
            mask).sum()
        num_total = mask.sum().float()

        label_correct = level_label.detach().eq(level_pred).float().sum()
        label_total = batch_size

        lossvalue = norm * loss.item()
        stats = Statistics(lossvalue, num_total, num_correct, label_total,
                           label_correct)

        trans = []
        for i, line in enumerate(pred.cpu().numpy().tolist()):
            sent_t = [wid for wid in line if wid != self.src_pad]
            x_tokens = []

            for wid in sent_t:
                if wid == self.src_eos:
                    break
                x_tokens.append(self.src_vocab.id2token(wid))

            if len(x_tokens) > 0:
                trans.append(
                    str(level_pred[i]) +
                    self.src_vocab.tokenizer.detokenize(x_tokens))
            else:
                trans.append(
                    str(level_pred[i]) +
                    '%s' % self.src_vocab.id2token(self.src_eos))

        return trans, stats

    def translate_batch(self, seqs_txt_t, seqs_label_t, pos_t, is_test=False, domain_idx: int = None):
        seqs_txt = [self.txt_data_id(txt_t) for txt_t in seqs_txt_t]
        seqs_label = [self.label_data_id(label_t) for label_t in seqs_label_t]
        if is_test is False:
            pos_t = [-1 for i in pos_t]
        if domain_idx is not None:
            pos_t = [domain_idx for i in pos_t]

        seqs, labels, crf_mask, bert_mask = self.pair_data_variable(seqs_txt, seqs_label)

        with torch.no_grad():
            res_dic = self.model(seqs, crf_mask, bert_mask, labels, pos=pos_t, inference=True)
            scores, tags, loss = res_dic['scores'], res_dic[
                'predicted_tags'], res_dic['loss']
            scores = torch.softmax(scores, dim=-1)
            loss = loss.sum()

        trans = []
        all_preds = []
        all_labels = []
        # Append result

        for i in range(len(tags)):
            tmp = [self.label_vocab.id2token(tag) for tag in tags[i]]
            all_preds += tmp
            all_labels += seqs_label_t[i]

            start = 0
            sent = []
            tag_label = ''
            tag_score = ''
            for j, tag in enumerate(tmp):
                if tag[0] == 'E' or tag[0] == 'S':
                    tag_label += tag[2:]
                    tag_score += ('%.2f' % scores[i, j + 1, tags[i][j]].item())
                    sent.append(''.join(seqs_txt_t[i][start:j + 1]) + '_' +
                                tag_label + '_' + tag_score)
                    start = j + 1
                    tag_label = ''
                    tag_score = ''
                else:
                    tag_label += tag[2:] + ','
                    tag_score += ('%.2f' % scores[i, j + 1, tags[i][j]].item()) + ','

            if start < len(tmp):
                sent.append(''.join(seqs_txt_t[i][start:]) + '_' +
                            tag_label[:-1] + '_' + tag_score[:-1])

            trans += [' '.join(sent)]

        return trans, all_preds, all_labels

    def translate_with_oov(self, seqs_txt_t, seqs_label_t, pos_t, oov_path: list = None, trainsets_len=1):
        seqs_txt = [self.txt_data_id(txt_t) for txt_t in seqs_txt_t]
        seqs_label = [self.label_data_id(label_t) for label_t in seqs_label_t]
        oov_dicts = None

        pos_t = [-1 for i in pos_t]

        if oov_path is not None:
            oov_dicts = []
            for path in oov_path:
                with open(path, 'r', encoding='utf-8') as f:
                    if len(oov_dicts) == 0:
                        oov_dicts.append(json.load(f))
                    else:
                        tmp_dict = oov_dicts[-1].copy().update(json.load(f))
                        if path == './debug/domain2.json':
                            print(tmp_dict)
                        oov_dicts.append(tmp_dict)

        trans = []
        all_preds = []
        all_labels = []
        STD_OOV = 0
        is_clear = False
        seqs, labels, crf_mask, bert_mask = self.pair_data_variable(seqs_txt, seqs_label)

        while is_clear is False:
            with torch.no_grad():
                res_dic = self.model(seqs, crf_mask, bert_mask, labels, pos=pos_t, inference=True)
                scores, tags, loss = res_dic['scores'], res_dic[
                    'predicted_tags'], res_dic['loss']
                scores = torch.softmax(scores, dim=-1)
                loss = loss.sum()

            no_std_idxs = []

            # Append result
            for i in range(len(tags)):
                oov_num = 0
                tmp = [self.label_vocab.id2token(tag) for tag in tags[i]]

                start = 0
                sent = []
                is_confidence = True
                last_tag = None
                tag_label = ''
                tag_score = ''
                for j, tag in enumerate(tmp):
                    if tag[0] == 'E' or tag[0] == 'S':
                        tag_label += tag[2:]
                        tag_score += ('%.2f' % scores[i, j + 1, tags[i][j]].item())
                        sent.append(''.join(seqs_txt_t[i][start:j + 1]) + '_' +
                                    tag_label + '_' + tag_score)

                        if scores[i, j + 1, tags[i][j]].item() < 0.2:
                            is_confidence = False
                        if last_tag is not None and last_tag != tag[2:]:
                            is_confidence = False
                        if ((''.join(seqs_txt_t[i][start:j + 1]) + '_' + tag[2:]) not in oov_dicts[STD_OOV]) or (is_confidence is False):
                            oov_num += 1

                        start = j + 1
                        is_confidence = True
                        last_tag = None
                        tag_label = ''
                        tag_score = ''
                    else:
                        tag_label += tag[2:] + ','
                        tag_score += ('%.2f' % scores[i, j + 1, tags[i][j]].item()) + ','
                        if scores[i, j + 1, tags[i][j]].item() < 0.2:
                            is_confidence = False
                        if last_tag is not None and last_tag != tag[2:]:
                            is_confidence = False
                        last_tag = tag[2:]

                if start < len(tmp):
                    sent.append(''.join(seqs_txt_t[i][start:]) + '_' +
                                tag_label[:-1] + '_' + tag_score[:-1])
                    oov_num += 1

                if oov_num <= STD_OOV or STD_OOV == (trainsets_len - 1):
                    trans += [' '.join(sent)]
                    all_preds += tmp
                    all_labels += seqs_label_t[i]
                else:
                    no_std_idxs.append(i)

            if len(no_std_idxs) == 0 or STD_OOV >= (trainsets_len - 1):
                is_clear = True
            else:
                STD_OOV += 1
                seqs_txt_t = [seqs_txt_t[idx] for idx in no_std_idxs]
                seqs_label_t = [seqs_label_t[idx] for idx in no_std_idxs]

                no_std_idxs = torch.tensor(no_std_idxs, device=self.device)
                seqs = seqs.index_select(dim=0, index=no_std_idxs)
                crf_mask = crf_mask.index_select(dim=0, index=no_std_idxs)
                bert_mask = bert_mask.index_select(dim=0, index=no_std_idxs)
                labels = labels.index_select(dim=0, index=no_std_idxs)
                pos_t = [trainsets_len - STD_OOV - 1 for _ in range(len(no_std_idxs))]

                del no_std_idxs

        return trans, all_preds, all_labels
