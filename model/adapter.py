from .bertCRF import CRF
import math
from typing import Dict, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import embedding_bag, linear
from modules.rnn import RNN

from transformers import BertModel


def set_requires_grad(module: nn.Module, status: bool = False):
    for param in module.parameters():
        param.requires_grad = status


class Adapter(nn.Module):
    def __init__(self, in_features, bottleneck_size, external_param=False, trainsets_len: int = -1):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_size = bottleneck_size
        self.act_fn = nn.GELU()

        self.pos = None
        self.linear1 = None
        self.linear2 = None
        self.bias1 = None
        self.bias2 = None

        if external_param:
            self.params = [None, None, None, None]
        else:
            if trainsets_len <= 1:
                self.params = nn.ParameterList([
                        nn.Parameter(torch.Tensor(in_features, bottleneck_size)),
                        nn.Parameter(torch.Tensor(bottleneck_size)),
                        nn.Parameter(torch.Tensor(bottleneck_size, in_features)),
                        nn.Parameter(torch.Tensor(in_features))
                    ])
            else:
                self.params = nn.ParameterList()
                for i in range(trainsets_len - 1):
                    self.params.extend(nn.ParameterList([
                        nn.Parameter(torch.Tensor(in_features, bottleneck_size)),
                        nn.Parameter(torch.Tensor(bottleneck_size)),
                        nn.Parameter(torch.Tensor(bottleneck_size, in_features)),
                        nn.Parameter(torch.Tensor(in_features))
                    ]))

            self.reset_parameters()

    def add_new_parameters(self):
        print(f'origin pos params size: {len(self.params)}')

        new_params = nn.ParameterList([nn.Parameter(self.params[i].data.clone()) for i in range(4)])
        for param in self.params:
            param.requires_grad = False

        self.params = new_params.extend(self.params)
        print(f'successful add a pos param at the first of the paramsList, cur size: {len(self.params)}')

    def init_linear(self, w, b):
        init.kaiming_uniform_(w, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(w.transpose(0, 1))
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(b, -bound, bound)

    def reset_parameters(self):
        for i in range(len(self.params) // 4):
            self.init_linear(self.params[i * 4], self.params[i * 4 + 1])
            self.init_linear(self.params[i * 4 + 2], self.params[i * 4 + 3])

    def set_pos(self, pos: List[int]):
        l1 = []
        b1 = []
        l2 = []
        b2 = []
        for p in pos:
            if len(self.params) == 4:
                p = 0
            l1 += [self.params[p * 4].unsqueeze(0)]
            b1 += [self.params[p * 4 + 1].unsqueeze(0)]
            l2 += [self.params[p * 4 + 2].unsqueeze(0)]
            b2 += [self.params[p * 4 + 3].unsqueeze(0)]
        self.linear1 = torch.cat(l1, dim=0)
        self.linear2 = torch.cat(l2, dim=0)
        self.bias1 = torch.cat(b1, dim=0).unsqueeze(-1)
        self.bias2 = torch.cat(b2, dim=0).unsqueeze(-1)

        assert len(self.bias1.size()) == 3, ValueError(f'the len of bias need to be 3 but get {self.bias1.size()}')

    def forward(self, hidden_states: torch.Tensor):
        x = (torch.matmul(hidden_states, self.linear1).transpose(1, 2) + self.bias1).transpose(1, 2)
        x = self.act_fn(x)
        x = (torch.matmul(x, self.linear2).transpose(1, 2) + self.bias2).transpose(1, 2)
        x = x + hidden_states

        return x


class AdapterBertOutput(nn.Module):
    """
    Replace BertOutput and BertSelfOutput
    """
    def __init__(self, base, adapter_forward):
        super().__init__()
        self.base = base
        self.adapter_forward = adapter_forward

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.base.dense(hidden_states)
        hidden_states = self.base.dropout(hidden_states)
        hidden_states = self.adapter_forward(hidden_states)
        hidden_states = self.base.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AdapterBertModel(nn.Module):
    def __init__(self,
                 name_or_path_or_model: Union[str, BertModel],
                 adapter_size: int = 192,
                 external_param: Union[bool, List[bool]] = False,
                 word_piece: str = 'first',
                 trainsets_len: int = -1,
                 adapter_layer: int = 9,
                 begin_layer: int = 12,
                 combine: bool = False,
                 **kwargs):
        super().__init__()
        self.last_pos = None
        self.adapter_layer = adapter_layer
        self.begin_layer = begin_layer
        self.trainsets_len = trainsets_len
        self.combine = combine
        print(f'adapter_layer begin at: {adapter_layer}')

        if isinstance(name_or_path_or_model, str):
            self.bert = BertModel.from_pretrained(name_or_path_or_model)
        else:
            self.bert = name_or_path_or_model

        set_requires_grad(self.bert, False)

        if isinstance(external_param, bool):
            param_place = [external_param for _ in range(
                self.bert.config.num_hidden_layers)]
        elif isinstance(external_param, list):
            param_place = [False for _ in range(
                self.bert.config.num_hidden_layers)]
            for i, e in enumerate(external_param, 1):
                param_place[-i] = e

        self.adapters = nn.ModuleList()
        self.adapter_lambda = None
        if self.combine is True:
            init_value = 1 / (self.bert.config.num_hidden_layers - self.adapter_layer + 1)
            self.adapter_lambda = nn.ParameterList([nn.Parameter(torch.FloatTensor(1).fill_(init_value)) for i in range(self.bert.config.num_hidden_layers)])
            print(f'init adapter lambda to {init_value}')

        for i, e in enumerate(param_place):
            if i < self.adapter_layer - 1:
                self.adapters.extend([None])
            elif i < self.begin_layer - 1:
                self.adapters.extend([nn.ModuleList([
                    Adapter(self.bert.config.hidden_size, adapter_size, e, -1),
                    Adapter(self.bert.config.hidden_size, adapter_size, e, -1)
                    ])
                ])
            else:
                self.adapters.extend([nn.ModuleList([
                    Adapter(self.bert.config.hidden_size, adapter_size, e, trainsets_len),
                    Adapter(self.bert.config.hidden_size, adapter_size, e, trainsets_len)
                    ])
                ])

        for i, layer in enumerate(self.bert.encoder.layer):
            if i < self.adapter_layer - 1:
                continue
            layer.output = AdapterBertOutput(
                layer.output, self.adapters[i][0].forward)
            set_requires_grad(layer.output.base.LayerNorm, True)
            layer.attention.output = AdapterBertOutput(
                layer.attention.output, self.adapters[i][1].forward)
            set_requires_grad(layer.attention.output.base.LayerNorm, True)

        self.output_dim = self.bert.config.hidden_size

        if word_piece == 'first':
            self.word_piece = None
        else:  # mean of pieces
            offset = torch.tensor([0], dtype=torch.long)

            self.word_piece = lambda x: embedding_bag(
                x, self.bert.embeddings.word_embeddings.weight, offset.to(x.device))

        if combine is True:
            self.bert.encoder.output_hidden_states = True

    def set_pos(self, pos: List[int]):
        for i, adapter in enumerate(self.adapters):
            if i < self.adapter_layer - 1:
                continue
            adapter[0].set_pos(pos)
            adapter[1].set_pos(pos)

    def forward(self,  
                input_ids: torch.Tensor,
                word_pieces: Dict[Tuple[int], torch.LongTensor] = None,
                mask: torch.Tensor = None,
                pos: List[int] = None,
                **kwargs) -> torch.Tensor:
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        if self.word_piece is not None and word_pieces is not None:
            for (s, w), pieces in word_pieces.items():
                inputs_embeds[s, w, :] = self.word_piece(pieces)

        self.set_pos(pos)

        attention_mask = None if mask is None else mask.float()
        if self.combine is False:
            bert_output = self.bert(attention_mask=attention_mask, inputs_embeds=inputs_embeds)
            return bert_output[0]
        else:
            bert_output, _, hidden_states = self.bert(attention_mask=attention_mask, inputs_embeds=inputs_embeds)
            bert_output = torch.zeros_like(bert_output)
            for i in range(self.adapter_layer - 1, self.bert.config.num_hidden_layers):
                bert_output += hidden_states[i + 1].contiguous() * self.adapter_lambda[i]
            return bert_output

    def init_model(self, param_path=None, device=None):
        if self.trainsets_len > 1:
            if param_path is not None:
                self._load_param(param_path, device)
            self._add_cur_pos_linear()
        else:
            print(f'trainsets len is: {self.trainsets_len}, do not need to init')

    def _save_pos_params(self, path):
        state_dict = {}
        for i, adapter in enumerate(self.adapters):
            if i < self.begin_layer - 1:
                continue
            state_dict[f'adapter_{i}_0'] = adapter[0].state_dict()
            state_dict[f'adapter_{i}_1'] = adapter[1].state_dict()
        torch.save(state_dict, path)
        print(f'save pos params successful, save at: {path}')

    def _load_param(self, path, device):
        state_dict = torch.load(path, map_location='cuda:' + str(device))
        for i, adapter in enumerate(self.adapters):
            if i < self.begin_layer - 1:
                continue
            adapter[0].load_state_dict(state_dict[f'adapter_{i}_0'])
            adapter[1].load_state_dict(state_dict[f'adapter_{i}_1'])
        print(f'load pos params from {path} into device cuda: {device}')

    def _add_cur_pos_linear(self):
        for i, adapter in enumerate(self.adapters):
            if i < self.begin_layer - 1:
                continue
            adapter[0].add_new_parameters()
            adapter[1].add_new_parameters()


class CWSPOSModel(nn.Module):
    def __init__(self, bert_path, label_vocab, d_model: int = 768, adapter_size: int = 192, trainsets_len=None, combine: bool = False):
        super(CWSPOSModel, self).__init__()

        self.num_layers = 1
        self.d_model = d_model
        self.hidden_size = 400

        self.label_vocab = label_vocab
        self.num_tags = label_vocab.max_n_words
        self.lstm_dropout = nn.Dropout(p=0.2)

        self.bert = AdapterBertModel(name_or_path_or_model=bert_path, adapter_size=adapter_size, trainsets_len=trainsets_len, combine=combine)

        self.linear_lstm = None
        self.lstm = nn.ModuleList([
            RNN(type="LSTM", batch_first=True, input_size=d_model if layer == 0 else 2 * self.hidden_size,
                hidden_size=self.hidden_size, bidirectional=True, dropout=0.2)
            for layer in range(self.num_layers)
        ])
        self.CRF = CRF(num_tags=self.num_tags, input_dim=2 * self.hidden_size)

        print(f'lstm layers size is: {len(self.lstm)}')

    def init_paramater(self):
        nn.init.kaiming_uniform_(self.linear_lstm.weight)

    def init_model(self, param_path=None, device=None):
        self.bert.init_model(param_path, device)

    def _save_pos_params(self, path):
        self.bert._save_pos_params(path)

    def _load_param(self, path, device):
        self.bert._load_param(path, device)

    def forward(self, seqs, mask, labels, finetune=False, pos=None):
        bert_outs = self.bert(seqs, mask=mask, pos=pos)

        ctx = self.linear_lstm(bert_outs) if self.linear_lstm is not None else bert_outs
        for i, rnn in enumerate(self.lstm):
            ctx, _ = rnn(ctx, ~mask)

        ctx = self.lstm_dropout(ctx)
        res_dic = self.CRF(ctx, mask, labels)

        return res_dic
