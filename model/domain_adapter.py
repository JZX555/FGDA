from .bertCRF import CRF
import math
from typing import Dict, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import embedding_bag, linear
from modules.rnn import RNN
from modules.common_modules import GRLClassifier, DomainVariational

from transformers import BertModel


def set_requires_grad(module: nn.Module, status: bool = False):
    for param in module.parameters():
        param.requires_grad = status


class Adapter(nn.Module):
    def __init__(self, in_features, bottleneck_size, external_param=False, domain_size: int = -1, domain_embedding: nn.Parameter = None):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_size = bottleneck_size
        self.domain_embedding_dim = 8
        self.domain_size = max(domain_size, 1)
        # self.domain = 'general'
        self.act_fn = nn.GELU()

        self.pos = None
        self.linear1 = None
        self.linear2 = None
        self.bias1 = None
        self.bias2 = None

        # self.domain_embedding = nn.Embedding(num_embeddings=self.domain_size, embedding_dim=self.domain_embedding_dim)
        if domain_embedding is None:
            self.domain_embedding = nn.Parameter(torch.Tensor(self.domain_size, self.domain_embedding_dim))
        else:
            self.domain_embedding = domain_embedding

        self.general_params = nn.ParameterList([
                        nn.Parameter(torch.Tensor(bottleneck_size, in_features)),
                        nn.Parameter(torch.Tensor(bottleneck_size)),
                        nn.Parameter(torch.Tensor(in_features, bottleneck_size)),
                        nn.Parameter(torch.Tensor(in_features))
                    ])

        if external_param:
            self.domain_params = [None, None, None, None]
        else:
            self.domain_params = nn.ParameterList([
                    nn.Parameter(torch.Tensor(in_features, bottleneck_size, self.domain_embedding_dim)),
                    nn.Parameter(torch.Tensor(bottleneck_size, self.domain_embedding_dim)),
                    nn.Parameter(torch.Tensor(bottleneck_size, in_features, self.domain_embedding_dim)),
                    nn.Parameter(torch.Tensor(in_features, self.domain_embedding_dim))
                ])

            self.reset_parameters()

    def init_linear(self, w, b=None, is_domain=True):
        init.kaiming_uniform_(w, a=math.sqrt(5))
        # init.uniform_(w)
        if is_domain is True:
            _, fan_in = init._calculate_fan_in_and_fan_out(w)
        else:
            fan_in, _ = init._calculate_fan_in_and_fan_out(w)
        if b is not None:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(b, -bound, bound)

    def load_general_params(self, weights):
        for i in range(len(self.general_params)):
            self.general_params[i].data = weights[f'general_params.{i}'].data
            self.domain_params[i].data = weights[f'domain_params.{i}'].data

    def reset_parameters(self):
        self.init_linear(self.general_params[0], self.general_params[1])
        self.init_linear(self.general_params[2], self.general_params[3])
        self.init_linear(self.domain_embedding)

        self.init_linear(self.domain_params[0], self.domain_params[1], is_domain=True)
        self.init_linear(self.domain_params[2], self.domain_params[3], is_domain=True)

    def set_domain(self, domain: str = 'general'):
        self.domain = domain

    def set_pos(self, pos: List[int]):
        l1 = []
        b1 = []
        l2 = []
        b2 = []
        for p in pos:
            l1 += [linear(self.domain_params[0].unsqueeze(0), self.domain_embedding[p].contiguous().unsqueeze(0)).squeeze(-1)]
            b1 += [linear(self.domain_params[1].unsqueeze(0), self.domain_embedding[p].contiguous().unsqueeze(0)).squeeze(-1)]
            l2 += [linear(self.domain_params[2].unsqueeze(0), self.domain_embedding[p].contiguous().unsqueeze(0)).squeeze(-1)]
            b2 += [linear(self.domain_params[3].unsqueeze(0), self.domain_embedding[p].contiguous().unsqueeze(0)).squeeze(-1)]
        self.linear1 = torch.cat(l1, dim=0)
        self.linear2 = torch.cat(l2, dim=0)
        self.bias1 = torch.cat(b1, dim=0).unsqueeze(-1)
        self.bias2 = torch.cat(b2, dim=0).unsqueeze(-1)

        assert len(self.bias1.size()) == 3, ValueError(f'the len of bias need to be 3 but get {self.bias1.size()}')

    def forward(self, hidden_states: torch.Tensor):
        if self.domain == 'general':
            x = linear(hidden_states, self.general_params[0], self.general_params[1])
            x = self.act_fn(x)
            x = linear(x, self.general_params[2], self.general_params[3])
        elif self.domain == 'domain':
            x = (torch.matmul(hidden_states, self.linear1).transpose(1, 2) + self.bias1).transpose(1, 2)
            x = self.act_fn(x)
            x = (torch.matmul(x, self.linear2).transpose(1, 2) + self.bias2).transpose(1, 2)
        else:
            raise NameError(f'domain must be general or domain, not {self.domain}')

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
                 begin_layer: int = 12,
                 **kwargs):
        super().__init__()
        self.last_pos = None
        self.begin_layer = begin_layer
        self.trainsets_len = max(trainsets_len, 1)
        self.domain_size = max(trainsets_len, 1)
        self.domain_embedding_dim = 8

        self.domain_embedding = nn.Parameter(torch.Tensor(self.domain_size, self.domain_embedding_dim))
        init.xavier_uniform_(self.domain_embedding)

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

        self.adapters = nn.ModuleList([
            Adapter(self.bert.config.hidden_size, adapter_size, False, trainsets_len),
            Adapter(self.bert.config.hidden_size, adapter_size, False, trainsets_len)
            ])

        for i, layer in enumerate(self.bert.encoder.layer):
            layer.output = AdapterBertOutput(
                layer.output, self.adapters[0].forward)
            set_requires_grad(layer.output.base.LayerNorm, True)
            layer.attention.output = AdapterBertOutput(
                layer.attention.output, self.adapters[1].forward)
            set_requires_grad(layer.attention.output.base.LayerNorm, True)

        self.output_dim = self.bert.config.hidden_size

        if word_piece == 'first':
            self.word_piece = None
        else:  # mean of pieces
            offset = torch.tensor([0], dtype=torch.long)

            self.word_piece = lambda x: embedding_bag(
                x, self.bert.embeddings.word_embeddings.weight, offset.to(x.device))

    def set_pos(self, pos: List[int]):
        self.adapters[0].set_pos(pos)
        self.adapters[1].set_pos(pos)

    def set_domain(self, domain: str = 'general'):
        self.adapters[0].set_domain(domain)
        self.adapters[1].set_domain(domain)

    def forward(self,
                input_ids: torch.Tensor,
                word_pieces: Dict[Tuple[int], torch.LongTensor] = None,
                mask: torch.Tensor = None,
                pos: List[int] = None,
                domain: str = 'general',
                **kwargs) -> torch.Tensor:
        self.set_pos(pos)
        self.set_domain(domain)
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        if self.word_piece is not None and word_pieces is not None:
            for (s, w), pieces in word_pieces.items():
                inputs_embeds[s, w, :] = self.word_piece(pieces)

        attention_mask = None if mask is None else mask.float()
        bert_output = self.bert(attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        return bert_output[0]

    def init_model(self, param_path=None, device=None):
        if self.trainsets_len > 1:
            if param_path is not None:
                self._load_param(param_path, device)
        else:
            print(f'trainsets len is: {self.trainsets_len}, do not need to init')

    def _save_pos_params(self, path):
        state_dict = {}
        state_dict[f'adapter_0'] = self.adapters[0].state_dict()
        state_dict[f'adapter_1'] = self.adapters[1].state_dict()
        torch.save(state_dict, path)
        print(f'save pos params successful, save at: {path}')

    def _load_param(self, state_dict, device):
        self.adapters[0].load_general_params(state_dict['adapter_0'])
        self.adapters[1].load_general_params(state_dict['adapter_1'])

        self.domain_embedding.data[1:] = state_dict['domain_embedding'].data
        self.domain_embedding.data[0] = state_dict['domain_embedding'].data[0]


class DomainModel(nn.Module):
    def __init__(self, bert_path, label_vocab, d_model: int = 768, adapter_size: int = 192, hidden_size: int = 400, trainsets_len=-1, combine: bool = True, share_weight: bool = True):
        super(DomainModel, self).__init__()

        self.d_model = d_model
        self.hidden_size = hidden_size
        self.domain_embedding_dim = 8
        self.num_layers = 1
        self.share_weight = share_weight

        self.label_vocab = label_vocab
        self.num_tags = label_vocab.max_n_words
        # self.linear_bridge = nn.Linear(in_features=d_model * 2, out_features=d_model)

        self.bert = AdapterBertModel(name_or_path_or_model=bert_path, adapter_size=adapter_size, trainsets_len=trainsets_len, combine=combine)
        self.domain_classifier = GRLClassifier(input_size=d_model, d_model=self.domain_embedding_dim, tag_size=trainsets_len)
        self.domain_vae = DomainVariational(input_size=d_model, d_model=d_model, domain_size=trainsets_len)

        self.lstm = nn.ModuleList([
            RNN(type="LSTM", batch_first=True, input_size=d_model if layer == 0 else 2 * self.hidden_size,
                hidden_size=self.hidden_size, bidirectional=True)
            for layer in range(self.num_layers)
        ])

        self.CRF = CRF(num_tags=self.num_tags, input_dim=self.hidden_size * 2)

        if share_weight is True:
            self.domain_classifier.share_domain_weight(self.bert.domain_embedding)
            self.domain_vae.share_domain_weight(self.bert.domain_embedding)

    def init_model(self, param_path=None, device=None):
        self.bert.init_model(param_path, device)

    def _save_pos_params(self, path):
        state_dict = {}
        state_dict['adapter_0'] = self.bert.adapters[0].state_dict()
        state_dict['adapter_1'] = self.bert.adapters[1].state_dict()
        state_dict['domain_embedding'] = self.bert.domain_embedding
        state_dict['lstm'] = self.lstm.state_dict()
        state_dict['domain_vae'] = self.domain_vae.state_dict()
        state_dict['domain_classifier'] = self.domain_classifier.state_dict()
        torch.save(state_dict, path)
        print(f'save pos params successful, save at: {path}')

    def dropout_state_dict(self, state_dict, ratio=0.1, train=True):
        for k, v in state_dict.items():
            if isinstance(v, dict):
                self.dropout_state_dict(v, ratio)
            elif isinstance(v, torch.Tensor):
                with torch.no_grad():
                    torch.dropout_(v, ratio, train)

    def _load_param(self, path, device, drop_ratio=0):
        state_dict = torch.load(path, map_location='cuda:' + str(device))
        if drop_ratio > 0:
            print(f'drop the continue params with ratio {drop_ratio}')
            self.dropout_state_dict(state_dict, drop_ratio)
        self.bert._load_param(state_dict, device)
        self.domain_vae._load_param(state_dict['domain_vae'], device)
        self.domain_classifier._load_param(state_dict['domain_classifier'], device)
        self.lstm.load_state_dict(state_dict=state_dict['lstm'])

        if self.share_weight is True:
            self.bert.adapters[0].domain_embedding = self.bert.domain_embedding
            self.bert.adapters[1].domain_embedding = self.bert.domain_embedding
            self.domain_classifier.share_domain_weight(self.bert.domain_embedding)
            self.domain_vae.share_domain_weight(self.bert.domain_embedding)

        print(f'load params from {path} successful !')

    def forward(self, seqs, crf_mask, bert_mask, labels, finetune=False, pos=None, inference: bool = False):
        if inference is True:
            # general_bert_outs = self.bert(seqs, mask=bert_mask, pos=pos, domain='general')
            # general_vae_outs = self.domain_vae(general_bert_outs, bert_mask, pos)
            # ctx = general_vae_outs

            domain_bert_outs = self.bert(seqs, mask=bert_mask, pos=pos, domain='domain')
            ctx = domain_bert_outs

            for i, rnn in enumerate(self.lstm):
                ctx, _ = rnn(ctx, ~bert_mask)

            res_dic = self.CRF(ctx, crf_mask, labels)

            return res_dic
        else:
            general_bert_outs = self.bert(seqs, mask=bert_mask, pos=pos, domain='general')
            domain_bert_outs = self.bert(seqs, mask=bert_mask, pos=pos, domain='domain')
            domain_logits = self.domain_classifier(general_bert_outs, bert_mask)

            general_vae_outs = self.domain_vae(general_bert_outs, bert_mask, pos)

            general_ctx = general_vae_outs
            domain_ctx = domain_bert_outs
            for i, rnn in enumerate(self.lstm):
                general_ctx, _ = rnn(general_ctx, ~bert_mask)
                domain_ctx, _ = rnn(domain_ctx, ~bert_mask)

            general_res_dic = self.CRF(general_ctx, crf_mask, labels)
            domain_res_dic = self.CRF(domain_ctx, crf_mask, labels)

            general_res_dic['adapter_out'] = general_vae_outs * bert_mask.unsqueeze(-1)
            domain_res_dic['adapter_out'] = domain_bert_outs * bert_mask.unsqueeze(-1)

            return general_res_dic, domain_res_dic, torch.log_softmax(domain_logits, dim=-1)
