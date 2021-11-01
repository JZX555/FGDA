import torch
import torch.nn as nn
import utils.init as init
from .bertCRF import CRF
from transformers import BertModel
from modules.rnn import RNN


class BaseModel(nn.Module):
    def __init__(self, bert_path, label_vocab, d_model: int = 768, trainsets=None):
        super(BaseModel, self).__init__()

        self.d_model = d_model
        self.hidden_size = 400

        self.label_vocab = label_vocab
        self.num_tags = label_vocab.max_n_words
        self.num_layers = 1

        self.bert = BertModel.from_pretrained(bert_path)
        self.lstm = nn.ModuleList([
            RNN(type="LSTM", batch_first=True, input_size=d_model if layer == 0 else 2 * self.hidden_size,
                hidden_size=self.hidden_size, bidirectional=True, dropout=0.2)
            for layer in range(self.num_layers)
        ])
        self.CRF = CRF(num_tags=self.num_tags, input_dim=self.d_model)

    def init_model(self, param_path=None, device=None):
        if param_path is not None:
            self._load_param(param_path, device)
        # self._add_cur_pos_linear()

    def _save_pos_params(self, path):
        pass
        # state_dict = self.pos_linear.state_dict()
        # torch.save(state_dict, path)
        # print(f'save pos params successful, save at: {path}')

    def _load_param(self, path, device):
        state_dict = torch.load(path, map_location='cuda:' + str(device))
        self.pos_linear.load_state_dict(state_dict)
        print(f'load pos params from {path} into device cuda: {device}')

    def _add_cur_pos_linear(self):
        print(f'origin pos params size: {len(self.pos_linear)}')
        linear = nn.Parameter(init.default_init(torch.zeros((self.d_model, self.d_model), dtype=torch.float32)))
        self.pos_linear = nn.ParameterList([linear]).extend(self.pos_linear)
        print(f'successful add a pos param at the first of the paramsList, cur size: {len(self.pos_linear)}')

    def forward(self, seqs, mask, labels, finetune=False, pos=None):
        batch_size = seqs.size(0)
        bert_outs, bert_cls = self.bert(seqs)

        if finetune is False:
            bert_outs = bert_outs.detach()

        ctx = bert_outs
        for i, rnn in enumerate(self.lstm):
            ctx, _ = rnn(ctx, ~mask, bert=True)

        # if isinstance(pos, list):
        #     assert len(pos) == batch_size, ValueError(f'the len of pos:{len(pos)} is not equal  the batch_size:{batch_size}')

        #     pos_weights = []
        #     for p in pos:
        #         pos_weights += [self.pos_linear[p].unsqueeze(0)]
        #     pos_weights = torch.cat(pos_weights, dim=0)

        #     bert_outs = torch.matmul(bert_outs, pos_weights)

        res_dic = self.CRF(ctx, mask, labels)

        return res_dic
