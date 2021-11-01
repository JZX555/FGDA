import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.functional import linear

from modules.grl import WarmStartGradientReverseLayer


class GRLClassifier(nn.Module):
    def __init__(self, input_size, d_model, tag_size, max_iters=4000, auto_step: bool = True) -> None:
        super(GRLClassifier, self).__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.tag_size = max(tag_size, 1)

        if input_size == d_model:
            self.linear_bridge = None
        else:
            self.linear_bridge = nn.Linear(in_features=input_size, out_features=d_model)

        self.projection = nn.Linear(in_features=d_model, out_features=self.tag_size)
        self.grl = WarmStartGradientReverseLayer(max_iters=max_iters, auto_step=auto_step)

    def forward(self, ctx, mask, reverse=True):
        mask = mask.float()
        if self.linear_bridge is not None:
            ctx = self.linear_bridge(ctx)
        ctx_mean = (ctx * mask.unsqueeze(2)).sum(1) / mask.unsqueeze(2).sum(1)

        if reverse is True:
            ctx_mean = self.grl(ctx_mean)

        tag = self.projection(ctx_mean)

        return tag

    def share_domain_weight(self, weight):
        self.projection.weight = weight

    def _load_param(self, state_dict, device):
        self.projection.bias.data[1:] = state_dict['projection.bias'].data
        self.projection.bias.data[0] = state_dict['projection.bias'].data[0]
        if self.linear_bridge is not None:
            self.linear_bridge.weight.data = state_dict['linear_bridge.weight']
            self.linear_bridge.bias.data = state_dict['linear_bridge.bias']


class DomainVariational(nn.Module):
    def __init__(self, input_size, d_model, domain_embedding_dim=8, domain_size: int = -1):
        super(DomainVariational, self).__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.domain_embedding_dim = domain_embedding_dim
        self.domain_size = max(domain_size, 1)

        self.domain_embedding = nn.Parameter(torch.Tensor(self.domain_size, self.domain_embedding_dim))

        self.params_mu = nn.ParameterList([
            nn.Parameter(torch.Tensor(input_size, d_model, self.domain_embedding_dim)),
            nn.Parameter(torch.zeros((d_model, self.domain_embedding_dim)))
        ])
        self.params_var = nn.ParameterList([
            nn.Parameter(torch.Tensor(input_size, d_model, self.domain_embedding_dim)),
            nn.Parameter(torch.zeros((d_model, self.domain_embedding_dim)))
        ])

        self.layer_norm = nn.LayerNorm(d_model)

        self.init_linear(self.params_mu[0])
        self.init_linear(self.params_var[0])
        self.init_linear(self.domain_embedding, is_domain=False)

    def init_linear(self, w, b=None, is_domain=True):
        init.xavier_uniform_(w)
        if is_domain is True:
            _, fan_in = init._calculate_fan_in_and_fan_out(w)
        else:
            fan_in, _ = init._calculate_fan_in_and_fan_out(w)
        if b is not None:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(b, -bound, bound) 

    def forward(self, ctx, mask, domain):
        linear_mu, linear_var, bias_mu, bias_var = [], [], [], []

        for d in domain:
            linear_mu += [linear(self.params_mu[0].unsqueeze(0), self.domain_embedding[d].contiguous().unsqueeze(0)).squeeze(-1)]
            linear_var += [linear(self.params_var[0].unsqueeze(0), self.domain_embedding[d].contiguous().unsqueeze(0)).squeeze(-1)]
            bias_mu += [linear(self.params_mu[1].unsqueeze(0), self.domain_embedding[d].contiguous().unsqueeze(0))]
            bias_var += [linear(self.params_var[1].unsqueeze(0), self.domain_embedding[d].contiguous().unsqueeze(0))]
        linear_mu = torch.cat(linear_mu, dim=0)
        linear_var = torch.cat(linear_var, dim=0)
        bias_mu = torch.cat(bias_mu, dim=0)
        bias_var = torch.cat(bias_var, dim=0)

        assert len(bias_var.size()) == 3, ValueError(f'the len of bias need to be 3 but get {bias_var.size()}')

        ctx_mu = (torch.matmul(ctx, linear_mu).transpose(1, 2) + bias_mu).transpose(1, 2)
        ctx_var = (torch.matmul(ctx, linear_var).transpose(1, 2) + bias_var).transpose(1, 2)

        sampled_z = torch.randn_like(ctx)

        vctx = ctx_mu + torch.exp(0.5 * ctx_var) * sampled_z

        return self.layer_norm(vctx)

    def share_domain_weight(self, weight):
        self.domain_embedding = weight

    def _load_param(self, state_dict, device):
        self.params_mu[0].data = state_dict['params_mu.0'].data
        self.params_mu[1].data = state_dict['params_mu.1'].data
        self.params_var[0].data = state_dict['params_var.0'].data
        self.params_var[1].data = state_dict['params_var.1'].data
        self.layer_norm.weight.data = state_dict['layer_norm.weight']
        self.layer_norm.bias.data = state_dict['layer_norm.bias']
