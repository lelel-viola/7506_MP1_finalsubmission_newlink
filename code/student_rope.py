"""From-scratch rotary causal LM, with no position-dependent inference state."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from student_cache import mixed_log_probs


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        return normalized.to(x.dtype) * self.weight


def rotary(x, cosine, sine):
    left, right = x.chunk(2, dim=-1)
    return torch.cat((left*cosine-right*sine, right*cosine+left*sine), dim=-1)


class RotaryBlock(nn.Module):
    def __init__(self, width, heads, hidden, dropout):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.qkv = nn.Linear(width, width*3, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.gate_value = nn.Linear(width, hidden*2, bias=False)
        self.out = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cosine, sine):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width//self.heads).permute(2, 0, 3, 1, 4)
        q, k = rotary(q, cosine, sine), rotary(k, cosine, sine)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        gate, value = self.gate_value(self.norm2(x)).chunk(2, dim=-1)
        return x + self.dropout(self.out(F.silu(gate)*value))


class RotaryGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width, heads, depth = config['width'], config['heads'], config['depth']
        assert width % heads == 0 and (width//heads) % 2 == 0
        self.token = nn.Embedding(config['vocab'], width)
        self.embedding_dropout = nn.Dropout(config.get('embedding_dropout', .1))
        self.blocks = nn.ModuleList([RotaryBlock(width, heads, config['ffn_hidden'],
                                                config.get('residual_dropout', .2)) for _ in range(depth)])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight
        with torch.no_grad():
            for block in self.blocks:
                block.proj.weight.div_(math.sqrt(2*depth))
                block.out.weight.div_(math.sqrt(2*depth))
        frequencies = 1. / (10000 ** (torch.arange(0, width//heads, 2).float()/(width//heads)))
        angles = torch.arange(self.context).float()[:, None] * frequencies[None, :]
        self.register_buffer('cosine', angles.cos()[None, None], persistent=False)
        self.register_buffer('sine', angles.sin()[None, None], persistent=False)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)

    def features(self, ids):
        x = self.embedding_dropout(self.token(ids))
        # Position tables are constants; no state survives an input window.
        for block in self.blocks:
            dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
            x = block(x, self.cosine[:, :, :ids.shape[1]].to(dtype), self.sine[:, :, :ids.shape[1]].to(dtype))
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        features = self.features(ids)
        return mixed_log_probs(features, self.head(features), ids, self.config)


def build_model(config):
    return RotaryGPT(config)
