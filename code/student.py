"""Student implementation: compact gated GPT blocks.

The baseline GPT is already a good small causal transformer.  This variant keeps
the same public interface and causal attention, but replaces the plain GELU MLP
with a SwiGLU-style gated feed-forward path.  The gate lets each token choose
which expanded features should pass through, improving expressiveness without
using external data, future tokens, or evaluation-time state.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model import GPT


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        hidden = 3 * width
        self.value = nn.Linear(width, hidden)
        self.gate = nn.Linear(width, hidden)
        self.out = nn.Linear(hidden, width)

    def forward(self, x):
        return self.out(F.silu(self.gate(x)) * self.value(x))


class StudentBlock(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = SwiGLU(width)
        self.attn_scale = nn.Parameter(torch.ones(()))
        self.mlp_scale = nn.Parameter(torch.ones(()))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attn_scale * self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp_scale * self.mlp(self.norm2(x))


class StudentGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        width = config.get('student_width', config['width'])
        heads = config.get('student_heads', config['heads'])
        depth = config.get('student_depth', config['depth'])
        self.context = config['context']
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList([StudentBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)


def build_model(config):
    return StudentGPT(config)
