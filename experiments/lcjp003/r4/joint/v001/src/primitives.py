import math
import torch
from torch import nn
from torch.nn import functional as F

class Cross(nn.Module):

    def __init__(self):
        super().__init__()
        self.qn = nn.LayerNorm(256)
        self.kn = nn.LayerNorm(256)
        self.att = nn.MultiheadAttention(256, 4, batch_first=True, dropout=0)
        self.fn = nn.LayerNorm(256)
        self.ff = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))

    def forward(self, q, kv, mask=None):
        q = q + self.att(self.qn(q), self.kn(kv), self.kn(kv), key_padding_mask=mask, need_weights=False)[0]
        return q + self.ff(self.fn(q))

class LoRA(nn.Module):

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, 16, bias=False)
        self.b = nn.Linear(16, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + self.b(self.a(x))

def mlp(*dims, ln=False):
    layers = []
    for (i, (a, b)) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            if ln:
                layers.append(nn.LayerNorm(b))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)

class KVInputAdapter(nn.Module):

    def __init__(self, base, source):
        super().__init__()
        self.base = base
        self.bridge = nn.Linear(720, 320, bias=False)
        with torch.no_grad():
            self.bridge.weight.copy_(source.weight)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(self.bridge(x))
