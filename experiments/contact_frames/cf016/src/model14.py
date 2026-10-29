import importlib.util
import torch
from torch import nn
from common14 import OLD
s = importlib.util.spec_from_file_location('cf010_frozen_model', OLD / 'src/model.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)

class ContactMotionNet(nn.Module):

    def __init__(self, with_motion):
        super().__init__()
        self.base = m.ContactFrameNet(50)
        self.motion = nn.Sequential(nn.Linear(24, 64, bias=False), nn.GELU(), nn.Linear(64, 128, bias=False))
        nn.init.zeros_(self.motion[-1].weight)
        self.with_motion = with_motion
        self._extra = None
        self.base.temporal.register_forward_pre_hook(self._inject)

    def _inject(self, module, args):
        return (args[0] + self._extra,) + args[1:]

    def forward(self, images, points, states, motion):
        x = motion if self.with_motion else torch.zeros_like(motion)
        x = x.clone()
        x[..., 21:23] = 0
        self._extra = self.motion(x)
        try:
            return self.base(images, points, states)
        finally:
            self._extra = None
