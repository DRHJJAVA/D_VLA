import torch
import torch
from torch import nn

class ContactFrameNet(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        self.rgbd = nn.Sequential(nn.Conv2d(4, 16, 3, 2, 1), nn.GELU(), nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.point = nn.Sequential(nn.Linear(10, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU())
        self.pool = nn.Sequential(nn.Linear(256, 128), nn.GELU())
        self.state = nn.Sequential(nn.Linear(state_dim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(256, 128), nn.GELU())
        self.temporal = nn.GRU(128, 128, batch_first=True)
        self.contact = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))
        self.dynamics = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 5))
        self.interaction = torch.nn.Sequential(torch.nn.Linear(128, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))

    def forward(self, images, points, states):
        states = states.clone()
        states[..., :9] = 0.0
        (B, T) = images.shape[:2]
        iv = self.rgbd(images.flatten(0, 1)).flatten(1).reshape(B, T, 64)
        pv = self.point(points)
        m = points[..., 9:10] > 0
        avg = (pv * m).sum(2) / m.sum(2).clamp_min(1)
        mx = pv.masked_fill(~m, -10000.0).max(2).values
        mx = torch.where(m.any(2), mx, torch.zeros_like(mx))
        pp = self.pool(torch.cat([mx, avg], -1))
        ss = self.state(states)
        (h, _) = self.temporal(self.fuse(torch.cat([iv, pp, ss], -1)))
        last = h[:, -1]
        return (self.interaction(last).squeeze(-1), self.dynamics(last), self.contact(last).squeeze(-1))