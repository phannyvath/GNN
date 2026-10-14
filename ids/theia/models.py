"""Independent node heads; never reuse StreamSpot graph labels or graph pooling."""
import torch
from torch import nn
from ids.models import DirectedSAGE, GraphAttention


class FlashNode(nn.Module):
    """One GraphSAGE node-type predictor; unexpected type probability is the score."""
    def __init__(self, dim, hidden, types):
        super().__init__()
        self.a = DirectedSAGE(dim, hidden); self.b = DirectedSAGE(hidden, hidden)
        self.head = nn.Linear(hidden, types)
    def forward(self, x, src, dst):
        return self.head(self.b(self.a(x, src, dst), src, dst))


class MagicNode(nn.Module):
    """Masked node feature reconstruction and node-embedding kNN distance."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.a = GraphAttention(dim, hidden); self.b = GraphAttention(hidden, hidden)
        self.decoder = nn.Linear(hidden, dim); self.token = nn.Parameter(torch.zeros(dim))
    def encode(self, x, src, dst):
        return self.b(self.a(x, src, dst), src, dst)
    def loss(self, x, src, dst, rate):
        mask = torch.rand(len(x),device=x.device) < rate
        if not mask.any(): mask[0] = True
        damaged = x.clone(); damaged[mask] = self.token
        z = self.encode(damaged,src,dst)
        return nn.functional.mse_loss(self.decoder(z)[mask],x[mask])
