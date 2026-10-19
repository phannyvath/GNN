"""Teaching implementations, not official paper reproductions."""
import torch
from torch import nn
import torch.nn.functional as F

def mean_neighbors(x,src,dst):
    out=torch.zeros_like(x); out.index_add_(0,dst,x[src])
    degree=torch.zeros(len(x),device=x.device,dtype=x.dtype)
    degree.index_add_(0,dst,torch.ones(len(dst),device=x.device,dtype=x.dtype))
    return out/degree.clamp_min(1).unsqueeze(1)

class DirectedSAGE(nn.Module):
    def __init__(self,dim,hidden):
        super().__init__(); self.linear=nn.Linear(dim*3,hidden)
    def forward(self,x,src,dst):
        return F.relu(self.linear(torch.cat([x,mean_neighbors(x,src,dst),mean_neighbors(x,dst,src)],1)))

class FlashInspired(nn.Module):
    """GraphSAGE and binary graph head; omits text and temporal encoders."""
    def __init__(self,dim,hidden):
        super().__init__(); self.a=DirectedSAGE(dim,hidden); self.b=DirectedSAGE(hidden,hidden)
        self.head=nn.Linear(2*hidden,1)
    def forward(self,x,src,dst):
        z=self.b(self.a(x,src,dst),src,dst)
        return self.head(torch.cat([z.mean(0),z.max(0).values])).squeeze()

class GraphAttention(nn.Module):
    """Incoming directed attention with self loops and destination softmax."""
    def __init__(self,dim,hidden):
        super().__init__(); self.linear=nn.Linear(dim,hidden,bias=False)
        self.attn_src=nn.Parameter(torch.empty(hidden)); self.attn_dst=nn.Parameter(torch.empty(hidden))
        nn.init.normal_(self.attn_src,std=.1); nn.init.normal_(self.attn_dst,std=.1)
    def forward(self,x,src,dst):
        z=self.linear(x); self_ids=torch.arange(len(x),device=x.device)
        src=torch.cat([src,self_ids]); dst=torch.cat([dst,self_ids])
        score=F.leaky_relu((z[src]*self.attn_src).sum(1)+(z[dst]*self.attn_dst).sum(1),.2)
        maximum=torch.full((len(x),),-float('inf'),device=x.device,dtype=x.dtype)
        maximum.scatter_reduce_(0,dst,score,reduce='amax',include_self=True)
        weight=(score-maximum[dst]).exp()
        den=torch.zeros(len(x),device=x.device,dtype=x.dtype); den.index_add_(0,dst,weight)
        weight=weight/den[dst].clamp_min(1e-12)
        out=torch.zeros_like(z); out.index_add_(0,dst,z[src]*weight.unsqueeze(1))
        return F.elu(out)

class MagicInspired(nn.Module):
    """Benign masked reconstruction -> graph embeddings -> external kNN.
    Omits structure reconstruction, adaptation and original multihead settings.
    """
    def __init__(self,dim,hidden):
        super().__init__(); self.a=GraphAttention(dim,hidden); self.b=GraphAttention(hidden,hidden)
        self.decoder=nn.Linear(hidden,dim); self.token=nn.Parameter(torch.zeros(dim))
    def encode(self,x,src,dst): return self.b(self.a(x,src,dst),src,dst)
    def embedding(self,x,src,dst):
        z=self.encode(x,src,dst); return torch.cat([z.mean(0),z.max(0).values])
    def loss(self,x,src,dst,mask_rate):
        mask=torch.rand(len(x),device=x.device)<mask_rate
        if not mask.any(): mask[0]=True
        damaged=x.clone(); damaged[mask]=self.token
        return F.mse_loss(self.decoder(self.encode(damaged,src,dst))[mask],x[mask])
