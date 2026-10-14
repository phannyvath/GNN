"""Identical graph inputs; never include graph ID, scenario or label as features."""
import hashlib
import numpy as np

def summary_features(g):
    x=g['x']
    return np.concatenate([x.mean(0),x.std(0),x.max(0),[np.log1p(len(x)),np.log1p(len(g['src']))]])

def shingle_features(g,bins=256):
    """Normalized hashed typed walks of lengths 1 and 2.
    StreamSpot-inspired offline approximation; no streaming sketches or clustering.
    """
    src=g['src']; dst=g['dst']; types=g['types']; rel=g['edge_types']
    out=np.zeros(bins,dtype=np.float64); outgoing={}
    for a,b,r in zip(src,dst,rel): outgoing.setdefault(int(a),[]).append((int(b),int(r)))
    def add(parts):
        key='|'.join(map(str,parts)).encode()
        idx=int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'little')%bins
        out[idx]+=1
    for a,b,r in zip(src,dst,rel):
        a=int(a); b=int(b); add([types[a],r,types[b]])
        for c,r2 in outgoing.get(b,[]): add([types[a],r,types[b],r2,types[c]])
    return out/max(np.linalg.norm(out),1e-12)
