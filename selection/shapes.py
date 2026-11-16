"""Classify 7x7 shapes and load the exact fair reference."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .combinations import TOTAL as UNIVERSE


POPCOUNT=np.asarray([int(i).bit_count() for i in range(64)],dtype=np.uint8)
BITS=np.asarray([1<<i for i in range(6)],dtype=np.uint8)
I,J=np.triu_indices(6,1)
RULE_VERSION='shape8-graph-partition-cardinal-diagonal-v1'
LEVELS=('component','partition','edge')


def classify_numbers(numbers: np.ndarray) -> dict[str,np.ndarray]:
    """D4-invariant, origin-independent exact combinatorial graph classes.

    Returns coded topology of all six points on the original 7x7 board:
    L1 = 8-neighbor connected component count;
    L2 = descending multiset of component sizes (integer packed 6x3 bits);
    L3 = L2 plus cardinal and diagonal 8-neighbor edge counts.

    Vectorized Nx6 closure, constant working memory per input chunk. Pure NumPy;
    no optional JIT/compiler/CUDA dependency for exact universe enumeration.
    """
    a=np.asarray(numbers)
    if a.ndim!=2 or a.shape[1]!=6 or len(a)<1:
        raise ValueError('expected [N,6] legal numbers 1..49')
    if not np.issubdtype(a.dtype,np.integer) or np.any(a<1) or np.any(a>49):
        raise ValueError('numbers must be 1..49 integers')
    a=np.sort(a.astype(np.uint8,copy=False),axis=1)
    if np.any(a[:,1:]==a[:,:-1]):raise ValueError('numbers must be distinct')
    pos=a-1
    rr=(pos//7).astype(np.int8)
    cc=(pos%7).astype(np.int8)
    dr=np.abs(rr[:,:,None]-rr[:,None,:]); dc=np.abs(cc[:,:,None]-cc[:,None,:])
    neighbor=(dr<=1)&(dc<=1)
    reach=np.sum(neighbor*BITS[None,None,:],axis=2,dtype=np.uint8)
    # Warshall transitive closure, using one pivot vertex per pass.
    for pivot in range(6):
        connected=(reach & BITS[pivot]) != 0
        reach |= np.where(connected,reach[:,pivot,None],np.uint8(0))
    # Only the lowest-index member represents each component.
    first=(reach & (BITS-1)[None,:])==0
    component_sizes=np.sort(POPCOUNT[reach]*first,axis=1)[:,::-1].astype(np.uint32)
    shifts=np.asarray([15,12,9,6,3,0],dtype=np.uint32)
    partition=np.bitwise_or.reduce(component_sizes << shifts[None,:],axis=1).astype(np.uint32)
    cardinal=np.count_nonzero((((dr==1)&(dc==0))|((dr==0)&(dc==1)))[:,I,J],axis=1).astype(np.uint8)
    diagonal=np.count_nonzero(((dr==1)&(dc==1))[:,I,J],axis=1).astype(np.uint8)
    code=((partition << 10) | (cardinal.astype(np.uint32)<<5) | diagonal.astype(np.uint32)).astype(np.uint32)
    ncomp=np.count_nonzero(component_sizes,axis=1).astype(np.uint8)
    return {'n_components':ncomp,'partition_code':partition,'edge_code':code,
            'cardinal_edges':cardinal,'diagonal_edges':diagonal}


def load_exact_baseline(path: Path):
    raw = path.read_bytes()
    baseline = json.loads(raw)
    if (baseline.get("complete") is not True
            or baseline.get("total") != UNIVERSE
            or baseline.get("schema") != RULE_VERSION):
        raise ValueError("need complete v0.16 exact 6/49 shape universe")
    counts = baseline.get("counts", {})
    if set(counts) != set(LEVELS):
        raise ValueError("wrong taxonomy levels")
    for level in LEVELS:
        values = counts[level]
        if (not isinstance(values, dict) or not values
                or any(int(k) < 0 or int(v) <= 0 for k, v in values.items())
                or sum(int(v) for v in values.values()) != UNIVERSE):
            raise ValueError(f"incomplete theoretical counts: {level}")
    return baseline, sha256(raw).hexdigest()
