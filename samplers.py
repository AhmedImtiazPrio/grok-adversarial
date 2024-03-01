## Neighborhood samplers for local complexity approximation

import torch
import numpy as np

@torch.no_grad()
def sample_ortho_random(n,d,seed=None):

    if seed is not None:
        torch.manual_seed(seed)

    orth_linear = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(d,n).cuda(),
                                                             use_trivialization=False)
    return orth_linear.weight

@torch.no_grad()
def get_ortho_hull_around_samples(x,r=1,n=10,seed=None):
    """
    x: batchsize x channels x dim1 x dim2
    r: max radius of hull
    n: number of samples

    returns:
    s: batchsize x n_samples x channels x dim1 x dim2
    """

    s = sample_ortho_random(n//2,
                            np.prod(x.shape[1:]),
                            seed=seed).to(x.device)
    s = torch.stack([s for i in range(x.shape[0])])

    s /= s.reshape(x.shape[0],n//2,-1).norm(keepdim=True,p=2,dim=-1)
    s *= r
    s1 = x.reshape(x.shape[0],1,-1) + s
    s2 = x.reshape(x.shape[0],1,-1) - s
    s = torch.cat([s1,s2],dim=1)
    return s.reshape(x.shape[0],n//2*2,*x.shape[1:])

@torch.no_grad()
def get_ortho_hull_around_samples_w_orig(x,r=1,n=11,seed=None):
    """
    x: batchsize x channels x dim1 x dim2
    r: max radius of hull
    n: number of samples

    returns:
    s: batchsize x n_samples x channels x dim1 x dim2
    """

    assert n % 2 == 1, 'n should be odd since centroid is included'
    n -= 1

    s = get_ortho_hull_around_samples(x,r=r,n=n,seed=seed)
    return torch.cat([s,x[:,None,...]],dim=1)

