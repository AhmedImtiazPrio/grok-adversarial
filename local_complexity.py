import torch
import numpy as np
import warnings
import time

@torch.no_grad()
@torch.jit.script
def get_intersection_from_activation_batched(act : torch.Tensor,batch_size : int = 1):
    """
    Input:
    act: batch of layer activations
    n_hulls: number of samples in hull. required for batching
    Out:
    Number of intersections with hull and percentage of layer hyperplanes intersecting
    """
    act = torch.sign(act).reshape(batch_size,act.shape[0]//batch_size,-1) ## get activation pattern
    match = act[:,1:,...] != act[:,:1,...] ## check if activation pattern identical across samples
    match = torch.any(match,dim=1)
    n_inter = torch.sum(match,dim=-1)
    return n_inter, n_inter/match.shape[-1]


@torch.no_grad()
def sample_ortho_random(n,d,seed=None):

    if seed is not None:
        torch.manual_seed(seed)

    orth_linear = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(d,n).cuda(),
                                                             use_trivialization=False)
    return orth_linear.weight


# @torch.jit.script
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


@torch.no_grad()
@torch.jit.script
def get_hull_around_samples(x : torch.Tensor,r:int = 1,n:int = 10, seed:int = 0):
    """
    x: batchsize x channels x dim1 x dim2
    r: max radius of hull
    n: number of samples

    returns:
    s: batchsize x n_samples x channels x dim1 x dim2
    """

#     s = torch.randn(n,*x.shape,dtype=x.dtype,device=x.device)
    s = torch.randn(n//2,x.shape[0],x.shape[1],x.shape[2],x.shape[3],
                    dtype=x.dtype,device=x.device)
    s /= s.reshape(n//2,x.shape[0],-1).norm(keepdim=True,p=2,dim=-1,)[...,None,None]
    s *= r
    s1 = x[None,...] + s
    s2 = x[None,...] - s
    s = torch.cat([s1,s2],dim=0)
    return s.transpose(0,1)

@torch.no_grad()
def get_layer_intersections_batched(layer_names,activation_buffer,batch_size=1):
    """
    Input:
    layer_names: list with elements as strings or tuples of strings.
    activation_buffer: dictionary containing all activations

    Output:
    Number and percentage of intersections for each member of layer_names
    """

    n_inters = torch.zeros(batch_size,len(layer_names),device='cpu')
    p_inters = torch.zeros(batch_size,len(layer_names),device='cpu')

    for i,name in enumerate(layer_names):

        if type(name) == tuple:

            fused_act = torch.stack([activation_buffer[each] for each in name]).sum(0)
            n_inter, p_inter = get_intersection_from_activation_batched(fused_act,
                                                                        batch_size=batch_size)

        else:
            n_inter, p_inter = get_intersection_from_activation_batched(activation_buffer[name],
                                                                        batch_size=batch_size)

        n_inters[:,i] = n_inter.cpu()
        p_inters[:,i] = p_inter.cpu()

    return n_inters,p_inters

@torch.no_grad()
def get_intersections_for_hulls(hulls,
                                model,
                                layer_names,
                                activation_buffer,
                                batch_size=32,
                                verbose=False
                                ):

    """
    sampler: sampling function to sample domain around each sample
    batch_size: number of samples to take for each forward pass.
    effective gpu batch_size is hull_n*batch_size
    """

    nsamples, n_frame = hulls.shape[:2]

    if nsamples % batch_size != 0:
        warnings.warn('number of samples not divisible by `batch_size`, last batch will be dropped')

    dataloader = torch.utils.data.DataLoader(hulls,
                                             batch_size=batch_size,
                                             pin_memory=False,
                                             shuffle=False,
                                             drop_last=True
                                            )

    n_inters = torch.zeros(nsamples,len(layer_names),device='cpu')
    p_inters = torch.zeros_like(n_inters)

    start = 0
    start_time = time.time()
    for batch in dataloader:

        end  = start+batch_size

        with torch.no_grad():

            concat_hulls = batch.reshape(batch_size*n_frame,*hulls.shape[2:])
            out = model.forward(concat_hulls.cuda())
            n_inter,p_inter = get_layer_intersections_batched(layer_names,
                                                              activation_buffer,
                                                              batch_size=batch_size)
        n_inters[start:end] = n_inter.cpu()
        p_inters[start:end] = p_inter.cpu()

        start = end
    
    if verbose:
        print(f"LC elapsed time:{time.time()-start_time:.5f}")
    
    return n_inters, p_inters