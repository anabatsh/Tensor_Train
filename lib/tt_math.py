import numpy as np
import torch
import torch.linalg as LA
from .tt_format import TensorTrain


# def svd(A, eps):
#     U, S, V = 
#     eps_s = np.cumsum(S.numpy()[::-1] ** 2)
    
#     print(eps_s, eps)
    
#     if eps_s[-1] < eps:
#         return U[:,0], S[0], V[0]
    
#     if eps_s[0] > eps:
#         return U, S, V
    
#     k = sum(eps_s < 31)
#     return U[:, :-k], S[:-k], V[:-k]


def tt_svd(A, max_rank=10):
    """
    A - d-dimensional torch.tensor
    """    
    modes = A.shape 
    d = len(modes)
    ranks = [1] + [max_rank] * (d - 1) + [1]    
    cores = []
    C = A.detach().clone()
    
    for k in range(d-1):
        C = C.reshape(ranks[k] * modes[k], -1)
        U, S, V = LA.svd(C, full_matrices=False)
        ranks[k + 1] = min(ranks[k + 1], U.shape[1], V.shape[0])
        U = U[:, 0:ranks[k + 1]]
        S = S[0:ranks[k + 1]]
        V = V[0:ranks[k + 1], :]
        core = U.reshape(ranks[k], modes[k], ranks[k+1])
        cores.append(core)
        C = torch.diag(S) @ V
        
    core = C.reshape(ranks[-2], modes[-1], ranks[-1])
    cores.append(core)
    
    return cores, ranks


def _dimensional_grid(A, d, inp_modes, out_modes):
    """
    """
    C = A.reshape(inp_modes + out_modes)
    transpose_idx = np.arange(2 * d).reshape(2, d).T.flatten()
    new_shape = np.array(inp_modes) * np.array(out_modes)
    C = C.permute(tuple(transpose_idx)).reshape(tuple(new_shape))
    return C


def tt_svd_2D(A, inp_modes, out_modes, max_rank=10):
    """
    """
#     assert np.prod(inp_modes) == A.shape[0], 'Incorrect inp_modes'
#     assert np.prod(out_modes) == A.shape[1], 'Incorrect out_modes'
#     assert len(inp_modes) == len(out_modes), 'Incorrect shapes'

    d = len(inp_modes)
    C = _dimensional_grid(A, d, inp_modes, out_modes)
    cores, ranks = tt_svd(C, max_rank)
    
    for k in range(d):
        cores[k] = cores[k].reshape(ranks[k], inp_modes[k], out_modes[k], ranks[k+1])
    
    return TensorTrain(cores, inp_modes, out_modes, ranks)