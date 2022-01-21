import numpy as np
import torch
import torch.linalg as LA


def _neg(core):
    return -core


def _abs(core):
    return abs(core)


def _t(core):
    return core.permute(0, 2, 1, 3)


def _norm(core):
    return LA.norm(core).item()


def _repr(core):
    return f'shape: {core.shape}\n'


def _H(core):
    return core.flatten(start_dim=1)


def _V(core):
    return core.flatten(end_dim=-2)


def _un_H(H, core):
    return H.reshape(*core.shape)


def _un_V(V, core):
    return V.reshape(*core.shape)


def _orthogonalize_right(A):
    cores = A.cores.copy()
    for k in range(1, A.d)[::-1]:
        Q, R = LA.qr(_H(cores[k]).t())
        cores[k] = _un_H(Q.t(), cores[k])
        cores[k-1] = _un_V(_V(cores[k-1]) @ R.t(), cores[k-1])
    return cores


def _orthogonalize_left(A):
    cores = A.cores.copy()
    for k in range(A.d - 1):
        Q, R = LA.qr(_V(cores[k]))
        cores[k] = _un_V(Q, cores[k])
        cores[k+1] = _un_H(R @ _H(cores[k+1]), cores[k+1])
    return cores


def _to_tensor(A):
    res = A.cores[0]
    for k in range(1, A.d):
        r_k = A.ranks[k]
        res = res.reshape(-1, r_k)
        core = A.cores[k].reshape(r_k, -1)
        res = torch.mm(res, core)

    shape = list(np.stack((A.inp_modes, A.out_modes)).T.ravel())
    res = res.reshape(shape)
    r_d = 2 * np.arange(A.d)
    shape = list(np.hstack((r_d, r_d + 1)))
    res = res.permute(shape)
    res = res.reshape(A.inp_size, A.out_size)   
    return res


########################################################################################


def _TT_TT_add(A, B):
    assert A.d == B.d, 'Incorrect dimension'
#     assert A.ranks == B.ranks, 'Incorrect ranks'
#     assert A.inp_modes == B.inp_modes, 'Incorrect inp_modes'
#     assert A.out_modes == B.out_modes, 'Incorrect out_modes'
    cores = []
    ranks = []
    for k in range(A.d):
        a_core = A.cores[k]
        b_core = B.cores[k]
        if k == 0:
            core = torch.cat([a_core, b_core], 3)
        elif k == A.d - 1:
            core = torch.cat([a_core, b_core], 0)
        else:
            upper_zeros = torch.zeros((A.ranks[k], A.inp_modes[k], A.out_modes[k], B.ranks[k+1]))
            lower_zeros = torch.zeros((B.ranks[k], A.inp_modes[k], A.out_modes[k], A.ranks[k+1]))
            upper = torch.cat([a_core, upper_zeros], 3)
            lower = torch.cat([lower_zeros, b_core], 3)
            core = torch.cat([upper, lower], 0)
        cores.append(core) 
        ranks.append(A.ranks[k] + B.ranks[k])
    ranks.append(1)
    ranks[0] = 1
    return cores, ranks


def _TT_TT_multiply(A, B):
    assert A.d == B.d, 'Incorrect dimension'
    ranks = [A.ranks[k] * B.ranks[k] for k in range(A.d+1)] 
    cores = []
    for k in range(A.d): 
        assert A.inp_modes[k] == B.out_modes[k], 'Incorrect shapes'
        core = torch.einsum('aijb, cjkd-> acikbd', A.cores[k], B.cores[k])
        cores.append(core.reshape(ranks[k], A.inp_modes[k], B.out_modes[k], ranks[k+1]))
    return cores, ranks


def _TT_tensor_multiply(A, x):
    assert A.out_size == x.shape[0], 'Incorrect shape'
    out = x.t()
    for k in range(A.d)[::-1]:
        out = out.reshape(-1, A.out_modes[k], A.ranks[k+1])
        out = torch.einsum('aijb,rjb->ira', A.cores[k], out)
    return out.reshape(A.inp_size, x.shape[1])


def _tensor_TT_multiply(x, A):
    assert A.inp_size == x.shape[1], 'Incorrect shape'
    return _TT_tensor_multiply(A.t(), x.t()).t()

