import numpy as np
import torch 
from torch import nn, optim 
import torch.nn.functional as F
from .tt_format import TensorTrain


class TT_linear(nn.Module):
    def __init__(self, inp_modes, out_modes, ranks):
        super().__init__()
        
        self.d = len(inp_modes)
        self.inp_modes = inp_modes 
        self.out_modes = out_modes 
        self.inp_size = np.prod(inp_modes)
        self.out_size = np.prod(out_modes)
        self.ranks = ranks
        
        self.W_cores = self.init_W_cores(inp_modes, out_modes)
        self.b = torch.nn.Parameter(torch.ones(self.out_size))
        
    def init_W_cores(self, inp_modes, out_modes):
        cores = torch.nn.ParameterList()
        for k in range(self.d):
            core = torch.randn(self.ranks[k], inp_modes[k], out_modes[k], self.ranks[k+1])
            core *= 2 / (inp_modes[k] + out_modes[k])
            cores.append(torch.nn.Parameter(core))
        return cores
            
    def forward(self, x):
        W = TensorTrain(self.W_cores, self.inp_modes, self.out_modes, self.ranks)
        return x * W + self.b
    
    
    
class TT_conv(nn.Module):
    def __init__(self, inp_modes, out_modes, ranks, wH, wW):
        super().__init__()

        self.c_inp_0, self.inp_modes = inp_modes[0], inp_modes[1:]
        self.c_out_0, self.out_modes = out_modes[0], out_modes[1:]

        self.c_inp = np.prod(self.inp_modes)
        self.c_out = np.prod(self.out_modes)
        self.C_inp = self.c_inp_0 * self.c_inp
        self.C_out = self.c_out_0 * self.c_out

        self.ranks = ranks

        self.wH = wH
        self.wW = wW

        self.kernel = self.init_kernel()
        self.cores = TT_linear(self.inp_modes, self.out_modes, self.ranks)

    def init_kernel(self):
        core = torch.randn(self.c_out_0, self.c_inp_0, self.wH, self.wW)
        core *= 2 / (self.c_inp_0 + self.c_out_0)
        return torch.nn.Parameter(core)

    def forward(self, x):
        batch_size, _, H, W = x.shape
        x = x.reshape(-1, self.c_inp_0, H, W)
        x = F.conv2d(x, self.kernel)
        h, w = x.shape[-2:]
        x = x.reshape(batch_size, self.c_inp, -1).permute(0, 2, 1)
        x = x.reshape(-1, self.c_inp)
        y = self.cores(x).reshape(batch_size, self.C_out, h, w)
        return y