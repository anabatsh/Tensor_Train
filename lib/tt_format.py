import numpy as np
import torch 
from . import tt_methods as M

class TensorTrain():
    def __init__(self, cores, inp_modes, out_modes, ranks):
        """
        Tensor Train format:
        y = Wx, 
            x: batch_size * inp_size
            W: inp_size * out_size
            y: batch_size * out_size
            
        W = G_1 * G_2 * ... * G_d - TT_cores:
            G_k shape is [r_k, n_k, m_k, r_k+1]
            r_k = TT_ranks[k]
            n_k = out_modes[k]
            m_k = inp_modes[k]
        """  
        
        self.cores = list(cores)            # TT_cores for W
        self.d = len(self.cores)            # TT_dimension - number of cores
        self.ranks = ranks                  # TT_ranks

        self.inp_modes = inp_modes          # cores input shapes
        self.out_modes = out_modes          # cores output shapes
        self.inp_size = np.prod(inp_modes)  # total input shape of W - product of inp_modes
        self.out_size = np.prod(out_modes)  # total output shape of W - product of out_modes
   
    ###### ---------------------- container methods ---------------------- ######
    
    def __len__(self):
        """
        <return> : len of the TT - number of cores
        """
        return self.d
   
    ###### ---------------------- unary operations ---------------------- ######
    
    def apply(self, fn):
        """
        Applies ``fn`` to every core 
        <args>   : fn - function to be applied to each core
        <return> : list of new cores
        """
        return [fn(core) for core in self.cores]    

    def __neg__(self):
        """
        <return> : negative -TT -self
        """
        return TensorTrain(self.apply(M._neg), self.inp_modes, self.out_modes, self.ranks)
        
    def __abs__(self):
        """
        <return> : absolute TT |self|
        """
        return TensorTrain(self.apply(M._abs), self.inp_modes, self.out_modes, self.ranks)
    
    def t(self):
        """
        <return> : transposed TT self^T
        """
        return TensorTrain(self.apply(M._t), self.out_modes, self.inp_modes, self.ranks)
            
    def norm(self):
        """
        <return> : Frobenius norm of the TT ||self||_F
        """
        return sum(self.apply(M._norm))

    def __repr__(self):
        """
        <return> : extra representation of the TT
        """
        return ''.join(self.apply(M._repr))
    
    def _orthogonalize_right(self):
        """
        <return> : right-orthogonalized TT
        """
        cores = M._orthogonalize_right(self)
        return TensorTrain(cores, self.inp_modes, self.out_modes, self.ranks)

    def _orthogonalize_left(self):
        """
        <return> : left-orthogonalized TT
        """
        cores = M._orthogonalize_left(self)
        return TensorTrain(cores, self.inp_modes, self.out_modes, self.ranks)
    
    def to_tensor(self):
        """
        <return> : (TT converted) corresponding Torch.Tensor
        """
        return M._to_tensor(self)    

    ###### ---------------------- binary operations ---------------------- ######
        
    def __add__(self, other):
        """
        Addition in the TT format: other + self
        <args>: other - object : [TT format] 
        <return>: TT = sum of self and other
        """
        if isinstance(other, TensorTrain):
            cores, ranks = M._TT_TT_add(self, other)
            return TensorTrain(cores, self.inp_modes, self.out_modes, ranks)
        
        else:
            raise ValueError(f'Incorrect type of the second argument')        
      
    def __mul__(self, other):
        """
        Multiplication in the TT format: self * other
        <args>: other - object : [TT format, Torch.Tensor] 
        <return>: TT = product of self and other
        """
        if isinstance(other, torch.Tensor):
            return M._TT_tensor_multiply(self, other)
        
        elif isinstance(other, TensorTrain):
            cores, ranks = M._TT_TT_multiply(self, other)
            return TensorTrain(cores, self.inp_modes, other.out_modes, ranks)
        
        else:
            raise ValueError(f'Incorrect type of the second argument')        

    def __rmul__(self, other):
        """
        Right multiplication in the TT format: other * self
        <args>: other - object : [Torch.Tensor] 
        <return>: TT = multiplication of other and self
        """
        if isinstance(other, torch.Tensor):
            return M._tensor_TT_multiply(other, self)
        
        else:
            raise ValueError(f'Incorrect type of the second argument')
