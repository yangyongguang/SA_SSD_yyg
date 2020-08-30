import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn

import myCudaFun_cuda as myCudaFun


import numpy as np

class myCudaFunClass(Function):
    @staticmethod
    def forward(ctx, tensorA: torch.Tensor, tensorB: torch.Tensor) -> torch.Tensor:
        """
        """
        assert tensorA.is_contiguous()
        assert tensorB.is_contiguous()
        
        n = tensorA.size(0)

        output = torch.ones_like(tensorA).cuda()
        myCudaFun.add_warpper(n, tensorA, tensorB, output)
        return output

    @staticmethod
    def backward(ctx, a=None, b=None):
        """
        """
        return None, None

cuda_fun = myCudaFunClass.apply

data1 = torch.ones([2000]).cuda().float() * 0.5
data2 = torch.ones([2000]).cuda().float() * 3.5

# data3 = torch.zeros([6]).cuda().float()
print(data1)
print(data2)

data3 = cuda_fun(data1, data2)

print(data3)

