import math
from torch import nn
from torch.autograd import Function
import torch

import lltm_cpp

torch.manual_seed(42)


def Enzyme(filename, function):

    class anon(Function):

        @staticmethod
        def forward(ctx, inp):
            outputs = lltm_cpp.forward(inp.contiguous(), filename, function)
            ctx.save_for_backward(inp)
            return outputs[0]

        @staticmethod
        def backward(ctx, grad_out):
            d_input = lltm_cpp.backward(grad_out.contiguous(), *ctx.saved_variables, filename, function)
            return d_input[0]

    return anon
