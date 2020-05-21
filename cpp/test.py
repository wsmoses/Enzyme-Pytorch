import torch
import math
import time
import lltm_cpp
import numpy as np
lltm_cpp.forward

# Our module!
import lltm

a = torch.from_numpy(np.array([[1,2,3,4.]], dtype=np.float32))
a.requires_grad_(True)
b = lltm.Enzyme("test.cpp", "f").apply(a).sum()
b.backward()
print(a.grad)
