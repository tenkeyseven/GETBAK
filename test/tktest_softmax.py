from platform import node
from numpy.core.fromnumeric import ptp
import torch.nn
import torch

m = torch.nn.Softmax(dim=None)

a = torch.FloatTensor([[1,2],[1,1]])

b = m(a)

print(b)
