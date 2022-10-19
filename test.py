import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F


action = torch.rand((3, 5, 1))
# print(action)
action = action.squeeze(-1).long()
print(action.shape)

b = F.one_hot(action, num_classes=10)
print(b.shape)