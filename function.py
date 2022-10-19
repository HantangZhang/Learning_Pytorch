import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F
from torch.distributions import Categorical

'''
F.one_hot
1. 只能输入longtensor
2. 返回（tesnsor.shape, num_classes)，例如输入的tensor shape是(3,5),num_class =10, 那么输出就是(3,5,10)
one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)
'''

'''
Categorical

x = torch.zeros((1, 3))
x[0][1] = 1
m = Categorical(x)
print(m.sample())
action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
action_log = distri.log_prob(action)

'''