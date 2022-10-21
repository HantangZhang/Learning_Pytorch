import torch
import torch.nn as nn

'''
关于pytorch中register_buffer and register_parameter
https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
简而言之，在整个模型中，会创建很多的参数，如果有一些参数你希望
saved and restored in the state_dict, but not trained by the optimizer, you should register them as buffers.
Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.

'''

n_agent = 3
x = torch.ones((n_agent + 1, n_agent + 1))
x = torch.tril(x)
x = x.view(1, 1, n_agent + 1, n_agent + 1)

class test(nn.Module):

    def __init__(self):
        # 这个例子也给出了如何在treansformer中构建mask操作的方法
        super(test, self).__init__()
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

    def forward(self, x):
        L = 3
        print(self.mask[:, :, :3, :3].shape)
        print(self.mask)
        mask = self.mask[:, :, :L, :L] == 0
        print(111111, mask)
        y = x.masked_fill(mask, float('-inf'))
        print(y)

model = test()
x = torch.ones((2, 1, 3, 3))
model.forward(x)
