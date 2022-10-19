import torch
import torch.nn as nn

'''
torch.tril(input, diagonal)
将input按对角线进行处理，对角线上面的元素都变成了0，对角线不变
diagonal正数指的是对角线上面几个斜杠不做处理
如果是负数指的是下面几个斜杠不做处理
如果是0，则对角线不变

x = torch.ones((5,5))
y = torch.tril(x, 2)
tensor([[1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]])

'''

x = torch.ones((5,5))
y = torch.tril(x, 2)

print(y)