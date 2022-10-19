import torch
import torch.nn as nn

'''
两种构建神经网络的方式
1. nn.Sequential()



tip1:假设有3个block，类似
layer = [layer1, layer2, layer3]
那么想让他们按次序运行，可以用*号
layer = nn.Sequential(* [layer1, layer2, layer3])

transformer中的应用
self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
'''







'''
进行layer normalization
nn.LayerNorm(normalized_shape)
normalized_shape的概念
For example, if normalized_shape is (3, 5) (a 2-dimensional shape), 
the mean and standard-deviation are computed over the last 2 dimensions of the input

'''
x = torch.randn(64, 3, 2, 2)
obs_dim = (3, 2, 2)

layer = nn.LayerNorm(obs_dim)
y = layer(x)

# 或者
nn.Sequential(nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, 64),
            nn.GELU()
            )
