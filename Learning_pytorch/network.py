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
2. 
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

'''
3. model.eval() and model.train()
在pytorch中，对于构建好的模型，其中的一些层在training和inference行为是不同的
例如dropout layer和BatchNorm Layers等等
所以在模型评估的时候，可以利用model.eval()来关掉这些层，他们在inference的时候将不再起作用
它通常和torch.no_grad()配合使用
# evaluate model:
model.eval()

with torch.no_grad():
    ...
    out_data = model(data)
    ...
    
但要记住在训练的时候记得把模型打开
model.train()
'''
