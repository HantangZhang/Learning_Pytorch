import torch
import torch.nn as nn

'''
对于神经网络，要用到权重初始化参数，下面介绍常用的初始化函数：
nn.init.orthogonal_：对于输入的tensor（输入的就是神经网络的权重矩阵），它可以使里面数值的分布成正交初始化，
    通常用在rnn网络的参数初始化， 来解决当中解决梯度消失或者爆炸的问题。（具体为什么能解决还没有梳理）
    它输入两个参数，tensor和gain
    gain 是一个scaling factor，把数据scaling到一个指定的范围，通常gain的值由nn.init.calculate_gain获得
        nn.init.calculate_gain 输入的是一个nonlinearity函数的str名称，通常就是激活函数的名称
        它会根据指定的激活函数，返回一个推荐的gain value
    
torch.nn.init.constant_(tensor, val)：会将给定的tensor里面的值全部换成val的值，常用来初始化bias

'''

# 首先定义一层网络，下面把它称作module
layer = nn.Linear(256, 256)
# 然后进行权重初始化-weight_init，下面用到的权重初始化参数是orthogonal_
gain = nn.init.calculate_gain('relu')
nn.init.orthogonal_(layer.weight.data, gain=gain)
# 对bias初始化
nn.init.constant_(layer.bias.data, 0)

# 下面给出一个见过的构建初始化参数流程的函数
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)
