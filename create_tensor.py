import torch
import torch.nn as nn

'''
根据指定size返回一个tensor，这个tensor里都是uninitialized data
uninitialized data但含义就是这个data都是memory block当中一些default value或者其他运算的结果，储存在memory block中
'''
x = torch.empty(size=(2, 2))
print(x)
