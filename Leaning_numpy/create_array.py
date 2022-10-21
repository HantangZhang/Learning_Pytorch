import numpy as np

'''
np.indices
这个函数比较复杂，它返回一个array的索引
例如我有一个shape是(2,3)的索引，那么每一个元素所处的位置，可以用2个数字代表出来，例如
第一行第一列的数字就是(0, 0)，第一行第三列的数字就是(0,2)
那么就可以这样用
x = np.indices((2,3))
x.shape = (2, 2, 3)
row, cols = np.indices((2,3))
row.shape = (2,3)
cols = (2, 3)
怎么理解呢，row第一个元素是0，cols第一个元素是0，组成就是(0, 0)代表就是array中第一个元素

https://stackoverflow.com/questions/32271331/can-anybody-explain-me-the-numpy-indices
看第二个答案
'''