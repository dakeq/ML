#  -*- coding:utf-8 -*-

""" \
 """
import numpy as np

__author__ = 'Dake'

# function out = mapFeature(X1, X2)
# % MAPFEATURE Feature mapping function to polynomial features
# %
# %   MAPFEATURE(X1, X2) maps the two input features
# %   to quadratic features used in the regularization exercise.
# %
# %   Returns a new feature array with more features, comprising of
# %   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
# %
# %   Inputs X1, X2 must be the same size
# %

def mapFeature(x1, x2):

    degree = 6
    # 共估计额外加1+2+3+4+5+6+7=28列
    out = np.ones((x1.size, sum(range(degree+2))))
    low_num = 0
    for i in range(1, degree+1):
        for j in range(i+1):
            # 不知有动态扩展函数吗 ？这里先扩展在赋值
            low_num += 1
            out[:, low_num] = (x1**(i-j)) * (x2**j)
    return out

if __name__ == '__main__':
    X = np.array([[0], [0]])
    res = mapFeature(X[0], X[1])
    print(res)