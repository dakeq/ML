#  -*- coding:utf-8 -*-
import numpy as np

from WuEnDa.ex3.sigmoid import sigmoid

""" \
 """

__author__ = 'Dake'

# [[1.  0.1 0.6 1.1]
#  [1.  0.2 0.7 1.2]
#  [1.  0.3 0.8 1.3]
#  [1.  0.4 0.9 1.4]
#  [1.  0.5 1.  1.5]]
# [ True False  True False  True]
def lrCostFunction(theta_t, X, y, lambda_t):
    theta_t = theta_t.reshape((-1, 1))
    y = np.double(y)
    m, n = X.shape
    theta_reg = theta_t.copy()
    theta_reg[0, 0] = 0
    h_theta = sigmoid(np.matmul(X, theta_t))
    J = 1/m * np.sum(-y*np.log(h_theta) - (1-y)*np.log(1-h_theta)) + lambda_t/2/m*np.sum(theta_reg**2)
    grad = 1/m * np.matmul(X.T, h_theta-y) + lambda_t/m * theta_reg
    grad = grad.flatten()
    return J, grad

if __name__ == '__main__':
    pass