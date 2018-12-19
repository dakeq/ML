#  -*- coding:utf-8 -*-

""" \
 """
from functools import reduce

import numpy as np

from WuEnDa.ex2.sigmoid import sigmoid

__author__ = 'Dake'


def costFunction(initial_theta, X, y):
    theta = np.array(initial_theta).reshape((-1, 1))
    m, n = X.shape
    h_theta = sigmoid(np.matmul(X, theta))
    cost = 1/m * np.sum(-y*np.log(h_theta)-(1-y)*np.log(1-h_theta))
    grad = 1 / m * np.matmul(X.T, h_theta - y)
    grad = grad.flatten()
    return cost, grad


# def gradient(initial_theta, X, y):
#     theta = np.array(initial_theta).reshape((-1, 1))
#     m, n = X.shape
#     h_theta = sigmoid(np.matmul(X, theta))
#     grad = 1/m * np.matmul(X.T, h_theta-y)
#     grad = grad.flatten()
#     return grad