#  -*- coding:utf-8 -*-

""" \
 """
import numpy as np

from WuEnDa.ex2.sigmoid import sigmoid

__author__ = 'Dake'

def costFunctionReg(initial_theta, X, y, lambda_):
    theta = np.array(initial_theta).reshape((-1, 1))
    # 此处数组的浅拷贝出错，浪费我大量时间，要用深拷贝。。。
    theta_reg = theta.copy()
    theta_reg[0, 0] = 0
    m, n = X.shape
    h_theta = sigmoid(np.matmul(X, theta))
    cost = 1/m * np.sum(-y*np.log(h_theta)-(1-y)*np.log(1-h_theta)) + lambda_/2/m*np.sum(theta_reg**2)
    grad = 1 / m * np.matmul(X.T, h_theta - y) + lambda_/m * theta_reg
    grad = grad.flatten()
    return cost, grad