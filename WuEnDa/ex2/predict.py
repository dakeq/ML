#  -*- coding:utf-8 -*-

""" \
 """
import numpy as np

from WuEnDa.ex2.sigmoid import sigmoid

__author__ = 'Dake'


def predict(theta, X):
    theta = np.array(theta).reshape((-1, 1))
    pred = sigmoid(np.matmul(X, theta))
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    return pred
