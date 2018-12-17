#  -*- coding:utf-8 -*-

""" \
 """
from functools import reduce

import numpy as np

__author__ = 'Dake'

def normalEqn(X, y):
    # theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T, X)), X.T), y)
    theta = reduce(np.matmul, [np.linalg.pinv(np.matmul(X.T, X)), X.T, y])
    return theta