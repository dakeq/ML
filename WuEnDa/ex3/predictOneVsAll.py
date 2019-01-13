#  -*- coding:utf-8 -*-

""" \
 """
import numpy as np
__author__ = 'Dake'

def predictOneVsAll(all_theta, X):
    m, n = X.shape
    num_labels = all_theta.shape[0]
    p = np.zeros((m, 1))
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    all_p = np.matmul(X, all_theta.T)
    p = np.argmax(all_p, axis=1) + 1
    return p