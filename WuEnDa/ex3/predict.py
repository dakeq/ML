#  -*- coding:utf-8 -*-

""" \
 """
import numpy as np

__author__ = 'Dake'

def predict(theta1, theta2, X):
    if len(X.shape) == 1:
        m = 1
        n, = X.shape
        X = np.concatenate((np.ones(m), X), axis=0)
    else:
        m, n = X.shape
        X = np.concatenate((np.ones((m, 1)), X), axis=1)
    num_labels = theta2.shape[0]
    p = np.zeros((m, 1))

    hidden_layer = np.matmul(X, theta1.T)
    if len(X.shape) == 1:
        hidden_layer = np.concatenate((np.ones(m), hidden_layer), axis=0)
    else:
        hidden_layer = np.concatenate((np.ones((m, 1)), hidden_layer), axis=1)
    output_layer = np.matmul(hidden_layer, theta2.T)
    if len(X.shape) == 1:
        p = np.argmax(output_layer) + 1
    else:
        p = np.argmax(output_layer, axis=1) + 1
    return p