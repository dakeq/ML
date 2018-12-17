#  -*- coding:utf-8 -*-

""" \
 """
import numpy as np

__author__ = 'Dake'


def featureNormalize(X):
    X_norm = X
    featureN = X.shape[1]
    mu = np.zeros(featureN)
    sigma = np.zeros(featureN)
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    # 没有考虑sigma为0
    X_norm = (X_norm-mu)/sigma
    return X_norm, mu, sigma

if __name__ == '__main__':
    X = np.array([[0, 2, 10], [1, 4, 11], [-1, 6, 9]])
    X = featureNormalize(X)
    print(X)