#  -*- coding:utf-8 -*-

""" \
 """
import numpy as np
import scipy.optimize as opt

from WuEnDa.ex3.lrCostFunction import lrCostFunction

__author__ = 'Dake'


# [all_theta] = oneVsAll(X, y, num_labels, lambda_)
def oneVsAll(X, y, num_labels, lambda_):
    m, n = X.shape
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    all_theta = np.zeros((num_labels, n+1))
    initial_theta = np.zeros((n+1, 1))
    for i in range(num_labels):
        # result = opt.fmin_cg(f=lrCostFunction, x0=initial_theta, fprime=None, args=(X, y == 0, lambda_), maxiter=50)
        result = opt.fmin_tnc(func=lrCostFunction, x0=initial_theta, fprime=None, args=(X, y==(i+1), lambda_))
        all_theta[i, :] = result[0]
    # print(all_theta)
    return all_theta
