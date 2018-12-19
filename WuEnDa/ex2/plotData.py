#  -*- coding:utf-8 -*-

""" \
 """

import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Dake'


def plotData(X, y):

    positive = X[np.where(y == 1)[0], :]
    negative = X[np.where(y == 0)[0], :]

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    plt.plot(positive[:, 0], positive[:, 1], "k+", label='Admitted', linewidth=2)
    plt.plot(negative[:, 0], negative[:, 1], "yo", label='Not Admitted', linewidth=2)
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]]).reshape((4, 2))
    y = np.array([1, 0, 1, 1]).reshape((-1, 1))
    plotData(X, y)