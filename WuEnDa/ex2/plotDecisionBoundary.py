#  -*- coding:utf-8 -*-

""" \
 """
import numpy as np
import matplotlib.pyplot as plt

from WuEnDa.ex2.mapFeature import mapFeature

__author__ = 'Dake'


def plotDecisionBoundary(theta, X, y):
    positive = X[np.where(y == 1)[0], 1:3]
    negative = X[np.where(y == 0)[0], 1:3]

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    plt.plot(positive[:, 0], positive[:, 1], "k+", label='Admitted', linewidth=2)
    plt.plot(negative[:, 0], negative[:, 1], "yo", label='Not Admitted', linewidth=2)

    if X.shape[1] <= 3:
        # % Only need 2 points to define a line, so choose two endpoints
        # x_1
        plot_x = np.array([min(X[:, 1])-2, max(X[:, 1])+2])

        # % Calculate the decision boundary line
        # x_2
        # -theta_2 * x_2 = theta_1 * x_1 + theta_0
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])

        # % Plot, and adjust axis for better viewing?
        plt.plot(plot_x, plot_y, label='Decision Boundary')

        # % Legend, specific for the exercise
        # legend('Admitted', 'Not admitted', 'Decision Boundary')
        # axis([30, 100, 30, 100])
    else:
        # % Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.size, v.size))
        # % Evaluate z = theta*x over the grid
        for i in range(u.size):
            for j in range(v.size):
                z[i][j] = np.dot(mapFeature(u[i], v[j]), theta)

        # 此处为什么转置？
        z = z.T   # % important to transpose z before calling contour
        # % Plot z = 0
        # % Notice you need to specify the range [0, 0]
        plt.contour(u, v, z, [0], colors='g')
    plt.legend(loc='upper right')
    plt.show()