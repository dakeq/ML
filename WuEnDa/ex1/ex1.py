#  -*- coding:utf-8 -*-

""" \
 """
from mpl_toolkits.mplot3d import Axes3D

from WuEnDa.ex1.computeCost import computeCost
from WuEnDa.ex1.gradientDescent import gradientDescent
from WuEnDa.ex1.plotData import plotData
from WuEnDa.ex1.warmUpExercise import warmUpExercise

__author__ = 'Dake'

import numpy as np
import matplotlib.pyplot as plt

# %% Machine Learning Online Class - Exercise 1: Linear Regression
#
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the
# %  linear exercise. You will need to complete the following functions
# %  in this exericse:
# %
# %     warmUpExercise.m
# %     plotData.m
# %     gradientDescent.m
# %     computeCost.m
# %     gradientDescentMulti.m
# %     computeCostMulti.m
# %     featureNormalize.m
# %     normalEqn.m
# %
# %  For this exercise, you will not need to change any code in this file,
# %  or any other files other than those mentioned above.
# %
# % x refers to the population size in 10,000s
# % y refers to the profit in $10,000s
# %
#
# %% Initialization
# clear ; close all; clc

if __name__ == '__main__':

    # %% ==================== Part 1: Basic Function ====================
    # % Complete warmUpExercise.py
    print('Running warmUpExercise ... \n')
    print('5x5 Identity Matrix: \n')

    print(warmUpExercise())

    print('Program paused. Press enter to continue.\n')

    # %% ======================= Part 2: Plotting =======================
    print('Plotting Data ...\n')
    X, y = np.loadtxt(".\\ex1data1.txt", delimiter=",", unpack=True)
    X = X.reshape((-1, 1))
    y = y.reshape((-1, 1))

    m = y.size # % number of training examples

    # % Plot Data
    # % Note: You have to complete the code in plotData.m
    # 加上x0
    # X = np.append(np.ones(m).reshape((-1, 1)), X, axis=1)
    # print(X)
    plotData(X, y)

    print('Program paused. Press enter to continue.\n');


    # %% =================== Part 3: Cost and Gradient descent ===================

    # % Add a column of ones to x
    X = np.append(np.ones((m, 1)), X, axis=1)
    theta = np.zeros((2, 1)) # % initialize fitting parameters

    # % Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    print('\nTesting the cost function ...\n')
    # % compute and display initial cost
    J = computeCost(X, y, theta)
    print('With theta = [0 ; 0]\nCost computed = %f\n', J)
    print('Expected cost value (approx) 32.07\n')

    # % further testing of the cost function
    J = computeCost(X, y, [[-1], [2]])
    print('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J)
    print('Expected cost value (approx) 54.24\n');

    print('Program paused. Press enter to continue.\n')


    print('\nRunning Gradient Descent ...\n')
    # % run gradient descent
    theta = gradientDescent(X, y, theta, alpha, iterations)[0]

    # % print theta to screen
    print('Theta found by gradient descent:\n');
    print('%f\n', theta);
    print('Expected theta values (approx)\n');
    print(' -3.6303\n  1.1664\n\n');

    # % Plot the linear fit
    # hold on; % keep previous plot visible
    plt.plot(X[:, 1], np.matmul(X, theta), '-')
    # legend('Training data', 'Linear regression')
    # ????????????????????????????
    plt.legend(['Training data', 'Linear regression'])
    plt.show()
    # hold off % don't overlay any more plots on this figure

    print(theta)
    # % Predict values for population sizes of 35,000 and 70,000
    predict1 = np.matmul([1, 3.5], theta)
    print('For population = 35,000, we predict a profit of %f\n', predict1*10000)
    predict2 = np.matmul([1, 7], theta)
    print('For population = 70,000, we predict a profit of %f\n', predict2*10000);

    print('Program paused. Press enter to continue.\n');

    # %% ============= Part 4: Visualizing J(theta_0, theta_1) =============
    print('Visualizing J(theta_0, theta_1) ...\n')

    # % Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100);
    theta1_vals = np.linspace(-1, 4, 100);

    # % initialize J_vals to a matrix of 0's
    J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

    # % Fill out J_vals
    for i in range(theta0_vals.size):
        for j in range(theta1_vals.size):
          t = [[theta0_vals[i]], [theta1_vals[j]]]
          J_vals[i,j] = computeCost(X, y, t)

    # % Because of the way meshgrids work in the surf command, we need to
    # % transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals.T
    # % Surface plot
    # figure;
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(theta0_vals, theta1_vals, J_vals)
    plt.xlabel('\\theta_0')
    plt.ylabel('\\theta_1')
    plt.show()
    # surf(theta0_vals, theta1_vals, J_vals)

    #

    # % Contour plot
    # figure;
    # % Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    # contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
    plt.xlabel('\\theta_0')
    plt.ylabel('\\theta_1')
    # plt.contourf(theta0_vals, theta1_vals, J_vals)
    plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
    # hold on;
    plt.plot(theta[0], theta[1], 'rx');
    plt.show()
