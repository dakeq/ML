#  -*- coding:utf-8 -*-

""" \
 """
from WuEnDa.ex1.featureNormalize import featureNormalize
from WuEnDa.ex1.gradientDescent import gradientDescent
from WuEnDa.ex1.normalEqn import normalEqn

__author__ = 'Dake'

import numpy as np
import matplotlib.pyplot as plt


# %% Machine Learning Online Class
# %  Exercise 1: Linear regression with multiple variables
# %
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the
# %  linear regression exercise.
# %
# %  You will need to complete the following functions in this
# %  exericse:
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
# %  For this part of the exercise, you will need to change some
# %  parts of the code below for various experiments (e.g., changing
# %  learning rates).
# %
#
# %% Initialization
if __name__ == '__main__':
    # %% ================ Part 1: Feature Normalization ================
    #
    # %% Clear and Close Figures
    # clear ; close all; clc
    #
    print('Loading data ...\n')

    # %% Load Data
    x1, x2, y = np.loadtxt(".\\ex1data2.txt", delimiter=",", unpack=True)
    m = y.size
    X = np.append(x1.reshape(-1, 1), x2.reshape(-1, 1), axis=1)
    y = y.reshape((-1, 1))

    # % Print out some data points
    print('First 10 examples from the dataset: \n')
    print(' x = [%.0f %.0f], y = %.0f \n', X[1:10, :], '\n', y[1:10, :])

    print('Program paused. Press enter to continue.\n')

    # % Scale features and set them to zero mean
    print('Normalizing Features ...\n')

    X, mu, sigma = featureNormalize(X)

    # % Add intercept term to X
    X = np.append(np.ones((m, 1)), X, axis=1)
    print(X)

    # %% ================ Part 2: Gradient Descent ================
    #
    # % ====================== YOUR CODE HERE ======================
    # % Instructions: We have provided you with the following starter
    # %               code that runs gradient descent with a particular
    # %               learning rate (alpha).
    # %
    # %               Your task is to first make sure that your functions -
    # %               computeCost and gradientDescent already work with
    # %               this starter code and support multiple variables.
    # %
    # %               After that, try running gradient descent with
    # %               different values of alpha and see which one gives
    # %               you the best result.
    # %
    # %               Finally, you should complete the code at the end
    # %               to predict the price of a 1650 sq-ft, 3 br house.
    # %
    # % Hint: By using the 'hold on' command, you can plot multiple
    # %       graphs on the same figure.
    # %
    # % Hint: At prediction, make sure you do the same feature normalization.
    # %
    #
    # fprintf('Running gradient descent ...\n');
    #
    # % Choose some alpha value
    alpha = 0.1
    num_iters = 400
    #
    # % Init Theta and Run Gradient Descent
    theta = np.zeros((3, 1))
    theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

    # % Plot the convergence graph
    # figure;
    plt.plot(np.arange(0, J_history.size), J_history, '-b')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()

    # % Display gradient descent's result
    print('Theta computed from gradient descent: \n')
    print(' %f \n', theta)
    print('\n')
    #
    # % Estimate the price of a 1650 sq-ft, 3 br house
    # % ====================== YOUR CODE HERE ======================
    # % Recall that the first column of X is all-ones. Thus, it does
    # % not need to be normalized.
    normEstimate = np.append([1], ([1650, 3]-mu)/sigma)
    print(normEstimate)
    price = np.matmul(normEstimate, theta) # % You should change this
    #
    #
    # % ============================================================
    #
    print(['Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n'], price);

    print('Program paused. Press enter to continue.\n')


    # %% ================ Part 3: Normal Equations ================
    #
    print('Solving with normal equations...\n');

    # % ====================== YOUR CODE HERE ======================
    # % Instructions: The following code computes the closed form
    # %               solution for linear regression using the normal
    # %               equations. You should complete the code in
    # %               normalEqn.m
    # %
    # %               After doing so, you should complete this code
    # %               to predict the price of a 1650 sq-ft, 3 br house.
    # %
    #
    # %% Load Data
    x1, x2, y = np.loadtxt(".\\ex1data2.txt", delimiter=",", unpack=True)
    m = y.size
    X = np.concatenate((np.ones((m, 1)), x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)
    y = y.reshape((-1, 1))

    # % Calculate the parameters from the normal equation
    theta = normalEqn(X, y)
    #
    # % Display normal equation's result
    print('Theta computed from the normal equations: \n')
    print(' %f \n', theta)
    print('\n')
    #
    #
    # % Estimate the price of a 1650 sq-ft, 3 br house
    # % ====================== YOUR CODE HERE ======================
    price = np.matmul([1, 1650, 3], theta) # % You should change this
    #
    #
    # % ============================================================
    #
    print(['Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f\n'], price)

