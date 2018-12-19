#  -*- coding:utf-8 -*-

""" \
 """
import numpy as np
import pandas as pd
import scipy.optimize as opt

from WuEnDa.ex2.costFunctionReg import costFunctionReg
from WuEnDa.ex2.mapFeature import mapFeature
from WuEnDa.ex2.plotData import plotData
from WuEnDa.ex2.plotDecisionBoundary import plotDecisionBoundary
from WuEnDa.ex2.predict import predict

__author__ = 'Dake'

# %% Machine Learning Online Class - Exercise 2: Logistic Regression
# %
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the second part
# %  of the exercise which covers regularization with logistic regression.
# %
# %  You will need to complete the following functions in this exericse:
# %
# %     sigmoid.m
# %     costFunction.m
# %     predict.m
# %     costFunctionReg.m
# %
# %  For this exercise, you will not need to change any code in this file,
# %  or any other files other than those mentioned above.
# %
#
# %% Initialization
# clear ; close all; clc
if __name__ == '__main__':
    # %% Load Data
    # %  The first two columns contains the X values and the third column
    # %  contains the label (y).
    #
    data = pd.read_csv('.\\ex2data2.txt', header=None, names=['Exam1', 'Exam2', 'Admitted'])
    print(data.head())
    data = np.array(data)
    X = data[:, 0:2]
    y = data[:, 2].reshape((-1, 1))
    plotData(X, y)
    #
    # % Put some labels
    # hold on;
    #
    # % Labels and Legend
    # xlabel('Microchip Test 1')
    # ylabel('Microchip Test 2')
    #
    # % Specified in plot order
    # legend('y = 1', 'y = 0')
    # hold off;
    #
    #
    # %% =========== Part 1: Regularized Logistic Regression ============
    # %  In this part, you are given a dataset with data points that are not
    # %  linearly separable. However, you would still like to use logistic
    # %  regression to classify the data points.
    # %
    # %  To do so, you introduce more features to use -- in particular, you add
    # %  polynomial features to our data matrix (similar to polynomial
    # %  regression).
    # %
    #
    # % Add Polynomial Features
    #
    # % Note that mapFeature also adds a column of ones for us, so the intercept
    # % term is handled
    X = mapFeature(X[:, 0], X[:, 1])
    print(X)

    # % Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1])
    #
    # % Set regularization parameter lambda to 1
    lambda_ = 1

    # % Compute and display initial cost and gradient for regularized logistic
    # % regression
    cost, grad = costFunctionReg(initial_theta, X, y, lambda_)

    print('Cost at initial theta (zeros): %f\n', cost)
    print('Expected cost (approx): 0.693\n')
    print('Gradient at initial theta (zeros) - first five values only:\n')
    print(' %f \n', grad[0:5])
    print('Expected gradients (approx) - first five values only:\n')
    print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

    print('\nProgram paused. Press enter to continue.\n')
    # pause;
    #
    # % Compute and display cost and gradient
    # % with all-ones theta and lambda = 10
    test_theta = np.ones(X.shape[1])

    cost, grad = costFunctionReg(test_theta, X, y, 10)

    print('\nCost at test theta (with lambda = 10): %f\n', cost)
    print('Expected cost (approx): 3.16\n')
    print('Gradient at test theta - first five values only:\n')
    print(' %f \n', grad[0:5])
    print('Expected gradients (approx) - first five values only:\n')
    print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

    print('\nProgram paused. Press enter to continue.\n')
    # pause;

    #
    # %% ============= Part 2: Regularization and Accuracies =============
    # %  Optional Exercise:
    # %  In this part, you will get to try different values of lambda and
    # %  see how regularization affects the decision coundart
    # %
    # %  Try the following values of lambda (0, 1, 10, 100).
    # %
    # %  How does the decision boundary change when you vary lambda? How does
    # %  the training set accuracy vary?
    # %
    #
    # % Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1])
    #
    # % Set regularization parameter lambda to 1 (you should vary this)
    lambda_ = 1
    result = opt.fmin_tnc(func=costFunctionReg, x0=initial_theta, fprime=None, args=(X, y, lambda_))
    print(result)
    theta = result[0]
    cost, grad = costFunctionReg(theta, X, y, lambda_)
    print(cost)
    print(grad)

    # % Plot Boundary
    plotDecisionBoundary(theta, X, y)
    # hold on;
    # title(sprintf('lambda = %g', lambda))
    #
    # % Labels and Legend
    # xlabel('Microchip Test 1')
    # ylabel('Microchip Test 2')
    #
    # legend('y = 1', 'y = 0', 'Decision boundary')
    # hold off;

    # % Compute accuracy on our training set
    p = predict(theta, X)

    print('\nTrain Accuracy: %f\n', np.mean(np.double(p == y)) * 100)
    print('Expected accuracy (with lambda = 1): 83.1 (approx)\n')

