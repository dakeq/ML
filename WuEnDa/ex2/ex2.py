#  -*- coding:utf-8 -*-

""" \
 """
import numpy as np
import pandas as pd
import scipy.optimize as opt

from WuEnDa.ex2.costFunction import costFunction
from WuEnDa.ex2.plotData import plotData
from WuEnDa.ex2.plotDecisionBoundary import plotDecisionBoundary
from WuEnDa.ex2.predict import predict
from WuEnDa.ex2.sigmoid import sigmoid

__author__ = 'Dake'
#
# %% Machine Learning Online Class - Exercise 2: Logistic Regression
# %
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the logistic
# %  regression exercise. You will need to complete the following functions
# %  in this exericse:
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
#

if __name__ == '__main__':
    # %% Load Data
    # %  The first two columns contains the exam scores and the third column
    # %  contains the label.
    #
    data = pd.read_csv('.\\ex2data1.txt', header=None, names=['Exam1', 'Exam2', 'Admitted'])
    print(data.head())
    data = np.array(data)
    X = data[:, 0:2]
    y = data[:, 2].reshape(-1, 1)


    #
    # %% ==================== Part 1: Plotting ====================
    # %  We start the exercise by first plotting the data to understand the
    # %  the problem we are working with.
    #
    print(['Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n'])

    plotData(X, y)

    print('\nProgram paused. Press enter to continue.\n')
    # pause;
    #
    #
    # %% ============ Part 2: Compute Cost and Gradient ============
    # %  In this part of the exercise, you will implement the cost and gradient
    # %  for logistic regression. You neeed to complete the code in
    # %  costFunction.m
    #
    # %  Setup the data matrix appropriately, and add ones for the intercept term
    m, n = X.shape
    #
    # % Add intercept term to x and X_test
    X = np.column_stack((np.ones((m, 1)), X))
    # % Initialize fitting parameters
    initial_theta = np.zeros(n + 1)
    #
    # % Compute and display initial cost and gradient
    cost, grad = costFunction(initial_theta, X, y)

    print('Cost at initial theta (zeros): %f\n', cost)
    print('Expected cost (approx): 0.693\n')
    print('Gradient at initial theta (zeros): \n')
    print(' %f \n', grad)
    print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

    # % Compute and display cost and gradient with non-zero theta
    test_theta = [-24, 0.2, 0.2]
    # test_theta = [1, 1, 1]
    cost, grad = costFunction(test_theta, X, y)

    print('\nCost at test theta: %f\n', cost)
    print('Expected cost (approx): 0.218\n')
    print('Gradient at test theta: \n')
    print(' %f \n', grad)
    print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

    print('\nProgram paused. Press enter to continue.\n')
    # pause;

    # %% ============= Part 3: Optimizing using fminunc  =============
    # %  In this exercise, you will use a built-in function (fminunc) to find the
    # %  optimal parameters theta.
    #
    # %  Set options for fminunc
    # 在练习中，使用名为“fminunc”的Octave函数来优化给定函数的参数，以计算成本和梯度。
    # 由于我们使用的是Python，我们可以使用SciPy的优化API来做同样的事情。
    # options = optimset('GradObj', 'on', 'MaxIter', 400);
    #
    # %  Run fminunc to obtain the optimal theta
    # %  This function will return theta and the cost
    # [theta, cost] = ...
    # 	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

    # 注意：theta gradient结果均为n维特征行向量
    # func : callable ``func(x, *args)``     Function to minimize.
    # x0 : array_like   Initial estimate of minimu
    # fprime : callable ``fprime(x, *args)``, optional  Gradient of `func`.
    # args : tuple, optional     Arguments to pass to function.
    # ...
    # Returns
    #     -------
    #     x : ndarray
    #         The solution.
    #     nfeval : int
    #         The number of function evaluations评估.
    #     rc : int
    #         Return code, see below
    result = opt.fmin_tnc(func=costFunction, x0=initial_theta, fprime=None, args=(X, y))
    print(result)
    theta = result[0]
    cost, grad = costFunction(theta, X, y)

    # % Print theta to screen
    print('Cost at theta found by fminunc: %f\n', cost)
    print('Expected cost (approx): 0.203\n')
    print('theta: \n')
    print(' %f \n', theta)
    print('Expected theta (approx):\n')
    print(' -25.161\n 0.206\n 0.201\n')

    # % Plot Boundary
    plotDecisionBoundary(theta, X, y)

    print('\nProgram paused. Press enter to continue.\n')
    # pause;
    #
    # %% ============== Part 4: Predict and Accuracies ==============
    # %  After learning the parameters, you'll like to use it to predict the outcomes
    # %  on unseen data. In this part, you will use the logistic regression model
    # %  to predict the probability that a student with score 45 on exam 1 and
    # %  score 85 on exam 2 will be admitted.
    # %
    # %  Furthermore, you will compute the training and test set accuracies of
    # %  our model.
    # %
    # %  Your task is to complete the code in predict.m
    #
    # %  Predict probability for a student with score 45 on exam 1
    # %  and score 85 on exam 2
    #
    prob = sigmoid(np.dot([1, 45, 85], theta))
    print(['For a student with scores 45 and 85, we predict an admission probability of %f\n'], prob)
    print('Expected value: 0.775 +/- 0.002\n\n')

    # % Compute accuracy on our training set
    p = predict(theta, X)

    print('Train Accuracy: %f\n', np.mean(np.double(p == y)) * 100)
    print('Expected accuracy (approx): 89.0\n')
    print('\n')


