#  -*- coding:utf-8 -*-

""" \
 """
import numpy as np

from scipy.io import loadmat

from WuEnDa.ex3.displayData import displayData
from WuEnDa.ex3.lrCostFunction import lrCostFunction
from WuEnDa.ex3.oneVsAll import oneVsAll
from WuEnDa.ex3.predictOneVsAll import predictOneVsAll

__author__ = 'Dake'

# %% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
#
# %  Instructions
# %  ------------
# %
# %  This file contains code that helps you get started on the
# %  linear exercise. You will need to complete the following functions
# %  in this exericse:
# %
# %     lrCostFunction.m (logistic regression cost function)
# %     oneVsAll.m
# %     predictOneVsAll.m
# %     predict.m
# %
# %  For this exercise, you will not need to change any code in this file,
# %  or any other files other than those mentioned above.
# %
#
# %% Initialization
# clear ; close all; clc
#

if __name__ == '__main__':

    # %% Setup the parameters you will use for this part of the exercise
    input_layer_size = 400   # % 20x20 Input Images of Digits
    num_labels = 10          # % 10 labels, from 1 to 10
    #                           % (note that we have mapped "0" to label 10)
    #
    # %% =========== Part 1: Loading and Visualizing Data =============
    # %  We start the exercise by first loading and visualizing the dataset.
    # %  You will be working with a data set that contains handwritten digits.
    # %
    #
    # % Load Training Data
    print('Loading and Visualizing Data ...\n')

    # load('ex3data1.mat'); % training data stored in arrays X, y
    # 数据文件采用MATLAB原生的格式，不能被pandas自动识别
    data = loadmat('.\\ex3data1.mat')
    X = data['X']
    y = data['y']
    m = y.shape[0]
    print(X)
    print(y)
    # % Randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    print(rand_indices)
    sel_X = X[rand_indices[0:100], :]
    sel_y = y[rand_indices[0:100], :]
    print(sel_X)
    print(sel_y)
    displayData(sel_X, int(np.round(np.sqrt(X.shape[1]))))

    print('Program paused. Press enter to continue.\n')
    # pause;
    #
    # %% ============ Part 2a: Vectorize Logistic Regression ============
    # %  In this part of the exercise, you will reuse your logistic regression
    # %  code from the last exercise. You task here is to make sure that your
    # %  regularized logistic regression implementation is vectorized. After
    # %  that, you will implement one-vs-all classification for the handwritten
    # %  digit dataset.
    # %
    #
    # % Test case for lrCostFunction
    print('\nTesting lrCostFunction() with regularization')
    #
    theta_t = np.array([-2, -1, 1, 2])
    X_t = np.concatenate((np.ones((5,1)), np.arange(1, 16).reshape((3, 5)).T/10), axis=1)
    print(X_t)
    y_t = np.array([1, 0, 1, 0, 1]) >= 0.5
    y_t = y_t.reshape((-1, 1))
    print(y_t)
    lambda_t = 3
    # J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
    J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
    print('\nCost: %f\n', J)
    print('Expected cost: 2.534819\n')
    print('Gradients:\n')
    print(' %f \n', grad)
    print('Expected gradients:\n')
    print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

    print('Program paused. Press enter to continue.\n')
    # pause;
    # %% ============ Part 2b: One-vs-All Training ============
    print('\nTraining One-vs-All Logistic Regression...\n')
    #
    lambda_ = 0.1
    all_theta = oneVsAll(X, y, num_labels, lambda_)

    print('Program paused. Press enter to continue.\n')
    # pause;
    #
    #
    # %% ================ Part 3: Predict for One-Vs-All ================

    pred = predictOneVsAll(all_theta, X)
    # print(pred)

    print('\nTraining Set Accuracy: %f\n', np.mean(np.double(pred.reshape((-1, 1)) == y)) * 100)

    print("You should see that the training set accuracy is about 94.9% ")
    #
