#  -*- coding:utf-8 -*-

""" \
 """
import numpy as np

from WuEnDa.ex3.displayData import displayData
from WuEnDa.ex3.predict import predict

__author__ = 'Dake'

from scipy.io import loadmat

# %% Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks#

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
# %#

if __name__ == '__main__':
    # %% Initialization
    # clear ; close all; clc#

    # %% Setup the parameters you will use for this exercise
    input_layer_size = 400       # % 20x20 Input Images of Digits
    hidden_layer_size = 25       # % 25 hidden units
    num_labels = 10              # % 10 labels, from 1 to 10
    #                           % (note that we have mapped "0" to label 10)#

    # %% =========== Part 1: Loading and Visualizing Data =============
    # %  We start the exercise by first loading and visualizing the dataset.
    # %  You will be working with a dataset that contains handwritten digits.
    # %#

    # % Load Training Data
    print('Loading and Visualizing Data ...\n')

    data = loadmat('.\\ex3data1.mat')
    print(data)
    X = data['X']
    y = data['y']
    m, n = X.shape

    # % Randomly select 100 data points to display
    sel = np.random.permutation(m)
    sel = sel[1:100]
    print(sel)

    displayData(X[sel, :], 20)

    print('Program paused. Press enter to continue.\n')
    # pause;#

    # %% ================ Part 2: Loading Pameters ================
    # % In this part of the exercise, we load some pre-initialized
    # % neural network parameters.#

    print('\nLoading Saved Neural Network Parameters ...\n')

    # % Load the weights into variables Theta1 and Theta2
    data = loadmat('.\\ex3weights.mat')
    print(data)
    theta1 = data['Theta1']
    theta2 = data['Theta2']

    # %% ================= Part 3: Implement Predict =================
    # %  After training the neural network, we would like to use it to predict
    # %  the labels. You will now implement the "predict" function to use the
    # %  neural network to predict the labels of the training set. This lets
    # %  you compute the training set accuracy.#

    pred = predict(theta1, theta2, X)
    print(pred)

    print('\nTraining Set Accuracy: %f\n', np.mean(np.double(pred.reshape((-1, 1)) == y)) * 100)
    print("You should see that the accuracy is about 97.5%")

    print('Program paused. Press enter to continue.\n')
    # pause;#

    # %  To give you an idea of the network's output, you can also run
    # %  through the examples one at the a time to see what it is predicting.#

    # %  Randomly permute examples
    rp = np.random.permutation(m)

    for i in range(m):
        # % Display
        print('\nDisplaying Example Image\n')
        displayData(X[rp[i], :], 20)

        pred = predict(theta1, theta2, X[rp[i], :])
        print('\nNeural Network Prediction: %d (digit %d)\n', pred)

        # % Pause with quit option
        s = input('Paused - press enter to continue, q to exit: s')
        if s == 'q':
          break

