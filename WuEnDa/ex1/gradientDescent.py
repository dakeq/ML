import numpy as np

from WuEnDa.ex1.computeCost import computeCost


# 梯度下降
def gradientDescent(X, y, theta, alpha, iterations):

    m = y.size
    J_history = np.zeros((iterations, 1))
    for i in range(iterations):
        errors = np.matmul(X, theta) - y
        Delta = np.matmul(X.T, errors)
        theta = theta - alpha / m * Delta
        J_history[i] = computeCost(X, y, theta)
    return theta, J_history


if __name__ == '__main__':
    pass
