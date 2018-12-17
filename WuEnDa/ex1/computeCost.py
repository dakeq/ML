import numpy as np


# 损失函数
def computeCost(X, y, theta):
    m = X.shape[0]

    predictions = np.matmul(X, theta)
    sqrErrors = (predictions - y) ** 2
    J = 1 / (2 * m) * np.sum(sqrErrors)

    return J


if __name__ == '__main__':
    x, y = np.loadtxt(".\\ex1data1.txt", delimiter=',', unpack=True)
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    m = x.size

    x0 = np.zeros((m, 1))+1
    X = np.append(x0, x, axis=1)

    theta = np.zeros((2, 1))

    j = computeCostMulti(X, y, theta)
    print(j)
