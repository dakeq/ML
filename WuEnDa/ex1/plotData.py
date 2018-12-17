import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y):
	plt.plot(X, y, "rx")
	plt.xlabel("Profit in $10,000s")
	plt.ylabel("Population of City in 10,000s")
	plt.show()


if __name__ == "__main__":
	x, y = np.loadtxt(".\\ex1data1.txt", delimiter=",", unpack=True)
	m = x.size