#  -*- coding:utf-8 -*-

""" \
 """
import numpy as np

__author__ = 'Dake'


def sigmoid(z):
    e_z = np.exp(z)
    g_z = e_z/(e_z+1)
    return g_z

if __name__ == '__main__':
    print(sigmoid(0))
    print(sigmoid(99))
    print(sigmoid([1,0,1]))