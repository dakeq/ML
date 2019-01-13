#  -*- coding:utf-8 -*-

""" \
 """
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Dake'

# function [h, display_array] = displayData(X, example_width)
# %DISPLAYDATA Display 2D data in a nice grid
# %   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
# %   stored in X in a nice grid. It returns the figure handle h and the
# %   displayed array if requested.#

# % Set example_width automatically if not passed in
# if ~exist('example_width', 'var') || isempty(example_width)
# 	example_width = round(sqrt(size(X, 2)));
# end#


def displayData(X, example_width):
    # % Gray Image
    # colormap(gray);#

    # % Compute rows=100, cols=400
    if len(X.shape) == 1:
        m = 1
        n, = X.shape
    else:
        m, n = X.shape
    example_height = n // example_width

    # % Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # % Between images padding
    pad = 1

    # % Setup blank display
    height = pad + display_rows * (example_height + pad)
    width = pad + display_cols * (example_width + pad)
    display_array = -np.ones((int(height), int(width)))
    # print(display_array)

    # % Copy each example into a patch on the display array
    # 当前行
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
    # 		% Copy the patch
    #
    # 		% Get the max value of the patch
            if len(X.shape) == 1:
                curr_grid = X
            else:
                curr_grid = X[curr_ex, :]
            max_val = np.max(np.abs(curr_grid))
            curr_grid = curr_grid.reshape((example_height, example_width))
    # 		display_array(pad + j * (example_height + pad) + (0:example_height-1), ...
    # 		              pad + i * (example_width + pad) + (0:example_width-1)) = ...
    # 						reshape(X(curr_ex, :), example_height, example_width) / max_val;
            display_array[pad + j * (example_height + pad) + np.arange(0, example_height),
                          (pad + i * (example_width + pad)): (pad + i * (example_width + pad) + example_width)] \
                = curr_grid.T / max_val
            curr_ex += 1
        if curr_ex > m:
            break

    # % Display Image
    # h = imagesc(display_array, [-1 1]);#
    plt.imshow(display_array, cmap=plt.cm.gray)
    plt.show()
    # % Do not show axis
    # axis image off#

    # drawnow;#

    # end
