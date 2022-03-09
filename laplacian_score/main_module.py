import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lp_score import *
import numpy as np


def laplacian_selection(data, y = None, neighbour_size=16, t_param=2):
    L = LaplacianScore(data, neighbour_size=neighbour_size, t_param=t_param)
    return -L


if __name__ == '__main__':
    X = np.loadtxt(
        "IRIS.csv", delimiter=',')
    n_samples, n_feature = X.shape
    data = X[:, 0:n_feature-1]
    L = LaplacianScore(data, neighbour_size=16, t_param=2)
    print(-L)
    print(feature_ranking(-L))
