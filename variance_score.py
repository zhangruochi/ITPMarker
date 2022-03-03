from sklearn.feature_selection import VarianceThreshold
import numpy as np

def variance_selection(data, y=None):
    selector = VarianceThreshold()
    selector.fit(data)
    
    return  selector.variances_


if __name__ == '__main__':
    X = np.loadtxt(
        "/data/zhangruochi/ITP-code/laplacian_score/IRIS.csv", delimiter=',')
    n_samples, n_feature = X.shape
    data = X[:, 0:n_feature-1]
    L = variance_selection(data, y=None)
    print(L)

