from sklearn.feature_selection import VarianceThreshold
import numpy as np

def variance_selection(data, y=None):
    selector = VarianceThreshold()
    selector.fit(data)
    
    return  selector.variances_
