import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator,TransformerMixin

from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold,KFold
from sklearn import set_config

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


from sklearn.metrics import accuracy_score,make_scorer,matthews_corrcoef,confusion_matrix,roc_auc_score
from sklearn.feature_selection import SelectFromModel,f_classif,RFE
from sklearn.feature_selection import SelectFdr, chi2

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from scipy.stats import ttest_ind_from_stats

def ttest_score_func(X, y):

    labels = np.unique(y)
    p_ = X[y == labels[0],:] 
    n_ = X[y == labels[1],:] 

    p_mean, n_mean = p_.mean(axis = 0), n_.mean(axis = 0)
    p_std, n_std = p_.std(axis = 0), n_.std(axis = 0)

    t_value, p_value = ttest_ind_from_stats(
        p_mean, p_std, p_.shape[0], n_mean, n_std, n_.shape[0])
    
    return (t_value,p_value)


def sn_func(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)

def sp_func(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn+fp)