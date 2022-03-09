import pandas as pd
import numpy as np

from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, KFold
from sklearn import set_config

from sklearn.metrics import accuracy_score, make_scorer, matthews_corrcoef, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectFromModel, f_classif, RFE
from sklearn.feature_selection import SelectFdr, chi2
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def data_split(data):

    transcriptome_data = data["transcriptome_data"]
    clinical_data = data["clinical_data"]
    clinical_data = clinical_data.loc[:, ["gender", "age"]]
    merged_data = pd.merge(transcriptome_data,
                           clinical_data,
                           left_index=True,
                           right_index=True,
                           how="left")
    label = np.array([
        1 if item.startswith("ITP") else 0
        for item in merged_data.index.tolist()
    ])

    X_train, X_test, y_train, y_test = train_test_split(merged_data,
                                                        label,
                                                        test_size=0.2,
                                                        stratify=label,
                                                        random_state=0)

    return X_train, X_test, y_train, y_test


class FeatureDecorrelated(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold
        self.decorrelated_features = None

    def fit(self, X, y=None):
        """All SciKit-Learn compatible transformers and classifiers have the
        same interface. `fit` always returns the same object."""
        data = pd.DataFrame(X)
        corrmat = data.corr()
        corrmat = corrmat.abs().unstack()  # absolute value of corr coef
        corrmat = corrmat.sort_values(ascending=False)
        corrmat = corrmat[corrmat >= self.threshold]
        corrmat = corrmat[corrmat < 1]  # remove the digonal
        corrmat = pd.DataFrame(corrmat).reset_index()
        corrmat.columns = ['feature1', 'feature2', 'corr']

        grouped_feature_ls = []
        correlated_groups = []

        for feature in corrmat.feature1.unique():
            if feature not in grouped_feature_ls:

                # find all features correlated to a single feature
                correlated_block = corrmat[corrmat.feature1 == feature]
                grouped_feature_ls = grouped_feature_ls + list(
                    correlated_block.feature2.unique()) + [feature]

                # append the block of features to the list
                correlated_groups.append(correlated_block)

        self.decorrelated_features = set(data.columns.tolist())

        for g in correlated_groups:
            self.decorrelated_features -= set(g["feature2"].values)

        self.decorrelated_features = list(self.decorrelated_features)

        return self

    def transform(self, X):
        return X[:, self.decorrelated_features]


class AllFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def get_feature_preprocessor(rna_features):
    rna_preprocessor = Pipeline(
        steps=[('rna_scaler', StandardScaler()),
               ("rna_selector",
                SelectFromModel(LogisticRegression(
                    penalty='l1', solver="liblinear", random_state=0),
                                max_features=50)
                ), ("rna_decorrelated", FeatureDecorrelated(threshold=0.9))])

    feature_preprocessor = ColumnTransformer(
        transformers=[('rna_preprocessor', rna_preprocessor,
                       rna_features), ('age_scaler', StandardScaler(),
                                       ['age']),
                      ('sex_encoder', OneHotEncoder(handle_unknown='ignore'),
                       ['gender'])])

    return feature_preprocessor


def get_feature_union():
    feature_union_pipleine = FeatureUnion([
        ('all_seletcted_features', AllFeatureSelector()),
        ('seletcted_features_lda', LinearDiscriminantAnalysis()),
        ("seletcted_features_svd", PCA(n_components=2)),
    ])

    return feature_union_pipleine


def get_pipeline(rna_features, selector=None):
    if selector is None:
        # default feature selection algorithm is lasso
        selector = SelectFromModel(LogisticRegression(penalty='l1',
                                                      solver="liblinear",
                                                      random_state=0),
                                   max_features=50)

    pipeline = Pipeline(
        steps=[("preprocessor",
                ColumnTransformer(transformers=[(
                    'rna_preprocessor',
                    Pipeline(
                        steps=[('rna_scaler',
                                StandardScaler()), ("rna_selector", selector),
                               ("rna_decorrelated",
                                FeatureDecorrelated(threshold=0.9))]),
                    rna_features), ('age_scaler', StandardScaler(), ['age']),
                                                ('sex_encoder',
                                                 OneHotEncoder(
                                                     handle_unknown='ignore'),
                                                 ['gender'])])
                ), ("feature_union", get_feature_union()
                    ), ("classifier",
                        RandomForestClassifier(random_state=42))])

    return pipeline


def get_ifs_pipeline(rna_features, features):
    ifs_pipeline = Pipeline(
        steps=[("preprocessor",
                ColumnTransformer(transformers=[(
                    "IFS", StandardScaler(),
                    features), ('age_scaler', StandardScaler(), ['age']),
                                                ('sex_encoder',
                                                 OneHotEncoder(
                                                     handle_unknown='ignore'),
                                                 ['gender'])])),
               ("feature_union",
                FeatureUnion([
                    ('all_seletcted_features', AllFeatureSelector()),
                    ('seletcted_features_lda', LinearDiscriminantAnalysis()),
                    ("seletcted_features_svd", PCA(n_components=2)),
                ])), ("classifier", RandomForestClassifier(random_state=42))])

    return ifs_pipeline


if __name__ == "__main__":
    with open("data.pkl", "rb") as f:
        data = pkl.load(f)

    X_train, X_test, y_train, y_test = data_split(data)

    rna_features = X_train.columns[:-2].tolist()

    pipeline = get_pipeline(rna_features)

    pipeline.fit(X_train, y_train)

    print("baseline: {}".format(pipeline.score(X_test, y_test)))
