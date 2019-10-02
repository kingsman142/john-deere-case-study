from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_columns", 100)

from utils import *

df = load_dataset()
X, y = split_data(df)

print("Performing PCA analysis")
n_component_values = [10, 15, 20, 30]
pca_clfs = []
for n_component in n_component_values:
    pca = PCA(n_components = n_component)
    pca.fit(X)
    pca_clfs.append(pca)
    #print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))
#X = pca_clfs[2].transform(X) # take n_components = 20 and transform the data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, shuffle = True)
lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
y_test = lab_enc.fit_transform(y_test)

'''c_values = [0.01, 0.1, 0.5, 1.0]
svm_clfs = []
for c in c_values:
    print("Training with c = {}...".format(c))
    clf = svm.SVC(C = c, decision_function_shape = 'ovr', kernel = 'linear')
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    svm_clfs.append(clf)
    print("C: {}, Num support vectors: {}, Score: {}".format(c, clf.n_support_, score))
    #print(clf.intercept_)
    #print(clf.coef_)'''

'''dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
dt_score = dt_clf.score(X_test, y_test)
print("Decision tree score: {}".format(dt_score))
print(dt_clf.feature_importances_)'''

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_score = rf_clf.score(X_test, y_test)
print("Random forest score: {}".format(rf_score))
print(rf_clf.feature_importances_)
rf_pred = rf_clf.predict(X_test)
mat = confusion_matrix(y_test, rf_pred)
heatmap = sb.heatmap(mat, cmap = "YlGnBu")
plt.xlabel("Predicted")
plt.ylabel("Ground-truth")
plt.show()

'''lr_penalties = ['l1', 'l2']
for lr_penalty in lr_penalties:
    lr_clf = LogisticRegression(penalty = lr_penalty, multi_class = 'ovr')
    lr_clf.fit(X_train, y_train)
    score = lr_clf.score(X_test, y_test)
    print("Logistic regression score with penalty = {}: {}".format(lr_penalty, score))
    print(lr_clf.coef_)'''
