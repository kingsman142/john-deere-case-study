from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from utils import *

df = load_dataset()
X, y = split_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, shuffle = True)

clfs = []
alphas = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
for i in range(len(alphas)):
    clf = Lasso(alpha = alphas[i])
    clfs.append(clf)

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test) # r^2 score
    print("Score of clf with alpha = {} : {}".format(alphas[i], score))

for clf in clfs:
    print(clf.intercept_)
    print(clf.coef_)
