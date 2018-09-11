from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X = pd.read_csv("MLB/X_sort.csv", encoding='latin-1')
y = pd.read_csv("MLB/y_sort.csv", encoding='latin-1', names=['Score'])
X2 = X['Expected_Runs']
X = X.drop('Expected_Runs', axis = 1)
#X = X.astype(float)
#y = y.astype(float)

class CatBoostRegressorInt(CatBoostRegressor):
    def predict(self, data, ntree_start=0, ntree_end=0, thread_count=1, verbose=None):
        predictions = self._predict(data, ntree_start, ntree_end, thread_count, verbose)

        # This line is the only change I did
        return np.asarray(predictions, dtype=np.int64).ravel()

clf1 = ExtraTreesRegressor(random_state=1)
clf2 = GradientBoostingRegressor()
clf3 = RandomForestRegressor()
clf4 = CatBoostRegressor()

eclf = VotingClassifier(estimators=[('et', clf1), ('gb', clf2), ('rf', clf3), ('cb', clf4)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, clf4, eclf], ['ExtraTrees', 'GradientBoosting', 'RandomForest', 'CatBoost']):
    scores = cross_val_score(clf, X, y, cv=5, scoring='neg_mean_squared_error')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

'''
Accuracy: -11.63 (+/- 0.79) [ExtraTrees]
Accuracy: -9.91 (+/- 0.53) [GradientBoosting]
Accuracy: -10.89 (+/- 0.59) [RandomForest]
Accuracy: -10.14 (+/- 0.60) [CatBoost]
Accuracy: 0.95 (+/- 0.05) [Ensemble]
'''

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling (Important for high intensity computations)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
CatBoostRegressor.predict()
model = CatBoostRegressor()
# Fit model
model.fit(X_train, y_train, plot=True)
# Get predictions
pred_cat = model.predict(X_test)
pred_cat_rd = np.round(pred_cat)

def eval_metrics(y_test, y_pred):
    mae = np.round(abs(y_test.values - y_pred).mean(),decimals=2)
    rmse = (((y_test.values - y_pred)**2).mean())**(1/2)
    r = (((y_test.values-y_test.values.mean())*(y_pred - y_pred.mean())).sum()) / ((1-len(y_test))*(y_test.values.std())*(y_pred.std()))
    print('mae: {0:.2f} \nrmse: {1:.2f} \nr: {2:.2f}'.format(mae, rmse, r))
    return mae, rmse, r

cat_eval = eval_metrics(y_test, pred_cat_rd)
