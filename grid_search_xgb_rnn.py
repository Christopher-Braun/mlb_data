from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.datasets import load_svmlight_files
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.learning_curve import validation_curve 
from sklearn.datasets import load_svmlight_files 
from sklearn.cross_validation import StratifiedKFold 
from sklearn.datasets import make_classification 
from xgboost.sklearn import XGBRegressor 
from scipy.sparse import vstack 
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

seed = 123
np.random.seed(seed)

X = pd.read_csv("MLB/X_sort.csv", encoding='latin-1')
y = pd.read_csv("MLB/y_sort.csv", encoding='latin-1', names=['Score'])
X1 = X
X = X.drop('Expected_Runs', axis = 1)

#X,y = make_classification(n_samples=1000, n_features=20, n_informative=8, n_redundant=3, n_repeated=2, random_state=seed)

cv = StratifiedKFold(y['Score'], n_folds=10, shuffle=True, random_state=seed)

'''
default_params = {
        'objective':'reg:linear',
        'max_depth':1,
        'learning_rate':0.3,
        'silent':1,
        }
n_estimators_range = np.linspace(1, 200, 10).astype('int')

train_scores, test_scores = validation_curve(
        XGBRegressor(**default_params),
        X,y,
        param_name = 'n_estimators',
        param_range = n_estimators_range,
        cv=cv,
        scoring = 'neg_mean_squared_error'
        )

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fig = plt.figure(figsize=(10, 6), dpi=100)

plt.title("Validation Curve with XGBoost (eta = 0.3)")
plt.xlabel("number of trees")
plt.ylabel("Accuracy")
plt.ylim(0, 20)

plt.plot(n_estimators_range,
             train_scores_mean,
             label="Training score",
             color="r")

plt.plot(n_estimators_range,
             test_scores_mean, 
             label="Cross-validation score",
             color="g")

plt.fill_between(n_estimators_range, 
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, 
                 alpha=0.2, color="r")

plt.fill_between(n_estimators_range,
                 test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std,
                 alpha=0.2, color="g")

plt.axhline(y=1, color='k', ls='dashed')

plt.legend(loc="best")
plt.show()

i = np.argmax(test_scores_mean)
print("Best cross-validation result ({0:.2f}) obtained for {1} trees".format(test_scores_mean[i], n_estimators_range[i]))
'''

params_grid = {
    'max_depth': [1, 2, 3],
    'n_estimators': [5, 10, 25, 50],
    'learning_rate': np.linspace(1e-16, 1, 3)
}


params_fixed = {
    'objective':'reg:linear',
    'silent': 1
}


bst_grid = GridSearchCV(
    estimator=XGBRegressor(**params_fixed, seed=seed),
    param_grid=params_grid,
    cv=cv,
    scoring='neg_mean_squared_error'
)

bst_grid.fit(X, y)
bst_grid.grid_scores_

print("Best accuracy obtained: {0}".format(bst_grid.best_score_))
print("Parameters:")
for key, value in bst_grid.best_params_.items():
    print("\t{}: {}".format(key, value))

params_dist_grid = {
    'max_depth': [1, 2, 3, 4],
    'gamma': [0, 0.5, 1],
    'n_estimators': randint(1, 1001), # uniform discrete random distribution
    'learning_rate': uniform(), # gaussian distribution
    'subsample': uniform(), # gaussian distribution
    'colsample_bytree': uniform() # gaussian distribution
}

rs_grid = RandomizedSearchCV(
    estimator=XGBRegressor(**params_fixed, seed=seed),
    param_distributions=params_dist_grid,
    n_iter=10,
    cv=cv,
    scoring='neg_mean_squared_error',
    random_state=seed
)

rs_grid.fit(X, y)
rs_grid.grid_scores_
rs_grid.best_estimator_
rs_grid.best_params_
rs_grid.best_score_

params = {'colsample_bytree': 0.028,
 'gamma': 1,
 'learning_rate': 0.1739,
 'max_depth': 1,
 'n_estimators': 791,
 'subsample': 0.4258}

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling (MUST BE APPLIED IN DIMENSIONALITY REDUCTION)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
# explained_variance - list of principle components and % of variance explained by each of them
# Run with n_components = None 1st (look at explained_variance)
from sklearn.decomposition import PCA
pca = PCA(n_components = 60)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

bst = XGBClassifier(**params).fit(X_train, y_train)
preds = bst.predict(X_test)

mae_test = np.round(abs(y_test.values - preds).mean(),decimals=2)
rmse_test = (((y_test.values - preds)**2).mean())**(1/2)
r_test = (((y_test.values-y_test.values.mean())*(preds - preds.mean())).sum()) / ((1-len(y_test))*(y_test.values.std())*(preds.std()))

def eval_metrics(y_test, y_pred):
    mae = np.round(abs(y_test.values - y_pred).mean(),decimals=2)
    rmse = (((y_test.values - y_pred)**2).mean())**(1/2)
    r = (((y_test.values-y_test.values.mean())*(y_pred - y_pred.mean())).sum()) / ((1-len(y_test))*(y_test.values.std())*(y_pred.std()))
    print('mae: {0:.2f} \nrmse: {1:.2f} \nr: {2:.2f}'.format(mae, rmse, r))
    return mae_test, rmse_test, r_test

metrics = eval_metrics(y_test, preds)


