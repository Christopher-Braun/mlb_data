import numpy as np
import pandas as pd

seed = 123
np.random.seed(seed)

X = pd.read_csv("MLB/X_sort.csv", encoding='latin-1')
y = pd.read_csv("MLB/y_sort.csv", encoding='latin-1', names=['Score'])

from constant_variables import features_top_list
X2 = X['Expected_Runs']
X = X.drop('Expected_Runs', axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
'''
# Feature Scaling (Important for high intensity computations)
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
'''
# Fitting XGBoost to the Training set
from xgboost import XGBRegressor
classifier = XGBRegressor()
classifier.fit(X_train, y_train)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier1 = XGBClassifier()
classifier1.fit(X_train, y_train)

def eval_metrics(y_test, y_pred):
    mae = np.round(abs(y_test.values - y_pred).mean(),decimals=2)
    rmse = (((y_test.values - y_pred)**2).mean())**(1/2)
    r = (((y_test.values-y_test.values.mean())*(y_pred - y_pred.mean())).sum()) / ((1-len(y_test))*(y_test.values.std())*(y_pred.std()))
    print('mae: {0:.2f} \nrmse: {1:.2f} \nr: {2:.4f}'.format(mae, rmse, r))
    return mae, rmse, r

# Predicting the Test set results
y_pred_X = classifier.predict(X_train)
y_pred_X_test = classifier.predict(X_test)
y_pred_X_test_rd = np.round(y_pred_X_test)

# Predicting the Test set results
y_pred_cls = classifier1.predict(X_train)
y_pred_cls_test = classifier1.predict(X_test)
y_pred_cls_test_rd = np.round(y_pred_cls_test)

metrics = eval_metrics(y_test, y_pred_X_test_rd)
metrics_class = eval_metrics(y_test, y_pred_cls_test_rd)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean_acc = accuracies.mean()
train_std = accuracies.std()
classifier.evals_result

feature_importance = classifier1.feature_importances_
#features = pd.Series(feature_importance, index = len(X_train[0]))
#features_sorted = pd.Series(features.sort_values())

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras import objectives
from keras import backend as K

# Initialising the ANN
regressor1 = Sequential()

# Adding the input layer and the first hidden layer with dropout (start 0.1 (10%))
regressor1.add(Dense(output_dim = 90, init = 'uniform', activation = 'relu', input_dim = len(X_train.columns)))
regressor1.add(Dropout(p=0.1))

# Adding the second hidden layer
regressor1.add(Dense(output_dim = 75, init = 'uniform', activation = 'relu'))
regressor1.add(Dropout(p=0.1))

# Adding the second hidden layer
regressor1.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu'))
regressor1.add(Dropout(p=0.1))

# Adding the third hidden layer
regressor1.add(Dense(output_dim = 45, init = 'uniform', activation = 'relu'))
regressor1.add(Dropout(p=0.1))

# Adding the output layer
regressor1.add(Dense(output_dim = 1, init = 'uniform'))

# Compiling the ANN (adam[SGD] - optimizer function to find optimal weights)
# Binary Dept Var(Binary_CrossEntropy) Dependent Var > 2 Outcomes (Categorical_CrossEntropy)
# [Accuracy] in brackets because list expected
regressor1.compile(optimizer = 'adam', loss = 'mse', metrics=['binary_crossentropy','acc'])

# Fitting the ANN to the Training set
regressor1.fit(X_train, y_train, batch_size = 35, epochs = 2000)


# Predicting the Test set results
y_pred_test = regressor1.predict(X_test)
y_pred_test_rd = np.round(y_pred_test)
y_test_index = [x for x in y_test.index]

# Predicting a new result
y_pred_train = regressor1.predict(X_train)
y_pred_train_rd = np.round(y_pred_train)
y_train_index = [x for x in y.index]

y_test_total_index = [x for x in pd.Series(y_test.index)]

expected = X2[:]
y_expected_test = expected[y_test_total_index]
y_expected_test_round = np.round(y_expected_test)

metrics_reg = eval_metrics(y_test, y_pred_test)
metrics_expect = eval_metrics(y_test, y_expected_test_round.values)

y_test_total = ((pd.DataFrame(y_pred_test_rd) + pd.DataFrame(y_pred_X_test_rd.ravel()))/2).astype(int)
metrics_total = eval_metrics(y_test, y_test_total[0].values)

from sklearn.neural_network import MLPRegressor
regressor2 = MLPRegressor(solver='adam', max_iter=1000, random_state=9)
regressor2.fit(X_train, y_train)
y_pred_mlp = regressor2.predict(X_test)
y_pred_mlp_rd = np.round(y_pred_mlp)

for i,x in enumerate(y_pred_mlp_rd):
    if x<0:
        y_pred_mlp_rd[i] = 0

#y_pred_mlp_rd = [0 for i,x in enumerate(y_pred_mlp_rd) if x<0]

test_eval = eval_metrics(y_test, y_pred_mlp_rd)

y_test_total2 = ((pd.DataFrame(y_pred_test_rd) + pd.DataFrame(y_pred_X_test_rd.ravel()) + pd.DataFrame(y_pred_mlp_rd.ravel()))/3).astype(int)
metrics_total2 = eval_metrics(y_test, y_test_total2[0].values)

from catboost import Pool, CatBoostRegressor, cv
from sklearn.metrics import accuracy_score
model = CatBoostRegressor()

# Fit model
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
#     logging_level='Verbose',  # you can uncomment this for text output
    plot=True
)

model.fit(X_train, y_train, plot=True)
# Get predictions
pred_cat = model.predict(X_test)
pred_cat_rd = np.round(pred_cat)
cat_eval = eval_metrics(y_test, pred_cat_rd)

y_test_total3 = ((pd.DataFrame(y_pred_test_rd) + pd.DataFrame(y_pred_X_test_rd.ravel()) + pd.DataFrame(y_pred_mlp_rd.ravel()) + pd.DataFrame(pred_cat_rd.ravel()))/4).astype(int)
metrics_total3 = eval_metrics(y_test, y_test_total3[0].values)

model.feature_importances_

params = {'iterations': 500, 'depth': 4, 'loss_function': 'MAE', 'eval_metric':'RMSE', 'learning_rate':0.1, 'logging_level': 'Silent'}
pool = Pool(X_train, y_train)
#scores = cv(params, pool)

metrics_expect
est1, est2, est3, est1s, est2s, est3s = [], [], [], [], [], []
estimators = [metrics, metrics_class, metrics_reg, metrics_total, test_eval, metrics_total2, cat_eval, metrics_total3]
estimator_names = ['metrics', 'metrics_class', 'metrics_reg', 'metrics_total', 'test_eval', 'metrics_total2', 'cat_eval', 'metrics_total3']
for i,est in enumerate(estimators):
    est1.append(est[0])
    est1s.append(estimator_names[i])
    est2.append(est[1])
    est2s.append(estimator_names[i])
    est3.append(est[2])
    est3s.append(estimator_names[i])
mae_total = pd.Series(est1, index = est1s, name = 'mae').sort_values()
rsme_total = pd.Series(est2, index = est2s, name = 'rsme').sort_values()
r_total = pd.Series(est3, index = est3s, name = 'r').sort_values()

y_cat_met = ((pd.DataFrame(y_pred_X_test_rd.ravel()) + pd.DataFrame(pred_cat_rd.ravel()))/2).astype(int)
cat_metrics = eval_metrics(y_test, y_cat_met[0].values)

y_cat_met_cls = ((pd.DataFrame(y_pred_X_test_rd.ravel()) + pd.DataFrame(pred_cat_rd.ravel()) + pd.DataFrame(y_pred_cls_test_rd.ravel()))/3).astype(int)
cat_metrics_cls = eval_metrics(y_test, y_cat_met_cls[0].values)

y_cat_met_cls_ratio = (pd.DataFrame((y_pred_X_test_rd*.4).ravel()) + pd.DataFrame((pred_cat_rd*.5).ravel()) + pd.DataFrame((y_pred_cls_test_rd.astype(float)*.1).ravel())).astype(int)
cat_metrics_cls_ratio = eval_metrics(y_test, y_cat_met_cls_ratio[0].values)

y_cat_met_cls_ratio = (pd.DataFrame((y_pred_X_test*.4).ravel()) + pd.DataFrame((pred_cat*.5).ravel()) + pd.DataFrame((y_pred_cls_test.astype(float)*.1).ravel())).astype(int)
cat_metrics_cls_ratio = eval_metrics(y_test, y_cat_met_cls_ratio[0].values)






