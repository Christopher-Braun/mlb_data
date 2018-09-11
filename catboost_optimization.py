from catboost import Pool, CatBoostRegressor, cv
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

X = pd.read_csv("MLB/X_sort.csv", encoding='latin-1')
y = pd.read_csv("MLB/y_sort.csv", encoding='latin-1', names=['Score'])
X = X.drop('Expected_Runs', axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

def eval_metrics(y_test, y_pred):
    mae = np.round(abs(y_test.values - y_pred).mean(),decimals=2)
    rmse = (((y_test.values - y_pred)**2).mean())**(1/2)
    r = (((y_test.values-y_test.values.mean())*(y_pred - y_pred.mean())).sum()) / ((1-len(y_test))*(y_test.values.std())*(y_pred.std()))
    print('mae: {0:.2f} \nrmse: {1:.2f} \nr: {2:.4f}'.format(mae, rmse, r))
    return mae, rmse, r

model = CatBoostRegressor()

# Fit model
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
#     logging_level='Verbose',  # you can uncomment this for text output
    plot=True
)

features = model.feature_importances_
feature_rank = pd.Series(features, index = list(X.columns)).sort_values(ascending=False)
pool = Pool(X, y)
pool_train = Pool(X_train, y_train)
object_importance = model.get_object_importance(pool, pool_train)
model.eval_metrics()
params = {'iterations': 500, 'depth': 10, 'loss_function': ['RMSE','MAE'], 'logging_level': 'Silent'}

params  =  {
    'iterations' : 500,
    'learning_rate': 0.1,
    'loss_function': 'RMSE',
    'eval_metric': 'R2',
    'random_seed': 42,
    'logging_level': 'Silent',
    'allow_writing_files' : True,
    'use_best_model': False
}
train_pool = Pool(X_train, y_train)
train_pool.set_baseline([[int(x)] for x in y_pred_X_train_rd])
validate_pool = Pool(X_test, y_test)
validate_pool.set_baseline([[int(x)] for x in y_pred_X_test_rd])

model = CatBoostRegressor(**params)
model.fit(train_pool, eval_set=validate_pool, plot=True)

best_model_params = params.copy()
best_model_params.update({
    'use_best_model': True
})
best_model = CatBoostRegressor(**best_model_params)
best_model.fit(train_pool, eval_set=validate_pool, plot=True);
best_pred2 = best_model.predict(X_test)
best_pred_total = np.round(y_pred_X_test_rd + best_pred1)
ev = eval_metrics(y_test, np.round(model.predict(X_test)))
ev1 = eval_metrics(y_test, np.round(best_model.predict(X_test)))
ev_total = eval_metrics(y_test, best_pred_total)

features1 = model.feature_importances_
feature_rank1 = pd.Series(features1, index = list(X.columns)).sort_values(ascending=False)
eval_metrics1 = model.eval_metrics(validate_pool, ['RMSE', 'MAE', 'R2'], plot=True)

model.save_model('catboost_model.dump')

import hyperopt

best_pred = best_model.predict(X_test)
model_score = best_model.score(X_test, y_test)
model.get_test_eval()
weights = train_pool.get_weight()
model.get_weight()


cv_data = cv(
    Pool(X,  y),
    model.get_params(),
    plot=True
)


print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
    np.max(cv_data['test-R2-mean']),
    cv_data['test-R2-std'][np.argmax(cv_data['test-RMSE-mean'])],
    np.argmax(cv_data['test-RMSE-mean'])
))

cv_data['test-RMSE-std'][np.argmax(cv_data['test-RMSE-mean'])]

params  =  {
    'iterations' : 500,
    'learning_rate': 0.2,
    'depth': 12,
    'loss_function': 'RMSE',
    'eval_metric': 'R2',
    'random_seed': 42,
    'logging_level': 'Silent',
    'l2_leaf_reg' : 2,
    'use_best_model': False
}
train_pool = Pool(X_train, y_train)
validate_pool = Pool(X_test, y_test)

model = CatBoostRegressor(**params)
model.fit(train_pool, eval_set=validate_pool)

best_model_params = params.copy()
best_model_params.update({
    'use_best_model': True
})
best_model1 = CatBoostRegressor(**best_model_params)
best_model1.fit(train_pool, eval_set=validate_pool);

ev = eval_metrics(y_test, np.round(model.predict(X_test)))
ev1 = eval_metrics(y_test, np.round(best_model.predict(X_test)))

feature_importance = best_model1.feature_importances_
features1 = pd.Series(feature_importance, index=list(X.columns))
features_sort = features1.sort_values(ascending=False)
features_zero = list(features_sort[features_sort==0].index.values)
