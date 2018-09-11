# Visualize training history
from keras.models import Sequential
from keras.layers import Dense, Activation, GRU, RNN, SimpleRNN, BatchNormalization, Dropout, AlphaDropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingRegressor, BaggingRegressor
from keras.datasets import mnist
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
#y = pd.read_csv("MLB/scores.csv", encoding='latin-1', names=['Score'])
#X = pd.read_csv("MLB/pitch_trial.csv", encoding='latin-1')
#X1 = pitch_trial

X = pd.read_csv("MLB/X_sort.csv", encoding='latin-1')
y = pd.read_csv("MLB/y_sort.csv", encoding='latin-1', names=['Score'])
X = X.drop('Expected_Runs', axis = 1)


#(X_train, y_train), (X_test, y_test)  ==  mnist.load_data()
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)
'''
# Feature Scaling (Important for high intensity computations)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
# create model
def mlp_model():
    model = Sequential()
    model.add(Dense(64, input_dim = len(X.columns), kernel_initializer='uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(45, kernel_initializer='uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(35, kernel_initializer='uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='uniform'))
    # Compile model
    model.compile(loss=['mse'], optimizer='adam', metrics=['accuracy', 'binary_crossentropy'])
    return model

model1 = KerasRegressor(build_fn = mlp_model, epochs = 100, verbose = 0)
model2 = KerasRegressor(build_fn = mlp_model, epochs = 100, verbose = 0)
model3 = KerasRegressor(build_fn = mlp_model, epochs = 100, verbose = 0)

ensemble_clf = BaggingRegressor(base_estimator = model1)

ensemble_clf.fit(X_train, y_train)
ensemble_clf.score(X_test, y_test)

y_pred = ensemble_clf.predict(X_test)
y_pred_rd = np.round(y_pred)

def eval_metrics(y_test, y_pred):
    mae = np.round(abs(y_test.values - y_pred).mean(),decimals=2)
    rmse = (((y_test.values - y_pred)**2).mean())**(1/2)
    r = (((y_test.values-y_test.values.mean())*(y_pred - y_pred.mean())).sum()) / ((1-len(y_test))*(y_test.values.std())*(y_pred.std()))
    print('mae: {0:.2f} \nrmse: {1:.2f} \nr: {2:.4f}'.format(mae, rmse, r))
    return mae, rmse, r

metrics1 = eval_metrics(y_test, y_pred)

# Fit the model
model = mlp_model()
history = model.fit(X_train, y_train, validation_split=0.15, epochs=2500, batch_size=30, verbose=0)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'], loc = 'upper left')
plt.show()




