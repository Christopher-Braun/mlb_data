# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


from constant_variables import features_top_list
X2 = X[features_top_list]
X3 = X2.drop('Expected_Runs', axis = 1)

X4 = X_train_weighted_total_norm
y1 = y_train_weighted

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

# Feature Scaling (Important for high intensity computations)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_full = sc.fit_transform(X4)


# create model
model = Sequential()
model.add(Dense(64, input_dim = len(X.columns), kernel_initializer='uniform', activation='relu'))
model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
model.add(Dense(45, kernel_initializer='uniform', activation='relu'))
model.add(Dense(35, kernel_initializer='uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform'))
# Compile model
model.compile(loss=['mse'], optimizer='adam', metrics=['accuracy', 'binary_crossentropy'])
# Fit the model
history = model.fit(X_train, y_train, validation_split=0.15, epochs=2500, batch_size=30, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Predicting the Test set results
y_pred_test = model.predict(X_test)
y_pred_test_rd = np.round(y_pred_test)

# Predicting a new result
y_pred_train = model.predict(X_train)
y_pred_train_rd = np.round(y_pred_train)

# Predicting the Full Results
y_pred_full = model.predict(X3)
y_pred_full_rd = np.round(y_pred_full)

# Predicting the Full Results
y_pred_full = history.predict(X3)
y_pred_full_rd = np.round(y_pred_full)