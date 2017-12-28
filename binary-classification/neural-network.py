import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read in dataset

data = pd.read_csv('pima-indians-diabetes.data', header=None)
data.head()

# note correlated features

correlations = data.corr()
print(correlations[8].sort_values(ascending=False))

# visualise highly correlated features

def visualise(data):
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:,1].values, data.iloc[:,5].values)
    ax.set_title('Highly Correlated Features')
    ax.set_xlabel('Plasma glucose concentration')
    ax.set_ylabel('Body mass index')

visualise(data)

# note missing values 
# mark zero values as missing or NaN
# drop rows with missing values

data[[1,2,3,4,5]] = data[[1,2,3,4,5]].replace(0, np.NaN)
data.dropna(inplace=True)

visualise(data)

# separate features and target 

X = data.iloc[:,[0,1,4,5,6,7]].values
y = data.iloc[:,-1].values

# apply feature scaling
 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# build a training set and a test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# build the ANN

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build_model(neurons=4, activation='relu', dropout_rate=0.0, weight_constraint=0, optimizer='adam'):
    model = Sequential()    
    model.add(Dense(units = neurons, activation = activation, kernel_initializer = 'uniform', input_dim = X.shape[1]))
    model.add(Dropout(rate = dropout_rate))
    model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])    
    return model

# SINGLE RUN # 

model = build_model()
model.fit(X_train, y_train, batch_size = 16, epochs = 500)

# PARAMETER TUNING #

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

parameters = { 
        'neurons': [2,4,6,8,10],
        'activation': ['softmax', 'relu', 'tanh', 'sigmoid'],
        'dropout_rate': [0.01, 0.1, 0.11], 
        'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adam'],
        'batch_size': [8,16], 
        'epochs': [500,700]}

keras_model = KerasClassifier(build_fn=build_model)
grid_search = GridSearchCV(estimator = keras_model, param_grid = parameters, scoring = 'accuracy', cv = 3)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best: %f using %s" % (best_accuracy, best_parameters))

means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# EVALUATION #

# predict y values for the test set and evaluate the confusion matrix

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix

def precision_recall(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)    
    tp = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]
    prec = tp / (tp+fp)
    rec = tp / (tp+fn)    
    return prec, rec

precision, recall = precision_recall(y_test, y_pred)
print('Precision: %f Recall %f' % (precision, recall))

