import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read in dataset

data = pd.read_csv('pima-indians-diabetes.data', header=None)
data.head()

# note correlated features

correlations = data.corr()
correlations[8].sort_values(ascending=False)

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

data[[1,5]] = data[[1,5]].replace(0, np.NaN)
data.dropna(inplace=True)

visualise(data)

# separate features and target 

X = data.iloc[:,[1,5]].values
y = data.iloc[:,-1].values

# apply feature scaling
 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# build a training set and a test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# build the logistic regression model

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# predict y values for the test set and evaluate the confusion matrix
y_pred = model.predict(X_test)

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

# evaluate model using cross validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=10, random_state=7)
train_result = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
test_result = cross_val_score(model, X_test, y_test, cv=kfold, scoring='accuracy')

print('Training set cv: %f Test set cv: %f' % (train_result.mean(), test_result.mean()))