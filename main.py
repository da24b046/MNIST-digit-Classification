import algorithms as alg
from algorithms import PCAModel
from algorithms import KNNClassifier
from algorithms import XGBoostMulticlass
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score , accuracy_score , precision_score , recall_score
import time

srt_time = time.time()

def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):
    
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)

    featurecols = list(dftrain.columns)
    featurecols.remove('label')
    featurecols.remove('even')
    targetcol = 'label'

    Xtrain = dftrain[featurecols]
    ytrain = dftrain[targetcol]
    
    Xval = dfval[featurecols]
    yval = dfval[targetcol]

    return (Xtrain, ytrain, Xval, yval)

def last_mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    max_count = counts.max()
    return values[counts == max_count][-1] 

def stack_models(x,y,z):

    if len(x) == len(y) == len(z) :        
         n = len(x) 
         pred = []
         for index in range(n):
              common_index_value = [x[index],y[index],z[index]]
              common_value = last_mode(common_index_value)
              pred.append(common_value)

    else:
         raise ValueError("invalid input")
    
    return np.array(pred)

X_train,y_train,X_val,y_val = read_data()

y_pred_sg = alg.softmax_regression(X_train.values, y_train.values,X_val.values, num_classes=10,mini_batch_size=56,learning_rate=0.01, n_epochs=30)

xgb = XGBoostMulticlass(n_classes=10, n_estimators=40, learning_rate=0.5,  max_depth=3, reg_lambda=1.0, gamma=0.1, subsample_features=0.1, n_thresholds=10)
xgb.fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_val)

pca = PCAModel(112)
pca.fit(X_train)
X_reduced = pca.reconstruct(X_train)

knn = KNNClassifier(6)
knn.fit(X_reduced,y_train)
y_pred_knn = knn.predict(X_val)

y_pred = stack_models(y_pred_sg,y_pred_xgb, y_pred_knn)
end_time = time.time()
print("F1 Score: ", f1_score(y_val, y_pred, average='macro'))
print("Accuracy: ", accuracy_score(y_val, y_pred))  
print("Precision: ", precision_score(y_val, y_pred, average='macro'))
print("Recall: ", recall_score(y_val, y_pred, average='macro'))
print("Training time: ",end_time-srt_time)
