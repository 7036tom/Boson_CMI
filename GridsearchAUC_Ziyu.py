# Create first network with Keras
from keras.callbacks import Callback
import sys
from keras.models import Sequential
from keras.layers import Dense, Reshape, Activation, Dropout,Layer, LocallyConnected1D, LocallyConnected2D, Convolution1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, MaxPooling2D, Merge
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import SGD
from keras.optimizers import RMSprop, Adamax, Nadam, Adamax, Adadelta, Adagrad
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import cohen_kappa_score, make_scorer
import math
from keras.utils import np_utils
from math import log

import csv

from keras import backend as K
from theano import tensor as T

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas
from keras.regularizers import l1, activity_l1, l1l2
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Def AUC for scoring

def AUC(estimator, y_true, y_probs):
    yhat = estimator.predict_proba(X_train_val, verbose=0)
    return roc_auc_score(Y_train_val, yhat, sample_weight=weight_train_val)


# Function to create model, required for KerasClassifier

def create_model(a1l2, a2l2):
    # create model
    #L=WinnerTakeAll1D_GaborMellis(spatial=1, OneOnX=WTAX)
   
    # create model(135/75/105)
    model_train=Sequential()
    model_train.add(Dense(280, input_dim=16, init='normal', activation='relu' ,W_regularizer=l1l2(l1=13E-6, l2=0), activity_regularizer=l1l2(l1=0, l2=0))) # ar : 0;1e-5 wr :5E-6, 5E-6   // wr : good, bad ar : bad, bad 
    model_train.add(Dropout(0.25))
    model_train.add(Dense(370,  activation ='relu', W_regularizer=l1l2(l1=0, l2=5E-6), activity_regularizer=l1l2(l1=0, l2=0))) # ar : 0;5e-5 // wr : bad, neutral ar : bad , bad
    model_train.add(Dropout(0)) 
    model_train.add(Dense(120,  activation ='relu', W_regularizer=l1l2(l1=0, l2=0), activity_regularizer=l1l2(l1=0, l2=0))) # wr : 0;5e-6 // wr : bad, bad.
    model_train.add(Dropout(0.55))  
    model_train.add(Dense(2))   
    model_train.add(Activation('softmax'))
    
    adagrd = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    model_train.compile(optimizer=adagrd, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent
    return model_train


# load dataset
dataframe = pandas.read_csv("DatabaseZiyu.csv", header=None)
dataframe_weights = pandas.read_csv("Weights.csv", header=None)
dataset_weights = dataframe_weights.values
dataset = dataframe.values


weights = dataset_weights[0:942548]
X = dataset[0:942548,0:16].astype(float)
Events_number = dataset[0:942548,16:17].astype(float)
Y = dataset[0:942548,17:18].astype(float)

# We shuffle the databases.
randomize = np.arange(len(X))
np.random.shuffle(randomize)
Events_number=Events_number[randomize]
X = X[randomize]
Y = Y[randomize]
weights= weights[randomize]

# Preprocessing

X -= np.mean(X, axis = 0) # center

#Normalisation of inputs.

X /= np.std(X, axis = 0) # normalize


# Count nbr of pairs
nbr_pairs = 0
for i in range(942548):
    if (Events_number[i]%2 == 0):
        nbr_pairs += 1


# Sample weights
sample_weights_train = np.copy(weights[0:nbr_pairs])
weights_train = np.copy(weights[0:nbr_pairs])
weights_test = np.copy(weights[0:942548 - nbr_pairs])
X_train = np.copy(X[0:nbr_pairs])
X_test = np.copy(X[0:942548-nbr_pairs]) #21
Y_train = np.copy(Y[0:nbr_pairs])
Y_test = np.copy(Y[0:942548-nbr_pairs])

p = 0
q = 0 
for i in range(942548):
    if (Events_number[i]%2 == 0):
        X_train[p]  = np.copy(X[i])
        Y_train[p] = Y[i]
        sample_weights_train[p] = np.copy(weights[i])
        weights_train[p] = np.copy(weights[i])
        if (Y_train[p] == 1):
            sample_weights_train[p] = sample_weights_train[p]*0.4*85
        p += 1
        
       
    else:
        X_test[q] = np.copy(X[i])
        Y_test[q] = Y[i]
        weights_test[q] = np.copy(weights[i])
       
        q += 1

Y_train = np_utils.to_categorical(Y_train, 2) # convert class vectors to binary class matrices
Y_test = np_utils.to_categorical(Y_test, 2) # convert class vectors to binary class matrices

c, r = weights_train.shape
weights_train = weights_train.reshape(c,)
c, r = weights_test.shape
weights_test = weights_test.reshape(c,)


X_train_val = X_test[int(0.8*len(X_test)):len(X_test)]
Y_train_val = Y_test[int(0.8*len(X_test)):len(X_test)]
weight_train_val =weights_test[int(0.8*len(X_test)):len(X_test)]



# create model
model = KerasClassifier(build_fn=create_model, nb_epoch=50, batch_size=400, verbose=1)


kappa_scorer = make_scorer(cohen_kappa_score)
# define the grid search parameters*
dr1 = [0.15, 0.2, 0.25]
dr2 = [0.45, 0.50,0.55,0.60]
dr3 = [0.45, 0.50,0.55,0.60]
dr4 = [0.1,0.2,0.3,0.4,0.5]
neurons1 = [70, 100, 120, 150]
neurons2 = [450, 480, 510, 540]
neurons3 = [250, 280, 310, 340]
neurons4 = [450, 480, 510, 540]

a1l2=[0]
a2l2 =[0, 0.0001]

w1l1 = [0, 0.00001]
w1l2 = [0, 0.00001]
w2l1 = [0, 0.00001]
w2l2 = [0, 0.0001, 0.0005 ,0.00001, 0.00005, 0.000001, 0.000005, 0.0000001, 0.0000005]
w3l1 = [0, 0.00001]
w3l2 = [0, 0.00001]

batch_size = [80,83,86,89,92,95,98, 100]
epochs = [100, 120, 140, 160, 180, 200]
WTAX=[3,4,5]
l1_value = [0, 0.0001, 0.0005 ,0.00001, 0.00005, 0.000001, 0.000005, 0.0000001, 0.0000005]
l2_value = [0, 0.0001, 0.0005 ,0.00001, 0.00005, 0.000001, 0.000005, 0.0000001, 0.0000005]
l_rate = [0.001, 0.002, 0.003 ,0.004]
decay = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0]

param_grid = dict(a1l2=a1l2, a2l2=a2l2)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=AUC, n_jobs=1, cv =2)#, verbose=1)

grid_result = grid.fit(X_train, Y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
