# Create first network with Keras
import sys
from keras.models import Sequential
from keras.layers import Dense, Reshape, Activation, Dropout,Layer, LocallyConnected1D, LocallyConnected2D, Convolution1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, MaxPooling2D, Merge
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import SGD
from keras.optimizers import RMSprop, Adamax
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import math
from math import log

import csv

from keras import backend as K
from theano import tensor as T

from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas
from keras.regularizers import l1, activity_l1, l1l2


# fix random seed for reproducibility
seed = int(sys.argv[1])
np.random.seed(seed)

from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D
K.set_image_dim_ordering('th')

# load dataset
dataframe = pandas.read_csv("DatabaseZiyu.csv", header=None)
dataset = dataframe.values

dataframe_weights = pandas.read_csv("Weights.csv", header=None)
dataset_weights = dataframe_weights.values

weights = dataset_weights[0:942548]

# split into input (X) and output (Y) variables

X = dataset[0:942548,0:16].astype(float)
Events_number = dataset[0:942548,16:17].astype(float)
Y = dataset[0:942548,17:18].astype(float)


# We shuffle the databases.
# Train
randomize = np.arange(len(X))
np.random.shuffle(randomize)
Events_number=Events_number[randomize]
X = X[randomize]
Y = Y[randomize]
weights= weights[randomize]

print(X[0:2])

# Preprocessing

X -= np.mean(X, axis = 0) # center

#Normalisation of inputs.

X /= np.std(X, axis = 0) # normalize

# Split X and Y into train/test datasets.
p = 0
q = 0

# Count nbr of pairs
nbr_pairs = 0
for i in range(942548):
	if (Events_number[i]%2 == 0):
		nbr_pairs += 1

# Sample weights
sample_weights_train = np.copy(weights[0:nbr_pairs])
sample_weights_test = np.copy(weights[0:942548 - nbr_pairs])

weights_train = np.copy(weights[0:nbr_pairs])
weights_test = np.copy(weights[0:942548 - nbr_pairs])

X_train = np.copy(X[0:nbr_pairs])
X_test = np.copy(X[0:942548-nbr_pairs]) #21

Y_train = np.copy(Y[0:nbr_pairs])
Y_test = np.copy(Y[0:942548-nbr_pairs])

for i in range(942548):
	if (Events_number[i]%2 == 0):
		X_train[p]  = np.copy(X[i])
		Y_train[p] = Y[i]
		sample_weights_train[p] = np.copy(weights[i])
		weights_train[p] = np.copy(weights[i])
		if (Y_train[p] == 1):
			sample_weights_train[p] = sample_weights_train[p]*0.4*100
		p += 1
		
		
	else:
		X_test[q] = X[i]
		Y_test[q] = Y[i]
		sample_weights_test[q] = np.copy(weights[i])
		weights_test[q] = np.copy(weights[i])
		if (Y_test[q] == 1):
			sample_weights_test[q] = sample_weights_test[q]*0.4*100
		q += 1

nbr_signals = 0.0
for i in range(942548):
	if (Y[i] == 1):
		nbr_signals = nbr_signals + 1

weight1 = 1
weight0 = (942548-627066)/nbr_signals
print(weight0)
print(weight1)
class_weight = {0 : 1/weight0,
    1: 1/weight1}

c, r = Y_train.shape
Y_train = Y_train.reshape(c,)

c, r = Y_test.shape
Y_test = Y_test.reshape(c,)

# Percentage definition
# The point is that we want to reduce the training base by a factor without touching the testing database size. Being said that the testing database is computed automatically with validation_split, it leads to the following definition.
"""
100 80 20 
80	60 20 
60	40 20
40	20 20


"""
#Percentage = float(sys.argv[2])
#Adapted_percentage = Percentage*0.8+0.2
"""
1 0.8/(4*Percentage) 
0.75 1/3
0.50 1/2
0.25 1/1
"""
c, r = Y.shape
Y = Y.reshape(c,)


# Sample weights

sample_weightx = np.copy(weights)

c, r = sample_weightx.shape
sample_weightx = sample_weightx.reshape(c,)
sample_weightx=np.absolute(sample_weightx)
for i in range(len(weights)):
	if (Y[i] == 1):
			sample_weightx[i] = sample_weightx[i]*0.4*85
ratio = 1/np.mean(sample_weightx)
for i in range(len(sample_weightx)):
	sample_weightx[i]*=ratio


# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=20, shuffle=False ,random_state=seed)
cvscores = []


Percentage = float(sys.argv[2])

j = 0
for train, test in kfold.split(X, Y):
	
	train = train[0:int(Percentage*len(train))]

	# create model(135/75/105)
	model_train=Sequential()
	model_train.add(Dense(280, input_dim=16, init='normal', activation='relu' ,W_regularizer=l1l2(l1=5e-06, l2=5e-06), activity_regularizer=l1l2(l1=0, l2=1e-5))) #W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))
	model_train.add(Dropout(0.25))
	model_train.add(Dense(370,  activation ='relu', activity_regularizer=l1l2(l1=0, l2=5e-5)))
	model_train.add(Dropout(0.5))
	model_train.add(Dense(120,  activation ='relu', W_regularizer=l1l2(l1=0, l2=5e-06)))
	model_train.add(Dropout(0.55))	
	model_train.add(Dense(2))	
	model_train.add(Activation('softmax'))

	admax = Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#decay ? 0.002
	reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.95, patience=1, min_lr=0.00001)
		
	callbacks = [
   		#EarlyStopping(monitor='val_loss', patience=30, verbose=0),
   		#ModelCheckpoint("/home/admin-7036/Documents/Projet python/bosongit/weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
   		reduce_lr 
	]

	model_train.compile(optimizer=admax, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent

	Y3 = np_utils.to_categorical(Y, 2) # convert class vectors to binary class matrices
				
	

	if (j == 0):
		Z = model_train.predict(X, batch_size=32, verbose=0)
		for i in range(len(X)):
			Z[i,0]=666
			
	# Fit the model		
	model_train.fit(X[train], Y3[train], nb_epoch=1, batch_size=400,  class_weight=class_weight,  shuffle=True, verbose=0, callbacks=callbacks)#sample_weight=sample_weightx[train]



	Z[test] = model_train.predict(X[test], batch_size=32, verbose=0)

	j = j + 1


			



Score_background = 0.0
Faux_background = 0.0
Score_signal = 0.0
Faux_signaux = 0.0
for i in range(len(Z)):

	if (Z[i][1] > Z[i][0] and Y[i] == 1):
		Score_signal += 1
		
	if (Z[i][1] < Z[i][0] and Y[i] == 0):
		Score_background += 1
		
	if (Z[i][1] < Z[i][0] and Y[i] == 1):
		Faux_background += 1
		
	if (Z[i][1] > Z[i][0] and Y[i] == 0):
		Faux_signaux += 1

print("Score_signal = "+str(Score_signal))
print("Faux_signaux = "+str(Faux_signaux))

print("Score_background = "+str(Score_background))
print("Faux_background = "+str(Faux_background))


Score_signal = Score_signal / 627066.0
Score_background = Score_background / 315482.0 

print("Score_signal = "+str(Score_signal))

print("Score_background = "+str(Score_background))

# Let's compute the AUC

Y_pred = np.zeros(len(Z))

for i in range(len(Z)):
	Y_pred[i]= Z[i,1]*2-1
	

c, r = weights.shape
weights = weights.reshape(c,)

r_score = roc_auc_score(Y, Y_pred, average='macro', sample_weight=weights)

print(r_score)