# coding=utf-8

from __future__ import unicode_literals
from __future__ import absolute_import

import sys
import copy
import inspect
#import random
import types
import inspect
from sklearn.utils.validation import column_or_1d
from keras.callbacks import Callback

import numpy as np

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from keras.optimizers import RMSprop, Adamax, Nadam, Adamax, Adadelta, Adagrad

from keras.callbacks import Callback
from sklearn import preprocessing

import sys
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Activation, Dropout,Layer, LocallyConnected1D, LocallyConnected2D, Convolution1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, MaxPooling2D, Merge, Input, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import SGD
from keras.optimizers import RMSprop, Adamax, Nadam, Adamax, Adadelta, Adagrad
from keras.constraints import maxnorm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import math
from math import log
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise, GaussianDropout
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import csv

from keras import backend as K
from theano import tensor as T

from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas
from keras.regularizers import l1,l2, activity_l1, l1l2
from datetime import datetime


# user inputs

nb_esti = int(sys.argv[2])
nb_epo = int(sys.argv[3])

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
			sample_weights_train[p] = sample_weights_train[p]*0.4*85
		p += 1
		
		
	else:
		X_test[q] = X[i]
		Y_test[q] = Y[i]
		sample_weights_test[q] = np.copy(weights[i])
		weights_test[q] = np.copy(weights[i])
		if (Y_test[q] == 1):
			sample_weights_test[q] = sample_weights_test[q]*0.4*85
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




# Definition du bagging 

nb_estimators = nb_esti

sample_weights_train = np.zeros(len(X_train))
sample_weights_test= np.zeros(len(X_test))


sample_weights_train_bagging=[]
sample_weights_test_bagging=[]

random_train = np.zeros(nb_estimators*len(X_train))
random_test = np.zeros(nb_estimators*len(X_test))

for i in range(len(random_train)):
	random_train[i]=np.random.random()

#print(random_train[len(X_train)])
#print(random_train[0])

for i in range(len(random_test)):
	random_test[i]=np.random.random()

# On genère sample_weight_train_bagging
for j in range(nb_estimators):
	custom_ratio = random_train[j]*2
	sample_weights_train_bagging.append(np.copy(sample_weights_train))
	for i in range(len(X_train)):

		pick = int(random_train[j*len(X_train)+i]*len(X_train))
	

		if (Y_train[pick]==1):
			sample_weights_train_bagging[j][pick]+=1*custom_ratio
 		elif (Y_train[pick]==0):
 			sample_weights_train_bagging[j][pick]+=1
 	

print(sample_weights_train_bagging[0][0:10])



 	#c, r = sample_weights_train_bagging[j].shape
	#sample_weights_train_bagging[j] = sample_weights_train_bagging[j].reshape(c,)
	

# On genère sample_weight_test_bagging
for j in range(nb_estimators):
	custom_ratio = random_test[j]*2
	sample_weights_test_bagging.append(np.copy(sample_weights_test))
	for i in range(len(X_test)):
		
		pick = int(random_test[j*len(X_test)+i]*len(X_test))
		if (Y_test[pick]==1):
			sample_weights_test_bagging[j][pick]+=1*custom_ratio
 		elif (Y_test[pick]==0):
 			sample_weights_test_bagging[j][pick]+=1


 		
 	#c, r = sample_weights_test_bagging[j].shape
	#sample_weights_test_bagging[j] = sample_weights_test_bagging[j].reshape(c,)



# Définition of the model

def create_model():

	model = Sequential()
	model.add(Dense(280, input_dim=16, init='normal', activation='relu' ,W_regularizer=l1l2(l1=5e-06, l2=5e-06), activity_regularizer=l1l2(l1=0, l2=1e-5))) #W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))
	model.add(Dropout(0.25))
	model.add(Dense(370,  activation ='relu',activity_regularizer=l1l2(l1=0, l2=5e-5)))
	model.add(Dropout(0.5))
	model.add(Dense(120,  activation ='relu',W_regularizer=l1l2(l1=0, l2=5e-06)))
	model.add(Dropout(0.55))
	model.add(Dense(1))	
	model.add(Activation('sigmoid'))

	admax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#decay ? 0.002
	
	"""
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.97, patience=1, min_lr=0.00001)
	callbacks = [
		EarlyStopping(monitor='val_loss', patience=25, verbose=0),
		ModelCheckpoint("/home/admin-7036/Documents/Projet python/bosongit/weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
		reduce_lr
	
	]
	"""
	model.compile(optimizer=admax, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent
	return model

def create_model_perceptron():

	model = Sequential()
	

	model.add(Dense(1, input_dim=16, init='normal', activation='relu'))#, W_regularizer=l1l2(l1=13E-6, l2=0), activity_regularizer=l1l2(l1=0, l2=0))) # ar : 0;0 wr :13E-6, 0  // wr : good, bad ar : bad, bad 
	#model.add(Dropout(0.25))
	#model.add(Dense(370,  activation ='relu',  name='output', W_regularizer=l1l2(l1=0, l2=5E-6), activity_regularizer=l1l2(l1=0, l2=0))) # ar : 0;5e-5 wr : 0 5E-6// wr : bad, neutral ar : bad , bad
	#model_train.add(Dropout(0)) 
	#model.add(Dense(120,  activation ='relu', W_regularizer=l1l2(l1=0, l2=0))) # wr : 0;5e-6 // wr : bad, bad.
	#model.add(Dropout(0.55))
	model.add(Dense(1))
		
	model.add(Activation('sigmoid'))

	admax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#decay ? 0.002
	adagrd = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)

	model.compile(optimizer=adagrd, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent
	#print(model.summary()) 
	return model



# Entrainement
List_models_train = []
List_models_test = []



for j in range(nb_estimators):
	List_models_train.append(create_model_perceptron())
	List_models_train[j].fit(X_train, Y_train, validation_split=0.2, nb_epoch=nb_epo, batch_size=400, shuffle=True, verbose=0, sample_weight=sample_weights_train_bagging[j])
	
	List_models_test.append(create_model_perceptron())
	List_models_test[j].fit(X_test, Y_test, validation_split=0.2, nb_epoch=nb_epo, batch_size=400, shuffle=True, verbose=0, sample_weight=sample_weights_test_bagging[j])



# Predictions 

Z_train = np.zeros(len(X_test))
Z_train = np.zeros(len(X_train))

for i in range(nb_estimators):
	if (i == 0):
		Z_train = List_models_train[i].predict(X_test, batch_size=32, verbose=0)
		Z_test = List_models_test[i].predict(X_train, batch_size=32, verbose=0)

	elif ( i > 0 and i != nb_estimators-1):
		Z_train += List_models_train[i].predict(X_test, batch_size=32, verbose=0)
		Z_test += List_models_test[i].predict(X_train, batch_size=32, verbose=0)

	elif (i == nb_estimators-1):
		Z_train = (Z_train + List_models_train[i].predict(X_test, batch_size=32, verbose=0))/nb_estimators
		Z_test = (Z_test + List_models_test[i].predict(X_train, batch_size=32, verbose=0))/nb_estimators


# Writting to csv (in case of future need)



tocsv = np.zeros((len(X_test),1))


for i in range(len(X_test)):
	tocsv[i]=Z_train[i]
	
np.savetxt("bagging_varyingSW"+str(seed)+"_"+str(nb_esti)+"_"+str(nb_epo)+".csv", tocsv, delimiter=",")


#Z_train = List_models_train[0].predict(X_test, batch_size=32, verbose=0)
#Z_test = List_models_test[0].predict(X_train, batch_size=32, verbose=0)
# Let's compute the AUC
Y_pred = np.zeros(len(X))

for i in range(len(Y_pred)):
	if (i < len(Z_test)):
		Y_pred[i]= Z_test[i]*2-1
	else:
		Y_pred[i]= Z_train[i-len(Z_test)]*2-1


Y_true = np.hstack([Y_train, Y_test])

c, r = weights_test.shape
weights_test = weights_test.reshape(c,)
c, r = weights_train.shape
weights_train = weights_train.reshape(c,)

weights_shuffled = np.hstack([weights_train, weights_test])

r_score = roc_auc_score(Y_true, Y_pred, average='macro', sample_weight=weights_shuffled)

print(r_score)



