

from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Activation, Dropout,Layer, LocallyConnected1D, LocallyConnected2D, Convolution1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, MaxPooling2D, Merge, Input, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import SGD
from keras.optimizers import RMSprop, Adamax
import matplotlib.pyplot as plt
import math
import sys
from math import log
from sklearn.metrics import roc_auc_score, roc_curve

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


# Data processing

# load dataset

dataframe_X = pandas.read_csv("DatabaseX100+_MIL.csv", header=None)
dataset_X = dataframe_X.values

weights = dataset_X[0:3*464182, 1:2]

X = dataset_X[0:3*464182,3:150].astype(float)

# Preprocessing

#X_best -= np.mean(X_best, axis = 0) # center
X -= np.mean(X , axis = 0) # center


#Normalisation of inputs.

#X_best /= np.std(X_best, axis = 0) # normalize
X /= np.std(X , axis = 0) # normalize

Events_number = dataset_X[0:3*464182,2:3].astype(float)
Y = dataset_X[0:3*464182,0:1]


# We shuffle the databases.
# Train
randomize = np.arange(len(X))
np.random.shuffle(randomize)
Events_number=Events_number[randomize]
X = X[randomize]
Y = Y[randomize]
weights= weights[randomize]


# Creation of Train and Test files.##################################################################################################################

# Count nbr of pairs
nbr_pairs = 0
for i in range(3*464182):
    if (Events_number[i]%2 == 0):
        nbr_pairs += 1

#X_train_best = np.copy(X_best[0:nbr_pairs/3])
#X_test_best = np.copy(X_best[0:464182-nbr_pairs/3]) #21

weights_train = np.copy(weights[0:nbr_pairs])
weights_test = np.copy(weights[0:464182*3 - nbr_pairs])

X_train = np.copy(X[0:nbr_pairs])
X_test = np.copy(X[0:3*464182-nbr_pairs]) #21

Y_train = np.copy(Y[0:nbr_pairs, 0:1])
Y_test = np.copy(Y[0:3*464182-nbr_pairs, 0:1])

#Y_train_best = np.copy(Y[0:nbr_pairs/3, 0:1])
#Y_test_best = np.copy(Y[0:464182-nbr_pairs/3, 0:1])


p = 0
q = 0

for i in range(3*464182):
    if (Events_number[i]%2 == 0):

    	weights_train[p] = np.copy(weights[i])
        
        X_train[p]  = np.copy(X[i])
        
        Y_train[p] = np.copy(Y[i])
        
        p += 1
        
    elif (Events_number[i]%2 == 1):

    	weights_test[q] = np.copy(weights[i])
        
        X_test[q]  = np.copy(X[i])
        
        Y_test[q] = np.copy(Y[i])
        q += 1


# Class weights definition

nbr_signals = 0.0
for i in range(3*464182):
    if (Y[i] == 1):
        nbr_signals = nbr_signals + 1

weight1 = 1
weight0 = (3*464182-nbr_signals)/nbr_signals

class_weight = {0 : 1/weight0,
    1: 1/weight1}



# Reshaping

model_train = Sequential()
model_test = Sequential()

c, r = Y_train.shape
Y_train = Y_train.reshape(c,)

c, r = Y_test.shape
Y_test = Y_test.reshape(c,)

c, r = weights_test.shape
weights_test = weights_test.reshape(c,)
c, r = weights_train.shape
weights_train = weights_train.reshape(c,)
weights_shuffled = np.hstack([weights_train, weights_test])

# Percentage definition
# The point is that we want to reduce the training base by a factor without touching the testing database size. Being said that the testing database is computed automatically with validation_split, it leads to the following definition.

Percentage = float(sys.argv[2])
Adapted_percentage = Percentage*0.8+0.2




		
# Model train ####################################################################################################################################		

# create model(135/75/105)
model_train.add(Dense(100, input_dim=147, init='normal', activation='relu' ,name='input_train',W_regularizer=l1l2(l1=1E-6, l2=1E-5), activity_regularizer=l1l2(l1=0, l2=1e-7))) #W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))	
model_train.add(Dropout(0.1))
model_train.add(Dense(540,  activation ='relu', W_regularizer=l1l2(l1=1e-07, l2=0), activity_regularizer=l1l2(l1=0, l2=5e-7)))
model_train.add(Dropout(0.3))
model_train.add(Dense(310,  activation ='relu',name='output_train'))
model_train.add(Dropout(0.5))
model_train.add(Dense(450,  activation ='relu' ))
model_train.add(Dropout(0.4))
		
model_train.add(Dense(2))
		
model_train.add(Activation('softmax'))

admax = Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#decay ? 0.002
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=1, min_lr=0.00001)
"""
def schedule(nb_epoch):
	return(0.97**nb_epoch)
"""

"""
X_train_val = X_train[int(0.8*len(X_train)):len(X_train)]
Y_train_val = Y_train[int(0.8*len(X_train)):len(X_train)]
weight_train_val =weights_train[int(0.8*len(X_train)):len(X_train)]


def AUC_train(y_true, y_pred):
    yhat = self.model.predict_proba(X_train_val, verbose=0).T[1]
    return roc_auc_score(Y_train_val, yhat, sample_weight=weight_train_val)

class MonitorAUC_train(Callback):
    def on_epoch_end(self, epoch, logs={}):
        yhat = self.model.predict_proba(X_train_val, verbose=0).T[1]
        print 'AUC', roc_auc_score(Y_train_val, yhat, sample_weight=weight_train_val)
"""

callbacks = [
   	EarlyStopping(monitor='val_loss', patience=30, verbose=0),
   	#MonitorAUC_train(),
   	ModelCheckpoint("weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
   	reduce_lr
   	
   	#LearningRateScheduler(schedule),
]

model_train.compile(optimizer=admax, loss='binary_crossentropy', metrics=['accuracy', AUC_train]) # Gradient descent
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) # Gradient descent
#model[j].compile(optimizer='adam', loss='categorical_crossentropy', metrics=[''sparse_categorical_accuracy'']) # Gradient descent



Y3 = np_utils.to_categorical(Y_train, 2) # convert class vectors to binary class matrices
			
"""
# Sample weights
if (j==0):
	c, r = sample_weights_train.shape
	sample_weights_train = sample_weights_train.reshape(c,)
sample_weights_train=np.absolute(sample_weights_train)
"""
		
print(len(X_train))
		
# Fit the model		
model_train.fit(X_train[0:int(len(X_train)*Adapted_percentage)], Y3[0:int(len(X_train)*Adapted_percentage)], validation_split=0.2/Adapted_percentage, nb_epoch=1, batch_size=400, shuffle=True, verbose=1, callbacks=callbacks)#, class_weight=class_weight)

		

# Model Test ###################################################################


# create model(135/75/105)
model_test.add(Dense(100, input_dim=147, init='normal', activation='relu' ,name='input_train',W_regularizer=l1l2(l1=1E-6, l2=1E-5), activity_regularizer=l1l2(l1=0, l2=1e-7))) #W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))	
model_test.add(Dropout(0.1))
model_test.add(Dense(540,  activation ='relu', W_regularizer=l1l2(l1=1e-07, l2=0), activity_regularizer=l1l2(l1=0, l2=5e-7)))
model_test.add(Dropout(0.3))
model_test.add(Dense(310,  activation ='relu',name='output_train'))
model_test.add(Dropout(0.5))
model_test.add(Dense(450,  activation ='relu' ))
model_test.add(Dropout(0.4))

model_test.add(Dense(2))
		
model_test.add(Activation('softmax'))

admax = Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#decay ? 0.002
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=1, min_lr=0.00001)
"""
def schedule(nb_epoch):
	return(0.97**nb_epoch)
"""
		
callbacks = [
   	EarlyStopping(monitor='val_loss', patience=30, verbose=0),
   	ModelCheckpoint("weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
   	reduce_lr
   	#LearningRateScheduler(schedule),
]

model_test.compile(optimizer=admax, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) # Gradient descent
#model[j].compile(optimizer='adam', loss='categorical_crossentropy', metrics=[''sparse_categorical_accuracy'']) # Gradient descent



Y3 = np_utils.to_categorical(Y_test, 2) # convert class vectors to binary class matrices
			
"""
# Sample weights
if (j==0):
	c, r = sample_weights_test.shape
	sample_weights_test = sample_weights_test.reshape(c,)
sample_weights_test=np.absolute(sample_weights_test)
"""
		
		
		
# Fit the model		
model_test.fit(X_test[0:int(len(X_test)*Adapted_percentage)], Y3[0:int(len(X_test)*Adapted_percentage)], validation_split=0.2/Adapted_percentage, nb_epoch=1, batch_size=400, shuffle=True, verbose=1, callbacks=callbacks)#, class_weight=class_weight)




Z_train = model_train.predict(X_test, batch_size=32, verbose=0)

Z_test = model_test.predict(X_train, batch_size=32, verbose=0)



Score_background = 0.0
Faux_background = 0.0
Score_signal = 0.0
Faux_signaux = 0.0
for i in range(len(X_test)):

	if (Z_train[i][1] > Z_train[i][0] and Y_test[i] == 1):
		Score_signal += 1
		
	if (Z_train[i][1] < Z_train[i][0] and Y_test[i] == 0):
		Score_background += 1
		
	if (Z_train[i][1] < Z_train[i][0] and Y_test[i] == 1):
		Faux_background += 1
		
	if (Z_train[i][1] > Z_train[i][0] and Y_test[i] == 0):
		Faux_signaux += 1

for i in range(len(X_train)):

	if (Z_test[i][1] > Z_test[i][0] and Y_train[i] == 1):
		Score_signal += 1
		
	if (Z_test[i][1] < Z_test[i][0] and Y_train[i] == 0):
		Score_background += 1
		
	if (Z_test[i][1] < Z_test[i][0] and Y_train[i] == 1):
		Faux_background += 1
		
	if (Z_test[i][1] > Z_test[i][0] and Y_train[i] == 0):
		Faux_signaux += 1
		


print("Score_signal = "+str(Score_signal))
print("Faux_signaux = "+str(Faux_signaux))

print("Score_background = "+str(Score_background))
print("Faux_background = "+str(Faux_background))


Score_signal = Score_signal / (347831*3)
Score_background = Score_background / (116351*3)

print("Score_signal = "+str(Score_signal))

print("Score_background = "+str(Score_background))

"""
model.fit(X2, Y2, validation_data=(X, Y), nb_epoch=200, batch_size=96)
# Final evaluation of the model
scores = model.evaluate(X, Y, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
"""


#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#test

#scores = model.evaluate(X, Y)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Let's compute the AUC
Y_pred = np.zeros(3*(347831+116351))

for i in range(len(Y_pred)):
	if (i < len(Z_test)):
		Y_pred[i]= Z_test[i,1]*2-1
	else:
		Y_pred[i]= Z_train[i-len(Z_test),1]*2-1


Y_true = np.hstack([Y_train, Y_test])



r_score = roc_auc_score(Y_true, Y_pred, average='macro', sample_weight=weights_shuffled)

print(r_score)
