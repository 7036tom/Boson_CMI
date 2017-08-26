# Create first network with Keras
import sys
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Activation, Dropout,Layer, LocallyConnected1D, LocallyConnected2D, Convolution1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, MaxPooling2D, Merge, Input, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import SGD
from keras.optimizers import RMSprop, Adamax
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from keras.layers.normalization import BatchNormalization
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

class WinnerTakeAll1D_GaborMellis(Layer):

	def __init__(self, spatial=1, OneOnX = 3,**kwargs):
		self.spatial = spatial
		self.OneOnX = OneOnX
		self.uses_learning_phase = True
		super(WinnerTakeAll1D_GaborMellis, self).__init__(**kwargs)

	def call(self, x, mask=None):
		R = T.reshape(x,(T.shape(x)[0],T.shape(x)[1]/self.OneOnX,self.OneOnX))
		M = K.max(R, axis=(2), keepdims=True)
		R = K.switch(K.equal(R, M), R, 0.)
		R = T.reshape(R,(T.shape(x)[0],T.shape(x)[1]))
		return R

	def get_output_shape_for(self, input_shape):
		shape = list(input_shape)
		return tuple(shape)
	
L=WinnerTakeAll1D_GaborMellis(spatial=1, OneOnX=3)

# load dataset
dataframe= pandas.read_csv("Database100+_best.csv", header=None)
dataset= dataframe.values


weights = dataset[0:464182, 1:2]

# split into input (X) and output (Y) variables
X1_best = dataset[0:464182,3:69].astype(float)
X2_best = dataset[0:464182,70:151].astype(float)
X= np.hstack([X1_best, X2_best])

Events_number = dataset[0:464182,2:3].astype(float)
Y = dataset[0:464182,0:1]

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
for i in range(464182):
	if (Events_number[i]%2 == 0):
		nbr_pairs += 1

# Sample weights
sample_weights_train = np.copy(weights[0:nbr_pairs])
sample_weights_test = np.copy(weights[0:464182 - nbr_pairs])

weights_train = np.copy(weights[0:nbr_pairs])
weights_test = np.copy(weights[0:942548 - nbr_pairs])


X_train = np.copy(X[0:nbr_pairs])
X_test = np.copy(X[0:464182-nbr_pairs]) #21

Y_train = np.copy(Y[0:nbr_pairs])
Y_test = np.copy(Y[0:464182-nbr_pairs])

for i in range(464182):
	if (Events_number[i]%2 == 0):
		X_train[p]  = np.copy(X[i])
		Y_train[p] = np.copy(Y[i])
		sample_weights_train[p] = np.copy(weights[i])
		weights_train[p] = np.copy(weights[i])
		if (Y_train[p] == 1):
			sample_weights_train[p] = sample_weights_train[p]*0.4*85
		p += 1
		
		
	else:
		X_test[q] = np.copy(X[i])
		Y_test[q] = np.copy(Y[i])
		sample_weights_test[q] = np.copy(weights[i])
		weights_test[q] = np.copy(weights[i])
		if (Y_test[q] == 1):
			sample_weights_test[q] = sample_weights_test[q]*0.4*85
		q += 1


model_train = Sequential()
model_test = Sequential()

nbr_signals = 0.0
for i in range(464182):
	if (Y[i] == 1):
		nbr_signals = nbr_signals + 1

weight1 = 1
weight0 = (464182-627066)/nbr_signals
print(weight0)
print(weight1)
class_weight = {0 : 1/weight0,
	1: 1/weight1}



"""
# We shuffle the databases.
# Train

randomize_train = np.random.permutation(len(X_train))
X_train = X_train[randomize_train]
Y_train = Y_train[randomize_train]
sample_weights_train = sample_weights_train[randomize_train]

# Test
randomize_test = np.random.permutation(len(X_test))
X_test = X_test[randomize_test]
Y_test = Y_test[randomize_test]
sample_weights_test = sample_weights_test[randomize_test]
"""

c, r = weights_test.shape
weights_test = weights_test.reshape(c,)
c, r = weights_train.shape
weights_train = weights_train.reshape(c,)
weights_shuffled = np.hstack([weights_train, weights_test])


c, r = Y_train.shape
Y_train = Y_train.reshape(c,)

c, r = Y_test.shape
Y_test = Y_test.reshape(c,)

# Percentage definition
# The point is that we want to reduce the training base by a factor without touching the testing database size. Being said that the testing database is computed automatically with validation_split, it leads to the following definition.


Percentage = float(sys.argv[2])
Adapted_percentage = Percentage*0.8+0.2


# Model train ####################################################################################

input_train = Input(shape=(147,))

x1 = Dense(130, init='normal', activation='relu', W_regularizer=l1l2(l1=0, l2=1e-05))(input_train)
x2 = Dropout(0.2)(x1)
		

x2 = Activation('relu')(x2)
x3 = Dense(130, activation ='relu')(x2)
#x3 = BatchNormalization()(x3)
x3 = Dropout(0.5)(x3)
x3 = Activation('relu')(x3)
x3 = Dense(130)(x3)
#x3 = BatchNormalization()(x3)
x4 = merge([x3, x2], mode='sum')
#x4 = Activation('relu')(x4)
x4= Dropout(0.2)(x4)

x4 = Activation('relu')(x4)		
x5 = Dense(130, activation ='relu')(x4)
#x5 = BatchNormalization()(x5)
x5 = Dropout(0.5)(x5)
x5 = Activation('relu')(x5)
x5 = Dense(130)(x5)
#x5 = BatchNormalization()(x5)
x6 = merge([x5, x4], mode='sum')
#x6 = Activation('relu')(x6)
x6 = Dropout(0.2)(x6)

	
x6 = Activation('relu')(x6)
x7 = Dense(130, activation ='relu')(x6)
#x7 = BatchNormalization()(x7)
x7 = Dropout(0.5)(x7)
x7 = Activation('relu')(x7)
x7 = Dense(130)(x6)
#x7 = BatchNormalization()(x7)
x8 = merge([x7, x6], mode='sum')
#x8 = Activation('relu')(x8)
x8 = Dropout(0.2)(x8)

x8 = Activation('relu')(x8)
x9 = Dense(130, activation ='relu')(x8)
#x9 = BatchNormalization()(x9)
x9 = Dropout(0.5)(x9)
x9 = Activation('relu')(x9)
x9 = Dense(130, activation ='relu')(x9)
#x9 = BatchNormalization()(x9)
x10 = merge([x9, x8], mode='sum')
#x10 = Activation('relu')(x10)
x10 = Dropout(0.2)(x10)

		
x31 = Dense(130, activation ='relu')(x10)
x31 = Dropout(0.4)(x31)
out_train = Dense(2, activation="softmax")(x31)
model_train = Model(input_train, out_train)



admax = Adamax(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#decay ? 0.002
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=1, min_lr=0.00001)

callbacks = [
	EarlyStopping(monitor='val_loss', patience=15, verbose=0),
	ModelCheckpoint("weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
	reduce_lr
]
model_train.compile(optimizer=admax, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent


Y3 = np_utils.to_categorical(Y_train, 2) # convert class vectors to binary class matrices
			
		
# Sample weights

c, r = sample_weights_train.shape
sample_weights_train = sample_weights_train.reshape(c,)
sample_weights_train=np.absolute(sample_weights_train)
		
		
# Fit the model		
model_train.fit(X_train[0:int(len(X_train)*Adapted_percentage)], Y3[0:int(len(X_train)*Adapted_percentage)],validation_split=0.2/Adapted_percentage, nb_epoch=100, batch_size=400, sample_weight=sample_weights_train[0:int(len(X_train)*Adapted_percentage)], shuffle=True, verbose=0, callbacks=callbacks)#, class_weight=class_weight)

	
# Model test ####################################################################################		


# create model(135/75/105)
input_test = Input(shape=(147,))

x1b = Dense(130, init='normal', activation='relu', W_regularizer=l1l2(l1=0, l2=1e-05))(input_test)
x2b = Dropout(0.2)(x1b)
	
x2b = Activation('relu')(x2b)	
x3b = Dense(130, activation ='relu')(x2b)
#x3b = BatchNormalization()(x3b)
x3b = Dropout(0.5)(x3b)
x3b = Activation('relu')(x3b)
x3b = Dense(130)(x3b)
#x3b = BatchNormalization()(x3b)
x4b = merge([x3b, x2b], mode='sum')
#x4b = Activation('relu')(x4b)
x4b= Dropout(0.2)(x4b)


x4b = Activation('relu')(x4b)		
x5b = Dense(130, activation ='relu')(x4b)
#x5b = BatchNormalization()(x5b)
x5b = Dropout(0.5)(x5b)
x5b = Activation('relu')(x5b)
x5b = Dense(130)(x5b)
#x5b = BatchNormalization()(x5b)
x6b = merge([x5b, x4b], mode='sum')
#x6b = Activation('relu')(x6b)
x6b = Dropout(0.2)(x6b)


x6b = Activation('relu')(x6b)
x7b = Dense(130, activation ='relu')(x6b)
#x6b = BatchNormalization()(x6b)
x7b = Dropout(0.5)(x7b)
x7b = Activation('relu')(x7b)
x7b = Dense(130)(x7b)
#x7b = BatchNormalization()(x7b)
x8b = merge([x7b, x6b], mode='sum')
#x8b = Activation('relu')(x8b)
x8b = Dropout(0.2)(x8b)
		

x8b = Activation('relu')(x8b)
x9b = Dense(130, activation ='relu')(x8b)
#x8b = BatchNormalization()(x8b)
x9b = Dropout(0.)(x9b)
x9b = Activation('relu')(x9b)
x9b = Dense(130)(x9b)
#x9b = BatchNormalization()(x9b)
x10b = merge([x9b, x8b], mode='sum')
#x10b = Activation('relu')(x10b)
x10b = Dropout(0.2)(x10b)
		


		
		
x21b = Dense(130, activation ='relu')(x10b)
x21b = Dropout(0.4)(x21b)
		
out_test = Dense(2, activation="softmax")(x21b)
		
model_test = Model(input_test, out_test)

admax = Adamax(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#decay ? 0.002
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=1, min_lr=0.00001)

		
callbacks = [
	EarlyStopping(monitor='val_loss', patience=15, verbose=0),
	ModelCheckpoint("weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
	reduce_lr
	
]

model_test.compile(optimizer=admax, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent



Y3 = np_utils.to_categorical(Y_test, 2) # convert class vectors to binary class matrices
			
		
# Sample weights

c, r = sample_weights_test.shape
sample_weights_test = sample_weights_test.reshape(c,)
sample_weights_test=np.absolute(sample_weights_test)
		
		
# Fit the model		
model_test.fit(X_test[0:int(len(X_test)*Adapted_percentage)], Y3[0:int(len(X_test)*Adapted_percentage)],validation_split=0.2, nb_epoch=100, batch_size=400, sample_weight=sample_weights_test[0:int(len(X_test)*Adapted_percentage)], shuffle=True, verbose=0, callbacks=callbacks)#, class_weight=class_weight)




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


Score_signal = Score_signal / 347831
Score_background = Score_background / 116351 

print("Score_signal = "+str(Score_signal))

print("Score_background = "+str(Score_background))

"""
model.fit(X2, Y2, validation_data=(X, Y), nb_epoch=1300, batch_size=96)
# Final evaluation of the model
scores = model.evaluate(X, Y, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
"""


#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#test

#scores = model.evaluate(X, Y)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Let's compute the AUC
Y_pred = np.zeros(464182)

for i in range(len(Y_pred)):
	if (i < len(Z_test)):
		Y_pred[i]= Z_test[i,1]*2-1
	else:
		Y_pred[i]= Z_train[i-len(Z_test),1]*2-1

Y_true = np.hstack([Y_train, Y_test])


r_score = roc_auc_score(Y_true, Y_pred, average='macro', sample_weight=weights_shuffled)

print(r_score)


