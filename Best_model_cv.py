# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Reshape, Activation, Dropout,Layer, LocallyConnected1D, LocallyConnected2D, Convolution1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, MaxPooling2D, Merge
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import SGD
from keras.optimizers import RMSprop, Adamax
import matplotlib.pyplot as plt
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
seed = 7
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
dataframe = pandas.read_csv("Database.csv", header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables

X = dataset[0:942548,0:16].astype(float)
Events_number = dataset[0:942548,16:17].astype(float)
Y = dataset[0:942548,17:18].astype(float)



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

X_train = dataset[0:nbr_pairs, 0:16]
X_test = dataset[0:942548-nbr_pairs, 0:16]
Y_train = dataset[0:nbr_pairs, 17:18]
Y_test = dataset[0:942548-nbr_pairs, 17:18]


for i in range(942548):
	if (Events_number[i]%2 == 0):
		X_train[p] = X[i]
		Y_train[p] = Y[i]
		#print(p)
		p += 1
		
		
	else:
		X_test[q] = X[i]
		Y_test[q] = Y[i]
		print(q, i)
		q += 1


model_train = [Sequential(), Sequential(),Sequential(), Sequential(), Sequential()]
model_test = [Sequential(), Sequential(),Sequential(), Sequential(), Sequential()]

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


# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []


c, r = Y_train.shape
Y_train = Y_train.reshape(c,)

c, r = Y_test.shape
Y_test = Y_test.reshape(c,)

j = 0

for i in range(1):
	for train, test in kfold.split(X_train, Y_train):
		print(j) 
		print("ieme fold")
		
		

		
		# create model(135/75/105)
		model_train[j].add(Dense(280, input_dim=16, init='normal', activation='relu' ,W_regularizer=l1l2(l1=1E-6, l2=1E-5))) #W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))
		#model[j].add(Dropout(0.4))
		#model[j].add(L)
		model_train[j].add(Dense(370,  activation ='relu'))
		#model[j].add(Dropout(0.4))
		#model[j].add(L)
		model_train[j].add(Dense(120,  activation ='relu'))
		#model[j].add(Dropout(0.4))
		#model[j].add(L)
		
		model_train[j].add(Dense(2))
		
		model_train[j].add(Activation('softmax'))

		admax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#decay ? 0.002
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.90, patience=1, min_lr=0.00001)
		"""
		def schedule(nb_epoch):
			return(0.97**nb_epoch)
		"""
		
		callbacks = [
   			EarlyStopping(monitor='val_loss', patience=15, verbose=0),
   			ModelCheckpoint("/home/admin-7036/Documents/Projet python/bosongit/weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
   			reduce_lr
   			#LearningRateScheduler(schedule),
		]

		model_train[j].compile(optimizer=admax, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent
		#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) # Gradient descent
		#model[j].compile(optimizer='adam', loss='categorical_crossentropy', metrics=[''sparse_categorical_accuracy'']) # Gradient descent

		print (model_train[j].summary()) # Affiche les details du reseau !

		Y3 = np_utils.to_categorical(Y_train, 2) # convert class vectors to binary class matrices
			
		
		# Early stopping.
		
		
		
		# Fit the model		
		model_train[j].fit(X_train[train], Y3[train],validation_data=(X_train[test], Y3[test]), nb_epoch=100, batch_size=93, class_weight=class_weight, shuffle=True, verbose=1, callbacks=callbacks)#, class_weight=class_weight)

		"""
		# list all data in history
		print(history.history.keys())
		# summarize history for accuracy
		plt.plot(history.history['sparse_categorical_accuracy'])
		plt.plot(history.history['val_sparse_categorical_accuracy'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		#	 summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		#model.fit(X2[train], Y3[train], nb_epoch=200, batch_size=96)
		"""
		
		scores = model_train[j].evaluate(X_train[test], Y3[test])
		print("%s: %.2f%%" % (model_train[j].metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
		j = j +1;

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))







j = 0

for i in range(1):
	for train, test in kfold.split(X_test, Y_test):
		print(j) 
		print("ieme fold")
		
		

		
		# create model(135/75/105)
		model_test[j].add(Dense(280, input_dim=16, init='normal', activation='relu' ,W_regularizer=l1l2(l1=1E-6, l2=1E-5))) #W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))
		#model[j].add(Dropout(0.4))
		#model[j].add(L)
		model_test[j].add(Dense(370,  activation ='relu'))
		#model[j].add(Dropout(0.4))
		#model[j].add(L)
		model_test[j].add(Dense(120,  activation ='relu'))
		#model[j].add(Dropout(0.4))
		#model[j].add(L)
		
		model_test[j].add(Dense(2))
		
		model_test[j].add(Activation('softmax'))

		admax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#decay ? 0.002
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.97, patience=1, min_lr=0.00001)
		"""
		def schedule(nb_epoch):
			return(0.97**nb_epoch)
		"""
		
		callbacks = [
   			EarlyStopping(monitor='val_loss', patience=15, verbose=0),
   			ModelCheckpoint("/home/admin-7036/Documents/Projet python/bosongit/weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
   			reduce_lr
   			#LearningRateScheduler(schedule),
		]

		model_test[j].compile(optimizer=admax, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent
		#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) # Gradient descent
		#model[j].compile(optimizer='adam', loss='categorical_crossentropy', metrics=[''sparse_categorical_accuracy'']) # Gradient descent

		print (model_test[j].summary()) # Affiche les details du reseau !

		Y3 = np_utils.to_categorical(Y_test, 2) # convert class vectors to binary class matrices
			
		
		# Early stopping.
		
		
		
		# Fit the model		
		model_test[j].fit(X_test[train], Y3[train],validation_data=(X_test[test], Y3[test]), nb_epoch=100, batch_size=93, class_weight=class_weight, shuffle=True, verbose=1, callbacks=callbacks)#, class_weight=class_weight)

		"""
		# list all data in history
		print(history.history.keys())
		# summarize history for accuracy
		plt.plot(history.history['sparse_categorical_accuracy'])
		plt.plot(history.history['val_sparse_categorical_accuracy'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		#	 summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		#model.fit(X2[train], Y3[train], nb_epoch=200, batch_size=96)
		"""
		
		scores = model_test[j].evaluate(X_test[test], Y3[test])
		print("%s: %.2f%%" % (model_test[j].metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
		j = j +1;

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))



merged_model_train = Sequential()
merged_model_train.add(Merge(model_train, mode='ave'))

Z_train = merged_model_train.predict([X_test, X_test, X_test, X_test, X_test], batch_size=32, verbose=0)

merged_model_test = Sequential()
merged_model_test.add(Merge(model_test, mode='ave'))

Z_test = merged_model_test.predict([X_train, X_train, X_train, X_train, X_train], batch_size=32, verbose=0)



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


Score_signal = Score_signal / 627066.0
Score_background = Score_background / 315482.0 

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

