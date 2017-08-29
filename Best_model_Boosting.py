from __future__ import unicode_literals
from __future__ import absolute_import

import copy
import inspect
import types
import inspect
from sklearn.utils.validation import column_or_1d

import numpy as np

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier


class BaseWrapper(object):
	"""Base class for the Keras scikit-learn wrapper.

	Warning: This class should not be used directly.
	Use descendant classes instead.

	# Arguments
		build_fn: callable function or class instance
		**sk_params: model parameters & fitting parameters

	The build_fn should construct, compile and return a Keras model, which
	will then be used to fit/predict. One of the following
	three values could be passed to build_fn:
	1. A function
	2. An instance of a class that implements the __call__ method
	3. None. This means you implement a class that inherits from either
	`KerasClassifier` or `KerasRegressor`. The __call__ method of the
	present class will then be treated as the default build_fn.

	`sk_params` takes both model parameters and fitting parameters. Legal model
	parameters are the arguments of `build_fn`. Note that like all other
	estimators in scikit-learn, 'build_fn' should provide default values for
	its arguments, so that you could create the estimator without passing any
	values to `sk_params`.

	`sk_params` could also accept parameters for calling `fit`, `predict`,
	`predict_proba`, and `score` methods (e.g., `nb_epoch`, `batch_size`).
	fitting (predicting) parameters are selected in the following order:

	1. Values passed to the dictionary arguments of
	`fit`, `predict`, `predict_proba`, and `score` methods
	2. Values passed to `sk_params`
	3. The default values of the `keras.models.Sequential`
	`fit`, `predict`, `predict_proba` and `score` methods

	When using scikit-learn's `grid_search` API, legal tunable parameters are
	those you could pass to `sk_params`, including fitting parameters.
	In other words, you could use `grid_search` to search for the best
	`batch_size` or `nb_epoch` as well as the model parameters.
	"""

	def __init__(self, build_fn=None,**sk_params):# sample_weight= None, 
		self.build_fn = build_fn
		#self.sample_weight=sample_weight
		self.sk_params = sk_params
		
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.97, patience=1, min_lr=0.00001)
		callbacks_pred = [
			#EarlyStopping(monitor='val_loss', patience=10, verbose=0),
			ModelCheckpoint("weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
			reduce_lr
		]
		self.callbacks=callbacks_pred
		self.classes_ = np.array([0, 1])
		self.n_classes_ = None

		self.check_params(sk_params)

	def check_params(self, params):
		"""Checks for user typos in "params".

		# Arguments
			params: dictionary; the parameters to be checked

		# Raises
			ValueError: if any member of `params` is not a valid argument.
		"""
		legal_params_fns = [Sequential.fit, Sequential.predict,
							Sequential.predict_classes, Sequential.evaluate]
		if self.build_fn is None:
			legal_params_fns.append(self.__call__)
		elif (not isinstance(self.build_fn, types.FunctionType) and
			  not isinstance(self.build_fn, types.MethodType)):
			legal_params_fns.append(self.build_fn.__call__)
		else:
			legal_params_fns.append(self.build_fn)

		legal_params = []
		for fn in legal_params_fns:
			legal_params += inspect.getargspec(fn)[0]
		legal_params = set(legal_params)

		for params_name in params:
			if params_name not in legal_params:
				raise ValueError('{} is not a legal parameter'.format(params_name))

	def get_params(self, **params):
		"""Gets parameters for this estimator.

		# Returns
			params : dict
				Dictionary of parameter names mapped to their values.
		"""
		res = copy.deepcopy(self.sk_params)
		res.update({'build_fn': self.build_fn})
		return res

	def set_params(self, **params):
		"""Sets the parameters of this estimator.

		# Arguments
			**params: Dictionary of parameter names mapped to their values.

		# Returns
			self
		"""
		self.check_params(params)
		self.sk_params.update(params)
		return self

	def fit(self, x, y, sample_weight=None, **kwargs):
		"""Constructs a new model with `build_fn` & fit the model to `(x, y)`.

		# Arguments
			x : array-like, shape `(n_samples, n_features)`
				Training samples where n_samples in the number of samples
				and n_features is the number of features.
			y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
				True labels for X.
			**kwargs: dictionary arguments
				Legal arguments are the arguments of `Sequential.fit`

		# Returns
			history : object
				details about the training history at each epoch.
		"""
		#print(inspect.getargspec(self.build_fn()))
		if self.build_fn is None:
			self.model = self.__call__(**self.filter_sk_params(self.__call__))
		elif (not isinstance(self.build_fn, types.FunctionType) and
			  not isinstance(self.build_fn, types.MethodType)):
			
			self.model = self.build_fn(
				**self.filter_sk_params(self.build_fn.__call__(None, None)))
		else:
			self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

		loss_name = self.model.loss
		if hasattr(loss_name, '__name__'):
			loss_name = loss_name.__name__
		if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
			y = to_categorical(y)

		fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
		fit_args.update(kwargs)
		#print(sample_weight[0:10])

		ratio = 1/np.mean(sample_weight)

		print(ratio)
		for i in range(len(sample_weight)):
			sample_weight[i]*=ratio
		fit_args.update({'sample_weight': sample_weight})

		print(fit_args)
		
		history = self.model.fit(x, y,  **fit_args)

		return history

	def filter_sk_params(self, fn, override=None):
		"""Filters `sk_params` and return those in `fn`'s arguments.

		# Arguments
			fn : arbitrary function
			override: dictionary, values to override sk_params

		# Returns
			res : dictionary dictionary containing variables
				in both sk_params and fn's arguments.
		"""
		override = override or {}
		res = {}
		fn_args = inspect.getargspec(fn)[0]
		for name, value in self.sk_params.items():
			if name in fn_args:
				res.update({name: value})
		res.update(override)
		return res


class KerasClassifier(BaseWrapper):
	"""Implementation of the scikit-learn classifier API for Keras.
	"""

	def fit(self, x, y, sample_weight=None, **kwargs):
		super(KerasClassifier, self).fit(
            x, y,
            sample_weight=sample_weight)
		return self

	def predict(self, x, **kwargs):
		"""Returns the class predictions for the given test data.

		# Arguments
			x: array-like, shape `(n_samples, n_features)`
				Test samples where n_samples in the number of samples
				and n_features is the number of features.
			**kwargs: dictionary arguments
				Legal arguments are the arguments
				of `Sequential.predict_classes`.

		# Returns
			preds: array-like, shape `(n_samples,)`
				Class predictions.
		"""
		kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)
		return self.model.predict_classes(x, **kwargs)

	def predict_proba(self, x, **kwargs):
		"""Returns class probability estimates for the given test data.

		# Arguments
			x: array-like, shape `(n_samples, n_features)`
				Test samples where n_samples in the number of samples
				and n_features is the number of features.
			**kwargs: dictionary arguments
				Legal arguments are the arguments
				of `Sequential.predict_classes`.

		# Returns
			proba: array-like, shape `(n_samples, n_outputs)`
				Class probability estimates.
				In the case of binary classification,
				tp match the scikit-learn API,
				will return an array of shape '(n_samples, 2)'
				(instead of `(n_sample, 1)` as in Keras).
		"""
		kwargs = self.filter_sk_params(Sequential.predict_proba, kwargs)
		probs = self.model.predict_proba(x, **kwargs)

		# check if binary classification
		if probs.shape[1] == 1:
			# first column is probability of class 0 and second is of class 1
			probs = np.hstack([1 - probs, probs])
		return probs

	def score(self, x, y, **kwargs):
		"""Returns the mean accuracy on the given test data and labels.

		# Arguments
			x: array-like, shape `(n_samples, n_features)`
				Test samples where n_samples in the number of samples
				and n_features is the number of features.
			y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
				True labels for x.
			**kwargs: dictionary arguments
				Legal arguments are the arguments of `Sequential.evaluate`.

		# Returns
			score: float
				Mean accuracy of predictions on X wrt. y.

		# Raises
			ValueError: If the underlying model isn't configured to
				compute accuracy. You should pass `metrics=["accuracy"]` to
				the `.compile()` method of the model.
		"""
		kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

		loss_name = self.model.loss
		if hasattr(loss_name, '__name__'):
			loss_name = loss_name.__name__
		if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
			y = to_categorical(y)

		outputs = self.model.evaluate(x, y, **kwargs)
		if not isinstance(outputs, list):
			outputs = [outputs]
		for name, output in zip(self.model.metrics_names, outputs):
			if name == 'acc':
				return output
		raise ValueError('The model is not configured to compute accuracy. '
						 'You should pass `metrics=["accuracy"]` to '
						 'the `model.compile()` method.')


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


def AMS(estimator, y_true, y_probs):
	
	Z = estimator.predict(y_true, batch_size=32, verbose=0)

	Y = y_probs

	s = 0
	b = 0
	
	for i in range(0,len(y_true)):
		if (Y[i][1]>Y[i][0]):
			if (Z[i] == 1):
				s = s + W[i][0]
			if (Z[i] == 0):
				b = b + W[i][0]
	
	br = 10.0
	radicand = 2 *( (s+b+br) * math.log(1.0 + s/(b+br)) - s)
	AMS = math.sqrt(radicand)
	return AMS


# load dataset
dataframe = pandas.read_csv("training.csv", header=None)
dataset = dataframe.values

dataframe2 = pandas.read_csv("test.csv", header=None)
dataset2 = dataframe2.values

# Parametres
Seuil = 0.5

# Let's move all the azymuth angle features at the end of X (12, 16, 19, 21, 16, 29)
# 12 <-> 28 PhiCentrality
dataset[:,[12,28]]
#dataset2[:,[12,28]]
# 16 <-> 16 Phitau
dataset[:,[16,16]]
#dataset2[:,[16,16]]
# 19 <-> 19 Philep
dataset[:,[19,19]]
#dataset2[:,[19,19]]
# 21 <-> 21 Phimet
dataset[:,[21,21]]
#dataset2[:,[21,21]]
# 16 Phijetleading
# 29 Phijetsubleading


# split into input (X) and output (Y) variables
W = dataset[0:250000:,31:32]
X = dataset[0:10000:,1:31].astype(float)
Y = dataset[0:10000:,32].astype(float)
X2 = dataset[0:250000:,1:31].astype(float)


Y2 = dataset[0:250000:,32].astype(float)
Z2 = dataset2[0:550000,1:31].astype(float)



# Implementation of advanced features in x and x2
# Notes : minv(tau,lep) (3) | 


for i in range(250000):
	# Implementation of Phi derived features
	X2[i,20] = min(dataset[i,16]-dataset[i,19],dataset[i,16]-dataset[i,21],dataset[i,19]-dataset[i,21])
	X2[i,18] = min(dataset[i,16]-dataset[i,21],dataset[i,19]-dataset[i,21])
	X2[i,15] = min(dataset[i,16]-dataset[i,19],dataset[i,16]-dataset[i,21])
	X2[i,27] = dataset[i,19]-dataset[i,21]

	# Implementation of mass based features
	#x2[i,4] = log(1)#+dataset[i, 3])

for i in range(550000):
	# Implementation of Phi derived features
	Z2[i,20] = min(dataset2[i,16]-dataset2[i,19],dataset2[i,16]-dataset2[i,21],dataset2[i,19]-dataset2[i,21])
	Z2[i,18] = min(dataset2[i,16]-dataset2[i,21],dataset2[i,19]-dataset2[i,21])
	Z2[i,15] = min(dataset2[i,16]-dataset2[i,19],dataset2[i,16]-dataset2[i,21])
	Z2[i,27] = dataset2[i,19]-dataset2[i,21]
"""
	# Implementation of mass based features
	#x2[i,4] = log(1)#+dataset[i, 3])
"""
# Replacing -999 by average of non-999

Missing = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
nb_missing = 0
for i in range(250000):
	for j in range(30):
		if (X2[i][j]==-999.0):
			Missing[j]=1

Missing_position = [0,0,0,0,0,0,0,0,0,0,0]
p = 0
for i in range(30):
	if (Missing[i]==1):
		Missing_position[p]=i
		p = p + 1


Mean = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

q = 0
for i in Missing_position:
	p = 0
	for j in range(250000):
		if (X2[j][i]!=-999.0):
			Mean[q]+=X2[j][i]
			p += 1			
	Mean[q] /= p
	q += 1

q = 0
for i in Missing_position:
	for j in range(250000):
		if (X2[j][i]==-999.0):
			X2[j][i] = Mean[q]
	q +=1
	
Mean = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

q = 0
for i in Missing_position:
	p = 0
	for j in range(550000):
		if (Z2[j][i]!=-999.0):
			Mean[q]+=Z2[j][i]
			p += 1		
	Mean[q] /= p
	q += 1

q = 0
for i in Missing_position:
	for j in range(550000):
		if (Z2[j][i]==-999.0):
			Z2[j][i] = Mean[q]
	q += 1			






X2 -= np.mean(X2, axis = 0) # center
Z2 -= np.mean(Z2, axis = 0) # center


#Normalisation of inputs.


X2 /= np.std(X2, axis = 0) # normalize
Z2 /= np.std(Z2, axis = 0) # normalize

"""
#In case of locallyConnected.
X = X.reshape(X.shape[0], 1, 7, 4).astype('float32')
X2 = X2.reshape(X2.shape[0],1, 7, 4).astype('float32')
Z2 = Z2.reshape(Z2.shape[0],1, 7, 4).astype('float32')
"""


Y = np_utils.to_categorical(Y, 2) # convert class vectors to binary class matrices
#Y2 = np_utils.to_categorical(Y2, 2)


#input_shape=(28,1)
model = [Sequential(), Sequential(),Sequential(), Sequential(), Sequential()]
#rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0001)


# Weights more adapted to the class imbalanced of the issue.

nbr_signals = 0.0
for i in range(250000):
	if (Y2[i] == 1):
		nbr_signals = nbr_signals + 1

weight1 = 1
weight0 = 250000/nbr_signals
print(weight0)
print(weight1)
class_weight = {0 : weight0 ,
	1: weight1}

"""
model1 = Sequential()

model1.add(Dense(285, input_dim=30, init='normal', activation='relu' ,W_regularizer=l1l2(l1=9E-7, l2=5e-07))) #W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))
model1.add(Dropout(0.35))
#model.add(L)
model1.add(Dense(360,  activation ='relu'))
model1.add(Dropout(0.35))
#model.add(L)
model1.add(Dense(270,  activation ='relu'))
model1.add(Dropout(0.35))
#model.add(L)

model1.add(Dense(1))
		
model1.add(Activation('sigmoid'))

admax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#decay ? 0.002
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.97, patience=1, min_lr=0.00001)
	
callbacks = [
 #EarlyStopping(monitor='val_loss', patience=15, verbose=0),
ModelCheckpoint("/home/admin-7036/Documents/Projet python/bosongit/weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
ReduceLROnPlateau(monitor='val_loss', factor=0.97, patience=1, min_lr=0.00001)
#LearningRateScheduler(schedule),
] 

model1.compile(optimizer=admax, loss='binary_crossentropy', metrics=['sparse_categorical_accuracy']) # Gradient descent

model1.fit(X2, Y2,validation_split=0.2, nb_epoch=200, batch_size=400, class_weight=class_weight, shuffle=True, verbose=1, callbacks=callbacks)

model1.save_weights('weights_boosting.h5')
"""

def create_model():

	model = Sequential()
	# 30
	model.add(Dense(285, input_dim=30, init='normal', activation='relu' ,trainable=True, W_regularizer=l1l2(l1=9E-7, l2=5e-07))) #W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))
	model.add(Dropout(0.35))
	model.add(L)
	model.add(Dense(360,  activation ='relu'))
	model.add(Dropout(0.35))
	model.add(L)
	model.add(Dense(270,  activation ='relu'))
	model.add(Dropout(0.35))
	model.add(L)

	model.add(Dense(1))
		
	model.add(Activation('sigmoid'))

	#model.load_weights('weights_boosting.h5')

	admax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#decay ? 0.002
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.97, patience=1, min_lr=0.00001)
	callbacks = [
		ModelCheckpoint("/home/admin-7036/Documents/Projet python/bosongit/weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
		ReduceLROnPlateau(monitor='val_loss', factor=0.97, patience=1, min_lr=0.00001)
	] 

	model.compile(optimizer=admax, loss='binary_crossentropy', metrics=['sparse_categorical_accuracy']) # Gradient descent
	#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) # Gradient descent
	#model[j].compile(optimizer='adam', loss='categorical_crossentropy', metrics=[''sparse_categorical_accuracy'']) # Gradient descent

	print (model.summary()) # Affiche les details du reseau !

	return model
	#return {'model': model}

model_train_sklearn = KerasClassifier(build_fn=create_model, batch_size=400, nb_epoch=200, class_weight=class_weight)

adaboost_model_train = BaggingClassifier(base_estimator=model_train_sklearn, n_estimators=10,random_state=7)#, learning_rate=1, algorithm='SAMME.R', 

adaboost_model_train.fit(X2, Y2)


Result_test = adaboost_model_train.predict(Z2)


c = csv.writer(open("Submission.csv", "wb"))
c.writerow(["EventId","RankOrder","Class"])
for i in range(550000):
	if (Result_test[i] > 0.5):
		c.writerow([350000+i,i+1,'s' ])
	else:
		c.writerow([i+350000,i+1,'b' ])


#----------------------------------------------------------------------------
# TO DO
# CV bagging with 10 repetitions DONE
# Momentum DONE
# Learning rate DONE
# L1 penalty DONE
# L2 penalty DONE
# AMS metric DONE
# Advanced features 4/10 (mail)
# Normalization DONE
# Elimination of Azymuth angles features DONE
# Winner takes all activation DONE 
# Constrain neurons in first layer
#
#
#
#----------------------------------------------------------------------------
