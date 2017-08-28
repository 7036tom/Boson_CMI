
from __future__ import unicode_literals
from __future__ import absolute_import

import sys
import copy
import inspect
import types
import inspect
from sklearn.utils.validation import column_or_1d

import numpy as np

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from sklearn.ensemble import AdaBoostClassifier
inspect.getfile(AdaBoostClassifier)

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
		print(sample_weight[0:10])
		#sample_weight= None # not necessary for bagging
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
from sklearn.ensemble import BaggingClassifier


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

# Sample weights

c, r = sample_weights_train.shape
sample_weights_train = sample_weights_train.reshape(c,)
sample_weights_train=np.absolute(sample_weights_train)
c, r = sample_weights_test.shape
sample_weights_test = sample_weights_test.reshape(c,)
sample_weights_test=np.absolute(sample_weights_test)

# Percentage definition
# The point is that we want to reduce the training base by a factor without touching the testing database size. Being said that the testing database is computed automatically with validation_split, it leads to the following definition.


Percentage = float(sys.argv[2])
Adapted_percentage = Percentage*0.8+0.2

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.97, patience=1, min_lr=0.00001)
callbacks = [
	#EarlyStopping(monitor='val_loss', patience=10, verbose=0),
	ModelCheckpoint("weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
	reduce_lr
]


nb_epoc = int(sys.argv[3])
nb_esti = int(sys.argv[4])

# Model_

def create_model():

	model =Sequential()

	model.add(Dense(100, input_dim=147, init='normal', activation='relu' ,W_regularizer=l1l2(l1=1E-6, l2=1E-5), activity_regularizer=l1l2(l1=0, l2=1e-7))) #W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))
		
	model.add(Dropout(0.1))
	model.add(Dense(540,  activation ='relu', W_regularizer=l1l2(l1=1e-07, l2=0), activity_regularizer=l1l2(l1=0, l2=5e-7)))
	model.add(Dropout(0.3))
	model.add(Dense(310,  activation ='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(450,  activation ='relu'))
	model.add(Dropout(0.4))
	
	
	model.add(Dense(1))	
	model.add(Activation('sigmoid'))

	admax = Adamax(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#decay ? 0.002
	
	model.compile(optimizer=admax, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent

	return model


# Train Boosting classifiers

model_train_sklearn = KerasClassifier(build_fn=create_model, batch_size=400, nb_epoch=nb_epoc, validation_split=0.2/Adapted_percentage)

adaboost_model_train = BaggingClassifier(base_estimator=model_train_sklearn, n_estimators=nb_esti, random_state=seed)

adaboost_model_train.fit(X_train[0:int(len(X_train)*Adapted_percentage)], Y_train[0:int(len(X_train)*Adapted_percentage)], sample_weight=sample_weights_train)



model_test_sklearn = KerasClassifier(build_fn=create_model,batch_size=400, nb_epoch=nb_epoc, validation_split=0.2/Adapted_percentage)

adaboost_model_test= BaggingClassifier(base_estimator=model_test_sklearn, n_estimators=nb_esti, random_state=seed)

adaboost_model_test.fit(X_test[0:int(len(X_test)*Adapted_percentage)], Y_test[0:int(len(X_test)*Adapted_percentage)], sample_weight=sample_weights_test)





Z_train = adaboost_model_train.predict_proba(X_test)


Z_test = adaboost_model_test.predict_proba(X_train)



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


