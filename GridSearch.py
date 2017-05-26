# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Reshape, Activation, Dropout,Layer
from keras.optimizers import Adamax
from keras import backend as K
from theano import tensor as T
import numpy as np
import pandas
import math
from keras.regularizers import l1l2
from keras.utils import np_utils
from sklearn.metrics import cohen_kappa_score, make_scorer


# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Function to create model, required for KerasClassifier

def create_model(w1l1, w1l2, w2l1, w2l2, w3l1, w3l2):
    # create model
    #L=WinnerTakeAll1D_GaborMellis(spatial=1, OneOnX=WTAX)
    model = Sequential()
    


    model.add(Dense(100, input_dim=95, init='normal', activation='relu', W_regularizer=l1l2(l1=0.0005, l2=0.0001)))# ,W_regularizer=l1l2(l1=9E-7, l2=5e-07))) #W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))
    #model.add(L)
    model.add(Dropout(0.1))
    model.add(Dense(540, activation ='relu', W_regularizer=l1l2(l1=w1l1, l2=0) ))
    #model.add(L)
    model.add(Dropout(0.3))
    model.add(Dense(310, activation ='relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(440, activation ='relu', W_regularizer=l1l2(l1=0, l2=w3l2)))
    model.add(Dropout(0.4))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    admax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(optimizer=admax, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent
    return model


# load dataset
dataframe = pandas.read_csv("Database_95_var.csv", header=None)
dataset = dataframe.values

np.random.shuffle(dataset)

X1 = dataset[0:50000,2:17].astype(float)
X2 = dataset[0:50000,22:102].astype(float)
X = np.hstack([X1, X2])

Y = dataset[0:50000,0:1].astype(float)


# Preprocessing

X -= np.mean(X, axis = 0) # center

#Normalisation of inputs.

X /= np.std(X, axis = 0) # normalize

Y = np_utils.to_categorical(Y, 2) # convert class vectors to binary class matrices

# create model
model = KerasClassifier(build_fn=create_model, nb_epoch=5, batch_size=80, verbose=1)


kappa_scorer = make_scorer(cohen_kappa_score)
# define the grid search parameters*
dr1 = [0.1,0.2,0.3,0.4,0.5]
dr2 = [0.1,0.2,0.3,0.4,0.5]
dr3 = [0.1,0.2,0.3,0.4,0.5]
dr4 = [0.1,0.2,0.3,0.4,0.5]
neurons1 = [70, 100, 120, 150]
neurons2 = [450, 480, 510, 540]
neurons3 = [250, 280, 310, 340]
neurons4 = [450, 480, 510, 540]

w1l1 = [0, 0.0001, 0.0005 ,0.00001, 0.00005, 0.000001, 0.000005, 0.0000001, 0.0000005]
w1l2 = [0, 0.00001]
w2l1 = [0, 0.00001]
w2l2 = [0, 0.00001]
w3l1 = [0, 0.00001]
w3l2 = [0, 0.0001, 0.0005 ,0.00001, 0.00005, 0.000001, 0.000005, 0.0000001, 0.0000005]

batch_size = [80,83,86,89,92,95,98, 100]
epochs = [100, 120, 140, 160, 180, 200]
WTAX=[3,4,5]
l1_value = [0, 0.0001, 0.0005 ,0.00001, 0.00005, 0.000001, 0.000005, 0.0000001, 0.0000005]
l2_value = [0, 0.0001, 0.0005 ,0.00001, 0.00005, 0.000001, 0.000005, 0.0000001, 0.0000005]
l_rate = [0.001, 0.002, 0.003 ,0.004]
decay = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0]

param_grid = dict(w1l1=w1l1, w1l2=w1l2, w2l1=w2l1, w2l2=w2l2, w3l1=w3l1, w3l2=w3l2)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv =2)#, verbose=1)

grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
