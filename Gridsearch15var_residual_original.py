# Create first network with Keras
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Activation, Dropout,Layer, Merge, Input, merge
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

def create_model(nr1, nr2, nr3, nr4, nr5, nr6, nr7, nr8):
    # create model
    #L=WinnerTakeAll1D_GaborMellis(spatial=1, OneOnX=WTAX)
   
    

    input_grid = Input(shape=(16,))

    x2 = Dense(nr1, init='normal', activation='relu')(input_grid)
    #x2 = Dropout(dr1)(x1)
    

    x3 = Dense(nr2, activation ='relu')(x2)
    x3 = Dense(nr2, activation ='relu')(x3)
    x4 = merge([x3, x2], mode='sum')
    #x4= Dropout(dr2)(x4)

    x5 = Dense(nr3, activation ='relu')(x4)
    x5 = Dense(nr3, activation ='relu')(x5)
    x6 = merge([x5, x4], mode='sum')
    #x6 = Dropout(dr2)(x6)

    x7 = Dense(nr4, activation ='relu')(x6)
    x7 = Dense(nr4, activation ='relu')(x6)
    x8 = merge([x7, x6], mode='sum')
    #x8 = Dropout(dr2)(x7)

    x9 = Dense(nr5, activation ='relu')(x8)
    x9 = Dense(nr5, activation ='relu')(x9)
    x10 = merge([x9, x8], mode='sum')
    #x10 = Dropout(dr3)(x9)

    x11 = Dense(nr6, activation ='relu')(x10)
    x11 = Dense(nr6, activation ='relu')(x11)
    x12 = merge([x11, x10], mode='sum')
    #x12 = Dropout(dr3)(x11)

    x13 = Dense(nr7, activation ='relu')(x12)
    x13 = Dense(nr7, activation ='relu')(x13)
    x13 = merge([x13, x12], mode='sum')
    #x12 = Dropout(dr3)(x11)

    x14 = Dense(nr8, activation ='relu')(x13)
    x14 = Dense(nr8, activation ='relu')(x14)
    x14 = merge([x14, x13], mode='sum')
    #x12 = Dropout(dr3)(x11)
    
    x15 = Dense(nr9, activation ='relu')(x14)
    #x14 = Dropout(dr3)(x13)
   
    output = Dense(2, activation='softmax')(x15)

    model = Model(input_grid, output)
    
    admax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(optimizer=admax, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent
    return model


# load dataset
dataframe = pandas.read_csv("DatabaseZiyu.csv", header=None)
dataset = dataframe.values

np.random.shuffle(dataset)

X = dataset[0:50000,0:16].astype(float)
Y = dataset[0:50000,17:18].astype(float)


# Preprocessing

X -= np.mean(X, axis = 0) # center

#Normalisation of inputs.

X /= np.std(X, axis = 0) # normalize

Y = np_utils.to_categorical(Y, 2) # convert class vectors to binary class matrices

# create model
model = KerasClassifier(build_fn=create_model, nb_epoch=5, batch_size=400, verbose=1)


kappa_scorer = make_scorer(cohen_kappa_score)
# define the grid search parameters*
nr1 = [50,100,150]
nr2 = [50,100,150]
nr4 = [50,100,150]
nr5 = [50,100,150]
nr6 = [50,100,150]
nr7 = [50,100,150]
nr8 = [50,100,150]


dr1 = [0.2,0.4]
dr2 = [0.2,0.4]
dr3 = [0.2,0.4]
dr4 = [0.2,0.4]
dr5 = [0.2,0.4]

lr = [0.001, 0.0001, 0.0005]


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

param_grid = dict(nr1=nr1, nr2=nr2, nr3=nr3, nr4=nr4, nr5=nr5, nr6=nr6, nr7=nr7, nr8=nr8)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv =2)#, verbose=1)

grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
