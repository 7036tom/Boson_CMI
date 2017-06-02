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

def create_model(nr1, nr2, nr3, nr4, nr5, dr1, dr2, dr3, dr4, dr5, lr):
    # create model
    #L=WinnerTakeAll1D_GaborMellis(spatial=1, OneOnX=WTAX)
   
    

    input_grid = Input(shape=(16,))

    x1 = Dense(nr1, init='normal', activation='relu')(input_grid)
    x2 = Dropout(dr1)(x1)
    x2 = merge([x2, input_grid], mode='concat')

    x3 = Dense(nr2, activation ='relu')(x2)
    x4= Dropout(dr2)(x3)
    x4 = merge([x4, x2], mode='concat')
    
    x5 = Dense(nr2, activation ='relu')(x4)
    x6 = Dropout(dr2)(x5)
    x6 = merge([x6, x4], mode='concat')
    
    x7 = Dense(nr2, activation ='relu')(x6)
    x8 = Dropout(dr2)(x7)
    x8 = merge([x8, x6], mode='concat')

    x9 = Dense(nr3, activation ='relu')(x8)
    x10 = Dropout(dr3)(x9)
    x10 = merge([x10, x8], mode='concat')

    x11 = Dense(nr3, activation ='relu')(x10)
    x12 = Dropout(dr3)(x11)
    x12 = merge([x12, x10], mode='concat')
    
    x13 = Dense(nr3, activation ='relu')(x12)
    x14 = Dropout(dr3)(x13)
    x14 = merge([x14, x12], mode='concat')
    
    x15 = Dense(nr4, activation ='relu')(x14)
    x16 = Dropout(dr4)(x15)
    x16 = merge([x16, x14], mode='concat')
    
    x17 = Dense(nr4, activation ='relu')(x16)
    x18 = Dropout(dr4)(x17)
    x18 = merge([x18, x16], mode='concat')

    x19 = Dense(nr5, activation ='relu')(x18)
    x20 = Dropout(dr5)(x19)

    output = Dense(2, activation='softmax')(x20)

    model = Model(input_grid, output)
    
    admax = Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
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
nr1 = [20,40,60]
nr2 = [20,40,60]
nr3 = [20,40,60]
nr4 = [20,40,60]
nr5 = [20,40,60]

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

param_grid = dict(nr1=nr1, nr2=nr2, nr3=nr3, nr4=nr4, nr5=nr5, dr1=dr1, dr2=dr2, dr3=dr3, dr4=dr4, dr5=dr5, lr=lr)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv =2)#, verbose=1)

grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
