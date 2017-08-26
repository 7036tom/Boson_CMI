from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Activation, Dropout,Layer, LocallyConnected1D, LocallyConnected2D, Convolution1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, MaxPooling2D, Merge, Input, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import SGD
from keras.optimizers import RMSprop, Adamax
import matplotlib.pyplot as plt
import math
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
    
L=WinnerTakeAll1D_GaborMellis(spatial=1, OneOnX=2)

# Data processing

# load dataset


dataframe_best1 = pandas.read_csv("Database100+_best1.csv", header=None)
dataset_best1 = dataframe_best1.values
dataframe_best = pandas.read_csv("Database100+_best.csv", header=None)
dataset_best = dataframe_best.values
dataframe_best2 = pandas.read_csv("Database100+_best2.csv", header=None)
dataset_best2 = dataframe_best2.values

# split into input (X) and output (Y) variables

weights = dataset_best[0:464182, 1:2]




X1_best = dataset_best[0:464182,3:69].astype(float)
X2_best = dataset_best[0:464182,70:151].astype(float)
X_best = np.hstack([X1_best, X2_best])


X1_best1 = dataset_best1[0:464182,3:69].astype(float)
X2_best1 = dataset_best1[0:464182,70:151].astype(float)
X_best1 = np.hstack([X1_best1, X2_best1])


X1_best2 = dataset_best2[0:464182,3:69].astype(float)
X2_best2 = dataset_best2[0:464182,70:151].astype(float)
X_best2 = np.hstack([X1_best2, X2_best2])




#X_best = np.random.rand(464182,1)
#X_best1 = np.random.rand(464182,1)

# Preprocessing

X_best -= np.mean(X_best, axis = 0) # center
X_best1 -= np.mean(X_best1, axis = 0) # center
X_best2 -= np.mean(X_best2, axis = 0) # center

#Normalisation of inputs.

X_best /= np.std(X_best, axis = 0) # normalize
X_best1 /= np.std(X_best1, axis = 0) # normalize
X_best2 /= np.std(X_best2, axis = 0) # normalize



Events_number = dataset_best[0:464182,2:3].astype(float)
Y = dataset_best[0:464182,0:1]




# Creation of Train and Test files.##################################################################################################################

# Count nbr of pairs
nbr_pairs = 0
for i in range(464182):
    if (Events_number[i]%2 == 0):
        nbr_pairs += 1

X_train_best = np.copy(X_best[0:nbr_pairs])
X_test_best = np.copy(X_best[0:464182-nbr_pairs]) #21
X_train_best1 = np.copy(X_best1[0:nbr_pairs])
X_test_best1 = np.copy(X_best1[0:464182-nbr_pairs]) #21
X_train_best2 = np.copy(X_best2[0:nbr_pairs])
X_test_best2 = np.copy(X_best2[0:464182-nbr_pairs]) #21

Y_train = np.copy(dataset_best[0:nbr_pairs, 0:1])
Y_test = np.copy(dataset_best[0:464182-nbr_pairs, 0:1])

weights_train = np.copy(weights[0:nbr_pairs])
weights_test = np.copy(weights[0:942548 - nbr_pairs])

p = 0
q = 0

for i in range(464182):
    if (Events_number[i]%2 == 0):
        
        #X_train[p] = X[i]
        weights_train[p] = np.copy(weights[i])
        X_train_best[p]  = np.copy(X_best[i])
        X_train_best1[p]  = np.copy(X_best1[i])
        X_train_best2[p]  = np.copy(X_best2[i])
        Y_train[p] = Y[i]
        #print(p)
        p += 1
        
    elif (Events_number[i]%2 == 1):
        
        #X_test[q] = X[i]
        weights_test[q] = np.copy(weights[i])
        X_test_best[q]  = np.copy(X_best[i])
        X_test_best1[q]  = np.copy(X_best1[i])
        X_test_best2[q]  = np.copy(X_best2[i])
        Y_test[q] = Y[i]
        q += 1

# We free memory
X_best = None
X_best1 = None
X_best2 =None

#X_train_best1 = np.random.rand(nbr_pairs,1)
#X_test_best1 = np.random.rand(464182 - nbr_pairs,1)



#print(X_train_best[0:2])
#print(X_train_best1[0:2])
# Class weights definition

nbr_signals = 0.0
for i in range(464182):
    if (Y[i] == 1):
        nbr_signals = nbr_signals + 1

weight1 = 1
weight0 = (464182-nbr_signals)/nbr_signals

print(weight0)
print(weight1)
class_weight = {0 : 1/weight0,
    1: 1/weight1}

Classification_model_train = [Sequential(), Sequential(), Sequential(), Sequential(), Sequential()]
Classification_model_test = [Sequential(), Sequential(), Sequential(), Sequential(), Sequential()]


c, r = Y_train.shape
Y_train = Y_train.reshape(c,)

c, r = Y_test.shape
Y_test = Y_test.reshape(c,)



X_train_true = [X_train_best, X_train_best1, X_train_best2]
X_test_true = [X_test_best, X_test_best1, X_test_best2]

# Adapted percentage to maintain test_split length yet reduce train_split the way the user wants
Percentage = float(sys.argv[2])
Adapted_percentage = Percentage*0.8+0.2


 # Model

 # Model_train #################################################################################################################################################

X_train_train = [X_train_best[0:int(len(X_train_best)*Adapted_percentage)], X_train_best1[0:int(len(X_train_best)*Adapted_percentage)], X_train_best2[0:int(len(X_train_best)*Adapted_percentage)]]

digit_input = Input(shape=(147,))

x = Dense(100, init='normal', activation='relu' ,W_regularizer=l1l2(l1=0.0005, l2=0.0001))(digit_input)
x = Dropout(0.1)(x)
x = Dense(540,  activation ='relu')(x)
x = Dropout(0.3)(x)
x = Dense(310,  activation ='relu')(x)
x = Dropout(0.5)(x)
x = Dense(450,  activation ='relu')(x)
x = Dropout(0.4)(x)
out1 = Dense(1,  activation ='sigmoid')(x)
model_MIL = Model(digit_input, out1)

# Let's define how we treat the bags now.
digit_a = Input(shape=(147,))
digit_b = Input(shape=(147,))
digit_c = Input(shape=(147,))

# The vision model will be shared, weights and all
out_a = model_MIL(digit_a)
out_b = model_MIL(digit_b)
out_c = model_MIL(digit_c)

concatenated = merge([out_a, out_b, out_c], mode='max') # By taking the max we take also the min of the output regarding background
#concatenated = Dense(200,  activation ='relu')(concatenated)

out2 = Dense(2, activation='softmax')(concatenated)

Classification_model_train = Model([digit_a, digit_b, digit_c], out2)

# Training 

admax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#decay ? 0.002
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.97, patience=1, min_lr=0.00001)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, verbose=0),
    ModelCheckpoint("weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
    reduce_lr
]

Classification_model_train.compile(optimizer=admax, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent

Y_train_true = np_utils.to_categorical(Y_train, 2) # convert class vectors to binary class matrices
                   
# Fit the model     
Classification_model_train.fit(X_train_train, Y_train_true[0:int(len(X_train_best)*Adapted_percentage)], validation_split=0.2/Adapted_percentage , nb_epoch=1, class_weight=class_weight, batch_size=400, shuffle=True, verbose=1, callbacks=callbacks)#, class_weight=class_weight)


# Model Test #######################################################################################################################################



X_test_train = [X_test_best[0:int(len(X_test_best)*Adapted_percentage)], X_test_best1[0:int(len(X_test_best)*Adapted_percentage)], X_test_best2[0:int(len(X_test_best)*Adapted_percentage)]]

        
# Model_test #################################################################################################################################################


digit_input_test = Input(shape=(147,))

x2 = Dense(200, init='normal', activation='relu' ,W_regularizer=l1l2(l1=0.0005, l2=0.0001))(digit_input_test)
x2 = Dropout(0.1)(x2)
x2 = Dense(540,  activation ='relu')(x2)
x2 = Dropout(0.3)(x2)
x2 = Dense(310,  activation ='relu')(x2)
x2 = Dropout(0.5)(x2)
x2 = Dense(450,  activation ='relu')(x2)
x2 = Dropout(0.4)(x2)
out1_test = Dense(1,  activation ='sigmoid')(x2)
model_MIL_test = Model(digit_input_test, out1_test)

# Let's define how we treat the bags now.
digit_a_test = Input(shape=(147,))
digit_b_test = Input(shape=(147,))
digit_c_test = Input(shape=(147,))

# The vision model will be shared, weights and all
out_a_test = model_MIL_test(digit_a_test)
out_b_test = model_MIL_test(digit_b_test)
out_c_test = model_MIL_test(digit_c_test)

concatenated_test = merge([out_a_test, out_b_test, out_c_test], mode='max') # By taking the max we take also the min of the output regarding background

#concatenated_test = Dense(200,  activation ='relu')(concatenated_test)

out2_test = Dense(2, activation='softmax')(concatenated_test)

Classification_model_test = Model([digit_a_test, digit_b_test, digit_c_test], out2_test)

# Training 

admax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)#decay ? 0.002
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.97, patience=1, min_lr=0.00001)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, verbose=0),
    ModelCheckpoint("weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
    reduce_lr
]

Classification_model_test.compile(optimizer=admax, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent
#model_MIL.compile(optimizer=admax, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent
             
print (Classification_model_test[j].summary()) # Affiche les details du reseau !
#print (model_MIL.summary()) # Affiche les details du reseau !

    
Y_test_true = np_utils.to_categorical(Y_test, 2)            
        
# Fit the model     
Classification_model_test.fit(X_test_train, Y_test_true[0:int(len(X_test_best)*Adapted_percentage)], validation_split=0.2/Adapted_percentage, nb_epoch=1, class_weight=class_weight, batch_size=400, shuffle=True, verbose=1, callbacks=callbacks)#, class_weight=class_weight)




Z_train = Classification_model_train.predict(X_test_true, batch_size=32, verbose=0)

Z_test= Classification_model_test.predict(X_train_true, batch_size=32, verbose=0)


Score_background = 0.0
Faux_background = 0.0
Score_signal = 0.0
Faux_signaux = 0.0

print(Y_test[0])
print(Z_test.shape)

for i in range(len(Z_train)):
    
    if (Z_train[i][1] > Z_train[i][0] and Y_test[i] == 1):
        Score_signal += 1.0
        
        
    if (Z_train[i][1] < Z_train[i][0] and Y_test[i] == 0):
        Score_background += 1.0
        
        
    if (Z_train[i][1] < Z_train[i][0] and Y_test[i] == 1):
        Faux_background += 1.0
        
        
    if (Z_train[i][1] > Z_train[i][0] and Y_test[i] == 0):
        Faux_signaux += 1.0


for i in range(len(Z_test)):
    
    if (Z_test[i][1] > Z_test[i][0] and Y_train[i] == 1):
        Score_signal += 1.0
        
        
    if (Z_test[i][1] < Z_test[i][0] and Y_train[i]== 0):
        Score_background += 1.0
        
        
    if (Z_test[i][1] < Z_test[i][0] and Y_train[i] == 1):
        Faux_background += 1.0
        
        
    if (Z_test[i][1] > Z_test[i][0] and Y_train[i] == 0):
        Faux_signaux += 1.0
        


print("Score_signal = "+str(Score_signal))
print("Faux_signaux = "+str(Faux_signaux))

print("Score_background = "+str(Score_background))
print("Faux_background = "+str(Faux_background))


Score_signal = Score_signal / (Score_signal+Faux_background)
Score_background = Score_background / (Score_background+Faux_signaux)


print("Score_signal = "+str(Score_signal))

print("Score_background = "+str(Score_background))



# Let's compute the AUC

Y_pred = np.zeros(942548)

for i in range(len(Y_pred)):
    if (i < len(Z_test)):
        Y_pred[i]= Z_test[i,1]*2-1
    else:
        Y_pred[i]= Z_train[i-len(Z_test),1]*2-1

Y_true = np.hstack([Y_train, Y_test])

c, r = weights_test.shape
weights_test = weights_test.reshape(c,)
c, r = weights_train.shape
weights_train = weights_train.reshape(c,)

weights_shuffled = np.hstack([weights_train, weights_test])

r_score = roc_auc_score(Y_true, Y_pred, average='macro', sample_weight=weights_shuffled)

print(r_score)
