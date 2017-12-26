# Module to train an autoencoder for hand tracking in videos
#
# Author: Prithvijit Chakrabarty (prithvichakra@gmail.com)

import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras import backend as K
from keras import optimizers

#Import module to load the oxford hand dataset: See hds.py
import hds

#Parameters
EPOCHS = 10
BATCH_SIZE = 50
MODEL_PATH = 'auto.h5'
LRATE = 1e-4
EPSILON = 1e-8

#Set input shape
if hds.COL == True:
    input_img = Input(shape=(hds.SIZE[0],hds.SIZE[1],3))
else:
    input_img = Input(shape=(hds.SIZE[0],hds.SIZE[1], 1))

#Convolution layers: Note the use of only strided convolutions, no max pooling
x = Conv2D(14, (5,5), activation='relu', padding='same', strides=(2,2))(input_img)
x = Conv2D(24, (3,3), activation='relu', padding='same', strides=(2,2))(x)
x = Conv2D(64, (3,3), activation='relu', padding='same', strides=(2,2))(x)
x = Conv2D(48, (3,3), activation='relu', padding='same')(x)

#Add dropout
x = Dropout(rate=0.5)(x)

#Upsampling layers
x = Conv2D(48, (3,3), activation='relu', padding='same')(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(24, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(14, (3,3), activation='relu',padding='same')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (5,5), activation='sigmoid', padding='same')(x)

#Set optimizer
opt = optimizers.Adam(lr=LRATE,epsilon=EPSILON)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=opt,loss='binary_crossentropy')

#Use the module to load the dataset
ds = hds.load_ds()
x,y = map(np.array,zip(*ds))
print 'Dataset loaded:',x.shape,'==',y.shape
#Fit the model
autoencoder.fit(x,y,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True
                )
#Save the model
autoencoder.save(MODEL_PATH)
