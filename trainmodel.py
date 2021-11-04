# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:03:07 2021

@author: ayesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Prof. Donald Patterson (Westmont College)
    Twitter Contact: @djp3
"""

#import the libraries
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization
import tensorflow as tf
from tensorflow.keras import backend as K

#Set the image size with are learning from
IMG_WIDTH, IMG_HEIGHT = 150,150

#Set the constants
TRAIN_DATA_DIR = 'Resources/chest_xray/train'
VALIDATION_DATA_DIR = 'Resources/chest_xray/test'
NB_TRAIN_SAMPLES = 200  #Must match number of files
NB_VALIDATION_SAMPLES = 200

EPOCHS = 50 #Higher for more time training model... diminishing returns
BATCH_SIZE = 5

# Machine Learning Model Filename
ML_MODEL_FILENAME = 'saved_model.h5'

def build_model():
    
    if K.image_data_format() == 'channels_first':
        input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
    else:
        input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
        
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add( BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add( BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add( BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add( BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add( BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
   

    model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ["accuracy"])
    return model


def train_model(model):
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rotation_range = 40,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            rescale = 1.0/255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale= 1. / 255)

    train_generator = train_datagen.flow_from_directory(
            TRAIN_DATA_DIR,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            VALIDATION_DATA_DIR,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='binary')
    
    model.fit_generator(
            train_generator,
            steps_per_epoch=NB_VALIDATION_SAMPLES // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=NB_VALIDATION_SAMPLES // BATCH_SIZE)
     
    return model



def main():
    myModel = None
    tf.keras.backend.clear_session()
    myModel = build_model()
    myModel = train_model(myModel)
    myModel.save(ML_MODEL_FILENAME)


main()
