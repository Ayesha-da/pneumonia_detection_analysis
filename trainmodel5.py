# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 11:33:43 2021

@author: ayesh
"""

from keras.preprocessing.image import ImageDataGenerator
#from multiprocessing import Pool
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
#from tensorflow.keras import backend as K
#from keras.callbacks import EarlyStopping
import cv2
import os
import numpy as np


path="Resources/chest_xray/train/"

def img_to_array(path, category_no, X, y):
    for p in os.listdir(path):
        if p == ".DS_Store":
            continue
        #print(p)
        img = cv2.imread(os.path.join(path, p), cv2.IMREAD_GRAYSCALE)
        img_np = cv2.resize(img, dsize = (150, 150))
        X.append(img_np)
        y.append(category_no)
        
path_train="Resources2/chest_xray/train/"


category_list = ["NORMAL", "PNEUMONIA"]

X_train = []
y_train = []

for i in range(len(category_list)):
    img_to_array(path_train + category_list[i], i, X_train, y_train)
    
X_train = np.array(X_train).reshape(-1,150,150,1)
y_train = np.array(y_train)


path_test = "Resources2/chest_xray/test/"

category_list = ["NORMAL", "PNEUMONIA"]
X_test = []
y_test = []

for i in range(len(category_list)):
    img_to_array(path_test + category_list[i], i, X_test, y_test)
    
X_test = np.array(X_test).reshape(-1,150,150,1)
y_test = np.array(y_test)

path_val = "Resources2/chest_xray/val/"

X_val = []
y_val = []

for i in range(len(category_list)):
    img_to_array(path_val + category_list[i], i, X_val, y_val)
    
X_val = np.array(X_val).reshape(-1,150,150,1)
y_val = np.array(y_val)


X_train = X_train / 255
X_val = X_val / 255
X_test = X_test / 255

datagen = ImageDataGenerator(
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip = False)  # randomly flip images


datagen.fit(X_train)

# tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
'''
model = Sequential([
    Conv2D(32, (3,3), activation = "relu", input_shape = (150,150,1)),
   # BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation = "relu"),
    Dropout(0.1),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation = "relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation = "relu"),
   # Dropout(0.2),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation = "relu"),
   # Dropout(0.2),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation = "relu"),
    Dropout(0.2),
    Dense(1, activation = "sigmoid")
])
'''
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1,activation = "sigmoid"))

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#early_stopping = EarlyStopping(monitor='loss', mode='min',patience=

history = model.fit(datagen.flow(X_train, y_train, batch_size = 32), epochs = 20, validation_data = datagen.flow(X_val, y_val))

model.save( 'saved_model7.h5')


train_loss = history.history['loss']
val_loss = history.history['val_loss']

train_auc_name = list(history.history.keys())[3]
val_auc_name = list(history.history.keys())[1]
train_auc = history.history[train_auc_name]
val_auc = history.history[val_auc_name]

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc = "upper left")
plt.show()


plt.title("Training vs. Validation AUC Score")
plt.plot(train_auc, label='training auc')
plt.plot(val_auc, label='validation auc')
plt.xlabel("Number of Epochs", size=14)
plt.legend()
plt.show()

#plt.subplot(12,12,11)
plt.title("Training vs. Validation Loss")
plt.plot(train_loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.xlabel("Number of Epochs", size=14)
plt.legend()
plt.show()
'''
train_auc_name = list(history.history.keys())[3]
val_auc_name = list(history.history.keys())[1]
train_auc = history.history[train_auc_name]
val_auc = history.history[val_auc_name]


## PLOT 2: TRAIN VS. VALIDATION AUC
    #plt.subplot(2,2,2)
plt.title("Training vs. Validation AUC Score")
plt.plot(train_auc, label='training auc')
plt.plot(val_auc, label='validation auc')
plt.xlabel("Number of Epochs", size=14)
plt.legend()

train_loss = history.history['loss']
val_loss = history.history['val_loss']

   ## PLOT 1: TRAIN VS. VALIDATION LOSS 
#plt.subplot(12,12,11)
plt.title("Training vs. Validation Loss")
plt.plot(train_loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.xlabel("Number of Epochs", size=14)
plt.legend()
'''