# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:23:10 2021
@author: William
"""


#This is model2 which tests Pneumonia vs everything (except Normal). In version A (this version) it assumes the image folder structure is:
#   Model_1A_Images
#       Test
#           Pneumonia
#           Other
#       Train
#           Pneumonia
#           Other            
#
import numpy as np 
import pandas as pd 
import random

# folder
import os
import glob

# image
from PIL import Image

# visu
import matplotlib.pyplot as plt
plt.rc('image', cmap='gray')

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

#There are two categories in two sets:
categories = ["Pneumonia", "Other"]
datasets = ["Train", "Test"]

#Compute the average size of the images among all categories and all datasets:


widths = []
heights = []

for set_ in datasets:
    for cat in categories:
        filelist = glob.glob('./Model_2A_Images/' + set_ + '/' + cat + '/*')
        widths.extend([Image.open(fname).size[0] for fname in filelist])
        heights.extend([Image.open(fname).size[1] for fname in filelist])

images_size = pd.DataFrame({"widths": widths, "heights": heights})
        
print("Average image width: " + f'{images_size["widths"].mean():.2f}')
print("Average image height: " + f'{images_size["heights"].mean():.2f}')

#divide the mean width and mean height of the images by 10:

im_width = int(images_size["widths"].mean()/10)
im_height = int(images_size["heights"].mean()/10)
print("image width: " + str(im_width))
print("image height: " + str(im_height))

#Resize images by the lengths described above, then load all the images from the two sets in one single dataframe.

data = []
target = []
filename = []

for set_ in datasets:
    for cat in categories:
        filelist = glob.glob('./Model_2A_Images/' + set_ + '/' + cat + '/*')
        target.extend([cat for _ in filelist])
        data.extend([np.array(Image.open(fname).convert('L').resize((im_width, im_height))) for fname in filelist])
        filename.extend([fname for fname in filelist])
#
data_array = np.stack(data, axis=0)


print("data array shape: " + str(data_array.shape))


pd.concat([pd.DataFrame(pd.DataFrame({"target" : target}).value_counts()).rename(columns={0:"count"}),
           pd.DataFrame(pd.DataFrame(target).value_counts()*100/len(target)).applymap(round).rename(columns={0:"%"})], axis=1)

#Review several random images and associated label of our dataset:

fig = plt.figure(figsize=(20,15))
gs = fig.add_gridspec(4, 4)
#
for line in range(0, 3):
    for row in range(0, 3):
        num_image = random.randint(0, data_array.shape[0])
        ax = fig.add_subplot(gs[line, row])
        ax.axis('off');
#        ax.set_title(target[num_image])
        ax.set_title("file: " + filename[num_image].split("/")[-1] + "/\n" + target[num_image])
        ax.imshow(data_array[num_image]);

#######THIS IS WHERE IT APPEARS TO BREAK (files don't match the "True Value" based on file names)###################################        
#Separate dataset into two sets. Test set consists in 20% of the dataset and the remaining is for the train set. The class repartition is kept by setting the parameter stratify to target.

X_train, X_test, y_train, y_test = train_test_split(data_array, np.array(target), random_state=43, test_size=0.2, stratify=target)
print("X_train.shape: " + str(X_train.shape))
print("X_test.shape: " + str(X_test.shape))
print("y_train.shape: " + str(y_train.shape))
print("y_test.shape: " + str(y_test.shape))

pd.DataFrame(y_train).value_counts()/len(y_train)

pd.DataFrame(y_test).value_counts()/len(y_test)

#normalize the data. 
#Check the maximum and minimum values in the data

print("X train max: " + str(X_train.max()))
print("X train min: " + str(X_train.min()))

#Normalize accordingly (Resulting image intensities should be between 0 and 1).
X_test_norm = np.round((X_test/255), 3).copy()
X_train_norm = np.round((X_train/255), 3).copy()

print("X_train_norm.max: " + str(X_train_norm.max()))
print("X_train_norm.min: " + str(X_train_norm.min()))
##############################################END OF BROKEN CODE###################################################################
#Check the normalised pictures randomly:
    
fig = plt.figure(figsize=(20,15))
gs = fig.add_gridspec(4, 4)
#
for line in range(0, 3):
    for row in range(0, 3):
        num_image = random.randint(0, X_train_norm.shape[0])
        ax = fig.add_subplot(gs[line, row])
        ax.axis('off');
#        ax.set_title(y_train[num_image])
        ax.set_title("file: " + filename[num_image].split("/")[-1] + "/\n" + target[num_image])
        ax.imshow(X_train_norm[num_image]);
        
#Here we convert targets from string to numerical values, each category becoming an integer - 0 or 1 - for NORMAL or PNEUMONIA: 

#display(np.array(y_train).shape)
#display(np.unique(y_train))
#display(np.array(y_test).shape)
#display(np.unique(y_test))

    
#Fitting the encoder on train set:
    
encoder = LabelEncoder().fit(y_train)

#Applying on train and test sets:

y_train_cat = encoder.transform(y_train)
y_test_cat = encoder.transform(y_test)


#model needs a 4 dimensions tensor to work
#current images are grayscale pictures with no channel: the matrices of our black and white pictures are of shape 3. 
#We need to add an extra dimension so algorithm can accept it.
#Identify the current shape:
X_train_norm.shape
print ("X train Norm Shape: " + str(X_train_norm.shape))

#Add 4th dim and set height and width values to normalized values
X_train_norm = X_train_norm.reshape(-1, im_height, im_width, 1)
X_test_norm = X_test_norm.reshape(-1, im_height, im_width, 1)
X_train_norm.shape

X_test_norm.shape

#define the Convolutional Neural Network.


def initialize_model():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(im_height, im_width, 1), padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding='same'))
    model.add(layers.MaxPool2D(pool_size=(3, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(60, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

model = initialize_model()
model.summary()

def compile_model(model):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics="accuracy")
    return model



#set an early stopping after 15 epochs and set the parameter restore_best_weights to True so that the weights of best score on monitored metric - here val_accuracy (accuracy on test set) - are restored when training stops. This way the model has the best accuracy possible on unseen data.

model = initialize_model()
model = compile_model(model)
es = EarlyStopping(patience=15, monitor='val_accuracy', restore_best_weights=True)
#
history = model.fit(X_train_norm, y_train_cat,
                    batch_size=8,
                    epochs=20,
                    validation_split=0.3,
                    callbacks=[es])

model.save("Model2A.h5")
#Results & Evaluation

def plot_history(history, title='', axs=None, exp_name=""):
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
   
    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label='train' + exp_name)
    ax1.plot(history.history['val_loss'], label='val' + exp_name)
    #ax1.set_ylim(-0.1, 0.1)
    ax1.set_title('loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train accuracy'  + exp_name)
    ax2.plot(history.history['val_accuracy'], label='val accuracy'  + exp_name)
    #ax2.set_ylim(0.9, 1.1)
    ax2.set_title('Accuracy')
    ax2.legend()
    return (ax1, ax2)
plot_history(history, title='', axs=None, exp_name="");

model.evaluate(X_test_norm, y_test_cat, verbose=0)


#plot random chest-x-ray picture alongside with true label and predicted label to check everything is ok:
#    
predictions = model.predict(X_test_norm)  #

fig = plt.figure(figsize=(20,25))
gs = fig.add_gridspec(8, 4)
#
for row in range(0, 8):
    for col in range(0, 3):
        num_image = random.randint(0, X_test_norm.shape[0])
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off');
#        ax.set_title("Predicted: " + categories[int(np.round(predictions)[num_image][0])] + " /\n True value: " + categories[y_test_cat[num_image]])
        ax.set_title("file: " + filename[num_image].split("/")[-1] + "/\n Predicted: " + categories[int(np.round(predictions)[num_image][0])] + " /\n True value: " + categories[y_test_cat[num_image]])
        ax.imshow(X_test_norm[num_image]);
fig.suptitle("Predicted label VS True label \n for the displayed chest X Ray pictures", fontsize=25, x=0.42);
plt.tight_layout;