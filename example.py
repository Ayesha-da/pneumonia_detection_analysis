# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 10:46:06 2021

@author: ayesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Prof. Donald Patterson (Westmont College)
    Twitter Contact: @djp3
"""

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

X = "Normal"
Y = "Pneumonia"

sample_Y_image = "Resources/chest_xray/train/PNEUMONIA/person1_bacteria_1.jpeg"

#Create a function that will tweak our imags to prevent overfitting

datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            rescale = 1.0/255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

img = load_img(sample_Y_image)

x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0

for batch in datagen.flow(x,
                          batch_size=1,
                          save_to_dir='preview',
                          save_prefix=Y,
                          save_format='jpeg'):
    i += 1
    if i > 20:
        break