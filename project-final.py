# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 15:22:58 2019

Portions of this code are referenced from the following link:
https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd

@author: deser
"""
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense



#Initialize CNN
classifier = Sequential()

#Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flatten
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#compile CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fit CNN to imgs
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
        )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
        r'dataset/trainingset',
        target_size = (64,64),
        batch_size = 32,
        class_mode = 'binary'
        )

testing_set = test_datagen.flow_from_directory(
        r'dataset/testingset',
        target_size = (64,64),
        batch_size = 32,
        class_mode = 'binary'
        )

from IPython.display import display
from PIL import Image

classifier.fit_generator(
        training_set,
        steps_per_epoch = 4000,
        epochs = 10,
        validation_data = testing_set,
        validation_steps = 800)

classifier.save('classifier2.h5')

keras.callbacks.ModelCheckpoint





