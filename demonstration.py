# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 22:13:09 2019

@author: deser
"""

'''
Portions of this code are referenced from the following links:
https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd
https://keras.io/visualization/
'''

import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
import glob
from keras.preprocessing import image


keras.callbacks.Callback()
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import model_from_json

newmodel = keras.models.load_model('classifier2.h5')

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

testlista = glob.glob('demo-images/*.jpg')
testlistb = glob.glob('demo-images/*.png')
testlist = testlista+testlistb

comparelist = []

for i in range(len(testlist)):
    test_image = image.load_img(testlist[i], target_size = (64, 64))
    test_image - image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = newmodel.predict(test_image)
    training_set.class_indices
    if result [0][0] >= 0.5:
        prediction = 'dog'
    else:
        prediction = 'cat'
        
    if i < 9:
        comparelist.append(['cat',prediction])
    else:
        comparelist.append(['dog',prediction])
        
#print(comparelist)

numcorrect = 0
for i in range(len(comparelist)):
    if(comparelist[i][0] == comparelist[i][1]):
        numcorrect = numcorrect + 1
print(numcorrect)
print(len(comparelist))
print(numcorrect/len(comparelist))


from PIL import Image

#im = Image.open(testlist[0])
#im.show()
print(comparelist[0])

#r'hand-drawn/cat.png'


from IPython.display import Image
from PIL import Image



for i in range (len(testlist)):
    path=testlist[i]

    size = 256, 256

    im = Image.open(path)
    im.thumbnail(size, Image.ANTIALIAS)
    display(im)

    print("Prediction: " + comparelist[i][1])
    

