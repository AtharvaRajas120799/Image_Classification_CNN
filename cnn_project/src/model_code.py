# -*- coding: utf-8 -*-
# Original Colab code (unmodified as requested)

import matplotlib.pyplot as plt

import seaborn as sns
import zipfile
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.__version__

from google.colab import drive
drive.mount('/content/drive')

path='/content/drive/MyDrive/Datasets/homer_bart_2.zip'
zip_object=zipfile.ZipFile(file=path, mode='r')
zip_object.extractall('./')
zip_object.close()

training_generator = ImageDataGenerator(rescale=1./255,rotation_range=7,zoom_range=0.2,horizontal_flip=True)

train_dataset= training_generator.flow_from_directory(directory='/content/homer_bart_2/training_set',
                                                     target_size=(64,64),
                                                     class_mode='categorical',
                                                     batch_size=8,shuffle=True)

train_dataset.classes

test_generator=ImageDataGenerator(rescale=1./255)
test_dataset=test_generator.flow_from_directory(directory='/content/homer_bart_2/test_set',
                                                     target_size=(64,64),
                                                     class_mode='categorical',
                                                     batch_size=1,shuffle=False)

network= Sequential()
network.add(Conv2D(input_shape=(64,64,3),filters=32,kernel_size=(3,3),activation='relu'))
network.add(MaxPool2D(pool_size=(2,2)))

network.add(Conv2D(input_shape=(64,64,3),filters=32,kernel_size=(3,3),activation='relu'))
network.add(MaxPool2D(pool_size=(2,2)))

network.add(Conv2D(input_shape=(64,64,3),filters=32,kernel_size=(3,3),activation='relu'))
network.add(MaxPool2D(pool_size=(2,2)))

network.add(Flatten())

network.add(Dense(units=577,activation='relu'))
network.add(Dense(units=577,activation='relu'))

network.add(Dense(units=2,activation='softmax'))

network.summary()

network.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = network.fit(train_dataset,epochs=50)

predictions=network.predict(test_dataset)

predictions

predictions=np.argmax(predictions,axis=1)
predictions

test_dataset.classes

from sklearn.metrics import accuracy_score
accuracy_score(test_dataset.classes,predictions)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(test_dataset.classes,predictions)
sns.heatmap(cm,annot=True);

from sklearn.metrics import classification_report
print(classification_report(test_dataset.classes,predictions))

image=cv2.imread('/content/homer_bart_2/test_set/homer/homer15.bmp')

cv2_imshow(image)

image=cv2.resize(image,(64,64))
image=image/255
image=image.reshape(-1,64,64,3)

result=network.predict(image)
result=np.argmax(result)

if result==0:
  print('Bart')
else:
  print('Homer')