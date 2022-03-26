import tensorflow as tf
import numpy as np
from glob import glob
import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten
from keras.models import Model


#importer la base de données
data = "/Users/l.abir/Documents/PersonalProjects/Melanoma Detection /skin-lesions"
#le chemin des bases de données train et test

train_dir = os.path.join(data, "train")
test_dir= os.path.join(data, "test")
#Préparer le modèle de base
#les dimensions de la photo + 3(rgb)
IMG_SHAPE = (224, 224, 3)
#créer le modèle entrainé
base_model=VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.summary()
#bloquer le modèle de base
base_model.trainable = False
folders = glob(train_dir+ '/*')
#Ajouter les couches de sorties
x = Flatten()(base_model.output)
prediction = Dense(len(folders), activation='softmax')(x)
model=Model(inputs=base_model.input, outputs=prediction)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Créer les photos augmentées
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale= 1./255)
trainning_set= train_datagen.flow_from_directory(train_dir, target_size=(224,224), batch_size=32, class_mode='categorical')
test_set= test_datagen.flow_from_directory(test_dir, target_size=(224,224), batch_size=32, class_mode='categorical')
model.fit_generator(trainning_set, epochs=5, validation_data=test_set)