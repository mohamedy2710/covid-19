import os
import shutil
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to your image datasets
TRAIN_DIR = r"D:\العلم نور33333\CoronaVirus-Prediction-using-CNN-master\CovidDataset"
VAL_DIR = r"D:\العلم نور33333\CoronaVirus-Prediction-using-CNN-master"

# Create directories if they don't exist
if not os.path.exists(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)
    print("Training folder created")

if not os.path.exists(VAL_DIR):
    os.mkdir(VAL_DIR)
    print("Validation folder created")

# CNN Model Building
model = Sequential()

# First convolutional block
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(224, 224, 3))) #1 => number of chanels
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))  #[[1, 3],[4, 2]] => we will take 4 from every 4 items matrix

# Adding dropout to prevent overfitting
model.add(Dropout(0.25))

# Second convolutional block
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Third convolutional block
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten and Fully connected layers
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

# Compile model
model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer="adam", metrics=['accuracy']) #adam used to change wehghts to avheve maximum accuracy
# end of training process

# Image Augmentation for training and validation data
train_datagen = ImageDataGenerator(
    rescale=1./255, #divide each pixel by 255
    shear_range=0.2, # shear image by 20% (امالة الصورة)
    zoom_range=0.2, #(تكبير الصورة بمقدار 20%)
    horizontal_flip=True,# بعكس الصور عشواءي لزيادة حجم بيانات التدريب 
)

test_datagen = ImageDataGenerator(rescale=1./255) #اجراء نفس التعديلا علي بيانات الاختبار حتي لا يتم الاختلاف بينها وبين بيانات التدريب )(القسمه عل255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

# Train the model
hist = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=2
)

# Save the trained model
model.save(r'model.h5')
print("Model saved to model.h5")


