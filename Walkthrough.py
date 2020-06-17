# This walkthrough shows the steps involved from preparing the dataset to training and saving the model

# The first step is to import the necessary packages

import os
import numpy as np
import glob
import shutil
import pathlib
import matplotlib.pyplot as pltimport tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

# Next the file was mounted from google drive to make it accessible in cola
from google.colab import drive
drive.mount('/content/drive')

# Then the necessary paths to the sub-directories in the dataset are created
normal_path = r'/content/drive/My Drive/Colab Notebooks/Pothole Dataset/normal'
pothole_path = r'/content/drive/My Drive/Colab Notebooks/Pothole Dataset/potholes'
train_pothole_path = r'/content/drive/My Drive/Colab Notebooks/Pothole Dataset/train/potholes'
train_normal_path = r'/content/drive/My Drive/Colab Notebooks/Pothole Dataset/train/normal'
val_pothole_path = r'/content/drive/My Drive/Colab Notebooks/Pothole Dataset/val/potholes'
val_normal_path = r'/content/drive/My Drive/Colab Notebooks/Pothole Dataset/val/normal'

# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224

# The next step is to label this images and group them in batches using the ImageDataGenerator class from Keras
train_dir = r'/content/drive/My Drive/Colab Notebooks/Pothole Dataset/train'
train_data_gen = image_generator.flow_from_directory(directory=train_dir,
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='sparse')
                                                     
# The image_generator object can now be used to call it's flow_from_directory method which groups the images in the training directory into batches,
# resizing the image, adding labels to the image.

# It is important to note that the labels created will strictly be named as each sub-directory in the training directory was named.

val_dir =  r'/content/drive/My Drive/Colab Notebooks/Pothole Dataset/val'
val_data_gen = image_generator.flow_from_directory(directory=val_dir,
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='sparse')

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
# This feature extractor is a CNN without the final classification layer. All the layers in this feature extractor will not be re-trained.
# The next line prevents the model from retraining the parameters in the feature extractor

feature_extractor.trainable = False

# The next step is to create the pothole detection  model

model = tf.keras.Sequential([
  feature_extractor,
  tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()

# Compile the model
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])
  
train_directory = pathlib.Path(train_dir)
no_of_training_images = len(list(train_directory.glob('*/*.jpg')))
no_of_training_images

val_directory = pathlib.Path(val_dir)
no_of_val_images = len(list(val_directory.glob('*/*.jpg')))
no_of_val_images

# Next we fit the model
EPOCHS = 6

history = model.fit_generator(train_data_gen,
                              steps_per_epoch=int(np.ceil(no_of_training_images/float(BATCH_SIZE))),
                              epochs=EPOCHS,
                              validation_data=val_data_gen,
                              validation_steps=int(np.ceil(no_of_val_images/float(BATCH_SIZE)))
                              )
                              

# Then Finally, we'll plot the training and validation accuracy/loss graphs to visualize how the model is performing..
def plot_graphs:
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(EPOCHS)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()
