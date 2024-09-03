import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import PIL
import glob
import os
import random
from pathlib import Path



BATCH_SIZE = 64
IMG_HEIGHT = 224
IMG_WIDTH = 224

# # Create a dataset
data_dir = Path(r"C:\Programming Files\TensorFlow\dataSet\PetImages")

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


# build model for image classification
model = tf.keras.models.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.5),  # Dropout
  tf.keras.layers.Dense(2) # dog and cat
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


model.save('saved_model/DogVsCat.keras') # save model 

def load_model():
  model = tf.keras.models.load_model('saved_model/DogVsCat.keras')

  test_path = r"C:\Programming Files\TensorFlow\dataSet\TestImage\KakaoTalk_20240901_214131075_05.jpg"
  img = tf.keras.utils.load_img(
    test_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
  )

  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch


  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
  )

load_model()