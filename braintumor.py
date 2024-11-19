import numpy as np
import tensorflow as tf
from tensorflow import keras 
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from data import data_augmentation

X_train, X_val = keras.utils.image_dataset_from_directory('Brain_Tumor', image_size=(240,240), batch_size=8, validation_split=0.2, subset='both', seed=8364)

X_train = X_train.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)

INPUT_SHAPE = (240, 240, 3)
FILTER1_SIZE = 16
FILTER2_SIZE = 32
KERNEL_SIZE1 = (5, 5)
POOL_SHAPE1 = (3, 3)
KERNEL_SIZE2 = (3, 3)
POOL_SHAPE2 = (2, 2)
FULLY_CONNECT_NUM = 128
# Model architecture implementation
model = keras.models.Sequential()
model.add(layers.Conv2D(FILTER1_SIZE, KERNEL_SIZE1, activation='relu', input_shape=INPUT_SHAPE))
model.add(layers.Conv2D(FILTER1_SIZE, KERNEL_SIZE1, activation='relu'))
model.add(layers.MaxPooling2D(POOL_SHAPE1))
model.add(layers.Conv2D(FILTER2_SIZE, KERNEL_SIZE2, activation='relu'))
model.add(layers.Conv2D(FILTER2_SIZE, KERNEL_SIZE2, activation='relu'))
model.add(layers.Conv2D(FILTER2_SIZE, KERNEL_SIZE2, activation='relu'))
model.add(layers.Conv2D(FILTER2_SIZE, KERNEL_SIZE2, activation='relu'))
model.add(layers.MaxPooling2D(POOL_SHAPE2))
model.add(layers.Conv2D(FILTER2_SIZE, KERNEL_SIZE2, activation='relu'))
model.add(layers.Conv2D(FILTER2_SIZE, KERNEL_SIZE2, activation='relu'))
model.add(layers.Conv2D(FILTER2_SIZE, KERNEL_SIZE2, activation='relu'))
model.add(layers.Conv2D(FILTER2_SIZE, KERNEL_SIZE2, activation='relu'))
model.add(layers.Conv2D(FILTER2_SIZE, KERNEL_SIZE2, activation='relu'))
model.add(layers.MaxPooling2D(POOL_SHAPE2))
model.add(layers.Flatten())
model.add(layers.Dense(FULLY_CONNECT_NUM, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.BinaryAccuracy()])
model.fit(X_train, batch_size=16, epochs=5)
model.save('brain_tumor.keras')

model.evaluate(X_val, batch_size=16)
