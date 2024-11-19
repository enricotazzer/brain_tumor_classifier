import numpy as np
import os
import pandas as pd
import tensorflow as tf
from keras import layers
from tensorflow import keras 
import shutil
import matplotlib.pyplot as plt

brain_tumor = pd.read_csv('Brain_Tumor.csv')

# splitting images into yes and no 
yes = os.path.join('Brain_Tumor', 'yes/')
no = os.path.join('Brain_Tumor', 'no/')
#os.mkdir(yes)
#os.mkdir(no)
for i in range(len(brain_tumor)):
		image_file = brain_tumor.iloc[i]['Image']+'.jpg'
		label  = brain_tumor.iloc[i]['Class']
		if label == 0:
			shutil.copy(os.path.join('Brain_Tumor/images', image_file), no)
		else: 
			shutil.copy(os.path.join('Brain_Tumor/images', image_file), yes)			


data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images
