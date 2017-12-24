#This file may optionally be used to augment the dataset

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import h5py

from keras import backend as K
K.set_image_dim_ordering('th')


datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.01,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest')


hdf5_file = os.path.join("PATH TO THE ORIGINAL DATASET", "NAME.h5")

with h5py.File(hdf5_file, "r") as hf:
    X = hf["data"] [()]


# the .flow() command generates batches of randomly transformed images
i = 0
for batch in datagen.flow(X, batch_size=64,
                          save_to_dir="FOLDER TO SAVE THE TRANSFORMED IMAGES IN", save_prefix='SET', save_format='jpg'):
    i += 1
    if i > 5:
        break
