'''
By Tim Ehrensberger

The base of the functions for the network's training is taken from https://github.com/Zackory/Keras-MNIST-GAN/blob/master/mnist_gan.py by Zackory Erickson

The network structure is inspired by https://github.com/aleju/face-generator by Alexander Jung
'''

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.layers import Input, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.datasets import cifar10
from keras.optimizers import Adam

from keras.regularizers import l1_l2

#------
# DATA
#------
from keras import backend as K
K.set_image_dim_ordering('th')

import h5py

# Get hdf5 file
hdf5_file = os.path.join("PATH TO DATASET", "CelebA_32_data.h5")

with h5py.File(hdf5_file, "r") as hf:
    X_train = hf["data"] [()] #[()] makes it read the whole thing
    X_train = X_train.astype(np.float32) / 255    
      

#----------------
# HYPERPARAMETERS
#----------------
randomDim = 100

adam = Adam(lr=0.0002, beta_1=0.5)

reg = lambda: l1_l2(l1=1e-7, l2=1e-7)

#dropout = 0.4

#-----------
# Generator
#-----------
h = 5

generator = Sequential()

#In: 100
generator.add(Dense(256 * 4 * 4, input_dim=100, kernel_regularizer=reg()))
generator.add(BatchNormalization())
generator.add(Reshape((256, 4, 4)))
#generator.add(Dropout(dropout))
#Out: 256 x 4 x 4

#In: 256 x 4 x 4
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(128, (h, h), padding='same', kernel_regularizer=reg())) #1
generator.add(BatchNormalization(axis=1))
generator.add(LeakyReLU(0.2))
#generator.add(Dropout(dropout))
#Out: 128 x 8 x 8

#In: 128 x 8 x 8
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(128, (h, h), padding='same', kernel_regularizer=reg())) #2
generator.add(BatchNormalization(axis=1))
generator.add(LeakyReLU(0.2))
#generator.add(Dropout(dropout))
#Out: 128 x 16 x 16

#In:  128 x 16 x 16
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(64, (h, h), padding='same', kernel_regularizer=reg())) #3
generator.add(BatchNormalization(axis=1))
generator.add(LeakyReLU(0.2))
#generator.add(Dropout(dropout))
#Out: 64 x 32 x 32

#In: 64 x 32 x 32
generator.add(Convolution2D(3, (h, h), padding='same', kernel_regularizer=reg())) #4
generator.add(Activation('sigmoid'))
#Out: 3 x 32 x 32

generator.compile(loss='binary_crossentropy', optimizer=adam)


#--------------
# Discriminator
#--------------
discriminator = Sequential()

#In: 3 x 32 x 32
discriminator.add(Convolution2D(64, (h, h), padding='same', input_shape=(3, 32, 32), kernel_regularizer=reg())) 
discriminator.add(MaxPooling2D(pool_size=(2, 2)))
discriminator.add(LeakyReLU(0.2))
#Out: 64 x 16 x 16

#In: 64 x 16 x 16
discriminator.add(Convolution2D(128, (h, h), padding='same', kernel_regularizer=reg())) 
discriminator.add(MaxPooling2D(pool_size=(2, 2)))
discriminator.add(LeakyReLU(0.2))
#Out: 128 x 8 x 8

#In: 128 x 8 x 8
discriminator.add(Convolution2D(256, (h, h), padding='same', kernel_regularizer=reg())) 
discriminator.add(MaxPooling2D(pool_size=(2, 2)))#Average?
discriminator.add(LeakyReLU(0.2))
#Out: 256 x 4 x 4

#In: 256 x 4 x 4
discriminator.add(Flatten())
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
#discriminator.add(Dropout(dropout))
discriminator.add(Dense(1))
discriminator.add(Activation('sigmoid'))
#Out: 1 (Probability)

discriminator.compile(loss='binary_crossentropy', optimizer=adam)

#-----
# GAN
#-----
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

#-----------
# FUNCTIONS
#-----------
dLosses = []
gLosses = []
def plotLoss(epoch):
    assertExists('images')

    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminative loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()    
    plt.savefig('images/dcgan_loss_epoch_%d.png' % epoch)

# Create a wall of generated images
noise = np.random.normal(0, 1, size=[examples, randomDim])
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):    
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.transpose(0, 2, 3, 1) #transpose is crucial

    assertExists('images')

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i, :, :, :], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()    
    plt.savefig('images/dcgan_generated_image_epoch_%d.png' % epoch)


# Save the generator and discriminator networks (and weights) for later use
def savemodels(epoch):
    assertExists('models')
    generator.save('models/dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/dcgan_discriminator_epoch_%d.h5' % epoch)


def train(epochs=1, batchSize=128):
    batchCount = X_train.shape[0] // batchSize
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            # Generate fake images
            generatedImages = generator.predict(noise)

            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing = not exactly 1
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis) # here only D is trained

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim]) 
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen) # here only G is trained because D is not trainable          

        # Store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)

        #plot after every epoch
        plotGeneratedImages(e)
        savemodels(e)

    # Plot losses from every epoch
    plotLoss(e)


def assertExists(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':	
    train(100, 128)

