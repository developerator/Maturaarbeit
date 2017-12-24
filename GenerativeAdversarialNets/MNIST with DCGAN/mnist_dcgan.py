'''
By Tim Ehrensberger

The base for this code was taken from https://github.com/Zackory/Keras-MNIST-GAN/blob/master/mnist_gan.py by Zackory Erickson
'''
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K

K.set_image_dim_ordering('th')

# Deterministic output.
#np.seed(?) Attention! Some seeds lead to random initial values that make training IMPOSSIBLE!


# The results are a little better when the dimensionality of the random vector is only 10.
# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim = 100

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train[:, np.newaxis, :, :]

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

# G
generator = Sequential()
generator.add(Dense(128*7*7, input_dim=randomDim, kernel_initializer='random_normal'))
generator.add(LeakyReLU(0.2))
generator.add(Reshape((128, 7, 7)))

generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(64, (5, 5), padding='same'))
generator.add(LeakyReLU(0.2))

generator.add(UpSampling2D(size=(2, 2)))
generator.add(Convolution2D(1, (5, 5), padding='same', activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)

# D
discriminator = Sequential()
discriminator.add(Convolution2D(64, (5, 5), padding='same', strides=(2, 2), input_shape=(1, 28, 28), kernel_initializer='random_normal')) 
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.5))

discriminator.add(Convolution2D(128, (5, 5), padding='same', strides=(2, 2)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.5)) 

discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# GAN (combined)
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

dLosses = []
gLosses = []

# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/gan_loss_epoch_%d.png' % epoch)


# Create a wall of generated MNIST images
examples=100
noise = np.random.normal(0, 1, size=[examples, randomDim]) #always the same noise

def plotGeneratedImages(epoch, dim=(10, 10), figsize=(10, 10)):
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)

    assertExists('images')

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/gan_generated_image_epoch_%d.png' % epoch)


# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    assertExists('models')
    generator.save('models/gan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/gan_discriminator_epoch_%d.h5' % epoch)


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

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
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

        
        plotGeneratedImages(e)
        saveModels(e)

    # Plot losses from every epoch
    plotLoss(e)

def assertExists(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
	train(200, 128)
