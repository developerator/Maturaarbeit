# Maturaarbeit
These experiments were carried out as part of a paper which Swiss students are requested to write in order to obtain general university admittance called 'Maturaarbeit'. The paper was written in German. Its title roughly translates to 'Generating Images using Artificial Neural Networks'. The main goal of this paper was to show that it is possible to apply transfer-learning as it is known from image recognition to Generative Adversarial Networks (GAN), where I have never seen it applied to.
Therefore, a GAN was first trained on the huge CelebA-Faces dataset (64x64 Pixel):

![alt text](https://raw.githubusercontent.com/developerator/Maturaarbeit/master/Images/CelebA64_results.png)

Then the dataset was radically reduced and trained only on a small so called 'transfer-dataset':
### CelebA64-GAN after 31 epochs of training on 276 images of Blondes:
![alt text](https://raw.githubusercontent.com/developerator/Maturaarbeit/master/Images/Blondes64_31.png)

In parallel, the same experiments were done with 32x32px images:

### CelebA32-GAN after 72 epochs of training on 276 images of Blondes:
![alt text](https://raw.githubusercontent.com/developerator/Maturaarbeit/master/Images/Blondes32_72.png)

### CelebA32-GAN on 500 images of flowers (Numbers denote relative order not epochs):
![alt text](https://raw.githubusercontent.com/developerator/Maturaarbeit/master/Images/Flower_evolution.png)

As a conclusion of my work, I state that transfer-learning is very well possible with GANs. In fact, I was able to get the 32-GAN to output reasonable pictures with a transfer-dataset with as few as 10 images:

### CelebA32-GAN on 10 images of flowers after 183 epochs:
![alt text](https://raw.githubusercontent.com/developerator/Maturaarbeit/master/Images/Flowers32_10.png)


## Information for recreating the experiments:

This is the setup with which the experiments were conducted:

• C# in Visual Studio 2015 (.NET Framework 4.5.2) 

• Python 3.5.3 

• Keras 2.0.5 

• Theano 0.9.0 

• Tensorflow 1.1.0 

• Windows 10 Home
