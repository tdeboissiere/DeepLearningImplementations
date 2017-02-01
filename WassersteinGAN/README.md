# Generative Adversarial Networks

Keras implementation of WassersteinGAN.

**Sources:**

- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)

# Requirements

## python modules

- keras, theano or tensorflow backend
- h5py
- matplotlib
- opencv 3
- numpy
- tqdm
- parmap


# Part 1. Processing the data

Follow [these instructions](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/WassersteinGAN/src/data).

# Part 2. Running the code

Follow [these instructions](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/WassersteinGAN/src/model)

# Part 3. Example results

**CelebA example results**

![figure](./figures/img_gan.png)<!-- 
![figure](./figures/img_gan2.png)
![figure](./figures/img_gan3.png) -->

**MNIST example results**

![figure](./figures/img_mnist.png)<!-- 
![figure](./figures/img_mnist2.png) -->

For each image:

- The first 2 rows are generated images
- The last 2 rows are real images
