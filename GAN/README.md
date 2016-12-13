# Generative Adversarial Networks

Keras implementation of some GAN models.

**Sources:**

- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- [GANHacks](https://github.com/soumith/ganhacks)

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

Follow [these instructions](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/GAN/src/data).

# Part 2. Running the code

Follow [these instructions](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/GAN/src/model)

# Part 3. Example results

**CelebA example results**

![figure](./figures/img_gan.png)
![figure](./figures/img_gan2.png)
![figure](./figures/img_gan3.png)

**MNIST example results**

![figure](./figures/img_mnist.png)
![figure](./figures/img_mnist2.png)

For each image:

- The first 2 rows are generated images
- The last 2 rows are real images
