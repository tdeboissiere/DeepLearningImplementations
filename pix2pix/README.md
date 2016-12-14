# pix2pix

Keras implementation of [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf) by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros


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

Follow [these instructions](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix/src/data).

# Part 2. Running the code

Follow [these instructions](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix/src/model)

# Part 3. Example results

**Facade example results (validation set)**

![figure](./figures/img_pix2pix.png)

For each image:

- The first row is the generator input
- The second row is the generator output
- The last row is the original image 
