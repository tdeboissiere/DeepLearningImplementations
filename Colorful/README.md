# Colorful

Keras implementation of [Colorful Image Colorization](https://arxiv.org/abs/1603.08511) by Richard Zhang, Phillip Isola and Alexei A. Efros

The technique is applied on the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) with minor modifications.

# Requirements

## python modules

- scikit-learn==0.19.1
- scikit-image==0.13.1
- tqdm==4.17.0
- opencv_python==3.3.0.10
- numpy==1.13.3
- matplotlib==2.0.2
- Keras==2.0.8
- Theano==0.9.0 or Tensorflow==1.3.0
- h5py==2.7.0
- parmap==1.5.1
- scipy==1.0.0

## System requirements

- Nvidia GPU with at least 2GB RAM
- At least 4GB RAM (when using the on_demand option for training)

The settings above should work well enough for small image size (32 x 32).
Above that, better GPU and more RAM are required.

# Part 1. Processing the data

Follow [these instructions](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/Colorful/src/data).

# Part 2. Running the code

Follow [these instructions](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/Colorful/src/model)

# Part 3. Example results

![figure](./figures/celeba_colorized_small.png)

For each triplet:

- First column is the original
- Second column is the B&W version
- Last column is the colorized output

# Part 4. Live colorization with webcam

Follow [these instructions](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/Colorful/src/app)