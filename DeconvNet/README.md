# Implementation of the Deconvnet algorithm

N.B. It's still early days for this code. Suggestions for improvement are welcome !

# Requirements

- Keras==2.0.8
- numpy==1.13.3
- opencv_python==3.3.0.10
- matplotlib==2.0.2
- Theano==0.9.0

# Configure

- Unzip the images in `./Data/Img` or add your own
- Download the [VGG16 keras weights](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) and put them in `./Data`

# How to use

- Open VGG_deconv.py in your text editor

## Get the deconvolution result of random image, for a specific layer and feature map

- Go to step 4, set the flag to `True`
- Specify your target_layer (keras default naming, i.e `convolution2d_5`, `convolution2d_7` etc)
- Specify the index of the filter map you want to look at
- Specify num_img (must be a perfect square) ex: 25 => plot 5x5
- run `python VGG_deconv.py`

## Get the deconvolution of the top 9 images that maximally activate a given target_layer/map

- Go to step 1, set the flag to `True`
- Specify which layer you want in `d_act` e.g. `d_act = {"convolution2d_13": {}}`
- By default, the algorithm will look for the images maximally activating the first 10 feature maps of this layer
- Go to step 2 and set the flag to `True`
- Then run `python VGG_deconv.py`. You may have to change the batch size depending on your GPU
- It will take a little while to run and find the top 9 images
- Once this step is complete
- Go to step 3 and set the flag to `True`
- Select the layer for which you want to plot the top9 deconv images
- run `python VGG_deconv.py`


# Examples

**N.B. The VGG weights have not been trained on the Kaggle dataset !**

Below, the top 9 images activating a selection of filters in convolutional layer 10.
Each block is 2x3x3 (1 block: original crop of the image, 1 block: deconvolution result)
Each 2x3x3 block corresponds to a specific feature map.

![conv10](./Figures/convolution2d_10.png)  

Below, the top 9 images activating a selection of filters in convolutional layer 13.
Each block is 2x3x3 (1 block: original crop of the image, 1 block: deconvolution result)
Each 2x3x3 block corresponds to a specific feature map.

![conv13](./Figures/convolution2d_13.png)

It's interesting to investigate what each feature map is looking for in the original image. For instance, the top right block of convolutional layer 13 strongly reacts to the steering wheel.