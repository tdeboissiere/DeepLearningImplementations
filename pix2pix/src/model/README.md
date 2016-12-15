# Training and evaluating

## Training

`python main.py`


positional arguments:
    
    patch_size            Patch size for D

optional arguments:

    -h, --help            show this help message and exit
    --backend BACKEND     theano or tensorflow
    --generator GENERATOR
                        upsampling or deconv
    --dset DSET           facades
    --batch_size BATCH_SIZE
                        Batch size
    --n_batch_per_epoch N_BATCH_PER_EPOCH
                        Number of training epochs
    --nb_epoch NB_EPOCH   Number of batches per epoch
    --epoch EPOCH         Epoch at which weights were saved for evaluation
    --nb_classes NB_CLASSES
                        Number of classes
    --do_plot             Debugging plot
    --bn_mode BN_MODE     Batch norm mode
    --img_dim IMG_DIM     Image width == height
    --use_mbd             Whether to use minibatch discrimination
    --use_label_smoothing
                        Whether to smooth the positive labels when training D
    --label_flipping LABEL_FLIPPING
                        Probability (0 to 1.) to flip the labels when training
                        D


**Example:**

`python main.py 64 64`


### Expected outputs:

- Weights are saved in  pix2pix/models
- Figures are saved in  pix2pix/figures
- Save model weights every few epochs

### Additional notes

You can choose the type of generator:

- The image dimension must be a multiple of the patch size (e.g. 256 is a multiple of 64)
- In the discriminator, each patch goes through the same feature extractor. Then the outputs are combined with a new dense layer + softmax
- `upsampling:` generate the image with a series of `Upsampling2D` and `Convolution2D` operations 
- `deconv:` use keras' transposed convolutions `Deconvolution2D`. This is closer to the original DCGAN implementation. 

At this stage, `deconv` only works with the `tensorflow` backend.