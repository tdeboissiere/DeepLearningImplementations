# Training and evaluating

## Training

`python main.py`


positional arguments:
    
    mode                  train or eval

optional arguments:

    -h, --help            show this help message and exit
    --backend BACKEND     theano or tensorflow
    --generator GENERATOR
                        upsampling or deconv
    --dset DSET           mnist or celebA
    --batch_size BATCH_SIZE
                        Batch size
    --n_batch_per_epoch N_BATCH_PER_EPOCH
                        Number of training epochs
    --nb_epoch NB_EPOCH   Number of batches per epoch
    --epoch EPOCH         Epoch at which weights were saved for evaluation
    --nb_classes NB_CLASSES
                        Number of classes
    --do_plot DO_PLOT     Debugging plot
    --bn_mode BN_MODE     Batch norm mode
    --img_dim IMG_DIM     Image width == height (only specify for CelebA)
    --noise_dim NOISE_DIM
                        noise dimension
    --cont_dim CONT_DIM   Latent continuous dimensions
    --cat_dim CAT_DIM     Latent categorical dimension
    --noise_scale NOISE_SCALE
                        variance of the normal from which we sample the noise
    --label_smoothing     smooth the positive labels when training D
    --use_mbd             use mini batch disc
    --label_flipping LABEL_FLIPPING
                        Probability (0 to 1.) to flip the labels when training
                        D



**Example:**

`python main.py train`

**N.B.** If using the CelebA dataset, make sure to specify the corresponding img_dim value. For instance, if you saved a 64x64 CelebA hdf5 dataset, call `python main.py --img_dim 64`


### Expected outputs:

- Weights are saved in  InfoGAN/models
- Figures are saved in  InfoGAN/figures
- Save model weights every few epochs


### Additional notes

You can choose the type of generator:

- `upsampling:` generate the image with a series of `Upsampling2D` and `Convolution2D` operations 
- `deconv:` use keras' transposed convolutions `Deconvolution2D`. This is closer to the original DCGAN implementation. 

At this stage, `deconv` only works with the `tensorflow` backend.