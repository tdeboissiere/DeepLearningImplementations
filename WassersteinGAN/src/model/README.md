# Training WGAN

## Usage

`python main.py`


optional arguments:

    --backend BACKEND     theano or tensorflow
    --generator GENERATOR
                          upsampling or deconv
    --dset DSET           mnist or celebA or cifar10 or toy
    --img_dim IMG_DIM     Image width == height
    --nb_epoch NB_EPOCH   Number of batches per epoch
    --batch_size BATCH_SIZE
                          Batch size
    --n_batch_per_epoch N_BATCH_PER_EPOCH
                          Number of training epochs
    --bn_mode BN_MODE     Batch norm mode
    --noise_dim NOISE_DIM
                          noise sampler dimension
    --noise_scale NOISE_SCALE
                          noise sampler variance
    --disc_iterations DISC_ITERATIONS
                          Number of discriminator iterations
    --clamp_lower CLAMP_LOWER
                          Clamp weights below this value
    --clamp_upper CLAMP_UPPER
                          Clamp weights above this value
    --opt_D OPT_D         Optimizer for the discriminator
    --opt_G OPT_G         Optimizer for the generator
    --lr_D LR_D           learning rate for the discriminator
    --lr_G LR_G           learning rate for the generator




**Example:**

`python main.py --backend tensorflow --generator deconv --dset celebA`

**N.B.** If using the CelebA dataset, make sure to specify the corresponding img_dim value. For instance, if you saved a 64x64 CelebA hdf5 dataset, call `python main.py --dset celebA --img_dim 64`


**Toy experiment:**

`python main.py --backend tensorflow --generator deconv --dset toy --lr_G 1E-3 --lr_D 1E-3 --clamp_lower -0.5 --clamp_upper 0.5 --batch_size 512 --noise_dim 128` generally gives good results.

To produce the `.gif`, let it run a few epochs to save some images. Then in `../../figures`:

- `python write_gif_script.py`
- `bash make_gif.sh`


## Expected outputs:

- Weights are saved in  `WassersteinGAN/models`
- Figures are saved in  `WassersteinGAN/figures`
- Save model weights every few epochs

## Implementation notes:

### Weight clipping:

    for l in discriminator_model.layers:
        weights = l.get_weights()
        weights = [np.clip(w, clamp_lower, clamp_upper) for w in weights]
        l.set_weights(weights)

### Wasserstein objective:

A new `keras` objective is defined:

    def wasserstein(y_true, y_pred):
        return K.mean(y_true * y_pred)


### Discriminator training:

Step1: **maximize**

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{1}{m}&space;\sum_i^nf_w(x^{(i)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{1}{m}&space;\sum_i^nf_w(x^{(i)})" title="\frac{1}{m} \sum_i^nf_w(x^{(i)})" /></a>

    disc_loss_real = discriminator_model.train_on_batch(X_disc_real, -np.ones(X_disc_real.shape[0]))

which is why there is a `-` sign in the train target

Step2: **minimize**

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{1}{m}&space;\sum_i^nf_w(g_{\theta}(z^{i}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{1}{m}&space;\sum_i^nf_w(g_{\theta}(z^{i}))" title="\frac{1}{m} \sum_i^nf_w(g_{\theta}(z^{i}))" /></a>

    disc_loss_gen = discriminator_model.train_on_batch(X_disc_gen, np.ones(X_disc_gen.shape[0]))

which is why there is *no* `-` sign in the train target

### Generator training:

The generator is trained to **maximize**

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{1}{m}&space;\sum_i^nf_w(g_{\theta}(z^{i}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{1}{m}&space;\sum_i^nf_w(g_{\theta}(z^{i}))" title="\frac{1}{m} \sum_i^nf_w(g_{\theta}(z^{i}))" /></a>

    gen_loss = DCGAN_model.train_on_batch(X_gen, -np.ones(X_gen.shape[0]))

which is why this time there is a `-` sign in the train target

## Additional notes

You can choose the type of generator:

- `upsampling:` generate the image with a series of `Upsampling2D` and `Convolution2D` operations 
- `deconv:` use keras' transposed convolutions `Deconvolution2D`. This is closer to the original DCGAN implementation. 

At this stage, `deconv` only works with the `tensorflow` backend.