# Training and evaluating

## Training

`python main.py`


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
      --img_dim IMG_DIM     Image width == height
      --noise_scale NOISE_SCALE
                            variance of the normal from which we sample the noise
      --disc_iterations DISC_ITERATIONS
                            Number of discriminator iterations



**Example:**

`python main.py --backend tensorflow --generator deconv --dset celebA`

**N.B.** If using the CelebA dataset, make sure to specify the corresponding img_dim value. For instance, if you saved a 64x64 CelebA hdf5 dataset, call `python main.py --dset celebA --img_dim 64`


### Expected outputs:

- Weights are saved in  `WassersteinGAN/models`
- Figures are saved in  `WassersteinGAN/figures`
- Save model weights every few epochs

### Implementation notes:

#### Weight clipping:

    for l in discriminator_model.layers:
        weights = l.get_weights()
        weights = [np.clip(w, -0.01, 0.01) for w in weights]
        l.set_weights(weights)

#### Wasserstein objective:

A new `keras` objective is defined:

    def wasserstein(y_true, y_pred):
        return K.mean(y_true * y_pred)


#### Training:

**Discriminator:**

Step1: **maximize**

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{1}{m}&space;\sum_i^nf_w(x^{(i)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{1}{m}&space;\sum_i^nf_w(x^{(i)})" title="\frac{1}{m} \sum_i^nf_w(x^{(i)})" /></a>

    disc_loss_real = discriminator_model.train_on_batch(X_disc_real, -np.ones(X_disc_real.shape[0]))

which is why there is a `-` sign in the train target

Step2: **minimize**

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{1}{m}&space;\sum_i^nf_w(g_{\theta}(z^{i}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{1}{m}&space;\sum_i^nf_w(g_{\theta}(z^{i}))" title="\frac{1}{m} \sum_i^nf_w(g_{\theta}(z^{i}))" /></a>

    disc_loss_gen = discriminator_model.train_on_batch(X_disc_gen, np.ones(X_disc_gen.shape[0]))

which is why there is *no* `-` sign in the train target

**Generator:**

The generator is trained to **maximize**

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{1}{m}&space;\sum_i^nf_w(g_{\theta}(z^{i}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{1}{m}&space;\sum_i^nf_w(g_{\theta}(z^{i}))" title="\frac{1}{m} \sum_i^nf_w(g_{\theta}(z^{i}))" /></a>

    gen_loss = DCGAN_model.train_on_batch(X_gen, -np.ones(X_gen.shape[0]))

which is why this time there is a `-` sign in the train target

### Additional notes

You can choose the type of generator:

- `upsampling:` generate the image with a series of `Upsampling2D` and `Convolution2D` operations 
- `deconv:` use keras' transposed convolutions `Deconvolution2D`. This is closer to the original DCGAN implementation. 

At this stage, `deconv` only works with the `tensorflow` backend.