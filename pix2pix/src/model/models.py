from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Convolution2D, Deconvolution2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K


def conv_block_unet(x, f, name, bn_mode, bn_axis, bn=True, dropout=False):

    x = Convolution2D(f, 3, 3, subsample=(2, 2), name=name, border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)
    if dropout:
        x = Dropout(0.5)(x)

    return x


def up_conv_block_unet(x1, x2, f, name, bn_mode, bn_axis, bn=True, dropout=False):

    x1 = UpSampling2D(size=(2, 2))(x1)
    x = merge([x1, x2], mode='concat', concat_axis=bn_axis)

    x = Convolution2D(f, 3, 3, name=name, border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = Activation("relu")(x)
    if dropout:
        x = Dropout(0.5)(x)

    return x


def generator_unet(img_dim, bn_mode, model_name="generator_unet"):

    nb_filters = 64

    if K.image_dim_ordering() == "th":
        bn_axis = 1
        nb_channels = img_dim[0]
    else:
        bn_axis = -1
        nb_channels = img_dim[-1]

    unet_input = Input(shape=img_dim, name="unet_input")

    # Encoder
    conv1 = conv_block_unet(unet_input, nb_filters, "unet_conv2D_1", bn_mode, bn_axis, bn=False)
    conv2 = conv_block_unet(conv1, nb_filters * 2, "unet_conv2D_2", bn_mode, bn_axis)
    conv3 = conv_block_unet(conv2, nb_filters * 4, "unet_conv2D_3", bn_mode, bn_axis)
    conv4 = conv_block_unet(conv3, nb_filters * 8, "unet_conv2D_4", bn_mode, bn_axis)
    conv5 = conv_block_unet(conv4, nb_filters * 8, "unet_conv2D_5", bn_mode, bn_axis)
    conv6 = conv_block_unet(conv5, nb_filters * 8, "unet_conv2D_6", bn_mode, bn_axis)
    conv7 = conv_block_unet(conv6, nb_filters * 8, "unet_conv2D_7", bn_mode, bn_axis)
    conv8 = conv_block_unet(conv7, nb_filters * 8, "unet_conv2D_8", bn_mode, bn_axis, dropout=True)

    # Decoder
    upconv1 = up_conv_block_unet(conv8, conv7, nb_filters * 8, "unet_upconv2D_1", bn_mode, bn_axis, dropout=True)
    upconv2 = up_conv_block_unet(upconv1, conv6, nb_filters * 8, "unet_upconv2D_2", bn_mode, bn_axis, dropout=True)
    upconv3 = up_conv_block_unet(upconv2, conv5, nb_filters * 8, "unet_upconv2D_3", bn_mode, bn_axis)
    upconv4 = up_conv_block_unet(upconv3, conv4, nb_filters * 4, "unet_upconv2D_4", bn_mode, bn_axis)
    upconv5 = up_conv_block_unet(upconv4, conv3, nb_filters * 2, "unet_upconv2D_5", bn_mode, bn_axis)
    upconv6 = up_conv_block_unet(upconv5, conv2, nb_filters * 1, "unet_upconv2D_6", bn_mode, bn_axis)
    upconv7 = up_conv_block_unet(upconv6, conv1, nb_filters * 1, "unet_upconv2D_7", bn_mode, bn_axis)

    x = UpSampling2D(size=(2, 2))(upconv7)
    x = Convolution2D(nb_channels, 1, 1, activation='tanh')(x)

    generator_unet = Model(input=[unet_input], output=[x])

    return generator_unet


def DCGAN_discriminator(img_dim, nb_patch, bn_mode, model_name="DCGAN_discriminator"):
    """
    Discriminator model of the DCGAN

    args : img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    list_input = [Input(shape=img_dim, name="disc_input_%s" % i) for i in range(nb_patch)]

    if K.image_dim_ordering() == "th":
        bn_axis = 1
    else:
        bn_axis = -1

    # First conv
    x_input = Input(shape=img_dim, name="discriminator_input")
    x = Convolution2D(64, 3, 3, subsample=(2, 2), name="disc_conv2d_1", border_mode="same")(x_input)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)

    # Next convs
    list_f = [128, 256, 512]
    for i, f in enumerate(list_f):
        name = "disc_conv2d_%s" % (i + 2)
        x = Convolution2D(f, 3, 3, subsample=(2, 2), name=name, border_mode="same")(x)
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(2, activation='softmax', name="disc_dense")(x)

    PatchGAN = Model(input=[x_input], output=[x], name="PatchGAN")

    list_output = [PatchGAN(patch) for patch in list_input]
    final_output = merge(list_output, mode="concat", name="merge_patch")
    final_output = Dense(2, activation="softmax", name="disc_output")(final_output)

    discriminator_model = Model(input=list_input, output=[final_output], name=model_name)

    return discriminator_model


def DCGAN(generator, discriminator_model, img_dim, patch_size, image_dim_ordering):

    gen_input = Input(shape=img_dim, name="DCGAN_input")

    generated_image = generator(gen_input)

    if image_dim_ordering == "th":
        h, w = img_dim[1:]
    else:
        h, w = img_dim[:-1]
    ph, pw = patch_size

    list_row_idx = [(i * ph, (i + 1) * ph) for i in range(h / ph)]
    list_col_idx = [(i * pw, (i + 1) * pw) for i in range(w / pw)]

    list_gen_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            if image_dim_ordering == "tf":
                x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(generated_image)
            else:
                x_patch = Lambda(lambda z: z[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]])(generated_image)
            list_gen_patch.append(x_patch)

    DCGAN_output = discriminator_model(list_gen_patch)

    DCGAN = Model(input=[gen_input],
                  output=[generated_image, DCGAN_output],
                  name="DCGAN")

    return DCGAN


def load(model_name, img_dim, nb_patch, bn_mode):

    if model_name == "generator_unet":
        model = generator_unet(img_dim, bn_mode, model_name=model_name)
        print model.summary()
        from keras.utils.visualize_util import plot
        plot(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model

    if model_name == "DCGAN_discriminator":
        model = DCGAN_discriminator(img_dim, nb_patch, bn_mode, model_name=model_name)
        model.summary()
        from keras.utils.visualize_util import plot
        plot(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model


if __name__ == '__main__':

    load("generator_unet", (100,), (3, 256, 256), 2)
