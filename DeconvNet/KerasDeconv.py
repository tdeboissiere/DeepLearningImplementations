import keras.backend as K
import time
import numpy as np
np.set_printoptions(precision=2)


class DeconvNet(object):
    """DeconvNet class"""

    def __init__(self, model):
        """Creates a DeconvNet instance

        :param model: Keras model
        :returns: deconvnet class
        :rtype: deconvnet

        """
        self.model = model
        list_layers = self.model.layers
        self.lnames = [l.name for l in list_layers]
        assert len(self.lnames) == len(
            set(self.lnames)), "Non unique layer names"
        # Dict of layers indexed by layer name
        self.d_layers = {}
        for l_name, l in zip(self.lnames, list_layers):
            self.d_layers[l_name] = l

        # Tensor for function definitions
        self.x = K.T.tensor4('x')

    def __getitem__(self, layer_name):
        try:
            return self.d_layers[layer_name]
        except KeyError:
            print "Erroneous layer name"

    def _deconv(self, X, lname, d_switch, feat_map=None):
        o_width, o_height = self[lname].output_shape[-2:]

        # Get filter size
        f_width = self[lname].W_shape[2]
        f_height = self[lname].W_shape[3]

        # Compute padding needed
        i_width, i_height = X.shape[-2:]
        pad_width = (o_width - i_width + f_width - 1) / 2
        pad_height = (o_height - i_height + f_height - 1) / 2

        assert isinstance(
            pad_width, int), "Pad width size issue at layer %s" % lname
        assert isinstance(
            pad_height, int), "Pad height size issue at layer %s" % lname

        # Set to zero based on switch values
        X[d_switch[lname]] = 0
        # Get activation function
        activation = self[lname].activation
        X = activation(X)
        if feat_map is not None:
            print "Setting other feat map to zero"
            for i in range(X.shape[1]):
                if i != feat_map:
                    X[:, i, :, :] = 0
            print "Setting non max activations to zero"
            for i in range(X.shape[0]):
                iw, ih = np.unravel_index(
                    X[i, feat_map, :, :].argmax(), X[i, feat_map, :, :].shape)
                m = np.max(X[i, feat_map, :, :])
                X[i, feat_map, :, :] = 0
                X[i, feat_map, iw, ih] = m
        # Get filters. No bias for now
        W = self[lname].W
        # Transpose filter
        W = W.transpose([1, 0, 2, 3])
        W = W[:, :, ::-1, ::-1]
        # CUDNN for conv2d ?
        conv_out = K.T.nnet.conv2d(
            input=self.x, filters=W, border_mode='valid')
        # Add padding to get correct size
        pad = K.function([self.x], K.spatial_2d_padding(
            self.x, padding=(pad_width, pad_height), dim_ordering="th"))
        X_pad = pad([X])
        # Get Deconv output
        deconv_func = K.function([self.x], conv_out)
        X_deconv = deconv_func([X_pad])
        assert X_deconv.shape[-2:] == (o_width, o_height),\
            "Deconv output at %s has wrong size" % lname
        return X_deconv

    def _forward_pass(self, X, target_layer):

        # For all layers up to the target layer
        # Store the max activation in switch
        d_switch = {}
        layer_index = self.lnames.index(target_layer)
        for lname in self.lnames[:layer_index + 1]:
            # Get layer output
            inc, out = self[lname].input, self[lname].output
            f = K.function([inc], out)
            X = f([X])
            if "convolution2d" in lname:
                d_switch[lname] = np.where(X <= 0)
        return d_switch

    def _backward_pass(self, X, target_layer, d_switch, feat_map):
        # Run deconv/maxunpooling until input pixel space
        layer_index = self.lnames.index(target_layer)
        # Get the output of the target_layer of interest
        layer_output = K.function(
            [self[self.lnames[0]].input], self[target_layer].output)
        X_outl = layer_output([X])
        # Special case for the starting layer where we may want
        # to switchoff somes maps/ activations
        print "Deconvolving %s..." % target_layer
        if "maxpooling2d" in target_layer:
            X_maxunp = K.pool.max_pool_2d_same_size(
                self[target_layer].input, self[target_layer].pool_size)
            unpool_func = K.function([self[self.lnames[0]].input], X_maxunp)
            X_outl = unpool_func([X])
            if feat_map is not None:
                for i in range(X_outl.shape[1]):
                    if i != feat_map:
                        X_outl[:, i, :, :] = 0
                for i in range(X_outl.shape[0]):
                    iw, ih = np.unravel_index(
                        X_outl[i, feat_map, :, :].argmax(), X_outl[i, feat_map, :, :].shape)
                    m = np.max(X_outl[i, feat_map, :, :])
                    X_outl[i, feat_map, :, :] = 0
                    X_outl[i, feat_map, iw, ih] = m
        elif "convolution2d" in target_layer:
            X_outl = self._deconv(X_outl, target_layer,
                                  d_switch, feat_map=feat_map)
        else:
            raise ValueError(
                "Invalid layer name: %s \n Can only handle maxpool and conv" % target_layer)
        # Iterate over layers (deepest to shallowest)
        for lname in self.lnames[:layer_index][::-1]:
            print "Deconvolving %s..." % lname
            # Unpool, Deconv or do nothing
            if "maxpooling2d" in lname:
                p1, p2 = self[lname].pool_size
                uppool = K.function(
                    [self.x], K.resize_images(self.x, p1, p2, "th"))
                X_outl = uppool([X_outl])

            elif "convolution2d" in lname:
                X_outl = self._deconv(X_outl, lname, d_switch)
            elif "padding" in lname:
                pass
            else:
                raise ValueError(
                    "Invalid layer name: %s \n Can only handle maxpool and conv" % lname)
        return X_outl

    def get_layers(self):
        """Returns the layers of the network

        :returns: returns the layers of the network
        :rtype: list

        """
        list_layers = self.model.layers
        list_layers_name = [l.name for l in list_layers]
        return list_layers_name

    def get_deconv(self, X, target_layer, feat_map=None):
        """ Starts the deconvolution process

        :param X: input image: the data to be deconv-ed
        :param target_layer: layer in the model up to which
        we want to be deconvolved
        :param feat_map: # @TODO: update
        :returns: deconvolved image
        :rtype: numpy array

        """

        # First make predictions to get feature maps
        self.model.predict(X)
        # Forward pass storing switches
        print "Starting forward pass..."
        start_time = time.time()
        d_switch = self._forward_pass(X, target_layer)
        end_time = time.time()
        print 'Forward pass completed in %ds' % (end_time - start_time)
        # Then deconvolve starting from target layer
        X_out = self._backward_pass(X, target_layer, d_switch, feat_map)
        return X_out
