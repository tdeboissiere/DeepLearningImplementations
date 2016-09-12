# Keras Implementation of DenseNet

Original idea and implementation:

[Densely Connected Convolutional Network](http://arxiv.org/abs/1608.06993)

The figures below are taken from the paper above.

![Model archi](./figures/dense_block.jpg)

Figure 1: A dense block with 5 layers and growth rate 4.

![Model archi](./figures/densenet_archi.jpg)


Figure 2: A deep DenseNet with three dense blocks. 

# To do:

- Reproduce paper results

# Using DenseNet

    import densenet.py
    model = densenet.DenseNet(nb_classes,
                              img_dim,
                              depth,
                              nb_dense_block,
                              growth_rate,
                              nb_filter,
                              dropout_rate=dropout_rate)
    nb_classes: number of classification targets.
    img_dim: (n_channels, height, width).
    depth: Network depth, must satisfy the following: (depth - 4) % 3 == 0.
    nb_dense_block: the number of dense blocks (typically 3).
    growth_rate: the number of new convolution filters to add at each step (typically 12 or 24).
    nb_filter: the number of convolution filters at the first convolution layer (typically 16).
    dropout_rate: the dropout rate (typically 0.2).


# Architecture

With two dense blocks and 2 convolution operations within each block, the model looks like this:

![Model archi](./figures/densenet_archi.png)
