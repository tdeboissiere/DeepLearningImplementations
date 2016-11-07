# Keras Implementation of DenseRecNet


Adding recurrent connections to the DenseNet model.

## Usage guide:

python run_cifar10.py

optional arguments:

    -h, --help show this help message and exit
    --batch_size BATCH_SIZE Batch size
    --nb_epoch NB_EPOCH  Number of epochs
    --depth DEPTH  Network depth
    --nb_dense_block NB_DENSE_BLOCK Number of dense blocks
    --nb_filter NB_FILTER Initial number of conv filters
    --growth_rate GROWTH_RATE Number of new filters added by conv layers
    --dropout_rate DROPOUT_RATE  Dropout rate
    --learning_rate LEARNING_RATE Learning rate
    --weight_decay WEIGHT_DECAY L2 regularization on weights
    --plot_architecture PLOT_ARCHITECTURE Save a plot of the network architecture

## Output

Trains a network for the specified number of epochs. Save the traing/validation loss in a .json file.
