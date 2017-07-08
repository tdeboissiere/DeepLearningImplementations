import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np


# network
class RELUNet(nn.Module):
    def __init__(self, n_inner_layers, input_dim, hidden_dim, output_dim, dropout=0, batchnorm=True):

        super(RELUNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.n_inner_layers = n_inner_layers

        # FC layers
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        # Hacky way to set inner layers. Ensures they are all converted to
        for k in range(n_inner_layers):
            setattr(self, "fc_%s" % k, nn.Linear(hidden_dim, hidden_dim))
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # BN layers
        self.bn_in = nn.BatchNorm1d(hidden_dim)

        for k in range(n_inner_layers):
            setattr(self, "bn_%s" % k, nn.BatchNorm1d(hidden_dim))

        # Initialize weights specifically for relu

        # First layer
        init.normal(self.fc_in.weight, std=2. / np.sqrt(np.float32(self.input_dim)))
        init.constant(self.fc_in.bias, 0.)

        # Inner layers
        for i in range(self.n_inner_layers):
            init.normal(getattr(self, "fc_%s" % i).weight, std=2. / np.sqrt(np.float32(self.hidden_dim)))
            init.constant(getattr(self, "fc_%s" % i).bias, 0.)

        # Last layer
        init.normal(self.fc_out.weight, std=2. / np.sqrt(np.float32(self.hidden_dim)))
        init.constant(self.fc_out.bias, 0.)

    def forward(self, x, training=False):

        # First layer
        x = self.fc_in(x)
        if self.batchnorm:
            x = self.bn_in(x)
        x = F.relu(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=training)

        # Inner layers
        for i in range(self.n_inner_layers):
            x = getattr(self, "fc_%s" % i)(x)
            if self.batchnorm:
                x = getattr(self, "bn_%s" % i)(x)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=training)

        # Output layers
        x = self.fc_out(x)

        return x


def alpha_dropout(input, p=0.5, training=False):
    """Applies alpha dropout to the input.

    See :class:`~torch.nn.AlphaDropout` for details.

    Args:
        p (float, optional): the drop probability
        training (bool, optional): switch between training and evaluation mode
    """
    if p < 0 or p > 1:
        raise ValueError("dropout probability has to be between 0 and 1, "
                         "but got {}".format(p))

    if p == 0 or not training:
        return input

    alpha = -1.7580993408473766
    keep_prob = 1 - p
    # TODO avoid casting to byte after resize
    noise = input.data.new().resize_(input.size())
    noise.bernoulli_(p)
    noise = Variable(noise.byte())

    output = input.masked_fill(noise, alpha)

    a = (keep_prob + alpha ** 2 * keep_prob * (1 - keep_prob)) ** (-0.5)
    b = -a * alpha * (1 - keep_prob)

    return output.mul_(a).add_(b)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)


class SELUNet(nn.Module):
    def __init__(self, n_inner_layers, input_dim, hidden_dim, output_dim, dropout=0.05):

        super(SELUNet, self).__init__()

        self.dropout = dropout

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_inner_layers = n_inner_layers
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        for k in range(n_inner_layers):
            setattr(self, "fc_%s" % k, nn.Linear(hidden_dim, hidden_dim))
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # Initialize weights specifically for selu

        # First layer
        init.normal(self.fc_in.weight, std=1. / np.sqrt(np.float32(self.input_dim)))
        init.constant(self.fc_in.bias, 0.)

        # Inner layers
        for i in range(self.n_inner_layers):
            init.normal(getattr(self, "fc_%s" % i).weight, std=1. / np.sqrt(np.float32(self.hidden_dim)))
            init.constant(getattr(self, "fc_%s" % i).bias, 0.)

        # Last layer
        init.normal(self.fc_out.weight, std=1. / np.sqrt(np.float32(self.hidden_dim)))
        init.constant(self.fc_out.bias, 0.)

    def forward(self, x, training=False):

        # First layer
        x = self.fc_in(x)
        x = selu(x)
        if self.dropout > 0:
            x = alpha_dropout(x, p=self.dropout, training=training)

        # Inner layers
        for i in range(self.n_inner_layers):
            x = getattr(self, "fc_%s" % i)(x)
            x = selu(x)
            if self.dropout > 0:
                x = alpha_dropout(x, p=self.dropout, training=training)

        # Output layers
        x = self.fc_out(x)

        return x
