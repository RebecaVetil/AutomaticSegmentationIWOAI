# MIT License
#
# Copyright (c) 2018 Miguel Monteiro
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from Layers import convolution, down_convolution, up_convolution, get_num_channels,prelu, get_spatial_rank


def compression_block(layer_input, num_conv_layer, keep_prob, activation_fn, is_training):
    """
    This function computes the ops corresponding to one level of the left path (compression)
    ---
    :param: layer_input: a tensor of shape [batch_size + input_spatial_shape + in_channels]
    :param: num_conv_layer: an int giving the number of convolution layers in this level
    (between 1 and 3 convolutions - according to the original vnet)
    :param: keep_prob: an int giving the probability that each element of x is kept 
    :param: activation_fn: the activation function for this block
    :param: is_training: a bool indicating wheter it is training or not
    ---- 
    :return: a tensor of shape [batch_size + output_spatial_shape + in_channels]
    (the number of channel is kept unchanged)
    """
    x = layer_input
    n_channels = get_num_channels(x)
    spatial_size = get_spatial_rank(x) 
    for i in range(num_conv_layer): 
        # variable sharing
        with tf.variable_scope('conv_' + str(i+1)): 
            x = convolution(x, spatial_size * [5] + [n_channels, n_channels]) 
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training) 
            if i == num_conv_layer - 1: 
                # at the end of the last convolution layer: residual learning
                layer_input = tf.layers.batch_normalization(layer_input, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training) 
                x = x + layer_input 
                x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training) 
            x = activation_fn(x) 
            x = tf.nn.dropout(x, keep_prob) 
    return x


def decompression_block(layer_input, fine_grained_features, num_conv_layer, keep_prob, activation_fn, is_training):
    """
    This function computes the ops corresponding to one level of the right path (decompression)
    It 'extracts features and expands the spatial support of the lower resolution feature maps in order to 
    gather and assemble the necessary information to output a two channel volumetric segmentation.'
    (quote from the original VNet paper)
    ---
    :param: layer_input: a tensor of shape [batch_size + input_spatial_shape + in_channels]
    :param: fine_grained_features: a tensor of features extracted from the left part
    :param: num_conv_layer: an int giving the number of convolution layers in this level
    :param: keep_prob: an int giving the probability that each element of x is kept 
    :param: activation_fn: the activation function for this block
    :param: is_training: a bool indicating wheter it is training or not
    ---- 
    :return: a tensor of shape [batch_size + output_spatial_shape + in_channels]
    (the number of channel is kept unchanged)
    """   
    # Concatenate the fine grained features
    x = tf.concat((layer_input, fine_grained_features), axis=-1) 

    n_channels = get_num_channels(layer_input)
    spatial_size = get_spatial_rank(layer_input) 
    # Special case: the block has only one convolutional layer
    # This is the case of the last decompressing block in the original vnet 
    if num_conv_layer == 1: 
        with tf.variable_scope('conv_' + str(1)):
            # First convolution
            x = convolution(x, spatial_size * [5] + [n_channels * 2, n_channels]) 
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training) 
            # Normalise the input layer, add it to the last convolution result, normalise again
            layer_input = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            x = x + layer_input
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)
        return x

    # Other cases: the block has more than one convolution layer
    # First convolution layer
    with tf.variable_scope('conv_' + str(1)):
        # in_channels = n_channels * 2 due to concatenation 
        x = convolution(x, spatial_size * [5] + [n_channels * 2, n_channels])
        x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
        x = activation_fn(x)
        x = tf.nn.dropout(x, keep_prob)

    # Remaining convolution layers
    for i in range(1, num_conv_layer):
        with tf.variable_scope('conv_' + str(i+1)):
            # the number of channel is kept unchanged
            x = convolution(x, spatial_size * [5] + [n_channels, n_channels])
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            if i == num_conv_layer - 1: 
                # at the end of the last convolution layer: residual learning
                layer_input = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
                x = x + layer_input 
                x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)

    return x


class VNet(object):
    def __init__(self,
                 num_classes =7,
                 keep_prob=1.0,
                 num_channels=16,
                 num_levels=4,
                 num_convolutions=(1, 2, 3, 3),
                 bottom_convolutions=3,
                 is_training = True,
                 activation_fn="relu"):
        """
        Implements VNet architecture https://arxiv.org/abs/1606.04797
        :param num_classes: Number of output classes.
        :param keep_prob: Dropout keep probability, set to 1.0 if not training or if no dropout is desired.
        :param num_channels: The number of output channels in the first level, this will be doubled every level. Default is 16 as in the paper.
        :param num_levels: The number of levels in the network. Default is 4 as in the paper.
        :param num_convolutions: An array with the number of convolutions at each level.
        :param bottom_convolutions: The number of convolutions at the bottom level of the network.
        :param activation_fn: The activation function.
        """
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.num_channels = num_channels
        assert num_levels == len(num_convolutions)
        self.num_levels = num_levels
        self.num_convolutions = num_convolutions
        self.bottom_convolutions = bottom_convolutions
        self.is_training = is_training

        if (activation_fn == "relu"):
            self.activation_fn = tf.nn.relu
        elif(activation_fn == "prelu"):
            self.activation_fn = prelu

    def network_fn(self, x):
        """
        This function passes x through the whole network
        ---
        :param: x: input tensor of shape [batch_size, input_spatial_shape, input_channels]
        ---- 
        :return: a tensor of shape [batch_size, input_spatial_shape, num_classes] 
        """

        keep_prob = self.keep_prob if self.is_training else 1.0
        spatial_size = get_spatial_rank(x) 
        # Input processing
        input_channels = int(x.get_shape()[-1])
        with tf.variable_scope('vnet/input_layer'):
            if input_channels == 1: 
                # Grayscale images
                # We replicate the input along the channel dimension to make it match with 
                # self.num_channels (number of output channels in the first level. default:16)
                x = tf.tile(x, [1] + spatial_size * [1] +  [self.num_channels]) 
                x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.is_training)

            else: 
                # other type of images (RGB)
                x = convolution(x, [5] * spatial_size + [input_channels, self.num_channels])
                x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.is_training)
                x = self.activation_fn(x)

        features = list()

        # Left path - compression
        for l in range(self.num_levels): 
            with tf.variable_scope('vnet/encoder/level_' + str(l + 1)):
                # Level l : compression block
                x = compression_block(x, self.num_convolutions[l], keep_prob, activation_fn=self.activation_fn, is_training=self.is_training)
                # Store the tensor as fine-grained features for the right path
                features.append(x) 
                # Down convolution: divide the spatial dimension and increase the number of features by a factor 2
                with tf.variable_scope('down_convolution'): 
                    x = down_convolution(x, factor=2, kernel_size=[2]* spatial_size)
                    x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.is_training)
                    x = self.activation_fn(x)

        # Bottom level
        with tf.variable_scope('vnet/bottom_level'):
            x = compression_block(x, self.bottom_convolutions, keep_prob, activation_fn=self.activation_fn, is_training=self.is_training)

        # Right path - decompression
        for l in reversed(range(self.num_levels)):
            with tf.variable_scope('vnet/decoder/level_' + str(l + 1)):
                # Level l 
                # Get the fine-grained features from the corresponding compressing level
                f = features[l] 
                # Start by up-convolution: 
                with tf.variable_scope('up_convolution'): 
                    # We generate a tensor with the same shape as the fine-grained feature tensor
                    x = up_convolution(x, tf.shape(f), factor=2, kernel_size= [2]* spatial_size)
                    x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.is_training)
                    x = self.activation_fn(x)
                # Decompression block 
                x = decompression_block(x, f, self.num_convolutions[l], keep_prob, activation_fn=self.activation_fn, is_training=self.is_training)

        # Output processing 
        with tf.variable_scope('vnet/output_layer'):
            # Compute the output features maps with a 1×1×1 kernel size 
            # and produce the raw outputs 
            logits = convolution(x, [1] * spatial_size + [self.num_channels, self.num_classes])
            logits = tf.layers.batch_normalization(logits, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.is_training)

        return logits