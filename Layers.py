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

# Modification by Rebeca V., Imperial College London, done during Summer 2019

import tensorflow as tf
import numpy as np


def xavier_initializer_convolution(shape, dist='uniform', lambda_initializer=True):
    """
    This function initializes the weights according to xavier initialization.
    ---
    :param: shape: The shape of the convolution patch (spatial_shape + [in_channels, out_channels]
    
    :param: dist: A string either 'uniform' or 'normal' determining the type of distribution    
    Uniform: lim = sqrt(3/(input_activations + output_activations))
    Normal: stddev =  sqrt(6/(input_activations + output_activations))
   
    :param: lambda_initializer: Whether to return the initial actual values of the parameters 
    (True) or placeholders that are initialized when the session is initiated
    ---- 
    :return: A numpy array with the initial values for the parameters in the patch
    """
    # rank of the filter
    s = len(shape) - 2 

    # num_activations   = in_activations + out_activations
    ## in_activations   = patch_volume (=x*y*z) * in_channels; 
    ## out_activations  = patch_volume (=x*y*z)* out_channels; 
    #                   = patch_volume * (in_channels + out_channels)
    num_activations = np.prod(shape[:s]) * np.sum(shape[s:]) 

    # initialising according to the type of distribution
    if dist == 'uniform': 
        # any number between -lim and lim has equiprobability to be drawn
        lim = np.sqrt(6. / num_activations)
        if lambda_initializer:
            return np.random.uniform(-lim, lim, shape).astype(np.float32) 
        else:
            return tf.random_uniform(shape, minval=-lim, maxval=lim)
    if dist == 'normal': 
        # any number is drawn with probability of the normal distribution
        stddev = np.sqrt(3. / num_activations)
        if lambda_initializer:
            return np.random.normal(0, stddev, shape).astype(np.float32)
        else:
            tf.truncated_normal(shape, mean=0, stddev=stddev)
    raise ValueError('Distribution must be either "uniform" or "normal".')


def constant_initializer(value, shape, lambda_initializer=True):
    """
    This functions initializes a constant
    ---
    :param: value: the value of the constant 
    
    :param :shape: the shape of the constant
   
    :param: lambda_initializer: Whether to return the initial actual values of the parameters 
    (True) or placeholders that are initialized when the session is initiated
    ---- 
    :return: a numpy array with the value of a placeholders that will be initialised later on
    """
    if lambda_initializer: 
        # returns the initial actual values of the parameters 
        return np.full(shape, value).astype(np.float32)
    else: 
        # returns placeholders that are initialised when the session is initiated
        return tf.constant(value, tf.float32, shape)


def get_spatial_rank(x):
    """
    This functions calculates the spatial rank of a tensor
    ---
    :param: x: an input tensor with shape [batch_size, ..., num_channels]
    ---- 
    :return: the spatial rank of the tensor 
    (i.e. the number of spatial dimensions between batch_size and num_channels)
    """
    # get_shape returns [D0, D1, ... Dn-1] for a tensor of rank n
    return len(x.get_shape()) - 2 


def get_num_channels(x):
    """
    This functions calculates the number of channels of a tensor
    ---
    :param: x: an input tensor with shape [batch_size, ..., num_channels]
    ---- 
    :return: the number of channels of x
    """
    return int(x.get_shape()[-1])


def get_spatial_size(x):
    """
    This functions calculates the number of channels of a tensor
    ---
    :param: x: an input tensor with shape [batch_size, ..., num_channels]
    ---- 
    :return: the spatial shape of x [...]
    """
    return x.get_shape()[1:-1]


# parametric leaky relu
def prelu(x):
    """
    This function implements the parametric ReLU
    ---
    :param: x: an input tensor 
    ---- 
    :return: the tensor after passing through the non-linearity
    """
    alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype, initializer=tf.constant_initializer(0.1)) 
    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)


def convolution(x, filter_shape, padding='SAME', strides=None, dilation_rate=None):
    """
    This function operates the convolution of two tensors
    ---
    :param: x: an (N+2)-D tensor of shape [ [batch_size] + input_spatial_shape + [in_channels]]
    :param: filter_shape: [spatial_filter_shape + [in_channels, out_channels]]
    :param: padding: wheter or not x has to be padded
    :param: strides: wheter or not strides have to be used 
    :param: dilatation_rate
    ---- 
    :return: the tensor that results from the convolution
    """
    w = tf.get_variable(name='weights', initializer=xavier_initializer_convolution(shape=filter_shape))
    b = tf.get_variable(name='biases', initializer=constant_initializer(0, shape=filter_shape[-1]))
    
    return tf.nn.convolution(x, w, padding, strides, dilation_rate) + b


def deconvolution(x, filter_shape, output_shape, strides, padding='SAME'):
    """
    This function operates the deconvolution of two tensors
    It reverses the effects of convolution on recorded data
    ---
    :param: x: an (N+2)-D tensor of shape [ [batch_size] + input_spatial_shape + [in_channels]]
    :param: filter_shape: a tuple giving [filter_spatial_shape + output_channels, in_channels]
    :param: output_shape: a 1-D Tensor representing the output shape of the deconvolution op
    :param: padding: wheter or not x has to be padded
    :param: strides: wheter or not strides have to be used 
    ---- 
    :return: the tensor that results from the deconvolution
    """
    w = tf.get_variable(name='weights', initializer=xavier_initializer_convolution(shape=filter_shape))
    b = tf.get_variable(name='biases', initializer=constant_initializer(0, shape=filter_shape[-2]))

    spatial_rank = get_spatial_rank(x)

    if spatial_rank == 2:
        return tf.nn.conv2d_transpose(x, w, output_shape, strides, padding) + b
    if spatial_rank == 3:
        return tf.nn.conv3d_transpose(x, w, output_shape, strides, padding) + b
    raise ValueError('Only 2D and 3D images supported.')


# More complex blocks

# down convolution
def down_convolution(x, factor, kernel_size):
    """
    This function operates the downconvolution of x
    ---
    :param: x: an (N+2)-D tensor of shape [ [batch_size] + input_spatial_shape + [in_channels]]
    :param: factor: an int saying by how much you want to divide the dim of the input tensor x,
    and by how much you want to multiply the feature channels
    :param: kernel_size: a 1-D Tensor specifying the spatial shape of the kernel
    ---- 
    :return: the tensor that results from the down convolution
    """
    in_channels = get_num_channels(x) 
    spatial_rank = get_spatial_rank(x)
    strides = spatial_rank * [factor]
    out_channels = in_channels * factor

    # compute the shape of the filter: kernel_size + [in_channels, out_channels]
    filter_shape = kernel_size + [in_channels, out_channels] 

    # compute the convolution
    x = convolution(x, filter_shape, strides=strides) 
    return x


# up convolution
def up_convolution(x, output_shape, factor, kernel_size):
    """
    This function operates the upconvolution of x
    ---
    :param: x: an (N+2)-D tensor of shape [ [batch_size] + input_spatial_shape + [in_channels]]
    :param: output_shape: a 1-D Tensor representing the output shape of the up convolution
    :param: factor: an int saying by how much you want to multiply the dim of the input tensor x,
    and by how much you want to divide the feature channels
    :param: kernel_size: a 1-D Tensor specifying the spatial shape of the kernel
    ---- 
    :return: the tensor that results from the up-convolution, of shape output_shape
    """
    in_channels = get_num_channels(x)
    spatial_rank = get_spatial_rank(x)
    strides = [1] + spatial_rank * [factor] + [1] #[1, factor, factor, factor, 1]
    out_channels = in_channels // factor
    
    # compute the shape of the filter: kernel_size + [out_channels, in_channels]
    filter_shape = kernel_size + [out_channels, in_channels]

    # compute the deconvolution
    x = deconvolution(x, filter_shape, output_shape, strides=strides)
    return x