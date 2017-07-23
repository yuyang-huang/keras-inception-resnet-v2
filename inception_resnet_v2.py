# -*- coding: utf-8 -*-
"""Inception-Resnet V2 model for Keras.

This code is modified from Keras' Inception V3 model (https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py).

Model layer naming and parameters follows TF-slim implementation (https://github.com/tensorflow/models/blob/master/slim/nets/inception_resnet_v2.py).

Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).

# Reference

- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape


WEIGHTS_PATH = 'https://github.com/myutwo150/keras-inception-resnet-v2/releases/download/v0.1/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/myutwo150/keras-inception-resnet-v2/releases/download/v0.1/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_Activation'`
            for the activation and `name + '_BatchNorm'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)

    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = name + '_BatchNorm' if name else None
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    if activation:
        ac_name = name + '_Activation' if name else None
        x = Activation(activation, name=ac_name)(x)

    return x


def block35(x, scale=0.17, activation='relu', name=None):
    name = name or 'Block35'
    name_fmt = name + '_Branch_{}_{}'

    branch_idx = 0
    tower_conv = conv2d_bn(x, 32, 1, name=name_fmt.format(branch_idx, 'Conv2d_1x1'))

    branch_idx = 1
    tower_conv1_0 = conv2d_bn(x, 32, 1, name=name_fmt.format(branch_idx, 'Conv2d_0a_1x1'))
    tower_conv1_1 = conv2d_bn(tower_conv1_0, 32, 3, name=name_fmt.format(branch_idx, 'Conv2d_0b_3x3'))

    branch_idx = 2
    tower_conv2_0 = conv2d_bn(x, 32, 1, name=name_fmt.format(branch_idx, 'Conv2d_0a_1x1'))
    tower_conv2_1 = conv2d_bn(tower_conv2_0, 48, 3, name=name_fmt.format(branch_idx, 'Conv2d_0b_3x3'))
    tower_conv2_2 = conv2d_bn(tower_conv2_1, 64, 3, name=name_fmt.format(branch_idx, 'Conv2d_0c_3x3'))

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    mixed = layers.concatenate([tower_conv, tower_conv1_1, tower_conv2_2],
                               axis=channel_axis,
                               name=name + '_Concatenate')
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=name + '_Conv2d_1x1')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=name + '_ScaleSum')([x, up])

    if activation:
        ac_name = name + '_Activation'
        x = Activation(activation, name=ac_name)(x)

    return x


def block17(x, scale=0.10, activation='relu', name=None):
    name = name or 'Block17'
    name_fmt = name + '_Branch_{}_{}'

    branch_idx = 0
    tower_conv = conv2d_bn(x, 192, 1, name=name_fmt.format(branch_idx, 'Conv2d_1x1'))

    branch_idx = 1
    tower_conv1_0 = conv2d_bn(x, 128, 1, name=name_fmt.format(branch_idx, 'Conv2d_0a_1x1'))
    tower_conv1_1 = conv2d_bn(tower_conv1_0, 160, [1, 7], name=name_fmt.format(branch_idx, 'Conv2d_0b_1x7'))
    tower_conv1_2 = conv2d_bn(tower_conv1_1, 192, [7, 1], name=name_fmt.format(branch_idx, 'Conv2d_0c_7x1'))

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    mixed = layers.concatenate([tower_conv, tower_conv1_2],
                               axis=channel_axis,
                               name=name + '_Concatenate')
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=name + '_Conv2d_1x1')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=name + '_ScaleSum')([x, up])

    if activation:
        ac_name = name + '_Activation'
        x = Activation(activation, name=ac_name)(x)

    return x


def block8(x, scale=0.20, activation='relu', name=None):
    name = name or 'Block8'
    name_fmt = name + '_Branch_{}_{}'

    branch_idx = 0
    tower_conv = conv2d_bn(x, 192, 1, name=name_fmt.format(branch_idx, 'Conv2d_1x1'))

    branch_idx = 1
    tower_conv1_0 = conv2d_bn(x, 192, 1, name=name_fmt.format(branch_idx, 'Conv2d_0a_1x1'))
    tower_conv1_1 = conv2d_bn(tower_conv1_0, 224, [1, 3], name=name_fmt.format(branch_idx, 'Conv2d_0b_1x3'))
    tower_conv1_2 = conv2d_bn(tower_conv1_1, 256, [3, 1], name=name_fmt.format(branch_idx, 'Conv2d_0c_3x1'))

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    mixed = layers.concatenate([tower_conv, tower_conv1_2],
                               axis=channel_axis,
                               name=name + '_Concatenate')
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=name + '_Conv2d_1x1')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=name + '_ScaleSum')([x, up])

    if activation:
        ac_name = name + '_Activation'
        x = Activation(activation, name=ac_name)(x)

    return x


def InceptionResNetV2(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                dropout_keep_prob=0.8):
    """Instantiates the Inception-ResNet v2 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input image size for this model is 299x299.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.
        dropout_keep_prob: dropout keep rate after pooling and before the
            classification layer, only to be specified if `include_top` is `True`.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    # stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid', name='Conv2d_1a_3x3')
    x = conv2d_bn(x, 32, 3, padding='valid', name='Conv2d_2a_3x3')
    x = conv2d_bn(x, 64, 3, name='Conv2d_2b_3x3')
    x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)

    x = conv2d_bn(x, 80, 1, padding='valid', name='Conv2d_3b_1x1')
    x = conv2d_bn(x, 192, 3, padding='valid', name='Conv2d_4a_3x3')
    x = MaxPooling2D(3, strides=2, name='MaxPool_5a_3x3')(x)

    # mixed 5b (Inception-A block): 35 x 35 x 320
    name_fmt = 'Mixed_5b_Branch_{}_{}'

    branch_idx = 0
    tower_conv = conv2d_bn(x, 96, 1, name=name_fmt.format(branch_idx, 'Conv2d_1x1'))

    branch_idx = 1
    tower_conv1_0 = conv2d_bn(x, 48, 1, name=name_fmt.format(branch_idx, 'Conv2d_0a_1x1'))
    tower_conv1_1 = conv2d_bn(tower_conv1_0, 64, 5, name=name_fmt.format(branch_idx, 'Conv2d_0b_5x5'))

    branch_idx = 2
    tower_conv2_0 = conv2d_bn(x, 64, 1, name=name_fmt.format(branch_idx, 'Conv2d_0a_1x1'))
    tower_conv2_1 = conv2d_bn(tower_conv2_0, 96, 3, name=name_fmt.format(branch_idx, 'Conv2d_0b_3x3'))
    tower_conv2_2 = conv2d_bn(tower_conv2_1, 96, 3, name=name_fmt.format(branch_idx, 'Conv2d_0c_3x3'))

    branch_idx = 3
    tower_pool = AveragePooling2D(3, strides=1, padding='same', name=name_fmt.format(branch_idx, 'AvgPool_0a_3x3'))(x)
    tower_pool_1 = conv2d_bn(tower_pool, 64, 1, name=name_fmt.format(branch_idx, 'Conv2d_0b_1x1'))

    x = layers.concatenate([tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1],
                           axis=channel_axis,
                           name='Mixed_5b')

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for r in range(1, 11):
        x = block35(x, name='Block35_{}'.format(r))

    # mixed 6a (Reduction-A block): 17 x 17 x 1088
    name_fmt = 'Mixed_6a_Branch_{}_{}'

    branch_idx = 0
    tower_conv = conv2d_bn(x, 384, 3, strides=2, padding='valid', name=name_fmt.format(branch_idx, 'Conv2d_1a_3x3'))

    branch_idx = 1
    tower_conv1_0 = conv2d_bn(x, 256, 1, name=name_fmt.format(branch_idx, 'Conv2d_0a_1x1'))
    tower_conv1_1 = conv2d_bn(tower_conv1_0, 256, 3, name=name_fmt.format(branch_idx, 'Conv2d_0b_3x3'))
    tower_conv1_2 = conv2d_bn(tower_conv1_1, 384, 3, strides=2, padding='valid', name=name_fmt.format(branch_idx, 'Conv2d_1a_3x3'))

    branch_idx = 2
    tower_pool = MaxPooling2D(3, strides=2, padding='valid', name=name_fmt.format(branch_idx, 'MaxPool_1a_3x3'))(x)

    x = layers.concatenate([tower_conv, tower_conv1_2, tower_pool],
                           axis=channel_axis,
                           name='Mixed_6a')

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for r in range(1, 21):
        x = block17(x, name='Block17_{}'.format(r))

    # mixed 7a (Reduction-B block): 8 x 8 x 2080
    name_fmt = 'Mixed_7a_Branch_{}_{}'

    branch_idx = 0
    tower_conv = conv2d_bn(x, 256, 1, name=name_fmt.format(branch_idx, 'Conv2d_0a_1x1'))
    tower_conv_1 = conv2d_bn(tower_conv, 384, 3, strides=2, padding='valid', name=name_fmt.format(branch_idx, 'Conv2d_1a_3x3'))

    branch_idx = 1
    tower_conv1 = conv2d_bn(x, 256, 1, name=name_fmt.format(branch_idx, 'Conv2d_0a_1x1'))
    tower_conv1_1 = conv2d_bn(tower_conv1, 288, 3, strides=2, padding='valid', name=name_fmt.format(branch_idx, 'Conv2d_1a_3x3'))

    branch_idx = 2
    tower_conv2 = conv2d_bn(x, 256, 1, name=name_fmt.format(branch_idx, 'Conv2d_0a_1x1'))
    tower_conv2_1 = conv2d_bn(tower_conv2, 288, 3, name=name_fmt.format(branch_idx, 'Conv2d_0b_3x3'))
    tower_conv2_2 = conv2d_bn(tower_conv2_1, 320, 3, strides=2, padding='valid', name=name_fmt.format(branch_idx, 'Conv2d_1a_3x3'))

    branch_idx = 3
    tower_pool = MaxPooling2D(3, strides=2, padding='valid', name=name_fmt.format(branch_idx, 'MaxPool_1a_3x3'))(x)

    x = layers.concatenate([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool],
                           axis=channel_axis,
                           name='Mixed_7a')

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for r in range(1, 10):
        x = block8(x, name='Block8_{}'.format(r))
    x = block8(x, scale=1.0, activation=None, name='Block8_10')

    # Final convolution block
    x = conv2d_bn(x, 1536, 1, name='Conv2d_7b_1x1')

    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='AvgPool_1a_8x8')(x)
        x = Dropout(1.0 - dropout_keep_prob, name='Dropout')(x)
        x = Dense(classes, name='Logits')(x)
        x = Activation('softmax', name='Predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='AvgPool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='MaxPool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='inception_resnet_v2')

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_path = get_file(
                'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='e693bd0210a403b3192acc6073ad2e96')
        else:
            weights_path = get_file(
                'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='d19885ff4a710c122648d3b5c3b684e4')
        model.load_weights(weights_path)

    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
