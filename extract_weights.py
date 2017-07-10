from __future__ import absolute_import
from __future__ import print_function

import os
import re
from glob import glob
import numpy as np
import tensorflow as tf
from keras.utils.data_utils import get_file


CKPT_URL = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'

checkpoint_tar = get_file(
    'inception_resnet_v2_2016_08_30.tar.gz',
    CKPT_URL,
    file_hash='9e0f18e1259acf943e30690460d96123',
    hash_algorithm='md5',
    extract=True,
    cache_subdir='models')
checkpoint_file = glob(os.path.join(os.path.dirname(checkpoint_tar), 'inception_resnet_v2_*.ckpt'))[0]

re_repeat = re.compile(r'Repeat_[0-9_]*b')
re_block8 = re.compile(r'Block8_[A-Za-z]')

def get_filename(key):
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('InceptionResnetV2_', '')
    filename = re_repeat.sub('B', filename)
    if re_block8.match(filename):
        filename = filename.replace('Block8', 'Block8_10')
    elif filename.startswith('Logits'):
        filename = filename.replace('Logits_' , '', 1)
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')
    return filename + '.npy'


def print_tensors_in_checkpoint_file(file_name, output_folder='weights'):
    """Prints tensors in a checkpoint file."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reader = tf.train.NewCheckpointReader(file_name)

    for key in reader.get_variable_to_shape_map():
        if key == 'global_step':
            continue
        if 'AuxLogit' in key:
            continue

        path = os.path.join(output_folder, get_filename(key))
        arr = reader.get_tensor(key)
        np.save(path, arr)
        print("tensor_name: ", key)

print_tensors_in_checkpoint_file(checkpoint_file)
