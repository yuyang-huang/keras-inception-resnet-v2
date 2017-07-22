import os
import re
import tempfile
from glob import glob
import numpy as np
import tensorflow as tf
from keras.utils.data_utils import get_file


# regex for renaming the tensors to their corresponding Keras counterpart
re_repeat = re.compile(r'Repeat_[0-9_]*b')
re_block8 = re.compile(r'Block8_[A-Za-z]')


def get_filename(key):
    """Rename tensor name to the corresponding Keras layer weight name.

    # Arguments
        key: tensor name in TF (determined by tf.variable_scope)
    """
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('InceptionResnetV2_', '')

    # remove "Repeat" scope from filename
    filename = re_repeat.sub('B', filename)

    if re_block8.match(filename):
        # the last block8 has different name with the previous 9 occurrences
        filename = filename.replace('Block8', 'Block8_10')
    elif filename.startswith('Logits'):
        # remove duplicate "Logits" scope
        filename = filename.replace('Logits_' , '', 1)

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def extract_tensors_from_checkpoint_file(filename, output_folder='weights'):
    """Extract tensors from a TF checkpoint file.

    # Arguments
        filename: TF checkpoint file
        output_folder: where to save the output numpy array files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reader = tf.train.NewCheckpointReader(filename)

    for key in reader.get_variable_to_shape_map():
        # not saving the following tensors
        if key == 'global_step':
            continue
        if 'AuxLogit' in key:
            continue

        # convert tensor name into the corresponding Keras layer weight name and save
        path = os.path.join(output_folder, get_filename(key))
        arr = reader.get_tensor(key)
        np.save(path, arr)
        print("tensor_name: ", key)


# download TF-slim checkpoint for Inception-ResNet v2 and extract
CKPT_URL = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
MODEL_DIR = './models'

checkpoint_tar = get_file(
    'inception_resnet_v2_2016_08_30.tar.gz',
    CKPT_URL,
    file_hash='9e0f18e1259acf943e30690460d96123',
    hash_algorithm='md5',
    extract=True,
    cache_subdir='',
    cache_dir=MODEL_DIR)

checkpoint_file = glob(os.path.join(MODEL_DIR, 'inception_resnet_v2_*.ckpt'))[0]
extract_tensors_from_checkpoint_file(checkpoint_file)
