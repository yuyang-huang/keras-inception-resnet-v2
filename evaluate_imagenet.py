from __future__ import division
from __future__ import print_function

import argparse
import math
import tensorflow as tf; slim = tf.contrib.slim
from keras import backend as K

# PYHTONPATH should contain the slim/ directory in the tensorflow/models repo.
from datasets import dataset_factory
from preprocessing import preprocessing_factory
from inception_resnet_v2 import InceptionResNetV2


def prepare_data(imagenet_dir, batch_size, num_threads):
    # setup image loading
    dataset = dataset_factory.get_dataset('imagenet', 'validation', imagenet_dir)
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              shuffle=False,
                                                              common_queue_capacity=batch_size * 5,
                                                              common_queue_min=batch_size)
    image, label = provider.get(['image', 'label'])

    # preprocess images and split into batches
    preprocess_input = preprocessing_factory.get_preprocessing('inception_resnet_v2',
                                                               is_training=False)
    image = preprocess_input(image, 299, 299)
    images, labels = tf.train.batch([image, label],
                                    batch_size=batch_size,
                                    num_threads=num_threads,
                                    capacity=batch_size * 5)

    # Keras label is different from TF
    labels = labels - 1  # remove the "background class"
    labels = K.cast(K.expand_dims(labels, -1), K.floatx())  # Keras labels are 2D float tensors
    return images, labels, dataset.num_samples


def evaluate(imagenet_dir, batch_size=100, steps=None, num_threads=4, verbose=False):
    with K.get_session().as_default():
        # setup data tensors
        images, labels, num_samples = prepare_data(imagenet_dir, batch_size, num_threads)
        tf.train.start_queue_runners(coord=tf.train.Coordinator())

        # compile model in order to provide `metrics` and `target_tensors`
        model = InceptionResNetV2(input_tensor=images)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy'],
                      target_tensors=[labels])

        # start evaluation
        if steps is None:
            steps = int(math.ceil(num_samples / batch_size))
        _, acc1, acc5 = model.evaluate(x=None, y=None, steps=steps, verbose=int(verbose))
        print()
        print('Top-1 Accuracy {:.1%}'.format(acc1))
        print('Top-5 Accuracy {:.1%}'.format(acc5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("imagenet_dir", type=str, help="where ImageNet data is located (i.e. the output of `download_and_convert_imagenet.sh` from TF-slim)")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size when evaluating, set this number according to your GPU memory")
    parser.add_argument("--steps", type=int, default=None, help="maximum number of batches to evaluate, if not specified, will go through the entire validation set by default")
    parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use for data loading, default 4")
    parser.add_argument("--verbose", action='store_true', help="if specified, print the progress bar")
    args = parser.parse_args()
    evaluate(**vars(args))
