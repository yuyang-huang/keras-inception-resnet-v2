import os
import tensorflow as tf; slim = tf.contrib.slim
import numpy as np
from PIL import Image
from keras.models import load_model
from nets import inception_resnet_v2 as slim_irv2  # PYHTONPATH should contain the slim/ directory in the tensorflow/models repo.
import inception_resnet_v2 as keras_irv2


IMAGES = ['elephant.jpg']
MODEL_DIR = './models'
SLIM_CKPT = os.path.join(MODEL_DIR, 'inception_resnet_v2_2016_08_30.ckpt')
KERAS_CKPT = os.path.join(MODEL_DIR, 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
ATOL = 1e-5
VERBOSE = True


def predict_slim(sample_images, print_func=print):
    """
    Code modified from here: [https://github.com/tensorflow/models/issues/429]
    """
    # Setup preprocessing
    input_tensor = tf.placeholder(tf.float32, shape=(None, 299, 299, 3), name='input_image')
    scaled_input_tensor = tf.scalar_mul((1.0 / 255), input_tensor)
    scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
    scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)

    # Setup session
    sess = tf.Session()
    arg_scope = slim_irv2.inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
        _, end_points = slim_irv2.inception_resnet_v2(scaled_input_tensor, is_training=False)

    # Load the model
    print_func("Loading TF-slim checkpoint...")
    saver = tf.train.Saver()
    saver.restore(sess, SLIM_CKPT)

    # Make prediction
    predict_values = []
    for image in sample_images:
        im = Image.open(image).resize((299, 299))
        arr = np.expand_dims(np.array(im), axis=0)
        y_pred = sess.run([end_points['Predictions']], feed_dict={input_tensor: arr})
        y_pred = y_pred[0].ravel()

        y_pred = y_pred[1:] / y_pred[1:].sum()  # remove background class and renormalize
        print_func("{} class={} prob={}".format(image, np.argmax(y_pred), np.max(y_pred)))
        predict_values.append(y_pred)

    return predict_values


def predict_keras(sample_images, print_func=print):
    # Load the model
    print_func("Loading Keras checkpoint...")
    model = keras_irv2.InceptionResNetV2(weights=None)
    model.load_weights(KERAS_CKPT)

    # Make prediction
    predict_values = []
    for image in sample_images:
        im = Image.open(image).resize((299, 299))
        arr = np.expand_dims(np.array(im), axis=0)
        y_pred = model.predict(keras_irv2.preprocess_input(arr.astype('float32')))
        y_pred = y_pred.ravel()
        print_func("{} class={} prob={}".format(image, np.argmax(y_pred), np.max(y_pred)))
        predict_values.append(y_pred)

    return predict_values


# test whether Keras implementation gives the same result as TF-slim implementation
verboseprint = print if VERBOSE else lambda *a, **k: None
slim_predictions = predict_slim(IMAGES, verboseprint)
keras_predictions = predict_keras(IMAGES, verboseprint)

for filename, y_slim, y_keras in zip(IMAGES, slim_predictions, keras_predictions):
    np.testing.assert_allclose(y_slim, y_keras, atol=ATOL, err_msg=filename)
    verboseprint('{} passed test. (tolerance={})'.format(filename, ATOL))
