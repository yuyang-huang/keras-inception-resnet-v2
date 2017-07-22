# keras-inception-resnet-v2
The Inception-ResNet v2 model using Keras.

Layers and namings basically follows the TF-slim implementation.
https://github.com/tensorflow/models/blob/master/slim/nets/inception_resnet_v2.py

## Extract layer weights from TF checkpoint
```
python extract_weights.py
```
By default, the TF checkpoint file will be downloaded to `./models` folder, and the layer weights (`.npy` files) will be saved to `./weights` folder.

## Load NumPy weight files and save to a Keras HDF5 weights file
```
python load_weights.py
```
The following weight files:
- models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5
- models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5

will be generated.

## Test model prediction
To test whether this implementation gives the same prediction as TF-slim implementation:
```
PYTHONPATH=../tensorflow-models/slim python test_inception_resnet_v2.py
```
`PYTHONPATH` should point to the `slim` folder under the https://github.com/tensorflow/models repo.

## Current status
- [X] Extract weights from TF-slim
- [X] Convert weights to HDF5 files
- [X] Test weight loading and image prediction (`elephant.jpg`)
- [ ] Release weight files
