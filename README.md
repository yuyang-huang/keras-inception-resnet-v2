# keras-inception-resnet-v2
The Inception-ResNet v2 model using Keras (with weight files)

Tested with `tensorflow-gpu==1.2.1` and `Keras==2.0.6`

Layers and namings follows the TF-slim implementation:
https://github.com/tensorflow/models/blob/master/slim/nets/inception_resnet_v2.py


## Usage
Basically the same with the `keras.applications.InceptionV3` model.
```
from inception_resnet_v2 import InceptionResNetV2

# ImageNet classification
model = InceptionResNetV2()
model.predict(...)

# Finetuning on another 100-class dataset
base_model = InceptionResNetV2(include_top=False, pooling='avg')
outputs = Dense(100, activation='softmax')(base_model.output)
model = Model(base_model.inputs, outputs)
model.compile(...)
model.fit(...)
```


### Extract layer weights from TF checkpoint
```
python extract_weights.py
```
By default, the TF checkpoint file will be downloaded to `./models` folder, and the layer weights (`.npy` files) will be saved to `./weights` folder.


### Load NumPy weight files and save to a Keras HDF5 weights file
```
python load_weights.py
```
The following weight files:
- models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5
- models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5

will be generated.


### Test model prediction on single image
To test whether this implementation gives the same prediction as TF-slim implementation:
```
PYTHONPATH=../tensorflow-models/slim python test_inception_resnet_v2.py
```
`PYTHONPATH` should point to the `slim` folder under the https://github.com/tensorflow/models repo.

The image file `elephant.jpg` (and basically the entire idea of converting weights from TF-slim to Keras) comes from:
https://github.com/kentsommer/keras-inception-resnetV2


## Current status
- [X] Extract weights from TF-slim
- [X] Convert weights to HDF5 files
- [X] Test weight loading and image prediction (`elephant.jpg`)
- [X] Release weight files
- [ ] Test accuracy on benchmark datasets
