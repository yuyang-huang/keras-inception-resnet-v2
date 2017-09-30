# keras-inception-resnet-v2
The Inception-ResNet v2 model using Keras (with weight files)

Tested with `tensorflow-gpu==1.3.0` and `Keras==2.0.8` under Python 2.7 and 3.6.

Layers and namings follow the TF-slim implementation:
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py


## News

This implementation has been merged into the `keras.applications` module!

Install the latest version Keras on GitHub and import it with:
```python
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
```


## Usage
Basically the same with the `keras.applications.InceptionV3` model.
```python
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
PYTHONPATH=../tensorflow-models/research/slim python test_inception_resnet_v2.py
```
`PYTHONPATH` should point to the `research/slim` folder under the https://github.com/tensorflow/models repo.

The image file `elephant.jpg` (and basically the entire idea of converting weights from TF-slim to Keras) comes from:
https://github.com/kentsommer/keras-inception-resnetV2


### Evaluate the model on ImageNet 2012 dataset
First, follow the
[instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
from TF-slim to download and process the data.

Suppose that the dataset is saved to the `imagenet_2012` directory, to evaluate:
```
PYTHONPATH=../tensorflow-models/research/slim python evaluate_imagenet.py ../tensorflow-models/research/slim/datasets/imagenet_2012 --verbose
```

The script should print out top-1 and top-5 accuracy on validation set:

Implementation | Top-1 Accuracy | Top-5 Accuracy
--- | --- | ---
[TF-slim](https://github.com/tensorflow/models/tree/master/research/slim) | 80.4 | 95.3
This repo (py27) | 80.4 | 95.2
This repo (py36) | 80.4 | 95.2


## Current status
- [X] Extract weights from TF-slim
- [X] Convert weights to HDF5 files
- [X] Test weight loading and image prediction (`elephant.jpg`)
- [X] Release weight files
- [X] Evaluate accuracy on ImageNet benchmark dataset
