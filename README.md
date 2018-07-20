# Face Recognition using Tensorflow [![Build Status][travis-image]][travis]

[travis-image]: http://travis-ci.org/davidsandberg/facenet.svg?branch=master
[travis]: http://travis-ci.org/davidsandberg/facenet

This is a TensorFlow implementation of the face recognizer described in the paper
["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832). The project also uses ideas from the paper ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) from the [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) at Oxford.

## Compatibility
The code is tested using Tensorflow r1.7 under Ubuntu 14.04 with Python 2.7 and Python 3.5. The test cases can be found [here](https://github.com/davidsandberg/facenet/tree/master/test) and the results can be found [here](http://travis-ci.org/davidsandberg/facenet).

## News
| Date     | Update |
|----------|--------|
| 2018-04-10 | Added new models trained on Casia-WebFace and VGGFace2 (see below). Note that the models uses fixed image standardization (see [wiki](https://github.com/davidsandberg/facenet/wiki/Training-using-the-VGGFace2-dataset)). |
| 2018-03-31 | Added a new, more flexible input pipeline as well as a bunch of minor updates. |
| 2017-05-13 | Removed a bunch of older non-slim models. Moved the last bottleneck layer into the respective models. Corrected normalization of Center Loss. |
| 2017-05-06 | Added code to [train a classifier on your own images](https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images). Renamed facenet_train.py to train_tripletloss.py and facenet_train_classifier.py to train_softmax.py. |
| 2017-03-02 | Added pretrained models that generate 128-dimensional embeddings.|
| 2017-02-22 | Updated to Tensorflow r1.0. Added Continuous Integration using Travis-CI.|
| 2017-02-03 | Added models where only trainable variables has been stored in the checkpoint. These are therefore significantly smaller. |
| 2017-01-27 | Added a model trained on a subset of the MS-Celeb-1M dataset. The LFW accuracy of this model is around 0.994. |
| 2017&#8209;01&#8209;02 | Updated to run with Tensorflow r0.12. Not sure if it runs with older versions of Tensorflow though.   |

## Pre-trained models
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [20180408-102900](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) | 0.9905        | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

NOTE: If you use any of the models, please do not forget to give proper credit to those providing the training dataset as well.

## Inspiration
The code is heavily inspired by the [OpenFace](https://github.com/cmusatyalab/openface) implementation.

## Training data
The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training. This training set consists of total of 453 453 images over 10 575 identities after face detection. Some performance improvement has been seen if the dataset has been filtered before training. Some more information about how this was done will come later.
The best performing model has been trained on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset consisting of ~3.3M faces and ~9000 classes.

## Pre-processing

### Face alignment using MTCNN
One problem with the above approach seems to be that the Dlib face detector misses some of the hard examples (partial occlusion, silhouettes, etc). This makes the training set too "easy" which causes the model to perform worse on other benchmarks.
To solve this, other face landmark detectors has been tested. One face landmark detector that has proven to work very well in this setting is the
[Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). A Matlab/Caffe implementation can be found [here](https://github.com/kpzhang93/MTCNN_face_detection_alignment) and this has been used for face alignment with very good results. A Python/Tensorflow implementation of MTCNN can be found [here](https://github.com/davidsandberg/facenet/tree/master/src/align). This implementation does not give identical results to the Matlab/Caffe implementation but the performance is very similar.

## Running training
Currently, the best results are achieved by training the model using softmax loss. Details on how to train a model using softmax loss on the CASIA-WebFace dataset can be found on the page [Classifier training of Inception-ResNet-v1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1) and .

## Pre-trained models
### Inception-ResNet-v1 model
A couple of pretrained models are provided. They are trained using softmax loss with the Inception-Resnet-v1 model. The datasets has been aligned using [MTCNN](https://github.com/davidsandberg/facenet/tree/master/src/align).

## Performance

The accuracy on LFW for the model [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) is 0.99650+-0.00252. A description of how to run the test can be found on the page [Validate on LFW](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw). Note that the input images to the model need to be standardized using fixed image standardization (use the option `--use_fixed_image_standardization` when running e.g. `validate_on_lfw.py`).

### References

* [VAL, FAR and Accuracy](https://github.com/davidsandberg/facenet/issues/288)
* [NMS——非极大值抑制](http://blog.csdn.net/shuzfan/article/details/52711706)
* [MTCNN（Multi-task convolutional neural networks）人脸对齐](http://blog.csdn.net/qq_14845119/article/details/52680940)

# Note


### MTCNN

```shell
python src/align/align_dataset_mtcnn.py \
    /datasets/ceibs/training-images-gen \
    /datasets/ceibs/mtcnn160 \
    --image_size 160 \
    --margin 32 \
    --random_order \
    --gpu_memory_fraction 0.2 \
```

### Classifier

```shell
python src/classifier.py TRAIN \
    /datasets/ceibs/mtcnn160/ \
    ./models/facenet/20170512-110547/20170512-110547.pb \
    ./models/ceibs_classifier.pkl \
    --test_data_dir=/datasets/ceibs/mtcnn160/ \
    --batch_size 1 \
    \
    --use_split_dataset=0 \
    --min_nrof_images_per_class=1 \
    --nrof_train_images_per_class=1 \
    ;
```

```shell
python src/classifier.py CLASSIFY \
    /datasets/ceibs/mtcnn160/ \
    ./models/facenet/20170512-110547/20170512-110547.pb \
    ./models/ceibs_classifier.pkl \
    --batch_size 1 \
    --image_size=160 \
    \
    --use_split_dataset=0 \
    --min_nrof_images_per_class=1 \
    --nrof_train_images_per_class=1 \
    ;
```

### Train Triple Loss

```shell
python src/train_tripletloss.py --logs_base_dir /facenet/logs/facenet/ --models_base_dir /facenet/models/facenet/ --data_dir /facenet/datasets/lfw/lfw_maxpy_mtcnnpy_182/ --image_size 160 --model_def models.inception_resnet_v1 --lfw_dir /facenet/datasets/lfw/lfw_mtcnnpy_160/ --optimizer RMSPROP --learning_rate 0.01 --weight_decay 1e-4 --max_nrof_epochs 500 --gpu_memory_fraction 0.5
```

### Train Soft Max

```shell
python src/train_softmax.py --logs_base_dir /facenet/logs/facenet/ --models_base_dir /facenet/models/facenet/20170512-110547/ --data_dir /facenet/datasets/lfw/lfw_maxpy_mtcnnpy_182/ --image_size 160 --model_def models.inception_resnet_v1 --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 80 --keep_probability 0.8 --random_crop --random_flip --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9
```


### Validate LFW

```shell
python3 src/validate_on_lfw.py \
    /datasets/lfw/lfw_mtcnnpy_160 \
    ./models/facenet/20170512-110547
```
