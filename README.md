# Face Recognition using Tensorflow
This is a TensorFlow implementation of the face recognizer described in the paper
["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832). The project also uses ideas from the paper ["A Discriminative Feature Learning Approach for Deep Face Recognition"](http://ydwen.github.io/papers/WenECCV16.pdf) as well as the paper ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) from the [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) at Oxford.

## Tensorflow release
Currently this repo is compatible with Tensorflow r1.0.

## News
| Date     | Update |
|----------|--------|
| 2017-05-13 | Removed a bunch of older non-slim models. Moved the last bottleneck layer into the respective models. Corrected normalization of Center Loss. |
| 2017-05-06 | Added code to [train a classifier on your own images](https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images). Renamed facenet_train.py to train_tripletloss.py and facenet_train_classifier.py to train_softmax.py. |
| 2017-03-02 | Added pretrained models that generate 128-dimensional embeddings.|
| 2017-02-22 | Updated to Tensorflow r1.0. Added Continuous Integration using Travis-CI.|
| 2017-02-03 | Added models where only trainable variables has been stored in the checkpoint. These are therefore significantly smaller. |
| 2017-01-27 | Added a model trained on a subset of the MS-Celeb-1M dataset. The LFW accuracy of this model is around 0.994. |
| 2017&#8209;01&#8209;02 | Updated to code to run with Tensorflow r0.12. Not sure if it runs with older versions of Tensorflow though.   |

## Pre-trained models
| Model name      | LFW accuracy | Training dataset | Architcture |
|-----------------|--------------|------------------|-------------|
| [20170511-185253](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE) | 0.987        | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk) | 0.992        | MS-Celeb-1M      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

## Inspiration
The code is heavly inspired by the [OpenFace](https://github.com/cmusatyalab/openface) implementation.

## Training data
The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training. This training set consists of total of 453 453 images over 10 575 identities after face detection. Some performance improvement has been seen if the dataset has been filtered before training. Some more information about how this was done will come later.
The best performing model has been trained on a subset of the [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/) dataset. This dataset is significantly larger but also contains significantly more label noise, and therefore it is crucial to apply dataset filtering on this dataset.

## Pre-processing

### Face alignment using MTCNN
One problem with the above approach seems to be that the Dlib face detector misses some of the hard examples (partial occlusion, siluettes, etc). This makes the training set to "easy" which causes the model to perform worse on other benchmarks.
To solve this, other face landmark detectors has been tested. One face landmark detector that has proven to work very well in this setting is the
[Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). A Matlab/Caffe implementation can be found [here](https://github.com/kpzhang93/MTCNN_face_detection_alignment) and this has been used for face alignment with very good results. A Python/Tensorflow implementation of MTCNN can be found [here](https://github.com/davidsandberg/facenet/tree/master/src/align). This implementation does not give identical results to the Matlab/Caffe implementation but the performance is very similar.

## Running training
Currently, the best results are achieved by training the model as a classifier with the addition of [Center loss](http://ydwen.github.io/papers/WenECCV16.pdf). Details on how to train a model as a classifier can be found on the page [Classifier training of Inception-ResNet-v1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1).

## Pre-trained model
### Inception-ResNet-v1 model
Currently, the best performing model is an Inception-Resnet-v1 model trained on CASIA-Webface aligned with [MTCNN](https://github.com/davidsandberg/facenet/tree/master/src/align).

## Performance
The accuracy on LFW for the model [20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk) is 0.992+-0.003. A description of how to run the test can be found on the page [Validate on LFW](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw).


### References

* [VAL, FAR and Accuracy](https://github.com/davidsandberg/facenet/issues/288)
* [NMS——非极大值抑制](http://blog.csdn.net/shuzfan/article/details/52711706)
* [MTCNN（Multi-task convolutional neural networks）人脸对齐](http://blog.csdn.net/qq_14845119/article/details/52680940)

# Note


### MTCNN

```shell
python src/align/align_dataset_mtcnn.py \
    /datasets/ceibs/training-images \
    /datasets/ceibs/mtcnn160 \
    --image_size 160 \
    --margin 32 \
    --random_order \
    --gpu_memory_fraction 0.2 \
    &
```

### Classifier

```shell
python src/classifier.py TRAIN \
    /datasets/ceibs/mtcnn160/ \
    ./models/facenet/20170512-110547/20170512-110547.pb \
    ./models/ceibs_classifier.pkl \
    --batch_size 1 \
    --test_data_dir=/datasets/ceibs/mtcnn160/ \
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
