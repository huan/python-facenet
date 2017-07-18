""" facenet euclidean distance calculator.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import os
import shelve
import sys
from typing import List, Iterable

import tensorflow as tf
import numpy as np
import facenet
# from tinydb import TinyDB

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def png(files: List[str]) -> Iterable[str]:
    """ png ext selector """
    for f in files:
        ext = os.path.splitext(f)[1]
        # print('f: ', f, ' ext: ', ext)
        if ext.lower() == '.png':
            yield f


def main(args):
    ''' Main entrypoint '''

    # db = TinyDB('embeddingdb.json')

    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=args.seed)  # pylint: disable=E1101

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph() \
                .get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph() \
                .get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph() \
                .get_tensor_by_name("phase_train:0")

            embedding_size = embeddings.get_shape()[1]

            paths = []  # type: List[str]
            for root, _, files in os.walk(args.data_dir):
                for file in png(files):
                    file_path = os.path.join(root, file)
                    print('adding image: ', file_path)
                    paths.append(file_path)

            # print(paths)

            # Run forward pass to calculate embeddings
            nrof_images = len(paths)
            print('Calculating features for %d images' % nrof_images)
            nrof_batches_per_epoch = int(
                math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(
                    paths_batch, False, False, args.image_size)
                feed_dict = {
                    images_placeholder:         images,
                    phase_train_placeholder:    False,
                }
                emb_array[start_index:end_index, :] = sess.run(
                    embeddings, feed_dict=feed_dict)
                sys.stdout.write('#')
                sys.stdout.flush()

            # data = []
            print('db writing...')
            with shelve.open('shelve.db') as db:
                for path, embedding in zip(paths, emb_array):
                    db[path] = embedding.tolist()

            # db.insert_multiple(data)

    print('done')


def parse_arguments(argv):
    """ args """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument(
        'model', type=str,
        help='Could be either a directory containing the meta_file '
        'and ckpt_file or a model protobuf (.pb) file')

    parser.add_argument(
        '--batch_size', type=int, default=90,
        help='Number of images to process in a batch.')
    parser.add_argument(
        '--image_size', type=int, default=160,
        help='Image size (height, width) in pixels.')
    parser.add_argument(
        '--seed', type=int, default=666,
        help='Random seed.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
