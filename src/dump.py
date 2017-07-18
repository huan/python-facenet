""" dump """
import argparse
import shelve
import sys

import numpy as np
# from tinydb import (
#     TinyDB,
#     Query,
# )


def main(args):
    """ main """
    # db = TinyDB('embeddingdb.json')
    # Embedding = Query()

    # file_path_name = '/datasets/ceibs/mtcnn160/沈岩/DSC_8570_0_7442.png'
    # doc1 = db.search(Embedding.id == args.image1)[0]
    # doc2 = db.search(Embedding.id == args.image2)[0]
    with shelve.open('shelve.db') as db:
        embedding1 = db[args.image1]
        embedding2 = db[args.image2]

    # print(doc1)

    embedding1 = np.asarray(embedding1)
    embedding2 = np.asarray(embedding2)

    dist = np.linalg.norm(embedding1 - embedding2)
    print(dist)


def parse_arguments(argv):
    """ args """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'image1', type=str,
        help='Path to the image file.')
    parser.add_argument(
        'image2', type=str,
        help='Path to the second image file')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

