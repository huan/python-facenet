"""
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
generate image data
"""
import argparse
import os
import sys

from keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)


def generate(
        from_dir: str,
        file: str,
        to_dir: str = None,
        gen_num: int = 10,
) -> None:
    """ gen data """

    file_path_name = os.path.join(from_dir, file)
    if not os.path.exists(file_path_name):
        raise ValueError('from_dir/file not exist')

    if to_dir:
        if not os.path.exists(to_dir):
            os.makedirs(to_dir)
    else:
        to_dir = from_dir

    name = os.path.splitext(file)[0]

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    # this is a PIL image
    img = load_img(file_path_name)
    # this is a Numpy array with shape (3, 150, 150)
    x = img_to_array(img)
    # this is a Numpy array with shape (1, 3, 150, 150)
    x = x.reshape((1,) + x.shape)
    # the .flow() command below generates batches of
    # randomly transformed images and
    # saves the results to the `preview/` directory
    image_generator = datagen.flow(
        x,
        batch_size=1,
        save_to_dir=to_dir,
        save_prefix=name,
        save_format='jpeg'
    )
    while gen_num > 0:
        next(image_generator)
        gen_num -= 1


def parse_arguments(argv):
    """ parse
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_dir', type=str,
        help='Directory where images stored.')
    parser.add_argument(
        'output_dir', type=str, default=None,
        help='Directory with generated new images.')

    parser.add_argument(
        '--num', type=int, default=10,
        help='How many new images will be generated for one original image.')

    return parser.parse_args(argv)


def main(argv):
    """ main """
    print('main')

    # strip the end '/'
    argv.input_dir = os.path.normpath(argv.input_dir)

    for root, _, files in os.walk(argv.input_dir):
        for file in files:
            path_diff = root.replace(argv.input_dir+'/', '')
            to_dir = os.path.join(argv.output_dir, path_diff)
            # print('argv.output_dir: ', argv.output_dir)
            # print('path_diff: ', path_diff)
            # print('to_dir: ', to_dir)
            print('%s/%s => %s' % (root, file, to_dir))
            generate(root, file, to_dir, argv.num)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
