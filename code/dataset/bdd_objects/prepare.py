"""Prepare BDD."""

import gc
import os
from pathlib import Path

import ijson
import numpy as np
from skimage import color, exposure, io, transform
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from ..prepare import divide_train, save_prepared_data


def prepare(root_path, output_path, divide_rate, random_state):
    """Prepare.

    :param root_path:
    :param output_path:
    :param divide_rate:
    :param random_state:
    :return:
    """
    train_images, train_labels, \
        repair_images, repair_labels, \
        test_images, test_labels = \
        _get_images_and_labels(
            divide_rate,
            random_state,
            root_path,
        )

    save_prepared_data(train_images,
                       train_labels,
                       repair_images,
                       repair_labels,
                       test_images,
                       test_labels,
                       output_path)


def _get_images_and_labels(
        divide_rate,
        random_state,
        img_path,
        target_size_h=32,
        target_size_w=32,
        classes=13,
        gray=False):
    images = []
    labels = []

    # Get images
    image_path = Path(img_path)
    train_image_paths = image_path.glob('train/*.jpg')
    train_image_paths = list(train_image_paths)
    train_image_paths = np.array(train_image_paths)
    val_image_paths = image_path.glob('val/*.jpg')
    val_image_paths = list(val_image_paths)
    val_image_paths = np.array(val_image_paths)
    all_image_paths = np.hstack((train_image_paths, val_image_paths))
    del val_image_paths
    del train_image_paths
    gc.collect()
    np.random.seed(random_state)
    np.random.shuffle(all_image_paths)

    # Get labels
    train_labels = _get_labels(img_path.joinpath(r'train/image_info.json'))
    val_labels = _get_labels(img_path.joinpath(r'val/image_info.json'))
    train_labels.update(val_labels)
    all_labels = train_labels
    del train_labels
    del val_labels
    gc.collect()

    data_count = 0
    for image_path in tqdm(all_image_paths):
        img = _preprocess_img(io.imread(image_path),
                              (target_size_h, target_size_w))
        label = _get_train_class(all_labels, image_path)
        try:
            labels.append(to_categorical(label, num_classes=classes))
            images.append(img)

            data_count += 1
        except TypeError:
            continue

    test_num = data_count // 6
    train_images = images[test_num:]
    train_labels = labels[test_num:]
    test_images = images[:test_num]
    test_labels = labels[:test_num]

    train_images, train_labels, repair_images, repair_labels = \
        divide_train(train_images, train_labels, divide_rate, random_state)

    return np.array(train_images, dtype='float32'), \
        np.array(train_labels, dtype='uint8'), \
        np.array(repair_images, dtype='float32'), \
        np.array(repair_labels, dtype='uint8'), \
        np.array(test_images, dtype='float32'), \
        np.array(test_labels, dtype='uint8')


def _preprocess_img(img, target_size):
    # Rescale to target size
    img = transform.resize(img, target_size)

    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    return img


def _get_labels(file_path):
    label_file = open(file_path)
    df = ijson.parse(label_file)
    labels = {}

    name = None
    attribute = None
    for prefix, _, value in df:
        if prefix == 'images.item.name' and name is None:
            name = value
        if name is not None and prefix == 'images.item.label':
            attribute = value
            labels[name] = attribute
            name = None
            attribute = None

    label_file.close()
    return labels


def _get_train_class(labels, img_path):
    img_name = os.path.basename(img_path)
    label_to_class = {'bicycle': 0,
                      'bus': 1,
                      'car': 2,
                      'motorcycle': 3,
                      'other person': 4,
                      'other vehicle': 5,
                      'pedestrian': 6,
                      'rider': 7,
                      'traffic light': 8,
                      'traffic sign': 9,
                      'trailer': 10,
                      'train': 11,
                      'truck': 12}
    attribute = label_to_class[labels[img_name]]

    return attribute
