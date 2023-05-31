"""Utility function: merge repair data."""

import shutil
from pathlib import Path

import h5py
import numpy as np


def merge_repair_data(dataset, kwargs):
    """Merge repair data.

    :param dataset:
    :param kwargs:
    """
    if 'input_dir1' in kwargs:
        input_dir1 = Path(kwargs['input_dir1'])
    else:
        raise Exception("Require --input_dir1")
    if 'input_dir2' in kwargs:
        input_dir2 = Path(kwargs['input_dir2'])
    else:
        raise Exception("Require --input_dir2")
    if 'output_dir' in kwargs:
        output_dir = Path(kwargs['output_dir'])
    else:
        raise Exception("Require --output_dir")
    # Load test data
    _test_images1, _test_labels1 = dataset.load_repair_data(input_dir1)
    _test_images2, _test_labels2 = dataset.load_repair_data(input_dir2)
    # Merge test data
    test_images = []
    test_labels = []
    for i in range(len(_test_images1)):
        test_images.append(_test_images1[i])
        test_labels.append(_test_labels1[i])
    for i in range(len(_test_images2)):
        test_images.append(_test_images2[i])
        test_labels.append(_test_labels2[i])
    # Format test data
    test_images = np.array(test_images, dtype='float32')
    test_labels = np.array(test_labels, dtype='float32')
    # Save test data
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    with h5py.File(output_dir.joinpath(r'repair.h5'), 'w') as hf:
        hf.create_dataset('images', data=test_images)
        hf.create_dataset('labels', data=test_labels)
