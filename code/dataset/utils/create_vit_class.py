"""Create dataset for training vision transformer model."""
import os
from pathlib import Path

import h5py
import numpy


def create_vit_class(dataset, kwargs):
    """Create dataset for training vision transformer model.

    :param dataset:
    """
    if 'data_dir' in kwargs:
        data_dir = Path(kwargs['data_dir'])
    else:
        Exception('Require --data_dir')

    _create_gate_file(dataset, data_dir, 'train.h5')


def _create_gate_file(dataset, data_dir, target_file):
    images, labels = dataset._load_data(data_dir, target_file)
    new_labels = _create_new_labels(labels)

    gate_dir = data_dir.joinpath('vision_transformer')
    if not gate_dir.exists():
        os.mkdir(gate_dir)

    with h5py.File(gate_dir.joinpath(target_file), 'w') as hf:
        hf.create_dataset('images', data=images)
        hf.create_dataset('labels', data=new_labels)


def _create_new_labels(labels):
    new_labels = numpy.argmax(labels, axis=1)
    return new_labels
