"""Create dataset for training gate model of HydraNet."""
import json
import os
from pathlib import Path

import h5py
import numpy


def create_gate_class(dataset, kwargs):
    """Create classes for gate in hydranet."""
    if 'hydra_setting_file' in kwargs:
        hydra_setting_file = Path(kwargs['hydra_setting_file'])
        with open(hydra_setting_file, 'r') as f:
            hydra_subtask = json.load(f)
    else:
        Exception('Require --hydra_setting_file')

    if 'data_dir' in kwargs:
        data_dir = Path(kwargs['data_dir'])
    else:
        Exception('Require --data_dir')

    _create_gate_file(dataset, data_dir, 'train.h5', hydra_subtask)
    _create_gate_file(dataset, data_dir, 'test.h5', hydra_subtask)
    _create_gate_file(dataset, data_dir, 'repair.h5', hydra_subtask)


def _create_gate_file(dataset, data_dir, target_file, hydra_subtask):
    images, labels = dataset._load_data(data_dir, target_file)
    new_labels = _create_new_labels(labels, hydra_subtask)

    gate_dir = data_dir.joinpath('gate')
    if not gate_dir.exists():
        os.mkdir(gate_dir)

    with h5py.File(gate_dir.joinpath(target_file), 'w') as hf:
        hf.create_dataset('images', data=images)
        hf.create_dataset('labels', data=new_labels)


def _create_new_labels(labels, hydra_subtask):
    new_labels = []
    for label in labels:
        new_label = numpy.zeros(len(hydra_subtask), dtype=int)
        for idx in range(len(label)):
            if label[idx] == 1:
                for sub_idx in range(len(hydra_subtask)):
                    if idx in hydra_subtask[sub_idx]:
                        new_label[sub_idx] = 1
                        break
        new_labels.append(new_label)
    return new_labels
