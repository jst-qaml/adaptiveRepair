"""Utility function: extract images from h5 file."""

import os
import shutil
from pathlib import Path

import h5py
import imageio


def extract_images(dataset, kwargs):
    """Extract images from H5.

    :param dataset:
    :param kwargs:
    """
    if 'input_dir' in kwargs:
        input_dir = Path(kwargs['input_dir'])
    else:
        raise Exception("Require --input_dir")
    if 'output_dir' in kwargs:
        output_dir = Path(kwargs['output_dir'])
    else:
        output_dir = input_dir
    if 'target_data' in kwargs:
        target_data = kwargs['target_data']
        if not target_data.endswith('.h5'):
            raise Exception("File type must be \'.h5\'")
    else:
        target_data = r'repair.h5'

    output_dir = output_dir.joinpath(r'images/')
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    with h5py.File(input_dir.joinpath(target_data), 'r') as hf:
        images = hf['images'][()]
        index = 0
        for image in images:
            filename = str(index).zfill(8) + '.ppm'
            imageio.imwrite(output_dir.joinpath(filename), image)
            index += 1
