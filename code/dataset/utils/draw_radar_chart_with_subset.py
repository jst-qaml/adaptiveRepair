"""Utility function:draw radar chart with subset."""

import json
from pathlib import Path

from tqdm import tqdm

from .plot_polar import plot_polar


def _draw_radar_chart_with_subset(dataset, kwargs):
    """Draw radar chart with labeled subset data.

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
    if 'model_dir' in kwargs:
        model_dir = Path(kwargs['model_dir'])
    else:
        raise Exception("Require --model_dir")
    # For radar chart
    min_lim = kwargs['min_lim'] if 'min_lim' in kwargs else 0
    max_lim = kwargs['max_lim'] if 'max_lim' in kwargs else 100
    filename = kwargs['filename'] if 'filename' in kwargs else r'radar.png'

    json_file = input_dir.joinpath(r'labels.json')
    with open(json_file, 'r') as fr:
        json_data = json.load(fr)
        labels = []
        values = []
        for key in tqdm(json_data, desc='Computing scores'):
            labels.append(key)
            data_dir = input_dir.joinpath(key)
            if 'score' in json_data[key]:
                values.append(json_data[key]['score'])
            elif data_dir.joinpath(r'repair.h5').exists():
                score = dataset.test(model_dir, data_dir)
                values.append(score[1] * 100)
                json_data[key]['score'] = score[1] * 100
            else:
                values.append(None)
        with open(json_file, 'w') as fw:
            sorted = sorted(json_data.items(), key=lambda x: x[0])
            json.dump(sorted, fw, indent=4)
        plot_polar(labels, values,
                   output_dir.joinpath(filename),
                   min_lim,
                   max_lim)
