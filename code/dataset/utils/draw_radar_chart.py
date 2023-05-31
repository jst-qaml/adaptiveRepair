"""Utility function: test a model and display results as a radar chart."""

import json
from pathlib import Path

from .plot_polar import plot_polar


def draw_radar_chart(dataset, kwargs):
    """Draw radar chart.

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
    if 'target_data' in kwargs:
        target_data = kwargs['target_data']
        if not target_data.endswith('.h5'):
            raise Exception("File type must be \'.h5\'")
    else:
        target_data = r'repair.h5'
    # For radar chart
    min_lim = kwargs['min_lim'] if 'min_lim' in kwargs else 0
    max_lim = kwargs['max_lim'] if 'max_lim' in kwargs else 100
    filename = kwargs['filename'] if 'filename' in kwargs else r'radar.png'

    # Load
    model = dataset.load_model(model_dir)
    test_images, test_labels = dataset._load_data(input_dir, target_data)

    dict = {}
    for test_label in test_labels:
        key = test_label.argmax()
        dict[str(key)] = {'success': 0, 'failure': 0}

    # Execute
    results = model.predict(test_images, verbose=0)

    # Parse
    for i in range(len(test_labels)):
        test_label = test_labels[i:i + 1]
        test_label_index = test_label.argmax()

        result = results[i:i + 1]

        if result.argmax() == test_label_index:
            current = dict[str(test_label_index)]['success']
            dict[str(test_label_index)]['success'] = current + 1
        else:
            current = dict[str(test_label_index)]['failure']
            dict[str(test_label_index)]['failure'] = current + 1
    labels = []
    values = []
    for key in dict:
        labels.append(key)
        success = dict[key]['success']
        failure = dict[key]['failure']
        score = (success * 100) / (success + failure)
        dict[key]['score'] = score
        values.append(score)

    # Save
    with open(output_dir.joinpath(r'results.json'), 'w') as f:
        dict_sorted = sorted(dict.items(), key=lambda x: x[0])
        json.dump(dict_sorted, f, indent=4)

    # Draw
    plot_polar(labels, values,
               output_dir.joinpath(filename),
               min_lim,
               max_lim)
