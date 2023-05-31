"""Utility function: draw bubble chart of repaired results."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_repaired_result(dataset, kwargs):
    """Draw repaired result.

    :param dataset:
    :param kwargs:
    """
    if 'model_dir' in kwargs:
        model_dir = Path(kwargs['model_dir'])
    else:
        raise Exception("Require --model_dir")

    if 'target_dir' in kwargs:
        target_dir = Path(kwargs['target_dir'])
    else:
        raise Exception("Require --target_dir")

    if 'output_dir' in kwargs:
        output_dir = Path(kwargs['output_dir'])
    else:
        output_dir = Path(r'outputs/')

    if 'target_data' in kwargs:
        target_data = kwargs['target_data']
        if not target_data.endswith('.h5'):
            raise Exception("File type must be \'.h5\'")
    else:
        target_data = r'repair.h5'

    filename = kwargs['filename'] if 'filename' in kwargs else f'repaired.png'

    repaired_model = dataset.load_model(model_dir)

    negative_label_dirs = \
        sorted([d for d in target_dir.iterdir() if d.is_dir()])

    negative_label_names = \
        [label_dir.name for label_dir in negative_label_dirs]

    def convert_labels(labels):
        """Convert labels.

        convert labels from array representation into number,
        then reconvert them into string to set ticks properly.
        e.g.) [[0,0,...,1],...,[1,...,0]] => [9,...,0]
        :param labels:
        """
        return np.array(list(map(lambda l: l.argmax(), labels)))

    results = {}
    for negative in negative_label_names:
        # discart test labels
        test_images, _ = \
            dataset._load_data(target_dir.joinpath(negative), target_data)

        pred_results = repaired_model.predict(test_images, verbose=0)
        pred_results_index = convert_labels(pred_results)
        total_elems = len(pred_results_index)

        def normalize(value):
            return value / total_elems * 500

        pred_results_summary = Counter(pred_results_index)
        results[negative] = \
            dict([(k, normalize(v)) for k, v in pred_results_summary.items()])

    df = pd.DataFrame.from_dict(results)
    df.fillna(0)
    x, y = np.meshgrid(df.columns.values, df.index.values)
    z = df.values.flatten()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x=x.flatten(), y=y.flatten(), s=z)

    ax.set_xlabel("negative labels")
    ax.set_xticks(df.columns.values)
    ax.set_ylabel("predicted labels")
    ax.set_yticks(df.index.values)
    ax.margins(.1)
    fig.savefig(output_dir.joinpath(filename), format='png', dpi=300)

    plt.close(fig)
