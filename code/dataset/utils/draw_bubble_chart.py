"""Utility function: test a model and display results as a bubble chart."""

from pathlib import Path

import numpy as np

from .plot_bubble_chart import plot_bubble_chart


def draw_bubble_chart(dataset, kwargs):
    """Draw bubble chart.

    :param dataset:
    :param kwargs:
    """
    if 'model_dir' in kwargs:
        model_dir = Path(kwargs['model_dir'])
    else:
        raise Exception("Require --model_dir")

    if 'test_dir' in kwargs:
        test_dir = Path(kwargs['test_dir'])
    else:
        raise Exception("Require --test_dir")

    if 'output_dir' in kwargs:
        output_dir = Path(kwargs['output_dir'])
    else:
        output_dir = Path(r'outputs/')

    if 'test_data' in kwargs:
        test_data = kwargs['test_data']
        if not test_data.endswith('.h5'):
            raise Exception("File type must be \'.h5\'")
    else:
        test_data = r'test.h5'

    filename = kwargs['filename'] if 'filename' in kwargs else f'bubble.png'

    # get model
    model = dataset.load_model(model_dir)

    # get test data.
    # test_lables are to be set as ground truth.
    test_images, test_labels = dataset._load_data(test_dir, test_data)

    def convert_labels(labels):
        """Convert labels.

        convert labels from array representation into number,
        then reconvert them into string to set ticks properly.
        e.g.) [[0,0,...,1],...,[1,...,0]] => ["9",...,"0"]
        :param labels:
        """
        return np.array(list(map(lambda l: l.argmax(), labels)))

    ground_truth = convert_labels(test_labels)

    # get predicted labels
    results = model.predict(test_images, verbose=0)
    pred_labels = convert_labels(results)

    # plot confusion matrix as bubble chart
    plot_bubble_chart(ground_truth,
                      pred_labels,
                      output_dir.joinpath(filename))
