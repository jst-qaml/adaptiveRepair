"""Utility function: test a model and display results as a bubble chart."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plot_bubble_chart import plot_each_bubble_chart


def overlay_bubble_chart(dataset, kwargs):
    """Draw bubble chart.

    :param dataset:
    :param kwargs:
    """
    model, model_overlay, model_overlay2 = _get_models(dataset, kwargs)
    test_images, test_labels = _get_test_data(dataset, kwargs)
    legends = _get_legends(kwargs, model_overlay2)

    if 'target_label' in kwargs:
        target_label = [str(num) for num in kwargs['target_label']]
    else:
        target_label = None

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
    results_overlay = model_overlay.predict(test_images, verbose=0)
    if model_overlay2 is not None:
        results_overlay2 = model_overlay2.predict(test_images, verbose=0)
    else:
        results_overlay2 = None

    pred_labels = convert_labels(results)
    pred_labels_overlay = convert_labels(results_overlay)
    if model_overlay2 is not None:
        pred_labels_overlay2 = convert_labels(results_overlay2)

    fig = plt.figure()

    # plot confusion matrix as bubble chart
    ax = plot_each_bubble_chart(fig,
                                ground_truth,
                                pred_labels,
                                target_label)
    plot_each_bubble_chart(fig,
                           ground_truth,
                           pred_labels_overlay,
                           target_label,
                           'h')

    if model_overlay2 is not None:
        plot_each_bubble_chart(fig,
                               ground_truth,
                               pred_labels_overlay2,
                               target_label,
                               '*')

    ax.legend(legends,
              bbox_to_anchor=(0, 1.15),
              fontsize=12,
              loc='upper left',
              borderaxespad=0)

    _save_fig(fig, kwargs)


def _get_models(dataset, kwargs):
    if 'model_dir' in kwargs:
        model_dir = Path(kwargs['model_dir'])
    else:
        raise Exception("Require --model_dir")

    if 'model_dir_overlay' in kwargs:
        model_dir_overlay = Path(kwargs['model_dir_overlay'])
    else:
        raise Exception("Require --model_dir_overlay")

    if 'model_dir_overlay2' in kwargs:
        model_dir_overlay2 = Path(kwargs['model_dir_overlay2'])
    else:
        model_dir_overlay2 = None
    # get models
    model = dataset.load_model(model_dir)
    model_overlay = dataset.load_model(model_dir_overlay)
    if model_dir_overlay2 is not None:
        model_overlay2 = dataset.load_model(model_dir_overlay2)
    else:
        model_overlay2 = None

    return model, model_overlay, model_overlay2


def _get_test_data(dataset, kwargs):
    if 'test_dir' in kwargs:
        test_dir = Path(kwargs['test_dir'])
    else:
        raise Exception("Require --test_dir")

    if 'test_data' in kwargs:
        test_data = kwargs['test_data']
        if not test_data.endswith('.h5'):
            raise Exception("File type must be \'.h5\'")
    else:
        test_data = r'test.h5'
    # get test data.
    # test_lables are to be set as ground truth.
    test_images, test_labels = dataset._load_data(test_dir, test_data)

    return test_images, test_labels


def _get_legends(kwargs, model_overlay2):
    legend = kwargs['legend'] if 'legend' in kwargs else 'Inputs (base)'
    legend_overlay = kwargs['legend_overlay'] \
        if 'legend_overlay' in kwargs \
        else 'Inputs (overlay)'
    legend_overlay2 = kwargs['legend_overlay2'] \
        if 'legend_overlay2' in kwargs \
        else 'Inputs (overlay2)'

    # Legend
    if model_overlay2 is None:
        legends = [legend, legend_overlay]
    else:
        legends = [legend, legend_overlay, legend_overlay2]

    return legends


def _save_fig(fig, kwargs):
    if 'output_dir' in kwargs:
        output_dir = Path(kwargs['output_dir'])
    else:
        output_dir = Path(r'outputs/')

    filename = kwargs['filename'] if 'filename' in kwargs else f'bubble.png'

    fig.savefig(output_dir.joinpath(filename), format='png', dpi=600)
