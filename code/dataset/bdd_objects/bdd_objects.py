"""Berkley Deep Drive (BDD100K).

cf. https://bdd-data.berkeley.edu/index.html
"""

import os
from pathlib import Path

from .. import dataset
from . import prepare


class BDD_Objects(dataset.Dataset):
    """API for DNN with BDD."""

    def __init__(self, name):
        """Initialize.

        :param name:
        """
        self.name = name
        self.target_label = 'weather'

    def _get_input_shape(self):
        """Set the input_shape and classes of BDD."""
        return (32, 32, 3), 13

    def prepare(self, input_dir, output_dir, divide_rate, random_state):
        """Prepare.

        :param input_dir:
        :param output_dir:
        :param divide_rate:
        :param random_state:
        :return:
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Make output directory if not exist
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass

        prepare.prepare(input_dir,
                        output_dir,
                        divide_rate,
                        random_state)

        return
