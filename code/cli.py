"""Command Line Interface (CLI).

handling commands with fire.

"""

import importlib
import json
import os
from pathlib import Path

import fire

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["KMP_AFFINITY"] = "noverbose"

tf = importlib.import_module('tensorflow')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class CLI(object):
    """Command Line Interface for automated repair of DNN models.

    Handling commands as defined below.

    """

    def prepare(self,
                dataset,
                input_dir=r'inputs/',
                output_dir=r'outputs/',
                divide_rate=0.2,
                random_state=None,
                **kwargs):
        """Prepare dataset to train.

        :param dataset: dataset name
        :param input_dir: path to input directory
                                containing train and test datasets
        :param output_dir: path to output directory
        :param divide_rate: The ratio of dividing training data
                            for using repair
        :param random_state: The seed value of random sampling
                             for reproducibility of dividing training data
        :return:
        """
        dataset = _get_dataset(dataset, kwargs=kwargs)
        dataset.prepare(input_dir, output_dir, divide_rate, random_state)

        return

    def train(self,
              dataset,
              model='base_cnn_model',
              batch_size=32,
              epochs=5,
              validation_split=0.2,
              gpu=False,
              data_dir=r'outputs/',
              output_dir=r'outputs/',
              **kwargs):
        """Train dataset prepared previously.

        :param dataset: dataset name
        :param model: model name
        :param batch_size: a size of batches
        :param epochs: the number of epochs
        :param validation_split: a proportion of data for validation
        :param gpu: True or False
        :param data_dir: path to directory containing train datasets
                         ('output_dir' on preparing) directory
        :param output_dir: path to output directory
        :return:
        """
        dataset = _get_dataset(dataset)
        kwargs['data_dir'] = data_dir
        model = _get_model(model, kwargs=kwargs)
        dataset.train(model,
                      batch_size,
                      epochs,
                      validation_split,
                      gpu,
                      data_dir,
                      output_dir)

        return

    def test(self,
             dataset,
             batch_size=32,
             model_dir=r'outputs/',
             data_dir=r'outputs/',
             target_data=r'test.h5',
             gpu_num=None,
             ):
        """Test DNN model generated previously.

        :param dataset: dataset name
        :param batch_size: a size of batches
        :param model_dir: path to directory containing DNN model
        :param data_dir: path to directory containing test dataset
        :param target_data: filename for target dataset
        :param gpu_num: if not none, constrains the program to only use the selected gpu
        :return: accuracy
        """
        if gpu_num is not None:
            self.select_gpu(gpu_num)
        dataset = _get_dataset(dataset)
        score = dataset.test(model_dir, data_dir, target_data, batch_size)

        return score

    def target(self,
               dataset,
               batch_size=32,
               model_dir=r'outputs/',
               data_dir=r'outputs/',
               dataset_type='repair',
               gpu_num=None,
               do_cleanup=True):
        """Find target dataset to reproduce failures on DNN models.

        :param dataset: dataset name
        :param batch_size: a size of batches
        :param model_dir: path to directory containing DNN model
        :param data_dir: path to directory containing test dataset
        :param dataset_type: type of dataset to divide into classes. Choose between train, repair, test
        :param gpu_num: if not none, constrains the program to only use the selected gpu
        :param do_cleanup: determines if we should erase previous target results
        :return:
        """
        if gpu_num is not None:
            self.select_gpu(gpu_num)
        dataset = _get_dataset(dataset)
        dataset.target(model_dir, data_dir, batch_size, dataset_type, do_cleanup)

        return

    def localize(self,
                 dataset,
                 method,
                 model_dir=r'outputs/',
                 target_data_dir=r'outputs/negative/0/',
                 verbose=1,
                 gpu_num=None,
                 **kwargs):
        """Repair DNN model with target and test data.

        :param dataset: dataset name
        :param method: repair method name
        :param model_dir: path to directory containing DNN model
        :param target_data_dir: path to directory
               containing target test dataset
        :param verbose: Log level
        :param gpu_num: if not none, constrains the program to only use the selected gpu
        :param kwargs:
                 `batch_size`:
                   a size of batches
                 `train_data_dir`:
                   (TBA) path to directory containing train dataset
                 `num_grad`:
                   (for Arachne only) number of neural weight candidates
                   to choose based on gradient loss
        :return:
        """
        # Parse optionals
        try:
            train_data_dir = kwargs['train_data_dir']
            raise Exception('To be implemented: {}'.format(train_data_dir))
        except KeyError:
            pass

        if gpu_num is not None:
            self.select_gpu(gpu_num)
        print(model_dir)
        dataset = _get_dataset(dataset)
        method = _get_repair_method(method, dataset, kwargs)

        model = dataset.load_model(model_dir)
        target_test_data = method.load_input_neg(target_data_dir)
        method.localize(model, target_test_data, target_data_dir, verbose)

        return

    def optimize(self,
                 dataset,
                 method,
                 model_dir=r'outputs/',
                 target_data_dir=r'outputs/negative/0/',
                 positive_inputs_dir=r'outputs/positive/',
                 output_dir=None,
                 risk_aware=False,
                 verbose=1,
                 gpu_num=None,
                 **kwargs):
        """Optimize neuron weights to repair.

        :param dataset: dataset name
        :param method: repair method name
        :param model_dir: path to directory containing DNN model
        :param target_data_dir: path to directory
               containing dataset for unexpected behavior
        :param positive_inputs_dir: path to directory
               containing dataset for correct behavior
        :param output_dir: path to directory where analysis results are saved
        :param risk_aware: selects between the classic Arachne fitness, and an ad_hoc
                risk aware fitness
        :param verbose: Log level
        :param gpu_num: if not none, constrains the program to only use the selected gpu
        :param kwargs:
                `batch_size`: a size of batches
        :return:
        """
        # Instantiate
        if gpu_num is not None:
            self.select_gpu(gpu_num)
        dataset = _get_dataset(dataset)
        method = _get_repair_method(method, dataset, kwargs)
        if risk_aware:
            method.make_risk_aware()
        if output_dir is None:
            output_dir = target_data_dir

        # Load
        model = dataset.load_model(model_dir)
        weights = method.load_weights(target_data_dir)
        target_data = method.load_input_neg(target_data_dir)
        positive_inputs = method.load_input_pos(positive_inputs_dir)

        # Analyze
        method.optimize(model,
                        dataset,
                        model_dir,
                        weights,
                        target_data,
                        positive_inputs,
                        output_dir,
                        verbose)

        return

    def evaluate(self,
                 dataset,
                 method,
                 model_dir=r'outputs/',
                 target_data_dir=r'outputs/negative/0/',
                 positive_inputs_dir=r'outputs/positive/',
                 output_dir=None,
                 num_runs=10,
                 verbose=1,
                 **kwargs):
        """Evaluate repairing performance.

        :param dataset: dataset name
        :param method: repair method name
        :param batch_size: a size of batches
        :param model_dir: path to directory containing DNN model
        :param target_data_dir: path to directory
               containing dataset for unexpected behavior
        :param positive_inputs_dir: path to directory
               containing dataset for correct behavior
        :param output_dir: path to directory where analysis results are saved
        :param num_runs: number of repair attempts
        :param verbose: Log level
        :param kwargs:
                `batch_size`: a size of batches
        :return:
        """
        # Instantiate
        dataset = _get_dataset(dataset)
        method = _get_repair_method(method, dataset, kwargs)
        if output_dir is None:
            output_dir = target_data_dir

        # Load
        target_data = dataset.load_repair_data(target_data_dir)
        positive_inputs = dataset.load_repair_data(positive_inputs_dir)

        # Evaluate
        method.evaluate(dataset,
                        method,
                        model_dir,
                        target_data,
                        target_data_dir,
                        positive_inputs,
                        positive_inputs_dir,
                        output_dir,
                        num_runs,
                        verbose)

        return

    def count_misclassifications(self, data_dir, data_type, dataset):
        dataset = _get_dataset(dataset)
        dataset.count_samples(data_type, data_dir)

    def select_gpu(self, gpu_num):
        print("GPU device number "+str(gpu_num)+" selected")
        gpu_num = int(gpu_num)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')


    def utils(self,
              dataset,
              **kwargs):
        """Utilities.

        This command is for dataset-specific functions.

        :param dataset: dataset name
        :param kwargs:
        :return:
        """
        dataset = _get_dataset(dataset)
        dataset.utils(kwargs)

        return


def _get_dataset(name, **kwargs):
    """Get dataset.

    :param name: identifier of dataset
    :param kwargs: extra config available for specific dataset
           (e.g. target_label for BDD dataset)
    :return: dataset
    """
    dataset = _load_instance('dataset', 'dataset', name, 'settings')
    if dataset is None:
        raise Exception('Invalid dataset: {}'.format(name))
    dataset.set_extra_config(kwargs)
    return dataset


def _get_model(name, **kwargs):
    """Get model.

    :param name: identifier of model
    :return: model
    """
    model = _load_instance('model', 'model', name, 'settings')
    if model is None:
        raise Exception('Invalid model: {}'.format(name))
    model.set_extra_config(kwargs)
    return model


def _get_repair_method(name, dataset, kwargs):
    """Get repair method.

    :param name: identifier of repair method
    :param dataset: identifier of dataset
    :param kwargs: repair method options
    :return: repair method
    """
    # Instantiate repair method
    method = _load_instance('method', 'repair', name, 'settings')
    if method is None:
        raise Exception('Invalid method: {}'.format(name))
    # Set optional parameters
    method.set_options(dataset, kwargs)

    return method


def _load_instance(key, parent_module, name, settings_dir):
    """Load settings.

    Loading a instance of repair methods, datasets, or models
    Settings are in "settings.json" under a given directory.

    :param key: key of settings.json (e.g. method)
    :param parent_module: directory containing modules (e.g. repair)
    :param name: available option name (e.g. Arachne for method)
    :param settings_dir: path to directory containing settings.json
    :return: instance of repair methods, datasets, or models
    """
    importlib.invalidate_caches()
    settings_dir = Path(settings_dir)
    file = settings_dir.joinpath(r'settings.json')

    with open(file, 'r') as f:
        try:
            settings = json.load(f)
            if name in settings[key]:
                name = settings[key][name]
            _names = name.split('.')
            module_name = name.split('.' + _names[-1])[0]
            methodclass = importlib.import_module('.' + module_name,
                                                  parent_module)
            instance = getattr(methodclass, _names[-1])(_names[0])

            return instance
        except BaseException:
            return None


if __name__ == '__main__':
    fire.Fire(CLI)
