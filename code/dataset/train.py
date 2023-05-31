"""Modules for training the DNN model."""

from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard


def train(model,
          input_shape,
          classes,
          batch_size=32,
          epochs=50,
          validation_split=0.2,
          gpu=False,
          data_dir=r'outputs/',
          output_dir=r'outputs/'
          ):
    """Train.

    :param model:
    :param input_shape:
    :param classes:
    :param batch_size:
    :param epochs:
    :param validation_split:
    :param gpu: configure GPU settings
    :param data_dir:
    :param output_dir:
    :return:
    """
    # GPU settings
    if gpu:
        config = tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(
                allow_growth=True,
                per_process_gpu_memory_fraction=0.8)
        )
        session = tf.compat.v1.Session(config=config)
        set_session(session)

    current_dir = Path(r'./')

    # callbacks
    mc_path = current_dir.joinpath(r'logs/model_check_points')
    mc_path.mkdir(parents=True, exist_ok=True)
    tb_path = current_dir.joinpath(r'logs/tensor_boards')
    tb_path.mkdir(parents=True, exist_ok=True)

    weight_path = mc_path.joinpath('weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    model_check_point = ModelCheckpoint(filepath=str(weight_path),
                                        save_best_only=True,
                                        save_weights_only=True)
    tensor_board = TensorBoard(log_dir=str(tb_path))

    lr_sc = LearningRateScheduler(__lr_schedule)

    callbacks = [model_check_point, tensor_board, lr_sc]
    train_path = data_dir.joinpath(r'train.h5')
    x_train = np.array(h5py.File(train_path, 'r')['images'])
    y_train = np.array(h5py.File(train_path, 'r')['labels'])

    # Load Model
    try:
        model = model.compile(x_train.shape[1:], y_train.shape[1])
    except IndexError:
        # The case of training non one-hot vector
        model = model.compile(x_train.shape[1:], 1)
    model.summary()

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              shuffle=True,
              validation_split=validation_split)

    model_json = model.to_json()
    with open(output_dir.joinpath(r'model.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(output_dir.joinpath(r'model.h5'))


def __lr_schedule(epoch, lr=0.01):
    return lr * (0.1 ** int(epoch / 10))
