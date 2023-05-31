import h5py
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Sequential


class KaminariUtils:
    """Module to reduce a model by removing its first self.depth layers using extract_tail.
     Also save_processed_images, changing them from the original value, to their output after layer self.depth
     Using translate_weights and de_translate_weights, we modify their first value (layer index) so repair
     approaches like Arachne can work with the models before and after the KaminariOptimization"""

    def __init__(self, depth):
        assert depth >= 0
        self.depth = depth

    def process_images(self, inputs, name, get_interm_output):
        """get the output from layer self.depth of the original model when given this inputs

        This computation done here is the part that is optimized and doesn't need to be done during all the evaluations
        of the repair approach

        :param inputs: the original images given to the model
        :param name: either positive or negative, the images set
        :param get_interm_output: function to get outputs of layer self.depth of the model
        :return: the outputs of layer self.depth when given these inputs
        """
        new_ones = get_interm_output(inputs[0])[0]
        print(f"Preprocessed {name} inputs")
        new_ones = np.array(new_ones)
        print(f"Compressed {name} inputs to shape {new_ones.shape}. Occupied {new_ones.nbytes} Bytes.")

        new_ones = [new_ones, inputs[1]]  # add back the labels that go with each image
        return new_ones

    def save_processed_images(self, model, input_neg, input_pos, output_path=None, input_path=None):
        """get the output from layer self.depth of the model when given this inputs

        Given a
            - keras model
            - negative images
            - positive images
            - depth of the layer whose output you want to retrieve
        it feeds the images to the portion of the model
        defined by the depth attributed
        and stores the output in lists, then returns them

        :param model: the model that will give output of layer self.depth
        :param input_neg: images that are incorrectly classified by the model
        :param input_pos: images that are correctly classified by the model
        :param output_path: path where we want to save the processed images
        :param input_path: path to load the processed images instead of computing them, if we previously saved it
        :return: the outputs of layer self.depth when given these inputs
        """

        print("Preprocessing images...")
        # If user specified an input path containing pre-computed data, load it
        if input_path is not None:
            hf = h5py.File(input_path)
            negative = hf.get('negative')
            positive = hf.get('positive')
            hf.close()
            return negative, positive

        if self.depth == 0:
            return input_neg, input_pos

        # Credit to this guy
        # https://stackoverflow.com/questions/51091106/correct-way-to-get-output-of-intermediate-layer-in-keras-model
        # from https://github.com/tensorflow/tensorflow/issues/34201 it seems including K.learning_phase() is no longer appropriate
        get_interm_output = K.function(
            [model.layers[0].input],
            [model.layers[self.depth - 1].output]
        )

        new_neg = self.process_images(input_neg, "negative", get_interm_output)
        new_pos = self.process_images(input_pos, "positive", get_interm_output)

        # Optionally save the compressed dataset to some files
        if output_path is not None:
            hf = h5py.File(output_path, 'w')
            hf.create_dataset(name='negative', data=new_neg)
            hf.create_dataset(name='positive', data=new_pos)
            hf.close()

        print("DONE")
        return new_neg, new_pos

    def copy_layer(self, layer):
        """return a copy of the layer

        Returns a copy of the layer.
        based on https://github.com/keras-team/keras/issues/13140

        :param layer: A DNN layer to copy
        :return: A true deep copy of the layer, that can be modified without trouble
        """
        config = layer.get_config()
        cloned_layer = type(layer).from_config(config)
        cloned_layer.build(layer.input_shape)
        return cloned_layer

    def extract_tail(self, model):
        """return the model but without the first self.depth layers

        It extracts the layers between depth and the last from model,
        and copies them to a new smaller model.

        :param model: the original model, that we will copy only the last layers
        :return: A new reduced model, which is the copy of the last layers of the input model,
                removing the first self.depth layers
        """
        assert self.depth < len(model.layers)

        input_shape = model.layers[self.depth].input_shape[1:]
        print("Extracting submodel...")
        submodel = Sequential()

        if self.depth > 0:  # if depth is 0, we don't need to modify the input
            submodel.add(
                Input(shape=input_shape, batch_size=None, name='input')
            )

        for layer in model.layers[self.depth:]:
            print(f"Adding layer {layer}")
            copy = self.copy_layer(layer)
            submodel.add(copy)
            copy.set_weights(layer.get_weights())

        submodel.build(input_shape)
        submodel.compile(
            optimizer='Adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        #tf.keras.utils.plot_model(submodel, show_shapes=True)  # useful for debugging, but requires installing extra library
        print(submodel.summary())
        print("DONE")
        return submodel

    def translate_weights(self, weights):
        """substract self.depth from every weight in the input

        If you have localized weights in the old model, you need to
        translate their coordinates to work with the new model.
        weights here still haven't got the value from the model, it's only the coordinates (layer, nw_i, nw_j)

        :param weights: the weights the translate to the reduced model
        :return: the same weights, but every layer index got self.depth substracted
        """

        print("Converting weights...")
        print(f"Initial weights:\n {weights}")
        if isinstance(weights[0][0], str):
            weights = [[int(value) for value in weight] for weight in weights]
        print(f"depth: {self.depth}")
        new_weights = []
        for weight in weights:
            if len(weight) > 3:  # later on we want to repair any layer, not only dense ones
                print("skipped susp weight because it wasn't a dense layer")
                continue
            new_weights.append(weight)
        new_weights = np.array(new_weights)
        new_weights[:, 0] -= self.depth
        print(f"New weights:\n {new_weights[:3]}...")
        return new_weights

    def de_translate_weights(self, weights):
        """add self.depth to every weight in the input

        Inverts the translation computed at translate_weights, i.e. moves the layer to the original complete model
        the input solution is already a list with each element in the location format
        weights returned by arachnev1: [weight value, layer index, weight coord. i, weight coord. j]
        weights returned by arachnve2: (layer_index, (weight coord. i, weight coord. j))

        :param weights: the weights the translate to the original model
        :return: the same weights, but every layer index got self.depth added to it
        """
        if isinstance(weights[0][1], tuple):  # if the indexes come in a tuple, like arachnev2 does
            weights = [ [w[0]] + list(w[1]) for w in weights]
            de_trans_weights = np.array(weights)
            de_trans_weights[:, 0] += self.depth
        else:
            de_trans_weights = np.array(weights)
            de_trans_weights[:, 1] += self.depth
        return de_trans_weights
