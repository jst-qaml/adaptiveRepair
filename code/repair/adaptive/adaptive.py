from pathlib import Path

import numpy as np
import tensorflow as tf
import csv
import warnings
from tqdm import tqdm, trange
from termcolor import colored
from ..kaminariOptimization import KaminariUtils

from ..arachne.arachne import Arachne

tf.compat.v1.disable_eager_execution()

class Weight:

    def __init__(self, coords=None, value=None):
        self.coords = coords
        self.value = float(value)

    def same_loc(self, other):

        return self.coords == other.coords

    def distance(self, other):

        if self.coords != other.coords:
            return 10 ** 9
        else:
            return abs(self.value - other.value)

class Adaptive(Arachne):

    def __init__(self, name):
        """Initialize.

        :param name:
        """
        self.name = name
        self.dataset = None
        self.num_grad = None
        self.num_particles = 100
        self.num_iterations = 10
        self.iterations_counter = 0
        self.num_macroiterations = 10
        self.num_input_pos_sampled = 200
        self.sampled_positive=None
        self.velocity_phi = 4.1
        self.min_iteration_range = 10
        self.target_layer = None
        self.output_files = set()
        self.batch_size = 32
        self.risk_aware = False
        self.fixed_stop = False
        self.switch_criterion = 'stagnation'
        self.window=8
        self.threshold = 0.01
        self.normalized_fitness = True
        self.switch_log = []
        self.current_macrogen = 0

        #kaminari things
        self.kaminari = None
        self.kaminari_depth = None
        self.original_complete_model = None
        self.original_reduced_model = None

    def set_options(self, dataset, kwargs):
        """Set options."""
        self.dataset = dataset
        if 'num_grad' in kwargs:
            self.num_grad = kwargs['num_grad']
        if 'num_particles' in kwargs:
            self.num_particles = kwargs['num_particles']
        if 'num_iterations' in kwargs:
            self.num_iterations = kwargs['num_iterations']
        if 'num_macroiterations' in kwargs:
            self.num_macroiterations = kwargs['num_macroiterations']
        if 'num_input_pos_sampled' in kwargs:
            self.num_input_pos_sampled = kwargs['num_input_pos_sampled']
        if 'velocity_phi' in kwargs:
            self.velocity_phi = kwargs['velocity_phi']
        if 'min_iteration_range' in kwargs:
            self.min_iteration_range = kwargs['min_iteration_range']
        if 'target_layer' in kwargs:
            self.target_layer = int(kwargs['target_layer'])
        if 'batch_size' in kwargs:
            self.batch_size = int(kwargs['batch_size'])
        if "window" in kwargs:
            self.window=int( kwargs['window'] )
        if 'threshold' in kwargs:
            self.threshold= kwargs['threshold']
        if 'fixed_stop' in kwargs:
            self.fixed_stop=True
        if 'switch_criterion' in kwargs:
            self.switch_criterion=kwargs['switch_criterion']
        if 'normalized_fitness' in kwargs:
            self.normalized_fitness = True

    def load_weights(self, output_dir):
        """Load neural weight candidates.

        :param output_dir: path to directory containing 'wights.csv'
        :return: Neural weight candidates
        """

        return None


    def eval_fitness(self,model,input_neg, input_pos):
        """
        Computes fitness
        """

        loss_input_neg, acc_input_neg = \
            model.evaluate(input_neg[0],
                           input_neg[1],
                           verbose=0,
                           batch_size=self.batch_size)
        n_patched = int(np.round(len(input_neg[1]) * acc_input_neg))

        input_pos_sampled = self._sample_positive_inputs(input_pos)

        # "N_{intact} is th number of inputs in I_{pos}
        # whose output is still correct"
        loss_input_pos, acc_input_pos = \
            model.evaluate(input_pos_sampled[0],
                           input_pos_sampled[1],
                           verbose=0,
                           batch_size=self.batch_size)
        n_intact = int(np.round(len(input_pos_sampled[1]) * acc_input_pos))

        neg_term = (n_patched + 1) / (loss_input_neg + 1)
        pos_term = (n_intact + 1) / (loss_input_pos + 1)

        # If you want to normalize the fitness, do it now
        if self.normalized_fitness:
            neg_term = neg_term / len(input_neg[0])
            pos_term = pos_term / len(input_pos[0])

        fitness = neg_term + pos_term

        return fitness

    def _sample_positive_inputs(self, input_pos):
        """Sample 200 positive inputs.

        :param input_pos:
        :return:
        """

        if not (self.sampled_positive is None):
            return self.sampled_positive

        labels = {}  # divide the index of the inputs depending on their label
        for index, label in np.ndenumerate(np.argmax(input_pos[1], axis=1)):
            if label in labels:
                labels[label].append(index[0])
            else:
                labels[label] = [index[0]]

        # at least put as many images of each label, as if you put a third of an equitative distribution
        amount_per_label = {}
        total = 0
        for label in labels:
            amount_per_label[label] = int((self.num_input_pos_sampled / len(labels)) / 3)
            amount_per_label[label] += int((2/3) * (self.num_input_pos_sampled * (len(labels[label]) / len(input_pos[1]))))
            total += amount_per_label[label]
        if total != self.num_input_pos_sampled:
            if abs(total - self.num_input_pos_sampled) > len(labels):
                warnings.warn("The sampling of positive images is not being accurate")
            amount_per_label[np.random.choice(list(amount_per_label.keys()), 1)[0]] += self.num_input_pos_sampled - total

        # actual sampling
        sample = np.empty(0, int)
        for label in amount_per_label:
            while amount_per_label[label] >= len(labels[label]):
                sample = np.concatenate((sample, labels[label]))
                amount_per_label[label] -= len(labels[label])
            sample = np.concatenate((sample, np.random.choice(labels[label], amount_per_label[label], replace=False)))

        assert len(sample) == self.num_input_pos_sampled
        input_pos_sampled = (input_pos[0][sample], input_pos[1][sample])

        self.sampled_positive=input_pos_sampled
        return input_pos_sampled

    def jacobi_index(self,s1,s2):
        inters = []
        union = []

        for x in s1:
            for y in s2:

                if x.same_loc(y):
                    inters.append(x)

        for x in s1:
            union.append(x)

        for y in s2:
            append=True

            for z in union:
                if z.same_loc(y):
                    append=False
                    break

            if append:
                union.append(y)

        return len(inters) / len(union)

    def switch_criterion_localisation(self, score_log, weights_log,
                                      input_neg, best_model=None, model=None,
                                      input_pos=None, input_pos_sampled=None,
                                      output_dir=None
                                      ):
        """
        Implements a switch criterion based on sharp differences in the F.L
        """

        # From the history, retrieve the best particle and the best without
        # considering the last self.window generations
        if len(score_log) < self.window + 1:
            return False

        score_list = [score_log[i][0] for i in range(len(score_log))]

        if best_model is None:
            # The best particle index
            best_index = np.argmax(score_list)
            best_particle = weights_log[best_index]
            #best_particle_model = self._copy_location_to_weights(best_particle,model)
        else:
            best_particle_model = best_model

        old_target_layer = self.target_layer
        #fl1 = Arachne.localize(self, model, input_neg)

        # The highest score up to window generations before
        score_list = score_list[0:len(score_list)-self.window]
        best_index_window = np.argmax(score_list)

        best_particle_window = weights_log[best_index_window]
        # Keep original weights and set specified weights on the model
        orig_location = np.copy(best_particle_window)
        orig_location = self._copy_weights_to_location(model, orig_location)

        best_particle_model_window = self._copy_location_to_weights(best_particle_window, model)

        # Find a new negative and positive set
        interm_input_neg, _ = self.find_new_neg_pos(
            best_particle_model_window, input_neg, input_pos_sampled,
            input_pos, output_dir)
        if len(interm_input_neg[0]) == 0:  # if no more neg inputs are left, we want to "swith" because we want to finish execution
            return True

        self.target_layer = old_target_layer
        fl2 = Arachne.localize(self, best_particle_model_window, interm_input_neg)
        self.target_layer = old_target_layer

        #print(best_particle_model_window == best_particle_window, best_index,best_index_window,weights_log,best_particle,best_particle_window)

        if 0 > self.threshold or self.threshold > 1:
            raise Exception(f"Invalid threshold: threshold {self.threshold} is not acceptable for FL-based criterion. Please specify a valid one (between 0 and 1)")

        # weights_log[p][w] has (weight_value, layer_index, nw_i, nw_j) of weight w in particle p
        fl1_weights= [ Weight(coords=w[1:],value=w[0]) for w in weights_log[-1] ]
        fl2_weights = [ Weight(coords=w[-4:-1],value=w[-1]) for w in fl2 ]

        jacobi_index = self.jacobi_index(fl1_weights,fl2_weights)

        fl1_aux = np.array(weights_log[-1])
        print(f"Jacobi index is {jacobi_index}, computed on {fl1_aux[:,1:]},"
              f" {fl2}",self.threshold)

        # Restore original weights to the model
        model = self._copy_location_to_weights(orig_location, model)

        return (jacobi_index < self.threshold)

    def switch_criterion_stagnation(self,log):
        """
        Implements a switch criterion based on fitness stagnation.
        """

        #
        # first one is the threshold to determine if we should switch or not
        # second one is the window to consider
        # Currently static and hardcoded, but we could implement an adaptive schedule

        if len(log) < self.window + 1:
            return False

        # The highest score
        score_list= [log[i][0] for i in range(len(log))]
        current = max(score_list)

        # The highest score up to window generations before
        score_list = score_list[0:len(score_list)-self.window]
        initial = max(score_list)

        # Compute relative improvement
        improvement = (current - initial)/initial

        print(log,initial,current,improvement)

        # Switch if improvement is higher than the threshold
        if (improvement < self.threshold):
            print(colored(f"Improvement {improvement} < {self.threshold}" ,'red') )

            # Decrease window and threshold if you stopped early
            #self.window = max(5,self.window-1)
            self.threshold *= 0.5
            print( colored(f"New window={self.window}, threshold={self.threshold}",'green'))
            return True

        return False

    def should_switch(self,log,weights_log=None,input_neg=None,model=None,
                      input_pos=None, input_pos_sampled=None,
                      output_dir=None):
        """
        Returns True iff we should stop PSO and do FL
        """

        if self.switch_criterion == 'stagnation':
            return self.switch_criterion_stagnation(log)

        if self.switch_criterion == 'localisation2':
            return self.switch_criterion_localisation(log,weights_log,
                                                      input_neg=input_neg,
                                                      model=model,
                                                      input_pos=input_pos,
                                                      input_pos_sampled=input_pos_sampled,
                                                      output_dir=output_dir
                                                      )

        return False

    def find_new_neg_pos(self,model,input_neg, input_pos,full_pos,output_dir=''):
        """
        Given a positive and negative sets, it returns new positive and negative sets.
        The new sets are:
            new_neg: inputs from repair set that are negative for model
            new_pos: inputs from input_pos that are still positive U repaired images from input_neg U unsampled positive inputs that remained positive
        """

        sample_pos_size = len(input_pos)

        # Predict both sets
        neg_predicted = model.predict(input_neg)
        pos_predicted = model.predict(input_pos)
        full_pos_predicted = model.predict(full_pos)
        neg_predicted = [np.argmax(x,axis=0) for x in neg_predicted]
        pos_predicted = [np.argmax(x, axis=0) for x in pos_predicted]
        full_pos_predicted = [np.argmax(x, axis=0) for x in full_pos_predicted]

        new_neg = [[],[]]
        new_pos = [[], []]
        fixed_neg = [[],[]]
        broken_pos = [[],[]]

        neg_true = [np.argmax(x, axis=0) for x in input_neg[1]]
        pos_true = [np.argmax(x, axis=0) for x in input_pos[1]]
        full_pos_true = [np.argmax(x, axis=0) for x in full_pos[1]]

        # Process negative set
        for i in range(len(neg_true)):

            # If prediction matches reality, the input was fixed
            if neg_true[i] == neg_predicted[i]:
                fixed_neg[0].append( input_neg[0][i] )
                fixed_neg[1].append(input_neg[1][i])
            else:
                # If prediction is wrong, the input is still negative. Keep it in negative set
                new_neg[0].append(input_neg[0][i])
                new_neg[1].append(input_neg[1][i])

        # Process positive set
        for i in range(len(pos_true)):

            # If prediction matches reality, the input is still positive, keep it in positive set
            if pos_true[i] == pos_predicted[i]:
                new_pos[0].append(input_pos[0][i])
                new_pos[1].append(input_pos[1][i])
            elif pos_true[i] == neg_true[0]:
                # If prediction is wrong, the input is now negative. Put it in negative set, but only
                # if the true label matches the class you are repairing
                broken_pos[0].append(input_pos[0][i])
                broken_pos[1].append(input_pos[1][i])

        # Process the set of all positive inputs (non-sampled)
        for i in range(len(full_pos_true)):

            # If prediction is wrong, the input is now negative. Put it in negative set, but only
            # if the true label matches the class you are repairing
            if full_pos_true[i] != full_pos_predicted[i] and full_pos_true[i] == neg_true[0]:
                broken_pos[0].append(full_pos[0][i])
                broken_pos[1].append(full_pos[1][i])
            elif len(new_pos[0])+len(fixed_neg[0]) < sample_pos_size:  # if prediction was right, replenish pos_inputs
                new_pos[0].append(full_pos[0][i])
                new_pos[1].append(full_pos[1][i])

        new_neg_images = np.array(new_neg[0] + broken_pos[0])
        new_pos_images = np.array(new_pos[0] + fixed_neg[0])
        new_neg_labels = np.array( new_neg[1] + broken_pos[1] )
        new_pos_labels = np.array( new_pos[1] + fixed_neg[1] )

        print([new_neg_images.shape,new_neg_labels.shape],[new_pos_images.shape,new_pos_labels.shape])

        #self.sampled_positive = [new_pos_images,new_pos_labels]

        print(f"Positive inputs: {len(self.sampled_positive[0])} in total")

        return [new_neg_images,new_neg_labels],[new_pos_images,new_pos_labels]


    def _criterion(self, model, location, input_pos, input_neg):
        """Compute fitness.

        :param model:  subject DNN model
        :param location: consists of a neural weight value to mutate,
                          an index of a layer of the model,
                          and a neural weight position (i, j) on the layer
        :param input_pos: positive inputs sampled
        :param input_neg: negative inputs targeted
        :return: fitness, n_patched and n_intact
        """
        # Keep original weights and set specified weights on the model
        orig_location = np.copy(location)
        orig_location = self._copy_weights_to_location(model, orig_location)
        model = self._copy_location_to_weights(location, model)

        if not self.risk_aware:  # use default Arachne
            # "N_{patched} is the number of inputs in I_{neg}
            # whose output is corrected by the current patch"
            loss_input_neg, acc_input_neg = \
                model.evaluate(input_neg[0],
                               input_neg[1],
                               verbose=0,
                               batch_size=self.batch_size)
            n_patched = int(np.round(len(input_neg[1]) * acc_input_neg))

            # "N_{intact} is th number of inputs in I_{pos}
            # whose output is still correct"
            loss_input_pos, acc_input_pos = \
                model.evaluate(input_pos[0],
                               input_pos[1],
                               verbose=0,
                               batch_size=self.batch_size)
            n_intact = int(np.round(len(input_pos[1]) * acc_input_pos))

            if self.normalized_fitness:
                fitness = (n_patched + 1) / (loss_input_neg + 1) / len(input_neg[0]) + \
                          (n_intact + 1) / (loss_input_pos + 1) / len(input_pos[0])
            else:

                fitness = (n_patched + 1) / (loss_input_neg + 1) + \
                      (n_intact + 1) / (loss_input_pos + 1)

        else:  # modify arachne to take into account risk levels into the fitness function

            # Session needed to convert to numpy with eager mode disabled
            sess = K.get_session()
            sess.as_default()

            requirements = {
                (6, 6): {'neg_amount': 0, 'pos_amount': 0, 'name': 'classif_person', 'risk': 3},
                (2, 2): {'neg_amount': 0, 'pos_amount': 0, 'name': 'classif_car', 'risk': 2},
                (0, 0): {'neg_amount': 0, 'pos_amount': 0, 'name': 'classif_bike', 'risk': 2},
                (7, 7): {'neg_amount': 0, 'pos_amount': 0, 'name': 'classif_rider', 'risk': 1},
                (6, -1): {'neg_amount': 0, 'pos_amount': 0, 'name': 'misclas_person', 'risk': 3},
                (2, 7): {'neg_amount': 0, 'pos_amount': 0, 'name': 'misclas_car_rider', 'risk': 3},
                (7, -1): {'neg_amount': 0, 'pos_amount': 0, 'name': 'misclas_rider', 'risk': 2},
                (2, 12): {'neg_amount': 0, 'pos_amount': 0, 'name': 'misclas_car_truck', 'risk': 2},
                (0, -1): {'neg_amount': 0, 'pos_amount': 0, 'name': 'misclas_bike', 'risk': 2},
                (6, 7): {'neg_amount': 0, 'pos_amount': 0, 'name': 'misclas_person_rider', 'risk': 1},
                (7, 6): {'neg_amount': 0, 'pos_amount': 0, 'name': 'misclas_rider_person', 'risk': 1},
                (3, 6): {'neg_amount': 0, 'pos_amount': 0, 'name': 'misclas_motor_person', 'risk': 1}}
            extra_risks = {'w1': 1, 'w2': 0.5, 'wa_total': 8, 'ws_total': 15}

            neg_predictions = \
                model.predict(input_neg[0],
                              verbose=0)
            pos_predictions = \
                model.predict(input_pos[0],
                              verbose=0)

            neg_predictions = np.argmax(neg_predictions, axis=1)
            pos_predictions = np.argmax(pos_predictions, axis=1)
            neg_classes = np.argmax(input_neg[1], axis=1)
            pos_classes = np.argmax(input_pos[1], axis=1)

            classes_amount = {}
            predicted_amount = {}

            # for each pair of [label, prediction], count how many instances we have
            def count(inputs, classes, predictions):
                for i in range(len(classes)):
                    clas = classes[i]
                    prediction = predictions[i]
                    pair = (clas, prediction)
                    if clas not in classes_amount:
                        classes_amount[clas] = 0
                    classes_amount[clas] += 1
                    if prediction not in predicted_amount:
                        predicted_amount[prediction] = 0
                    predicted_amount[prediction] += 1

                    if pair in requirements:
                        requirements[pair][inputs + '_amount'] += 1
                    if pair[0] in [6, 7, 0] and pair[0] != pair[1]:
                        requirements[(pair[0], -1)][inputs + '_amount'] += 1

            count('pos', pos_classes, pos_predictions)
            count('neg', neg_classes, neg_predictions)

            # for each [label, prediction] we care about how many instances we had of that (miss)classification
            # as a percentage, so divided by the total amount of images of that label
            # we've tried using the loss function to avoid jumps in the fitness function, but it got worse results
            for pair in requirements:
                if pair[0] == pair[1]:  # case w1, precision of a class C, we divide by all images predicted as C
                    denominator = predicted_amount[pair[0]] + 1
                else:  # case w2, misclassification rate of a type of misclassification c1 -> c2
                    denominator = classes_amount[pair[0]] + 1
                requirements[pair]['total'] = ((requirements[pair]['neg_amount'] + requirements[pair][
                    'pos_amount'] + 1)
                                               * requirements[pair]['risk']) / denominator

            fitness = 0
            fitness -= ((requirements[(6, -1)]['total'] / extra_risks['ws_total']) +
                        (requirements[(2, 7)]['total'] / extra_risks['ws_total']) +
                        (requirements[(7, -1)]['total'] / extra_risks['ws_total']) +
                        (requirements[(2, 12)]['total'] / extra_risks['ws_total']) +
                        (requirements[(0, -1)]['total'] / extra_risks['ws_total']) +
                        (requirements[(6, 7)]['total'] / extra_risks['ws_total']) +
                        (requirements[(7, 6)]['total'] / extra_risks['ws_total']) +
                        (requirements[(3, 6)]['total'] / extra_risks['ws_total']))

            n_patched = None  # these are never used anyway
            n_intact = None

            """  w1 * (
                  (wa1/wa_total) * <The precision value of classification of person> + 
                  (wa2/wa_total) * <The precision value of classification of car> +
                  (wa2/wa_total) * <The precision value of classification of bike> +
                  (wa3/wa_total) * <The precision value of classification of rider> )
                 -
                 w2 * (
                  (ws1/ws_total) * <The rate of misclassification of person to another> +
                  (ws1/ws_total) * <The rate of misclassification of car to rider> +
                  (ws2/ws_total) * <The rate of misclassification of rider to another> +
                  (ws2/ws_total) * <The rate of misclassification of car to truck> +
                  (ws2/ws_total) * <The rate of misclassification of bike to another> +
                  (ws3/ws_total) * <The rate of misclassification of person to rider> +
                  (ws3/ws_total) * <The rate of misclassification of rider to person> +
                  (ws3/ws_total) * <The rate of misclassification of motor to person> )
                 where w1=1, w2=0.5, wa1=3, wa2=2, wa3=1, ws1=3, ws2=2, ws3=1,
                  wa_total = wa1+w2+w2+w3, ws_total = ws1+ws1+ws2+ws2+ws2+ws3+ws3+ws3.
                 """

        # Restore original weights to the model
        model = self._copy_location_to_weights(orig_location, model)

        return fitness, n_patched, n_intact

    def old_optimize(self,
                 model,
                 dataset,
                 model_dir,
                 weights,
                 input_neg,
                 input_pos,
                 output_dir,
                 verbose=1):
        """Optimize.

        cf. https://qiita.com/sz_dr/items/bccb478965195c5e4097

        :param model: a DNN model to repair
        :param weights: a set of neural weights to target for repair
        :param input_neg: dataset for unexpected behavior
        :param input_pos: dataset for correct behavior
        :param output_dir:
        :param verbose: Log level

        :return: model repaired
        """
        # Initialize particle positions.
        # locations[p][w] has (weight_value, layer_index, nw_i, nw_j) of weight w in particle p
        locations = self._get_initial_particle_positions(weights,
                                                         model,
                                                         self.num_particles)

        # "The initial velocity of each particle is set to zero"
        velocities = np.zeros((self.num_particles, len(weights)))

        # Compute velocity bounds
        velocity_bounds = self._get_velocity_bounds(model)

        # "We sample 200 positive inputs"
        input_pos_sampled = self._sample_positive_inputs(input_pos)

        # Initialize for PSO search
        personal_best_positions = list(locations)  # position i has the best configuration particle i ever found
        personal_best_scores = self._initialize_personal_best_scores(locations, model,
                                                                     input_pos_sampled,
                                                                     input_neg)  # pos i has best score particle i ever got
        best_particle = np.argmax(np.array(personal_best_scores)[:, 0])  # the particle that got the best score ever
        global_best_position = personal_best_positions[best_particle]  # best configuration ever found
        self.create_optimization_history(output_dir, len(weights),
                                         file_name=f'optimization_history_{self.current_macrogen}.csv')

        # Search
        history = []
        best_particle_locations=[]

        # "PSO uses ... the maximum number of iterations is 100"
        for t in range(self.num_iterations):

            self.iterations_counter += 1

            g = self._get_weight_values(global_best_position)
            current_scores = [0] * self.num_particles

            print(f'\nUpdating particle positions: {self.iterations_counter }/{self.num_iterations}')

            # "PSO uses a population size of 100"
            for n in range(self.num_particles):

                print(f"\rUpdated particles %i/{self.num_particles}" % n, end="")

                new_weights, new_v, score, n_patched, n_intact = \
                    self._update_particle(locations[n],
                                          velocities[n],
                                          velocity_bounds,
                                          personal_best_positions[n],
                                          g,
                                          model,
                                          input_pos_sampled,
                                          input_neg)

                # Update position
                locations[n] = new_weights
                # Update velocity
                velocities[n] = new_v
                # Update score
                if personal_best_scores[n][0] < score:
                    personal_best_scores[n] = [score, n_patched, n_intact]
                    personal_best_positions[n] = locations[n]
                current_scores[n] = score

            # Update global best
            best_particle = np.argmax(np.array(personal_best_scores)[:, 0])
            global_best_position = personal_best_positions[best_particle]
            # add current best
            history.append(personal_best_scores[best_particle])
            best_particle_locations.append(global_best_position)

            self.update_optimization_history(locations, output_dir, t, personal_best_scores, current_scores,
                                             file_name=f'optimization_history_{self.current_macrogen}.csv')

            # Stop earlier
            if (not self.fixed_stop) and \
                    self.should_switch(history,best_particle_locations,
                                       model=model,input_neg=input_neg,
                                       input_pos=input_pos,
                                       input_pos_sampled=input_pos_sampled,
                                       output_dir=output_dir): # NEW: stopping condition
                print( colored(f'Optimization stopped by {self.switch_criterion}. SWITCHING','red') )
                self.switch_log.append(self.iterations_counter)
                break

            if self.iterations_counter == self.num_iterations:
                break

        # Reduce window if you made it to the end of optimization
        if (t == self.num_iterations):
            self.window = int(self.window * 0.75)
            self.threshold *= 1.2
            print(colored(f"New window={self.window}, threshold={self.threshold}", 'green'))

        self._append_weights(model, weights)
        model = self._copy_location_to_weights(global_best_position, model)
        self._append_weights(model, weights)
        self.save_weights(weights, output_dir)
        self.finish_optimization_history(self.num_iterations + 1, best_particle, personal_best_scores[best_particle],
                                         locations[best_particle], output_dir,
                                         file_name=f'optimization_history_{self.current_macrogen}.csv')

        self._output_repaired_model(output_dir, model)
        self._log_optimize(global_best_position, verbose)

        return model, global_best_position

    def kaminari_reduce_model(self, model, input_pos, input_neg, weights):
        self.original_complete_model = model

        # Preprocess the images
        new_neg, new_pos = self.kaminari.save_processed_images(
            model,
            input_neg,
            input_pos
        )

        input_shape = [new_neg[0].shape[1]]
        # Extract the tail submodel
        submodel = self.kaminari.extract_tail(model)
        self.original_reduced_model = submodel

        if weights is not None:
            # Translate the localized weights' coordinates to
            # deal with the new shrinked model
            new_weights = self.kaminari.translate_weights(weights)
        else:
            new_weights = None

        return submodel, new_pos, new_neg, new_weights

    def kaminari_setup(self, model, input_pos, input_neg, weights):
        if not self.kaminari_depth == 0:
            self.kaminari = KaminariUtils(self.kaminari_depth)
            model, input_pos, input_neg, weights = self.kaminari_reduce_model(model, input_pos, input_neg, weights)
        else:  # depth 0 means use the whole model, ignore kaminari
            self.original_complete_model = model  # we don't change the model
            self.original_reduced_model = model  # we don't change the model
        return model, input_pos, input_neg, weights

    def optimize(self,
                 model,
                 dataset,
                 model_dir,
                 weights,
                 input_neg,
                 input_pos,
                 output_dir,
                 verbose=1):

        self.model_dir = model_dir

        # Data structures for logging purpose
        log = []
        weight_log = []
        best_score = 0
        best_model = model

        if self.kaminari_depth is None:
            # "only considers the neural weights connected
            # to the final output layer"
            self.kaminari_depth = len(model.layers) - 2
        model, input_pos, input_neg, weights = self.kaminari_setup(model, input_pos, input_neg, weights)

        # Model and faulty weights variables to use
        intermediate_model = model
        old_target_layer = self.target_layer
        faulty_weights = Arachne.localize(self,model,input_neg)
        self.target_layer = old_target_layer

        # Iterate for n macrogenerations
        for macrogen in range(self.num_macroiterations):

            self.current_macrogen = macrogen
            print( colored(f"MACROGENERATION {macrogen}",'red') )

            print( colored(f"Optimization: {macrogen}",'green') )

            # Run PSO
            intermediate_model, best_position = self.old_optimize(
                intermediate_model,
                self.dataset,
                None,
                faulty_weights,
                input_neg,
                input_pos,
                output_dir,
                verbose
            )

            if self.iterations_counter == self.num_iterations:
                break

            # Find a new negative and positive set
            interm_input_neg, interm_input_pos = self.find_new_neg_pos(intermediate_model, input_neg,self.sampled_positive,input_pos,output_dir)

            print(f"Recomputing negative and positive sets... Found {len(interm_input_neg[0])} and {len(interm_input_pos[0])}.")

            if len(interm_input_neg[0])==0:
                print(colored('No more negative inputs. Aborting.','red'))
                break

            print(colored(f"Fault localization: {macrogen}", 'green'))

            # Localize new Pareto front
            old_target_layer = self.target_layer
            faulty_weights = Arachne.localize(
                self,
                intermediate_model,
                interm_input_neg
            )
            self.target_layer = old_target_layer

            # Compute and save fitness
            fitness = self.eval_fitness(intermediate_model,input_neg,input_pos)
            log.append(fitness)
            weight_log.append(faulty_weights)
            print( colored(f"Best fitness {fitness}",'red'))

            if fitness > best_score:
                best_model=intermediate_model
                best_score=fitness
                print( colored("New highscore!!!",'red'))

        best_position = self.kaminari.de_translate_weights(best_position)
        best_model = self._copy_location_to_weights(best_position, self.original_complete_model)
        self.save_fitness_log(output_dir,log)
        self._output_repaired_model(output_dir,best_model)
        return intermediate_model

    def save_fitness_log(self,output_dir,log,weights_log=[]):

        path = Path(output_dir).joinpath('fitness.log')

        f = open(path,'w')

        # Write header
        f.write(f"Method: Adaptive\n")
        f.write(f"Dataset: {self.dataset.name}\n")
        f.write(f"Particles: {self.num_particles}\n")
        f.write(f"Macrogenerations: {self.num_macroiterations}\n")
        f.write(f"Switching by: {self.switch_criterion}\n")
        f.write(f"Threshold:{self.threshold}; Window:{self.window}\n")
        f.write(f"Switched at generations: {self.switch_log}\n")
        f.write(f"Found Pareto fronts: {weights_log}\n")
        f.write("#"*20+"\n\n")

        for i in range(len(log)):

            f.write(f'Macrogeneration {i}: {log[i]}\n')

        f.close()