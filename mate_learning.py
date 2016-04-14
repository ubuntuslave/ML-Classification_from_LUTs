'''
@summary:   Supervised Learning method for binary classification based on mating (grouping of values) across the supplied data set of components and classification labels.
            The algorithm continues to exhaust all the possibilities for as long as accuracy of correct classification can be increased.

@author: Juan Pablo Munoz
@author: Carlos Jaramillo

Machine Learning Course at CUNY Graduate Center - Final Project - Fall 2014 - Prof. Robert Haralick
'''

from __future__ import division
from __future__ import print_function

import numpy as np
import itertools
import time

class MateFinder(object):

    '''
    Used for the particular binary classification problem based on mates (usually groups of 2 - pairs) across each component of a discrete-valued data set.
    This class provides procedures such as the removal of irrelevant components (selection of relevant components),
    as well as essential learning (training and validation) and testing procedures from learned decision tables of pairs.

    Once a MateFinder object has been instantiated through the constructor, the interfacing methods employed by an external classifier are:
    learn_by_mate_discovery() and compute_prediction_accuracy()
    '''

    def __init__(self, component_data, classification_labels, mating_size=2):
        '''
        Constructor for the PairFinder object out of the given data set.
        @summary: The data set (and corresponing classification labels) are split into the following portions:
            50% for learning phase:  25% training + 25% validation
            50% for testing phase:  the real test using the learned table of decision rules

            This constructor initializes the reusable member variables that are called from other functions.

        @param component_data: A 2D matrix (numpy ndarray) of measurements (observations) to work with. A copy is created of this.
        @param classification_labels: The category classification_labels as a 1D ndarray. Expected to be nonnegative integers
        @param mating_size: It should default to 2 (for pairs), but as in life, mating (a.k.a grouping) can be done using larger number of members.
        Note, the number of values (10 digits for this exercise) must be divisible by the mating_size. For now, we basically can have only groups of 2 or 5.
        '''

        self.all_data = np.copy(component_data)
        self.all_classes = classification_labels
        self.num_of_instances = self.all_data.shape[0]
        self.num_of_components = self.all_data.shape[1]
        self.num_of_values = 10  # digits
        self.mating_size = mating_size
        self.num_of_mating_groups = self.num_of_components // self.mating_size
        self.map_base = np.repeat(np.arange(self.num_of_mating_groups), repeats=mating_size)  # produces [0,0,1,1,2,2,3,3,4,4]

        N = self.num_of_instances
        # Get 50% for training:
        N_split = N // 2
        self.feat_data_learning, self.class_learning = self.all_data[:N_split], self.all_classes[:N_split]
        self.feat_data_test, self.class_test = self.all_data[N_split:], self.all_classes[N_split:]

        N_split_train_validate = N_split // 2
        self.feat_data_learning_train = self.feat_data_learning[:N_split_train_validate]
        self.class_learning_train = self.class_learning[:N_split_train_validate]
        self.feat_data_learning_validate = self.feat_data_learning[N_split_train_validate:]
        self.class_learning_validate = self.class_learning[N_split_train_validate:]
        self.N_train = N_split_train_validate

        # Segregate training data by class:
        self.c0_indices_train = np.where(self.class_learning_train == 0)[0].reshape(-1, 1)
        self.c1_indices_train = np.where(self.class_learning_train == 1)[0].reshape(-1, 1)
        self.feat_train_c0 = np.take(self.feat_data_learning_train[:, 0], self.c0_indices_train)
        self.feat_train_c1 = np.take(self.feat_data_learning_train[:, 0], self.c1_indices_train)
        for c in range(1, self.feat_data_learning_train.shape[1]):
            self.feat_train_c0 = np.hstack((self.feat_train_c0, np.take(self.feat_data_learning_train[:, c], self.c0_indices_train)))
            self.feat_train_c1 = np.hstack((self.feat_train_c1, np.take(self.feat_data_learning_train[:, c], self.c1_indices_train)))

        self.values_list = np.arange(self.num_of_values)

        mates_candidates = list(itertools.combinations(self.values_list, self.mating_size))
        all_groupings = np.array(list(itertools.combinations(mates_candidates, self.num_of_mating_groups)))
        self.mates_candidates_per_component = self.validate_groupings(all_groupings)

        # The address LUT multiplication factor (base)
        self.address_multiples = np.array([self.num_of_mating_groups ** i for i in range(self.num_of_components)]).reshape(-1, 1)


    def validate_groupings(self, candidate_lists):
        '''
        It removes overlapping groupings (matings) among the list of groupings provided.

        @param candidate_lists: An ndarray of all possible groupings (in our particular exercise, these are all possible combinations of 2 out of 10 digits)

        @return: the filtered list of non-overlapping groupings. The reduction in length for this list is crucial for faster results during learning procedure.
        '''

        print("Removing overlapping groupings...", end="")
        t_start = time.clock()
        grouping_size = len(candidate_lists[0].ravel())  # A constant length
        skimmed_list = []
        for candidate in candidate_lists:
            c_unraveled = candidate.ravel()
            if len(np.unique(c_unraveled)) == grouping_size:  # It means no overlaps exist, so it's a valid grouping
                skimmed_list.append(c_unraveled)  # Provide unraveled list instead of the array of groups (the candidate)
        t_end = time.clock()
        t_elapsed = t_end - t_start
        print("done! Elapsed: %.9f s" % (t_elapsed))

        return np.array(skimmed_list)

    def learn_by_mate_discovery(self):
        '''
        Brute force discovery of mates and relevant components during the training phase so that accuracy is the highest among all possibilities.

        @return The 1-d numpy array (table) of learned rules.
        @return List of best combination of mates (pairs) across the discovered relevant components
        @return The list of discovered relevant components from the training phase
        '''

        best_accuracy = 0
        best_matings = None
        relevant_components = None

        min_num_of_relevant_comps = 1
        for i in range(min_num_of_relevant_comps, self.num_of_components + 1):
            t_start = time.clock()
            current_accuracy, current_matings, comps = self.select_K_components(k_comps=i, verbose=True)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_matings = current_matings
                relevant_components = comps
            else:  # When accuracy starts to decrease, it's time to stop because we have found the plateau concerning the number of relevant components.
                break

        t_end = time.clock()
        t_elapsed = t_end - t_start
        print("\nElapsed: %.9f s" % (t_elapsed))
        print("TRAINED Accuracy=%.2f%s with %d Selected best features:" % (best_accuracy * 100., "%", i), relevant_components)
        print("Using matings:", best_matings)

        decision_rules = self.generate_decision_rules_LUT(best_matings, relevant_components)

        return decision_rules, best_matings, relevant_components

    def select_K_components(self, k_comps, verbose=True):
        '''
        Find the k number of components that produce the highest prediction accuracy from the training procedure:

        Learning of rules is performed on the training portion of the data set (technically, the 25% of the entire data set)
        Accuracy validation to choose best is performed on the validation data set (technically, the other 25% of the entire data set)

        @param k_comps: Indicates the number of components to be attempted the "brute-force" search upon.

        @return The accuracy (normalized value from 0 to 1.0) obtained using the best selection of k components
        @return List of best combination of mates (pairs) across the best selected components
        @return The list of the best selected components
        '''

        components_list = range(self.num_of_components)

        best_accuracy = 0
        best_matings = None
        best_comps = None

        print("Performing %d-component selection" % (k_comps))
        t_start = time.clock()

        comps_combs = list(itertools.combinations(components_list, k_comps))
        for matings in self.mates_candidates_per_component:
            for subset in comps_combs:
                decisions_LUT = self.generate_decision_rules_LUT(matings, subset)
                current_accuracy = self.compute_prediction_accuracy(matings, decisions_LUT, subset, self.feat_data_learning_validate, self.class_learning_validate, verbose=False)

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_matings = matings
                    best_comps = subset

            print(".", end="")

        t_end = time.clock()
        t_elapsed = t_end - t_start
        if verbose:
            print("\nAccuracy =  %.2f%s" % (best_accuracy * 100., "%"), "with matings:", best_matings)
            print("\tand components:", best_comps)
            print("\tIt took %.9f s" % (t_elapsed))

        return best_accuracy, best_matings, best_comps

    def _generate_training_LUT(self, cols):
        '''
        Generates an empty look-up table that will be addressed to train the learned with the given data.

        @param cols: the number of columns (components) in the data set that are being used for the training.

        @return: An empty 2-dimensional matrix that can be addressed while performing decision statistics
        '''
        len_of_table = self.num_of_mating_groups ** cols
        LUT = np.zeros((len_of_table, 2))
        return LUT

    def get_address(self, matings, component_data):
        '''
        Resolves the address in the training LUT pertaining the given matings (a list of candidate matches) on the data set portion provided.

        @param matings: List of indices associated with the groupings (matches)
        @param component_data: The ndarray of component data for which addresses are resolved for

        @return: the numpy n-dimensional array of resolved addresses for the provided data and desired matings
        '''

        address = np.empty_like(component_data, dtype="uint8")

        # Get value indices in matings array
        value_indices = np.zeros_like(matings)  # + self.values_list

        # TODO: vectorize this code (without using loops, they are too slow!)
        for i in self.values_list:
            value_indices[i] = np.where(matings == i)[-1]

        # semi-vectorized address look-up
        for comp in range(component_data.shape[1]):
            address[:, comp] = self.map_base[value_indices[component_data[:, comp]]]

        return address

    def get_index_address_table(self, address):
        '''
        This is a vectorized function that resolves the index (a.k.a. hash address) on the 1-D array of rules
        employing in the learning/testing procedure).

        @param address: An array of address arrays to be hashed for on a 1-D table using the pertaining base (such as base 5 when dealing with groups of 2)

        @return: the resolved 1-D array of indices corresponding the input address.
        '''
        hashed_address = np.dot(address, self.address_multiples[:len(address[-1])])  # Nice!
        return hashed_address

    def generate_decision_rules_LUT(self, matings, comp_subset):
        '''
        A decision rules are learned from the provided list of matings (non-overlapping groups) and the chosen components

        @param matings: An ordered list of indices used to group employing the map_based of choice set in the constructor.
        By default, the mapping is pair-wise ordered from left to right.
        @param comp_subset: The list of selected component indices upon where the decision table is generated

        @return: The 1-D table of decision rules extracted from frequency analysis out of the learning portion of the data set
        '''
        k_comps = len(comp_subset)
        learning_LUT_of_probs = self._generate_training_LUT(k_comps)  # For relative statistics work

        addresses_0 = self.get_address(matings, self.feat_train_c0[:, comp_subset])  # Mapped addresses
        address_LUT_0 = self.get_index_address_table(addresses_0)
        addresses_1 = self.get_address(matings, self.feat_train_c1[:, comp_subset])  # Mapped addresses
        address_LUT_1 = self.get_index_address_table(addresses_1)

        # Count per catagory addresses and fill the table
        count_c0 = np.bincount(address_LUT_0[:, 0])
        count_c1 = np.bincount(address_LUT_1[:, 0])

        learning_LUT_of_probs[:len(count_c0), 0] = count_c0
        learning_LUT_of_probs[:len(count_c1), 1] = count_c1
        # Normalize
        learning_LUT_of_probs = learning_LUT_of_probs / self.N_train

        # Fill up decision rules LUT
        decision_rules_LUT = np.argmax(learning_LUT_of_probs, axis=-1)
        # WISH: For now, the LUT will get assigned zeros if there are any ties! which may appear biased towards class 0!

        return decision_rules_LUT

    def compute_prediction_accuracy(self, matings, decisions_LUT, relevant_components=None, comp_data_test=None, class_test=None, verbose=False):
        '''
        The accuracy of correct classification using the provided decision rules is computed statistically.

        @param matings: An ordered list of indices used to group employing the map_based of choice set in the constructor.
        By default, the mapping is pair-wise ordered from left to right.
        @param decisions_LUT: The learned rules in a LUT
        @param relevant_components: By default is set to None, so all components will be used. Otherwise, this specifies the selection of components to work with.
        @param comp_data_test: If any, this is the component data set to use for predicting upon and testing the classification accuracy with the provided matings and decision rules. Otherwise, it defaults to using the test data set (the 50% of the entire data set passed in the constructor)
        @param class_test: The classification labels associated with comp_data_test if any.
        @param verbose: Indicates whether to enable message verbosity.

        @return: The normalized accuracy of correct classification computed according to the specified parameters.
        '''
        if comp_data_test == None:
            observations, labels_true = self.feat_data_test, self.class_test
        else:
            observations, labels_true = comp_data_test, class_test

        if relevant_components != None:
            observations = observations[:, relevant_components]

        N = observations.shape[0]

        predictions = self.predict_category(matings, decisions_LUT, observations)
        labels_true = labels_true.reshape(*predictions.shape)
        errors = predictions - labels_true
        correct_count = N - np.count_nonzero(errors)
        prob_correct = correct_count / float(N)
        if verbose:
            print("Accuracy =  %.2f%s" % (prob_correct * 100., "%"), "with matings:", matings)

        return prob_correct

    def predict_category(self, matings, decisions_LUT, instance_vector):
        '''
        With the given list of matings (groupings) and the decision rules table, the list of measurement vectors (instances) are classified accordingly.

        @param matings: An ordered list of indices used to group employing the map_based of choice set in the constructor.
        By default, the mapping is pair-wise ordered from left to right.
        @param decisions_LUT: The learned rules in a LUT
        @param instance_vector: The list of measurement vectors (instances, a.k.a observations) to classify.

        @return: The classification results corresponding to the supplied instance vectors for the specified rules and matings.
        '''
        addresses = self.get_address(matings, instance_vector)  # Hashed address
        address_LUT = self.get_index_address_table(addresses)
        return decisions_LUT[address_LUT]


