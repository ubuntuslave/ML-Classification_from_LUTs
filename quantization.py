'''
Created on Dec 18, 2014

@author: Carlos Jaramillo
@author: Pablo Munoz
'''

from __future__ import division
from __future__ import print_function

import numpy as np

class Quantizer(object):
    '''
    Quantizes a 2D matrix (numpy ndarray) of measurements (observations)
    '''

    def __init__(self, component_data, classification_labels, min_level_per_comp, max_level_per_comp, **kwargs):
        '''
        Constructor for the quantizer out of the given component_data set

        @param component_data: A Copy of the 2D matrix (numpy ndarray) of measurements (observations) to quantize about
        @param classification_labels: The category classification_labels as a 1D ndarray. Expected to be nonnegative integers
        @param min_level_per_comp: The minimum number of levels per component allowed
        @param max_level_per_comp: The maximum number of levels per component allowed
        '''

        self.all_data = np.copy(component_data)
        self.category_data = classification_labels
        self.num_of_observations = self.all_data.shape[0]
        self.num_of_components = self.all_data.shape[1]

        self.max_level_per_comp = max_level_per_comp
        self.min_level_per_comp = min_level_per_comp
#         self.entropy_change_thresh = kwargs.get("entropy_change_thresh", 0.5)

        # Initialize reference table of sorted components
        self.init_sorted_reference_data()

        # Initialize the measurement dimension components_c0 as a list of objects
        self.components_bounds = np.empty((self.num_of_components), dtype=object)  # [None] * self.num_of_components

        # Initiliaze to a small number of  uniform level boundaries
        self.set_boundaries_of_uniform_density(levels_list=[self.min_level_per_comp] * self.num_of_components)

    def init_sorted_reference_data(self):
        self.sorted_data = np.zeros_like(self.all_data)
        for c in range(self.num_of_components):
            self.sorted_data[:, c] = self.all_data[self.all_data[:, c].argsort(), c]

    def set_boundaries_of_uniform_density(self, levels_list):
        '''
        Split based on density of count per level (using sorted components vectors)

        @param use_uniform_density: To have the same number of data points per level in the component
        '''
        N = self.num_of_observations
        for col in range(len(levels_list)):
            K = levels_list[col]
            self.components_bounds[col] = np.zeros((K, 2))

            right_idx = 0  # Used only for the leftmost boundary at first
            for k in range(K):
                left_idx = right_idx
                right_idx = (k + 1) * np.floor(N / K) - 1
                # print("(", left_idx, ",", right_idx, ")")

                self.components_bounds[col][k, :] = [self.sorted_data[left_idx, col], self.sorted_data[right_idx, col]]
            # The first left boundary must be negative infinity (or a very small number)
            self.components_bounds[col][0, 0] = -np.infty
            # The last right boundary must be infinity (or a large number)
            self.components_bounds[col][-1, -1] = np.infty

        self.max_num_levels = np.max(levels_list)
        self.current_num_bins = self.get_number_of_bins()

    def set_boundaries_brute_force(self, levels_list):
        '''
        Split based on density of count per level (using sorted components vectors)

        @param use_uniform_density: To have the same number of data points per level in the component
        '''
        for col in range(len(levels_list)):
            min_col_value = levels_list[col].min
            max_col_value = levels_list[col].max

            K = levels_list[col]
            self.components_bounds[col] = np.zeros((K, 2))

            # TODO: not implemented yet

        self.max_num_levels = np.max(levels_list)
        self.current_num_bins = self.get_number_of_bins()

    def get_number_of_bins(self):

        current_num_bins = 1
        for comp in  self.components_bounds:
            current_num_bins = current_num_bins * len(comp)

        return current_num_bins

    def optimize_num_levels(self, feat_data=None, cat_labels=None, num_of_candidates=1, is_discrete=False, use_visualization=False, verbose=False):
        '''
        Brute force optimization to find the number of levels on each component

        @return: queue of best levels
        '''
        # Queue with max length
        from collections import deque

        q = deque(maxlen=num_of_candidates)

        if feat_data == None:
            feat_data = self.all_data
            cat_labels = self.category_data

        best_accuracy = 0  # Initial accuracy
        # Generate permutation table for level quantities

        max_num_of_bins = self.max_level_per_comp ** self.num_of_components
        levels_LUT = np.zeros((max_num_of_bins, self.num_of_components), dtype='int8')
        indices = np.arange(max_num_of_bins)
        # Break up the indices in to its digits
        base_multiple = max_num_of_bins
        for comp in range(self.num_of_components):
            levels_LUT[:, comp] = (indices % base_multiple) // int(base_multiple / self.max_level_per_comp)
            base_multiple = base_multiple / self.max_level_per_comp

        if is_discrete:
            boundary_setter = self.set_boundaries_brute_force  # Tries all possible permutations with the given levels
        else:
            boundary_setter = self.set_boundaries_of_uniform_density

        decider = DecisionRule(self.all_data, self.category_data, use_visualization)
        for levels in levels_LUT:
            # Hack: skipping indices with zero elements for now
            if np.all(levels >= self.min_level_per_comp):
                boundary_setter(levels_list=levels)
                decider.fill_likelihood_LUT_bins(self)
                # NOTE: prediction accuracy must happen with a different dataset
                new_accuracy = decider.predict_batch(feat_data, cat_labels, verbose=verbose)
                if new_accuracy > best_accuracy:
                    best_accuracy = new_accuracy
                    best_levels = levels
                    q.append(best_levels)

        # Set optimal number of levels quantizer state on components
        self.set_boundaries_of_uniform_density(levels_list=best_levels)
        return q

    def test_entropy_in_components(self):
        '''
        Test using uniform levels
        '''
        for comp_c0 in self.components_c0:
            comp_c0.get_entropy()

        for comp_c1 in self.components_c1:
            comp_c1.get_entropy()


    def get_address(self, meas_data):
        address = np.ndarray((len(meas_data), self.num_of_components), dtype="uint8")

        # vectorized address look-up
        # TODO: needs true vectorization, but it's hard due to the non-uniformity of the internal arrays sizes
        for i in range(self.num_of_components):
#             address[i] = np.where(np.logical_and(self.components_bounds[i][:, 0] <= meas_data[i], meas_data[i] < self.components_bounds[i][:, 1]))
#             address[:, i] = np.searchsorted(self.components_bounds[i][1:, 0], meas_data[:, 0], side="right")  # Don't compare to the left of -infty!
            address[:, i] = np.searchsorted(self.components_bounds[i][1:, 0], meas_data[:, i], side="right")  # Don't compare to the left of -infty!

        return address


    def slide_discrete_boundary(self, feat_data, cat_labels, use_visualization=False, verbose=False):
        # Randomly choose a component j
        random_component_index = np.random.random_integers(0, self.num_of_components - 1)
        # Randomly choose quantizing interval k (excluding the last interval because right boundaries are what will be optimized)
        random_quantizing_interval = np.random.random_integers(0, len(self.components_bounds[random_component_index]) - 2)

        # Initialize the new boundary with the perturbation
        best_boundary = self.components_bounds[random_component_index][random_quantizing_interval, 1]  # Target as the right boundary of the level
        lim_bound_low = self.components_bounds[random_component_index][random_quantizing_interval, 0]
        lim_bound_high = self.components_bounds[random_component_index][random_quantizing_interval + 1, 1]

        # Current best accuracy (Without perturbation)
        decider = DecisionRule(self.all_data, self.category_data, use_visualization)
        decider.fill_likelihood_LUT_bins(self)
        best_accuracy = decider.predict_batch(feat_data, cat_labels, verbose=False)
        initial_accuracy = best_accuracy

        # TODO: change implementation for dicrete boundaries
        success = np.nan  # TODO: not implemented yet
        return self.components_bounds, best_accuracy, success

    def perturb_boundary_greedy(self, feat_data, cat_labels, use_visualization=False, verbose=False):
        '''
        @param target_accuracy: The accuracy for the global optimization
        '''
        # Randomly choose a component j
        random_component_index = np.random.random_integers(0, self.num_of_components - 1)
        # Randomly choose quantizing interval k (excluding the last interval because right boundaries are what will be optimized)
        random_quantizing_interval = np.random.random_integers(0, len(self.components_bounds[random_component_index]) - 2)

        # Initialize the new boundary with the perturbation
        best_boundary = self.components_bounds[random_component_index][random_quantizing_interval, 1]  # Target as the right boundary of the level
        lim_bound_low = self.components_bounds[random_component_index][random_quantizing_interval, 0]
        lim_bound_high = self.components_bounds[random_component_index][random_quantizing_interval + 1, 1]

        # Current best accuracy (Without perturbation)
        decider = DecisionRule(self.all_data, self.category_data, use_visualization)
        decider.fill_likelihood_LUT_bins(self)
        best_accuracy = decider.predict_batch(feat_data, cat_labels, verbose=False)
        initial_accuracy = best_accuracy

        # Random iteration (in some integer interval)
        # Tweaking iterations and offset parameter using dynamic heuristics
#         M = 2 * np.random.random_integers(self.min_level_per_comp, len(self.components_bounds[random_component_index]))
        level_width = len(self.components_bounds[random_component_index])
        max_rand_int = 50 / level_width  # TODO: Parametrize this value
        M = np.random.random_integers(max_rand_int / 2, max_rand_int)

        # Randomly choose a small pertubation  ( can be positive or negative)
        # Pick direction:
        search_direction = (2 * np.random.random_integers(0, 1) - 1)
        if search_direction < 0:  # Searching below the current target boundary
            level_width = best_boundary - np.max([0, lim_bound_low])
        else:
            level_width = np.min([1, lim_bound_high]) - best_boundary
        search_offset = level_width / (M)
        search_offset = search_direction * search_offset
#         print(M, "Iterations on Component:", random_component_index, "Level:", random_quantizing_interval, "Search Offset:", search_offset)
        num_iters = M
        best_boundary, best_accuracy = self.slide_boundary_iteratively(feat_data, cat_labels, random_component_index, random_quantizing_interval, search_offset, num_iters, best_boundary, best_accuracy)

        # Try search with smaller granularity (sublevel) from new boundary (on same direction)
        # *********************************
#         max_rand_int = 20
#         # *****************************
#         M = np.random.random_integers(10, max_rand_int)
        search_offset_sublevel = search_offset / (M)
#         print("     %d SUBLEVEL Iterations" % (M), "Search Offset:", search_offset_sublevel)
        msg_custom = "\t(Sublevel)"
        old_best_boundary = best_boundary
        num_iters = M
        best_boundary, best_accuracy = self.slide_boundary_iteratively(feat_data, cat_labels, random_component_index, random_quantizing_interval, search_offset_sublevel, num_iters, best_boundary, best_accuracy, msg_custom)
        # Try search with same sublevel granularity from new boundary (on opposite direction from the older higher level best boundary)
#         M = np.random.random_integers(4, max_rand_int)
#         num_iters = M
#         search_offset_sublevel = -1. * search_offset / (M)  # reverse direction
        search_offset_sublevel = -1. * search_offset_sublevel
        msg_custom = "\t(Sublevel REVERSED)"
        best_boundary_reverse, best_accuracy_reverse = self.slide_boundary_iteratively(feat_data, cat_labels, random_component_index, random_quantizing_interval, search_offset_sublevel, num_iters, old_best_boundary, best_accuracy, msg_custom)

        if best_accuracy < best_accuracy_reverse:
            best_boundary = best_boundary_reverse
            best_accuracy = best_accuracy_reverse

        # OLD way with too many bells and whistles:
# #         perturbation = 2 * np.random.random_sample() - 1  # Gives a random number in the range [-1, +1]
#         best_accuracy_rel_perc = (100. * best_accuracy) / target_accuracy  # Relative percentage from target accuracy
#         # Set perturbation values for exploration during the initial states (when distance to target is within some percentage)
#         explorative_percentage = 90.
#         if best_accuracy_rel_perc < explorative_percentage:
#             perturbation_weight = 0.2  # A high weight
#         else:
#             # Be more conservative and prefer exploitation
#             perturbation_weight = 0.1  # A low weight
#
#         random_perturbation = (2 * np.random.random_sample() - 1) * perturbation_weight  # FIXME: Gives a random number in the range [-1, +1]
#         target_boundary_new = best_boundary + random_perturbation  # * m
#         # Randomly choose a small integer M (No collision with neighboring boundaries)
#         # TODO: When dealing with discrete values, M should be a small integer
#         print("Trying", M, "loops. With component:", random_component_index, "Level:", random_quantizing_interval, "(Using perturbation):", random_perturbation)
#
#         greedy_accuracy_gain_percentage = 0.05
#         i = 0
#         no_progress_counter = 0
#         no_progress_max_count = M * perturbation_weight
#         is_going_well = True  # Being greedy
#         while i < M or is_going_well:
#             i += 1
#             if lim_bound_low < target_boundary_new < lim_bound_high:
#                 # Update boundaries on intervals
#                 self.components_bounds[random_component_index][random_quantizing_interval, 1] = target_boundary_new
#                 self.components_bounds[random_component_index][random_quantizing_interval + 1, 0] = target_boundary_new
#                 # Compute New Probabilities
#                 decider = DecisionRule(self.all_data, self.category_data, use_visualization)
#                 decider.fill_likelihood_LUT_bins(self)
#                 # NOTE: prediction accuracy is done (usually) on a different dataset
#                 new_accuracy = decider.predict_batch(feat_data, cat_labels, verbose=False)
#                 if new_accuracy > best_accuracy:
#                     # When the change was high, don't be too greedy and slow down the search
# #                     change_in_accuracy = 100 - (100. * best_accuracy) / new_accuracy  # Relative percentage from target accuracy
# #                     # Set perturbation values for exploration during the initial states (when distance to target is within some percentage)
# #                     if change_in_accuracy > greedy_accuracy_gain_percentage:
# #                         # Be more conservative and prefer exploitation
# #                         random_perturbation = random_perturbation * 0.9  # TODO: A more conservative perturbation
#
#                     best_accuracy = new_accuracy
#                     best_boundary = target_boundary_new
#
#                     print("Loop ", i, ", component:", random_component_index, "Level:", random_quantizing_interval, "Accuracy:", best_accuracy)
#                     no_progress_counter = no_progress_counter / 2  # Reduced the lack of progress counter by half
#                     i = 0
#                 else:
#                     no_progress_counter += 1
#                     # Consider it to be going well if the lack of progress is less than some% of the remaining total number of loops
#                     no_progress_max_count = (M - i) * perturbation_weight
#                     if no_progress_counter > no_progress_max_count:
#                         is_going_well = False
#
#                 target_boundary_new = target_boundary_new + random_perturbation
#             else:
#                 break

        # Set optimal boundaries from recent optimization pass
        self.components_bounds[random_component_index][random_quantizing_interval, 1] = best_boundary
        self.components_bounds[random_component_index][random_quantizing_interval + 1, 0] = best_boundary
        success = 0
        if best_accuracy > initial_accuracy:
            success = 1
            if verbose:
                print("WINNER best boundary = ", best_boundary, "with accuracy of", best_accuracy)

        return self.components_bounds, best_accuracy, success

    def slide_boundary_iteratively(self, feat_data, cat_labels, random_component_index, random_quantizing_interval, search_offset, num_iters, best_boundary, best_accuracy, msg_custom="", use_visualization=False, verbose=False):
        lim_bound_low = self.components_bounds[random_component_index][random_quantizing_interval, 0]
        lim_bound_high = self.components_bounds[random_component_index][random_quantizing_interval + 1, 1]

        target_boundary_new = best_boundary + search_offset
        i = 0
        while i < num_iters:
            i += 1
            if lim_bound_low < target_boundary_new < lim_bound_high:
                # Update boundaries on this interval and the next above
                self.components_bounds[random_component_index][random_quantizing_interval, 1] = target_boundary_new
                self.components_bounds[random_component_index][random_quantizing_interval + 1, 0] = target_boundary_new
                # Compute New Probabilities
                decider = DecisionRule(self.all_data, self.category_data, use_visualization)
                decider.fill_likelihood_LUT_bins(self)
                # NOTE: prediction accuracy is done (usually) on a different dataset
                new_accuracy = decider.predict_batch(feat_data, cat_labels, verbose=False)
                if new_accuracy > best_accuracy:
                    best_accuracy = new_accuracy
                    best_boundary = target_boundary_new
                    if verbose:
                        print("%s %d out of %d Iterations" % (msg_custom, i, num_iters), "--> Improved accuracy to:", best_accuracy)
                    i = 0  # Reset count
                target_boundary_new = target_boundary_new + search_offset
            else:
                break

        return best_boundary, best_accuracy

    def optimize_boundaries(self, feat_data, cat_labels, target_accuracy, is_discrete=False, use_visualization=False, verbose=False):
        '''
        The goal is to maximize correctness (target_accuracy) of learning_driver by random perturbation of boundaries
        '''

        decider = DecisionRule(self.all_data, self.category_data, use_visualization)
        decider.fill_likelihood_LUT_bins(self)
        current_accuracy = decider.predict_batch(feat_data, cat_labels, verbose=False)
        print("INITIAL accuracy prior to optimization procedure:", current_accuracy)

        if is_discrete:
            perturbation_function = self.slide_discrete_boundary
        else:
            perturbation_function = self.perturb_boundary_greedy

        # Repeat until no change
        count_trials = 1.
        count_success = 1.
        success_rate = (count_success / count_trials) / target_accuracy
        while (current_accuracy < target_accuracy) and (0.2 < success_rate):
            current_boundaries, current_accuracy, success_flag = perturbation_function(feat_data, cat_labels, use_visualization, verbose)
            count_success += success_flag
            count_trials += 1
            success_rate = (count_success / count_trials) / target_accuracy
            print("SUCCESS RATE =", success_rate, "| ACCURACY =", current_accuracy)

        print("-"*80)
        print("DONE optimizing after %d trial-iterations with ACCURACY = %f" % (count_trials, current_accuracy))
        print("-"*80)
        return current_boundaries, current_accuracy

class DecisionRule(object):

    def __init__(self, component_data, classes, use_visualization):
        '''
        Constructor for the quantizer_c0 out of the given component_data (without labels)

        @param component_data: The component component_data
        @param classes: The category labels for the observations
        '''

        self.data = component_data
        self.N = self.data.shape[0]
        # Segregate categories
        self.c0_indices_train = np.where(classes == 0)[0].reshape(-1, 1)
        self.c1_indices_train = np.where(classes == 1)[0].reshape(-1, 1)
        self.feat_train_c0 = np.take(self.data[:, 0], self.c0_indices_train)
        self.feat_train_c1 = np.take(self.data[:, 0], self.c1_indices_train)
        for c in range(1, self.data.shape[1]):
            self.feat_train_c0 = np.hstack((self.feat_train_c0, np.take(self.data[:, c], self.c0_indices_train)))
            self.feat_train_c1 = np.hstack((self.feat_train_c1, np.take(self.data[:, c], self.c1_indices_train)))

        self.use_visualization = use_visualization

    def _build_address_table(self):
        self.address_base = self.quantizer.max_num_levels
        self.address_multiples = np.array([self.address_base ** i for i in range(self.quantizer.num_of_components)]).reshape(-1, 1)
        len_of_table = self.address_base ** self.quantizer.num_of_components
        self.bin_probabilities_LUT = np.zeros((len_of_table, 2))
        # Initialize decision rules LUT to -1 (an invalid class)
        self.decision_rules_LUT = np.zeros(self.address_base ** self.quantizer.num_of_components, dtype="int8") - 1

    def get_index_address_table(self, address):
        # hashed_address = np.sum(address * self.address_multiples)
        hashed_address = np.dot(address, self.address_multiples)  # Nice!
        return hashed_address

    def fill_likelihood_LUT_bins(self, quantizer):
        '''
        @param quantizer: The quantizer object
        '''
        self.quantizer = quantizer
        self._build_address_table()
        self.address_base = self.quantizer.max_num_levels

        q_addresses_0 = self.quantizer.get_address(self.feat_train_c0)  # Quantized addresses
        address_LUT_0 = self.get_index_address_table(q_addresses_0)
        q_addresses_1 = self.quantizer.get_address(self.feat_train_c1)  # Quantized addresses
        address_LUT_1 = self.get_index_address_table(q_addresses_1)

        # Count per catagory addresses and fill the table
        count_c0 = np.bincount(address_LUT_0[:, 0])
        count_c1 = np.bincount(address_LUT_1[:, 0])

        # WISH: Randomize those addresses with empty counts
        # So far, it seems that there is always counts in the bins due to initialization using uniform densities
        # zero_count_indices = np.where((count_c0 + count_c1) == 0)

        self.bin_probabilities_LUT[:len(count_c0), 0] = count_c0
        self.bin_probabilities_LUT[:len(count_c1), 1] = count_c1
        # Normalize
        self.bin_probabilities_LUT = self.bin_probabilities_LUT / self.N


        # Fill up decision rules LUT
        self.decision_rules_LUT = np.argmax(self.bin_probabilities_LUT, axis=-1)  # Puts always zeros if ties! Wrong!

    def predict_category_from_bins(self, meas_vector):
        q_addresses = self.quantizer.get_address(meas_vector)  # Quantized addresses
        address_LUT = self.get_index_address_table(q_addresses)
        return self.decision_rules_LUT[address_LUT]

    def predict_batch(self, observations, labels_true, verbose=False):
        N = observations.shape[0]
        predicting_function = self.predict_category_from_bins
        predictions = predicting_function(observations)
        labels_true = labels_true.reshape(*predictions.shape)
        errors = predictions - labels_true
        correct_count = N - np.count_nonzero(errors)
        prob_correct = correct_count / float(N)
        if verbose:
            levels_str = "["
            for c in self.quantizer.components_bounds:
                levels_str = levels_str + " " + str(len(c))
            levels_str += "]"
            print("Accuracy = ", prob_correct, "with levels", levels_str)

        return prob_correct

def learn_with_quantization(features_data, classes_data, min_level_per_component, max_levels_per_component, target_training_accuracy, num_of_candidates, is_discrete, use_visualization=False, verbose=False):
    '''

    '''
    N = features_data.shape[0]
    N_split = np.floor(N / 2)
    feat_data_train, class_train = features_data[:N_split], classes_data[:N_split]
    feat_data_validate, class_validate = features_data[N_split:2 * N_split], classes_data[N_split:2 * N_split]

    quantizer = Quantizer(data=feat_data_train, classification_labels=class_train, min_level_per_comp=min_level_per_component, max_level_per_comp=max_levels_per_component)  # entropy_change_thresh=entropy_change_thresh)

    # TODO: save best 4 and try them up to some time out (lack of progress):
    candidate_levels_queue = quantizer.optimize_num_levels(feat_data=feat_data_train, cat_labels=class_train, num_of_candidates=num_of_candidates, is_discrete=is_discrete, use_visualization=use_visualization, verbose=verbose)
#     quantizer.set_boundaries_of_uniform_density(levels_list=[6, 5, 6, 5])  # IMPORTANT: Gives 0.94 accuracy!
#     quantizer.set_boundaries_of_uniform_density(levels_list=[6, 6, 6, 6])  # IMPORTANT: Gives 0.94 accuracy!
#     quantizer.set_boundaries_of_uniform_density(levels_list=[7, 7, 7, 7])  # IMPORTANT: Gives 0.94 accuracy!
#     quantizer.set_boundaries_of_uniform_density(levels_list=[5, 8, 9, 11])  # IMPORTANT: Gives 0.94 accuracy!
#     quantizer.set_boundaries_of_uniform_density(levels_list=[5, 8, 10, 8])  # IMPORTANT: Gives 0.94 accuracy!
#     quantizer.set_boundaries_of_uniform_density(levels_list=[5, 5, 6, 6])

    best_boundaries = None
    best_accuracy = 0.
    best_levels = None
    print("Queue of best boundaries:", candidate_levels_queue)
    for l in candidate_levels_queue:
        print("PREDICTING test on 2/3 with", l, "levels optimization:")
        print("/"*80)
        quantizer.set_boundaries_of_uniform_density(levels_list=l)  # IMPORTANT: Gives 0.94 accuracy!

        current_boundaries, current_train_accuracy = quantizer.optimize_boundaries(feat_data=feat_data_train, cat_labels=class_train, target_accuracy=target_training_accuracy, is_discrete=is_discrete, use_visualization=use_visualization, verbose=True)
        trained_decider = DecisionRule(feat_data_train, class_train, use_visualization)
        trained_decider.fill_likelihood_LUT_bins(quantizer)
        current_internal_test_accuracy = trained_decider.predict_batch(feat_data_validate, class_validate, verbose=verbose)
        if best_accuracy < current_internal_test_accuracy:
            best_accuracy = current_internal_test_accuracy
            best_boundaries = current_boundaries
            best_levels = l

    # PREDICTION test using the bin-wise approach
    # set best_boundaries into quantizer for the final test against the test dataset!!!!
    quantizer.components_bounds = best_boundaries

    return quantizer, best_levels


#     def get_entropy(self, num_levels=None):
#         '''
#         Computes the entropy in the component using the indicated boundary intervals
#
#         @return: the entropy in obtained due to the assigned intervals (boundaries)
#         '''
#         if num_levels == None:
#             num_levels = self.num_levels
#
#         level_probabilities = self.get_level_probabilities(num_levels)
#         entropy_sum_levels = 0.
#         for p in level_probabilities:
#             if p > 0:
#                 entropy_sum_levels += p * np.log2(p)
#
# #         level_entropies = np.where(level_probabilities > 0, level_probabilities * np.log2(level_probabilities), 0)
#         n0 = len(self.boundaries) - np.count_nonzero(level_probabilities)
#         N = self.N
#         zero_entropy = (n0 - 1) / (2 * N * np.log(2))
# #         Hj = -np.sum(level_entropies) + zero_entropy
#         Hj = -entropy_sum_levels + zero_entropy
#         return Hj
