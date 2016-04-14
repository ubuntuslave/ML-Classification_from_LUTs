from __future__ import division
from __future__ import print_function

import numpy as np
import quantization
import common_tools

def test_real_dataset(use_visualization=False, verbose=False, load_data_from_pickle=False):

    data_file_prefix = "pr_data"

    if load_data_from_pickle:
        all_comp_data, all_classes = common_tools.load_dataset_from_pickle(data_file_prefix)
    else:
        all_comp_data, all_classes = common_tools.read_data("./InputData/" + data_file_prefix + ".csv", ",", \
                        use_string_labels=False, given_string_labels=True, num_components=4, dtype='float', has_header=True, shuffle_rows=True)
        common_tools.save_dataset_to_pickle(all_comp_data, all_classes, data_file_prefix)

    # WISH: Seems useful as if all features are relevant:
#     new_feat_data_train = mltools.filter_irrelevant_features(features_data, classes, wanted_num_of_features=2)

    # Get 50% for training:
    N = all_comp_data.shape[0]
    N_split = np.floor(N / 2)
    feat_data_learning, class_learning = all_comp_data[:N_split], all_classes[:N_split]
    feat_data_test, class_test = all_comp_data[N_split:], all_classes[N_split:]

    # ***** Hyper Parameters for quantization *************************
    min_level_per_component = 3
    max_levels_per_component = 8
    target_training_accuracy = 0.95
    num_of_candidates = 1
    is_discrete = False
    # ******************************************************************
    quantizer, best_levels = quantization.learn_with_quantization(feat_data_learning, class_learning, min_level_per_component, max_levels_per_component, target_training_accuracy, num_of_candidates, is_discrete, use_visualization, verbose)

    final_decider = quantization.DecisionRule(feat_data_learning, class_learning, use_visualization)
    final_decider.fill_likelihood_LUT_bins(quantizer)
    final_accuracy = final_decider.predict_batch(feat_data_test, class_test, verbose=False)
    print("~"*80)
    print("~"*80)
    print("Final PREDICTION TEST accuracy = %f using levels:" % (final_accuracy), best_levels)

# def test_real_with_visualization():
#     import
#     visualizer = visualization.Visualization()

if __name__ == '__main__':

    test_real_dataset(verbose=True, load_data_from_pickle=True)
#     test_real_with_visualization()

    pass
