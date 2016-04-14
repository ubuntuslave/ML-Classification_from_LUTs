'''
@summary:   Decision rules and data set generation using 10 components and discrete values (from 0 to 9) for supervised learning learning_driver.
            The number of decision rules is reduced exponentially based upon the level of mating (grouping of values) applied across the selected relevant components.

@author: Juan Pablo Munoz
@author: Carlos Jaramillo

Machine Learning Course at CUNY Graduate Center - Final Project - Fall 2014 - Prof. Robert Haralick1
'''
from __future__ import division
from __future__ import print_function

import numpy as np
import datetime
import common_tools

def generate_rules(class_labels, num_of_components, relevant_components, num_of_groups):
    '''
    Generate learning_driver rules as a big LUT where irrelevant values get an invalid value of -1
    Valid values for components are only nonnegative integers in the interval [0, 9].
    Here, each possible combination of values among the relevant components becomes a rule.
    Thus, a table of decision rules is generated using random learning_driver.
    The length of the decision table depends on the number of groups on the components and the total number of relevant components.
    For example, for groups of 2 (pairs) and only 3 relevant components, there exists (10/2)^3 = 5^3 = 125 decision rules.

    @param class_labels: list of nonnegative integers for allowed classes (categories)
    @param num_of_components: Total number of components (a.k.a features)
    @param relevant_components: A list of those relevant components in the rules and any data set generated employing the table of decision rules created here.
    @param num_of_groups: If set to 10, then grouping does not have an effect since every value is a singleton.

    @return: The numpy array of rules where the last column corresponds to learning_driver labels.
    '''
    # Create a LUT of num_of_components by n_rules
    n_rules = num_of_groups ** len(relevant_components)
    components_LUT = np.zeros((n_rules, num_of_components), dtype='int8') - 1

    # Fill up the relevant LUT of components with indices on the respective places
    indices = np.arange(n_rules)

    multiple = n_rules
    # Break up the indices in to its digits
    for comp in relevant_components:
        components_LUT[:, comp] = (indices % multiple) // int(multiple / num_of_groups)
        multiple = multiple / num_of_groups

    num_of_classes = len(class_labels)

    # WISH: use string labels:
    # labels = np.ndarray((total_num_of_instances), dtype=np.dtype({'classlabel':('S1', 0)}))  # , dtype=str)

    # Create an array of labels of total_num_of_instances length
    classes = np.asarray(class_labels, dtype='int8')
    labels = classes[np.random.randint(num_of_classes, size=(n_rules))]

    # Shuffle labels
    np.random.shuffle(labels)

    # Initialize all values in dataset to -1
    complete_LUT = np.zeros((n_rules, num_of_components + 1), dtype='int8') - 1
    complete_LUT[:, :num_of_components] = components_LUT
    complete_LUT[:, -1] = labels

    return complete_LUT

def generate_dataset(rules, relevant_components, total_num_of_instances, instances_percentages, num_of_groups):
    '''
    Irrelevant components are filled up with random values from a uniform distribution
    Last, the generated instance vectors (measurement records or rows including their associated class label) of the data set are shuffled and returned as the final result.

    @param rules: Numpy ndarray of rules organized as a 2D table of components and classes (last column)
    @param relevant_components: The list of relevant components
    @param total_num_of_instances: The desired number of instances
    @param instances_percentages: The split (percentage-wise) of instances from each category (class)
    @param num_of_groups: If set to 10, then grouping does not have an effect because groups would have a single element each.

    @return: The generated data set as a numpy array in which the last column corresponds to learning_driver labels.
    '''
    low = 0  # Lowest valid measurement value
    high = 9  # Highest valid measurement value

    num_of_components = rules.shape[1] - 1
    dataset = np.ndarray((total_num_of_instances, num_of_components + 1), dtype='int8')

    rules_c0 = rules[np.where(rules[..., -1] == 0)]
    rules_indices_c0 = np.random.randint(len(rules_c0), size=(instances_percentages[0] * total_num_of_instances / 100))
    dataset[:len(rules_indices_c0)] = rules_c0[rules_indices_c0]

    rules_c1 = rules[np.where(rules[..., -1] == 1)]
    rules_indices_c1 = np.random.randint(len(rules_c1), size=(instances_percentages[1] * total_num_of_instances / 100))
    dataset[len(rules_indices_c0):] = rules_c1[rules_indices_c1]

    # WISH: done manually for now: as proof of concept!
    if num_of_groups < 10:
        for f in relevant_components:
            # Butterfly approach:
            for q in range(num_of_groups):  # Random reassignment done in reverse order to avoid overwrites
                target_idx = np.where(dataset[:, f] == q)[0]
                dataset[target_idx, f] = dataset[target_idx, f] + num_of_groups * np.random.randint(2, size=len(target_idx))

    # Fill random values on irrelevant components
    for f in range(num_of_components):
        if f not in relevant_components:
            # Draw samples from a uniform distribution.
            irrelevant_comp_data = np.random.uniform(low, high, size=total_num_of_instances)
            dataset[:, f] = irrelevant_comp_data

    # Shuffle rows
    np.random.shuffle(dataset)

    return dataset


if __name__ == '__main__':
    class_labels = [0, 1]
    instances_percentages = [50, 50]  # Must add to 100%
    relevant_components = [0, 1, 8]  # Arbitrary component numbers
    total_num_of_instances = 100000
    num_of_components = 10
    dataset_header = ["d" + str(fname + 1) for fname in range(num_of_components)]
    dataset_header = ', '.join(dataset_header) + ', class'

    path_to_output_files = "./InputData/"

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    will_generate_rules = True  # <<<<< Important to SET
    quant_levels = 5  # If set to 10, then quantization does not have an effect.
    save_class_as_string = True  # <<<<< Determines whether to save classes as Strings!
    k_comp_str = str(len(relevant_components))
    filename_time = k_comp_str + "rel_components-quantized_butterfly-2014-12-27"
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    if save_class_as_string:
        custom_fmt = ["%s" for fname in range(num_of_components)] + ["%s"]

    if will_generate_rules:
        rules = generate_rules(class_labels, num_of_components, relevant_components, quant_levels)
        # Use current time for filename suffix
        now = datetime.datetime.now()
        # filename_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        rules_filename = path_to_output_files + "rules-" + filename_time + ".csv"

        if save_class_as_string:
            rules_labels = np.where(rules[:, -1] == 0, 'A', 'B')  # Save table of rules to file
            rules_as_str = np.hstack((rules[:, :-1].astype('S2'), rules_labels.reshape(-1, 1)))
            np.savetxt(rules_filename, rules_as_str, delimiter=",", header=dataset_header, fmt=custom_fmt)
        else:
            np.savetxt(rules_filename, rules, delimiter=",", header=dataset_header, fmt='%d')

        print("Saving rules to: %s" % rules_filename)
    else:
        # Load rules from file
        rules_filename = path_to_output_files + "rules-" + filename_time + ".csv"
        print("Loading rules from: %s" % rules_filename)
        all_comp_data, all_classes = common_tools.read_data(rules_filename, ",", \
                    use_string_labels=False, given_string_labels=save_class_as_string, \
                    num_components=num_of_components, dtype='int8', has_header=False, shuffle_rows=False)

        rules = np.hstack((all_comp_data, all_classes.reshape(-1, 1)))

    # Generate instance from LUT of rules filling up irrelevant components
    dataset = generate_dataset(rules, relevant_components, total_num_of_instances, instances_percentages, quant_levels)

    dataset_filename = path_to_output_files + "dataset-" + filename_time + ".csv"
    # Export data set to CSV file
    if save_class_as_string:
        dataset_labels = np.where(dataset[:, -1] == 0, 'A', 'B')  # Save table of rules to file
        dataset_as_str = np.hstack((dataset[:, :-1].astype('S2'), dataset_labels.reshape(-1, 1)))
        np.savetxt(dataset_filename, dataset_as_str, delimiter=",", header=dataset_header, fmt=custom_fmt)
    else:
        np.savetxt(dataset_filename, dataset, delimiter=",", header=dataset_header, fmt='%u')  # save only unsigned decimal integers

    print("Done. Saved data set file to: %s" % dataset_filename)
