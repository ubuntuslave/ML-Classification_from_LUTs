"""
==================
SVM classification
==================

Demonstration of Support Vector Machines with various kernel tricks
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import time
from sklearn import svm

import common_tools


data_file_prefix = "dataset-3rel_components-quantized_butterfly-2014-12-27"
load_data_from_pickle = True

if load_data_from_pickle:
    all_comp_data, all_classes = common_tools.load_dataset_from_pickle(data_file_prefix)
else:
    all_comp_data, all_classes = common_tools.read_data("./InputData/" + data_file_prefix + ".csv", ",", \
                    use_string_labels=False, given_string_labels=True, \
                    num_components=10, dtype='uint8', has_header=True, shuffle_rows=True)

# Partition the data in order to train the classifier
percentage_to_train = 50
num_instances = len(all_classes)
holdout_number = int(num_instances * percentage_to_train / 100.)  # hold out percentage (%) of the dataset instances for testing
X_train = all_comp_data[:holdout_number]
y_train = all_classes[:holdout_number]
X_test = all_comp_data[holdout_number:]
y_test = all_classes[holdout_number:]

n_classes = len(np.unique(y_train))

# Try SVM using different types of kernels.
classifiers = dict((kernel_type, svm.SVC(kernel=kernel_type))
                    for kernel_type in ['linear', 'poly', 'rbf', 'sigmoid'])

for index, (name, classifier) in enumerate(classifiers.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    print('%d) Using SVM with %s kernel' % (index + 1, name))

    t_start = time.clock()

    classifier.fit(X_train, y_train)
    t_end = time.clock()
    t_elapsed = t_end - t_start
    print('Building model took: %.9f seconds' % t_elapsed)

    y_test_pred = classifier.predict(X_test)
    y_test_true = y_test.reshape(*y_test_pred.shape)

    test_accuracy = classifier.score(X_test, y_test) * 100
    print('Test accuracy using score: %.1f' % test_accuracy)

    errors = y_test_pred - y_test_true
    N = len(y_test_pred)
    correct_count = N - np.count_nonzero(errors)
    prob_correct = correct_count / float(N)

    print("FINAL Test Accuracy = %.2f%s" % (prob_correct * 100., "%"))
    print("-"*80)

