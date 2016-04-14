'''
@summary:   This script drives the supervised learning project for binary classification. It uses the data set generated in order to learn and validate the decision rules with extremely high accuracy (almost perfect classification of 100%).
            Popular classification techniques, such as logistic regression, bayes multinomial classification, and kernel-based support vector machines (SVM) fail to classify accurately (with less than 60% correctness).

@author: Juan Pablo Munoz
@author: Carlos Jaramillo

Machine Learning Course at CUNY Graduate Center - Final Project - Fall 2014 - Prof. Robert Haralick
'''
from __future__ import division
from __future__ import print_function

import common_tools
import time

def learn_sk_models(data_file_prefix, percentage_to_train=90, load_data_from_pickle=False):
    '''
    Simple training (fitting) of the data set using the Naive Bayesian classifier.

    @param filename: the database filename path
    @param porcentage_to_train: The percentage from the database dedicated for training. The complimentary percentage is used for testing.
    '''
    if load_data_from_pickle:
        features_data, classifications = common_tools.load_dataset_from_pickle(data_file_prefix)
    else:
        features_data, classifications = common_tools.read_data("./InputData/" + data_file_prefix + ".csv", ",", \
                        use_string_labels=False, given_string_labels=True, \
                        num_components=10, dtype='uint8', has_header=True, shuffle_rows=True)

    # Partition the data in ordert to train the classifier
    num_instances = len(classifications)
    holdout_number = int(num_instances * percentage_to_train / 100.)  # hold out percentage (%) of the dataset instances for testing
    train_instances = features_data[0:holdout_number]
    train_labels = classifications[0:holdout_number]
    test_instances = features_data[holdout_number:]
    test_labels = classifications[holdout_number:]

    # Fit the dataset using a Bayes classifiers
    from sklearn import naive_bayes

    # Naive Bayes Model Construction
    # MultinomialNB & Complement NB
    t_start = time.clock()
    mnb = naive_bayes.MultinomialNB()
    mnb.fit(train_instances, train_labels)
    t_end = time.clock()
    t_elapsed = t_end - t_start
    print('Building model took: %.9f seconds' % t_elapsed)
    mnb_classification_accuracy = mnb.score(test_instances, test_labels)
    print("Classification accuracy of Multinomial Naive Bayes =", mnb_classification_accuracy)
    print("\tTraining class count: [A=%d], [B=%d]" % tuple(mnb.class_count_))

    print("-"*80)

    # Fisher's Linear Discriminant Analysis (LDA)
    from sklearn.lda import LDA
    t_start = time.clock()
    gnb = naive_bayes.GaussianNB()
    gnb.fit(train_instances, train_labels)
    t_end = time.clock()
    t_elapsed = t_end - t_start
    print('Building model took: %.9f seconds' % t_elapsed)
    gnb_classification_accuracy = gnb.score(test_instances, test_labels)
    print("Classification accuracy of Naive Bayes Gaussian =", gnb_classification_accuracy)

    print("-"*80)

    # Naive Bayes Gaussian Classifier Construction
    t_start = time.clock()
    fisher_lda = LDA(n_components=3)
    fisher_lda.fit(train_instances, train_labels)
    t_end = time.clock()
    t_elapsed = t_end - t_start
    print('Building model took: %.9f seconds' % t_elapsed)
    fisher_lda_accuracy = fisher_lda.score(test_instances, test_labels)
    print("Classification accuracy of Fisher's LDA = ", fisher_lda_accuracy)


    pass


def classify_dataset(data_file_prefix, load_data_from_pickle=False):
    '''
    Reads the data set from the file name provided in order to perform learning and final classification test providing the resulting accuracy of correct classification.

    @param data_file_prefix: The prefix of the data set file name
    @param load_data_from_pickle: When set to True, a Python pickle (.pkl extension) will be used instead of the traditional comma separated valued (.csv extension) data set.
    '''

    from mate_learning import MateFinder

    if load_data_from_pickle:
        all_comp_data, all_classes = common_tools.load_dataset_from_pickle(data_file_prefix)
    else:
        all_comp_data, all_classes = common_tools.read_data("./InputData/" + data_file_prefix + ".csv", ",", \
                        use_string_labels=False, given_string_labels=True, \
                        num_components=10, dtype='uint8', has_header=True, shuffle_rows=True)

        common_tools.save_dataset_to_pickle(all_comp_data, all_classes, data_file_prefix)

    t_start = time.clock()
    match_maker = MateFinder(all_comp_data, all_classes, mating_size=2)

    # Learning phase
    decision_rules_LUT, best_matings, relevant_components = match_maker.learn_by_mate_discovery()

    # Final test
    final_accuracy = match_maker.compute_prediction_accuracy(best_matings, decision_rules_LUT, relevant_components)
    t_end = time.clock()
    t_elapsed = t_end - t_start
    print("_"*80)
    print("FINAL Accuracy = %.2f%s" % (final_accuracy * 100., "%"))
    print("_"*80)
    print("TOTAL elapsed time: %.9f s" % (t_elapsed))


if __name__ == '__main__':
    data_file_prefix = "dataset-3rel_components-quantized_butterfly-2014-12-27"  # relevant_features = [0, 1, 8]

    test_simple_sk_classifiers = True

    if test_simple_sk_classifiers:
        learn_sk_models(data_file_prefix, percentage_to_train=50, load_data_from_pickle=True)
        print("~"*80)

    classify_dataset(data_file_prefix, load_data_from_pickle=True)

    pass
