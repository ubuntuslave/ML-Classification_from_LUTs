'''
Created on Aug 20, 2014

@author: carlos
'''
from __future__ import division
from __future__ import print_function

import numpy as np
import dill as pickle
# import cPickle as pickle
# import pickle

def get_namestr(obj, namespace):
    '''
    Finds and returns the variable name for the object instance in question in the desired namespace.
    For example

        >>> get_namestr(my_wow, globals())
        Out: 'my_wow'
    '''
    if namespace:
        results = [name for name in namespace if namespace[name] is obj]
        if len(results) == 1:
            return results[0]
        else:
            return results
    else:
        return ""

def save_obj_in_pickle(obj_instance, filename, namespace=None):
    print("Saving %s instance to pickle file %s ..." % (get_namestr(obj_instance, namespace), filename), end="")
    f = open(filename, 'wb')  # Create external f
    pickle.dump(obj_instance, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    print("done!")

def load_obj_from_pickle(filename, namespace=None):
    f = open(filename, 'rb')  # Create external f
    obj_instance = pickle.load(f)
    print("Loading %s instance from pickle file %s ... " % (get_namestr(obj_instance, locals()), filename), end="")
    f.close()
    print("done!")
    return obj_instance

def save_dataset_to_pickle(all_comp_data, all_classes, filename_prefix):
    save_obj_in_pickle([all_comp_data, all_classes], "./InputData/" + filename_prefix + ".pkl", namespace=locals())

def load_dataset_from_pickle(filename_prefix):
    all_comp_data, all_classes = load_obj_from_pickle("./InputData/" + filename_prefix + ".pkl", namespace=locals())
    return all_comp_data, all_classes

def read_data(filename, delimiter=",", feature_indices=None, use_string_labels=True, given_string_labels=False, num_components=10, dtype='uint8', has_header=True, shuffle_rows=False):
    '''
    @param filename: the database filename path
    @param delimiter: the string usefile delimeter (e.g. ",")
    @param feature_indices: The index of the columns for the desired indices. If nothing is passed, then all features are used.
    @param use_string_labels: indicates whether labels are kept as strings, or if False, they should be indexed instead
    @param given_string_labels: indicates whether class labels are strings. Use False (default) to imply the use of digit labels
    @param num_components: number of columns (feature dimensions) in the file.
    @return the np array of features and the single-column array of the labels from the dataset being read.
    '''

    if given_string_labels:
        all_feature_data = np.genfromtxt(filename, delimiter=delimiter, usecols=tuple(range(num_components)), dtype=dtype)
        all_labels = np.genfromtxt(filename, delimiter=delimiter, usecols=(-1), dtype='str')  # [('str_class_label', 'S1')])
    else:
        all_data = np.loadtxt(filename, dtype=dtype, delimiter=delimiter)
        all_feature_data = all_data[:, :-1]
#         labels = all_data[:, -1]


    if has_header:
        first_data_row = 1
    else:
        first_data_row = 0

    if feature_indices == None:
        feature_data = all_feature_data[first_data_row:]
    else:
        feature_data = all_feature_data[first_data_row:, feature_indices]

    labels = all_labels[first_data_row:]

    if use_string_labels == False and given_string_labels:
        uniq_keys = np.unique(labels)
        uniq_values = range(len(uniq_keys))
        # create a dictionary of the unique values
        indexed_keys_dict = dict([(k, v) for k, v in zip(uniq_keys, uniq_values)])
        # Translate every element in numpy array according to key
        labels = np.vectorize(indexed_keys_dict.get)(labels)

    if shuffle_rows:
        # FIXME: it may not work if labels are of type "str"
        # Must shuffle as a whole
        whole_table = np.hstack((feature_data, labels.reshape(-1, 1)))
        np.random.shuffle(whole_table)
        feature_data = whole_table[:, :-1]
        labels = whole_table[:, -1].astype('uint8')

    return feature_data, labels

def filter_irrelevant_features(feature_data, labels, wanted_num_of_features, use_chi2=True):
    '''
    @param use_sklearn: Indicates whether the filtering is done with scikit-learn's SelectKBesr or a custom bruteforce procedure.
    '''
    from sklearn.feature_selection import SelectKBest
    X, y = feature_data, labels
    if use_chi2:
        from sklearn.feature_selection import chi2
        X_new = SelectKBest(chi2, k=wanted_num_of_features).fit_transform(X, y)
    else:
        from sklearn.feature_selection import f_classif  # Compute the Anova F-value for the provided sample
        X_new = SelectKBest(f_classif, k=wanted_num_of_features).fit_transform(X, y)

    return X_new


def pdf(point, cons, mean, det_sigma):
    if isinstance(mean, np.ndarray):
        return cons * np.exp(-np.dot(np.dot((point - mean), det_sigma), (point - mean).T) / 2.)
    else:
        return cons * np.exp(-((point - mean) / det_sigma) ** 2 / 2.)

def reverse_axis_elems(arr, k=0):
    '''
    @param arr: the numpy ndarray to be reversed
    @param k: The axis to be reversed
        Reverse the order of rows: set axis k=0
        Reverse the order of columns: set axis k=1

    @return: the reversed numpy array
    '''
    reversed_arr = np.swapaxes(np.swapaxes(arr, 0, k)[::-1], 0, k)
    return reversed_arr

def rms(x, axis=None):
    return np.sqrt(np.mean(np.square(x), axis=axis))


def nanrms(x, axis=None):
    '''
    If you have nans in your data, you can do
    '''
    return np.sqrt(np.nanmean(np.square(x), axis=axis))

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)  # Fast and numerically precise
    return average, np.sqrt(variance)

def mean_and_std(values):
    """
    Return the arithmetic mean or unweighted average and the standard deviation.

    @param values: Numpy ndarrays of values
    """
#     average = np.mean(values)
    average = np.mean(values)
    std_dev = np.std(values)

    return average, std_dev
