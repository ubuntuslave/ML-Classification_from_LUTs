Multidimensional Classification using LUTs 
========

## Final Project For Machine Learning Course 

- **Course Name**: Machine Learning (Fall 2014) by [Prof. Robert Haralick](https://en.wikipedia.org/wiki/Robert_Haralick)
- **Authors**: Carlos Jaramillo **and** Juan Pablo Munoz
- **Contact**: <cjaramillo@gradcenter.cuny.edu> **and** <jpablomch@gmail.com>

*Copyright (C)* 2014 under the *Gnu Public License version 3 (GPL3)*
 
## Overview
The **goal** is to design a *labeled two class data set* of 10-dimensional vectors that has test set classification accuracy less than 60% on some popular classifiers. 
However, our decision rule was designed such that it can perform with greater than 90% accuracy on the test set.

### Report

The [project report](https://github.com/ubuntuslave/ML-Classification_from_LUTs/blob/master/ML_Final_Project_Report.pdf) for our solution.

### Data Set Generation

`data_generation.py` is a *toolbox* written in Python for data set generation satisfying the problem specifications.
We generate a data set of 100,000 10-dimensional vectors (components), where 3 out them are relevant.

Decision rules and data set generation using 10 components and discrete values (from 0 to 9) for supervised learning learning_driver.
The number of decision rules is reduced exponentially based upon the level of mate_learning (grouping of values) applied across the selected relevant components. 

### Learning Method

Supervised Learning method for binary classification based on mating (grouping of values) across the supplied data set of components and classification labels.
The learning algorithm is implemented in 'mate_learning.py`. It consists of exhausting all the possibilities as long as accuracy of correct classification can be increased.

We define a `MateFinder` class for the particular binary classification problem based on mates (usually groups of 2 - pairs) across each component of a discrete-valued data set.
The `MateFinder` class provides procedures such as the removal of irrelevant components (selection of relevant components),
as well as essential learning (training and validation) and testing procedures from learned decision tables of pairs.

Once a `MateFinder` object has been instantiated through the constructor, the interfacing methods employed by an external classifier are:
`learn_by_mate_discovery()` and `compute_prediction_accuracy()`

The program driving the learning from data and estimating the accuracy of the learned decision rule on the testing set 
is driven by `learning_driver.py`

## Installation

### From source ###

It's **Python**! Only the [Numpy](http://numpy.org) module is required.

After fulfilling the module dependencies, just run the scripts, such as:

    $ python data_generation.py

## Wish List

- Gimme some!

## More info

***Documentation***:  can be found in the 'docs' folder or regenerated via [doxypy](http://code.foosel.org/doxypy)


