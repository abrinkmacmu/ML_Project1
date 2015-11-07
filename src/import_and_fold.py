# -*- coding: utf-8 -*-

import scipy.io as sio
file_loc = '/home/apark/Homework/ML_project1/'

# Import test data and labels
import_data = sio.loadmat(file_loc + 'data/project1data.mat')
import_labels = sio.loadmat(file_loc + 'data/project1data_labels.mat')

X = import_data['X']
Y = import_labels['Y']

X

#Segment into 5 folds

