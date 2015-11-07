# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
file_loc = '/home/apark/Homework/ML_project1/'

# Import test data and labels
import_data = sio.loadmat(file_loc + 'data/project1data.mat')
import_labels = sio.loadmat(file_loc + 'data/project1data_labels.mat')

X = import_data['X']
Y = import_labels['Y']

#Segment into 5 folds
fold = np.zeros((5,99,5903), dtype=np.int16)

for i in range(0,5):
    print(i)    
    l = i*100
    h = (i+1)*100-1
    print("%s, %s" % ( l,h))
    arr = fold[i:,l:h,:]
    print arr.shape
    xarr = X[l:h,:]
    print xarr.shape
    
    fold[i,:] = X[l:h] 
    


