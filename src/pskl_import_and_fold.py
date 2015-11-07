# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 09:29:12 2015

@author: apark
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
file_loc = '/home/apark/Homework/ML_project1/'
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.feature_selection import SelectKBest
import matplotlib.cm as cm
# Import test data and labels
import_data = sio.loadmat(file_loc + 'data/project1data.mat')
import_labels = sio.loadmat(file_loc + 'data/project1data_labels.mat')

X = import_data['X']
Y = import_labels['Y']


''' 
# SelectKBest 
skb = SelectKBest()
skb.fit(X,Y)
print("Best Features: ", skb.get_support)
X_downsampled = skb.transform(X)
print("size of X_downsampled: ", X_downsampled.shape)
'''

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(X)
print(pca.explained_variance_ratio_)
X_pca = pca.transform(X)
print( " X_pca share: ", X_pca.shape)


# plot principal components
index_1 = np.where(Y==1)[0]
index_2 = np.where(Y==2)[0]
index_3 = np.where(Y==3)[0]
index_4 = np.where(Y==4)[0]
index_5 = np.where(Y==5)[0]

x = X_pca[index_1,:1]
y = X_pca[index_1,1:2]
print x
print y
plt.figure()
plt.hold('on')

plt.scatter(X_pca[index_1,:1],X_pca[index_1,1:2], color="r")
plt.scatter(X_pca[index_2,:1],X_pca[index_2,1:2], color="b")
plt.scatter(X_pca[index_3,:1],X_pca[index_3,1:2], color="g")
plt.scatter(X_pca[index_4,:1],X_pca[index_4,1:2], color="c")
plt.scatter(X_pca[index_5,:1],X_pca[index_5,1:2], color="m")

plt.legend(("1","2","3","4","5"))
plt.title("PCA of 2 largest components")

plt.show()
    
#plt.plot()


'''
kf = KFold(len(Y),n_folds = 5)
for train_index,test_index in kf:
    #print( "Train: ", train_index, "  Test: ", test_index)
    X_train, X_test = X_pca[train_index], X_pca[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    print("X_train shape: ", X_train.shape)
    clf = svm.SVC()
    clf.fit(X_train, Y_train)
    Y_predict = clf.predict(X_test)
    right = 0
    for i in range(0,len(Y_test)):
        if(abs(Y_predict[i] - Y_train[i]) < 1):
            right +=1
            print("correct label")
    performance = float(right)/float(len(Y_train))
    print("performance of : ", performance)

      '''      