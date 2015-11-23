# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 09:29:12 2015

@author: apark
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
file_loc = '/home/apark/Homework/ML_Project1/'
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.feature_selection import SelectKBest
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
# Import test data and labels
import_test = sio.loadmat(file_loc + 'data/Test.mat')
import_train = sio.loadmat(file_loc + 'data/Train.mat')


Ytrain = import_train['Ytrain']
eventsTrain = import_train['eventsTrain']
subjectsTrain = import_train['subjectsTrain']
x = import_train['x']
y = import_train['y']
z = import_train['z']




Xtrain = import_train['Xtrain']
'''
Xtrain = np.zeros((501,5904))
Xtrain[:,0:5903] = import_train['Xtrain']
Xtrain[:,5903:5904] = import_train['eventsTrain']/1000.0
'''


# PCA
'''
from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
pca.fit(Xtrain)
print(pca.explained_variance_ratio_)
Xtrain_pca = pca.transform(Xtrain)
print( " Xtrain_pca share: ", Xtrain_pca.shape)
'''
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(kernel="rbf", degree=5, gamma=10)
Xtrain_pca = kpca.fit_transform(Xtrain)
print( " Xtrain Kernel PCA share: ", Xtrain_pca.shape)


index_0 = np.where(Ytrain==0)[0]
index_1 = np.where(Ytrain==1)[0]
index_3 = np.where(Ytrain==3)[0]


# Brain plots
voxel_global_avg = np.zeros((5903,1))
voxel_avg = np.zeros((5903,3))
color_val = ['r','g','b']
vals = [0,1,3]
fig = plt.figure()
plt.hold('on')

ax = fig.add_subplot(221,projection='3d')
ax.scatter(x,y,z)
plt.title('all voxels')
# get averages for each voxel
for i in range(0,5903):
        voxel_global_avg[i] = Xtrain[:,i].mean()
        
# get averages of ech voxel by label
for j in range(0,3):
    print j
    index = np.where(Ytrain==vals[j])[0]
    print index
    for i in range(0,5903):
        voxel_avg[i,j] = Xtrain[index,i].mean()
      
thresh = .1

subplot_val=[222,223,224]
titles = ['class 0', 'class 1', 'class 3']
# see if mean for label exceed threshold then plot
for j in range(0,3):
    ax = fig.add_subplot(subplot_val[j],projection='3d')
    plt.title(titles[j])
    for i in range(0,5903):
        if( abs(voxel_avg[i,j] - voxel_global_avg[i]) > thresh):
            ax.scatter(x[i],y[i],z[i],c=color_val[j])
    
plt.show()

# try another visualization
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
for i in range(0,5903):
    max_val = voxel_avg[i,:].max()
    if( voxel_avg[i,0] == max_val):
        ax.scatter(x[i],y[i],z[i],c=color_val[0])
    elif( voxel_avg[i,1] == max_val):
        ax.scatter(x[i],y[i],z[i],c=color_val[1])
    elif( voxel_avg[i,2] == max_val):
        ax.scatter(x[i],y[i],z[i],c=color_val[2])
plt.show()
