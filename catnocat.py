# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:30:56 2019

@author: adarshm
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import load_dataset as ld
import model as sys

################################
#Load Data
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes=ld.load_data()
################################

################################
#To show the loaded image
index = 1
#plt.imshow(train_set_x_orig[index])
#print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
################################

###############################
#nput Shapes
#train_set_x shape: (209, 64, 64, 3)
#train_set_y shape: (1, 209)
#test_set_x shape: (50, 64, 64, 3)
#test_set_y shape: (1, 50)
###############################

################################
#Flattening Input
#A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b ∗∗ c ∗∗ d, a) is to use:
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print(test_set_x_flatten.shape)
################################

################################
#Standardized
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255
################################

d = sys.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.01, print_cost = True)

# Plot learning curve (with costs)
#costs = np.squeeze(d['costs'])
#plt.plot(costs)
#plt.ylabel('cost')
#plt.xlabel('iterations (per hundreds)')
#plt.title("Learning rate =" + str(d["learning_rate"]))
#plt.show()

