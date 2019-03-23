# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:50:45 2019

@author: adarshm
"""


import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import os
from PIL import Image
from scipy import ndimage
import load_dataset as ld
import model as sys
import os
import h5py
import glob
from keras_preprocessing import image



train_path   = "C:\\Users\\adarshm\\Documents\\Python Scripts\\Programming\\LR\\datasets\\human\\train"
test_path    = "C:\\Users\\adarshm\\Documents\\Python Scripts\\Programming\\LR\\datasets\\human\\test"
train_labels = os.listdir(train_path)
test_labels  = os.listdir(test_path)
# tunable parameters
image_size       = (64, 64)
num_train_images = 500
num_test_images  = 40
num_channels     = 3

# train_x dimension = {(64*64*3), 1500}
# train_y dimension = {1, 1500}
# test_x dimension  = {(64*64*3), 100}
# test_y dimension  = {1, 100}
train_x = np.zeros(((image_size[0]*image_size[1]*num_channels), num_train_images))
train_y = np.zeros((1, num_train_images))
test_x  = np.zeros(((image_size[0]*image_size[1]*num_channels), num_test_images))
test_y  = np.zeros((1, num_test_images))

#----------------
# TRAIN dataset
#----------------
count = 0
num_label = 0
for i, label in enumerate(train_labels):
	cur_path = train_path + "\\" + label
	for image_path in glob.glob(cur_path + "/*.jpg"):
		img = image.load_img(image_path, target_size=image_size)
		x   = image.img_to_array(img)
		x   = x.flatten()
		x   = np.expand_dims(x, axis=0)
		train_x[:,count] = x
		train_y[:,count] = num_label
		count += 1
	num_label += 1
  
print(train_x.shape)  
#--------------
# TEST dataset
#--------------
count = 0 
num_label = 0 
for i, label in enumerate(test_labels):
	cur_path = test_path + "\\" + label
	for image_path in glob.glob(cur_path + "/*.jpg"):
		img = image.load_img(image_path, target_size=image_size)
		x   = image.img_to_array(img)
		x   = x.flatten()
		x   = np.expand_dims(x, axis=0)
		test_x[:,count] = x
		test_y[:,count] = 0
		count += 1
	num_label += 1

#------------------
# standardization
#------------------
train_x = train_x/255.
test_x  = test_x/255.

print ("train_labels : " + str(train_labels))
print ("train_x shape: " + str(train_x.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x shape : " + str(test_x.shape))
print ("test_y shape : " + str(test_y.shape))

#-----------------
# save using h5py
#-----------------
h5_train = h5py.File("train_x.h5", 'w')
h5_train.create_dataset("data_train_x", data=np.array(train_x))
h5_train.create_dataset("data_train_y", data=np.array(train_y))
h5_train.close()

h5_test = h5py.File("test_x.h5", 'w')
h5_test.create_dataset("data_test", data=np.array(test_x))
h5_test.create_dataset("data_train_y", data=np.array(test_y))
h5_test.close()

d = sys.model(train_x, train_y, test_x, test_y, num_iterations = 2000, learning_rate = 0.001, print_cost = True)