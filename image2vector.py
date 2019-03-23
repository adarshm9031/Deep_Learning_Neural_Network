# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:51:58 2019

@author: adarshm
"""

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2],1))
    ### END CODE HERE ###
    
    return v