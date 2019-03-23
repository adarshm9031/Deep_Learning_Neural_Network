# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:32:52 2019

@author: adarshm
"""

import numpy as np
def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.dot(np.abs(yhat-y),np.abs(yhat-y))
    ### END CODE HERE ###
    
    return loss