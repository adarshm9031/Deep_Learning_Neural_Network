# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 15:11:31 2019

@author: adarshm
"""
import numpy as np
import predict as pre
import optimize as op
import initialize_with_zeros as ini





def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost = False):
    #X_train --(num_px * num_px * 3, m_train)
    #Y_train --(1, m_train)
    #X_test -- (num_px * num_px * 3, m_test)
    #Y_test -- (1, m_test)
    #Returns:
    #d -- dictionary containing information about the model.
    
    w, b = ini.initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = op.optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = pre.predict(w, b, X_test)
    Y_prediction_train = pre.predict(w, b, X_train)


    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d