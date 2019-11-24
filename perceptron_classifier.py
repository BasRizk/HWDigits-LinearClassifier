# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
import os

# =============================================================================
# ------------------------------Pathes
# =============================================================================
train_path = "Train"
test_path = "Test"
train_labels_path = "Train/Training Labels.txt"
test_labels_path = "Test/Test Labels.txt"
weights_dir_path = "perceptron_trained_weights"

# =============================================================================
# -------------------------------Utils
# =============================================================================
def get_data(dir_path, labels_path, num_of_imgs): 
    labels = open(labels_path).readlines()
    X = np.zeros([num_of_imgs, 784])
    T_labels = np.zeros([num_of_imgs,1])    
    for filepath in os.listdir(dir_path):
        if filepath.endswith(".jpg"):
            index = int(filepath[:-4]) - 1
            image_array = plt.imread(os.path.join(dir_path, filepath))\
                            .flatten()
            X[index] = image_array
            T_labels[index] = int(labels[index])
    X = np.insert(X, X.shape[1], values=1, axis=1)
    return X, T_labels

def print_progress(iteration_type, iteration_value):
    print( '\r ' + iteration_type + ' %s' % (str(iteration_value)), end = '\r')

# =============================================================================
# -----------------------ALGORITHM IMPLMENTATION
# =============================================================================  

def activation_function(x):
    if(x > 0):
        return 1
    return -1


def train_perceptron(X_train, T_train, learning_rate):
    # TODO implement algorithm dah! 
    weights = np.zeros((X_train.shape[1],1))
    weights[0] = 1
    
    weights_updated = True
    for epoch in range(500):
#        print("Epoch " + str(epoch))
        print_progress("@ Epoch", epoch)
        if not(weights_updated):
            break
        weights_updated = False
        for row in range(X_train.shape[0]):
            
            x_sample = np.expand_dims(X_train[row],axis = 1)
            t_sample = T_train[row]
            
            predicted_label =\
                activation_function(np.mat(weights.T)*np.mat(x_sample))
                
            if(predicted_label != t_sample):
                weights = weights +\
                    (learning_rate*np.mat(x_sample)*t_sample)
                weights_updated = True
    print()
    print("Finished training.")
    return weights


# =============================================================================
# --------------------------------Task
# =============================================================================
X_train, T_train_labels = get_data(train_path, train_labels_path, 2400)
X_test, T_test_labels = get_data(test_path, test_labels_path, 200)

all_weights = np.zeros((10, 10, X_train.shape[1]))
for i in range(0, 10):
    print("=> Begin training on handwritten digit '" + str(i) + "'")
    for learning_phase in range(0, 10):
        learning_rate = math.pow(learning_phase, learning_phase)
        print("..with learning_rate = " + str(learning_rate))
        T_train =  np.where(T_train_labels == i, 1, -1)
        weights = train_perceptron(X_train, T_train, learning_rate)
        all_weights[i][learning_phase] = weights.reshape(weights.shape[0])[0]

if not(os.path.exists(weights_dir_path)):    
    os.makedirs(weights_dir_path)
    
for i in range(all_weights.shape[0]):
    weights_filepath = os.path.join(weights_dir_path, "digit_" + str(i) + "_weights.csv")
    np.savetxt(weights_filepath, all_weights[i], delimiter=",")

# TODO CONFUSION MATRIX CALCULATION











