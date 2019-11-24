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
weights_dir_path = "naive_bayes_trained_weights"

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

# =============================================================================
# -----------------------ALGORITHM IMPLMENTATION
# =============================================================================  

def calculate_u_and_sigma_sqr(X_column, T_train):
    X_class = X_column[np.where(T_train = 1)]
    X_nonclass = X_column[np.where(T_train = -1)]
    
    u_class = (1/X_class.shape[0]) * sum(X_class) 
    u_nonclass  = (1/X_nonclass.shape[0]) * sum(X_nonclass)
    
    sigma_sqr_class =\
        (1/X_class.shape[0]) * sum((X_class - u_class)**2) 
    sigma_sqr_nonclass =\
        (1/X_nonclass.shape[0]) * sum((X_nonclass - u_nonclass)**2) 
    
    return u_class, u_nonclass, sigma_sqr_class, sigma_sqr_nonclass

    
def train_naive_bayes_classifier(X_train, T_train):
    X_train = X_train/255
    
    u_vector = np.zeros(2, X_train.shape[1])
    sigma_sqr_vector = np.zeros(2, X_train.shape[1])
    
    X_train_t = X_train.T
    for column_num in range(X_train_t.shape[0]):
        u_vector[:], sigma_sqr_vector[:] =\
            calculate_u_and_sigma_sqr(X_train_t[column_num], T_train)
            
    print("Finished training.")
    return u_vector, sigma_sqr_vector

def gaussian_distribution_probability(x, u, sigma_sqr):
    result = 1
    result = result * (1/(np.sqrt(2*np.pi*sigma_sqr)))
    result = result * np.e**(-((x-u)**2)/(2*sigma_sqr))
    return result

def naive_bayes_predict(X_sample, u_vector, sigma_sqr_vector):
    p_class = 1
    p_nonclass = 1
    for i in range(X_sample.shape[0]):
        p_class *= gaussian_distribution_probability(X_sample[i],
                                                     u_vector[0][i],
                                                     sigma_sqr_vector[0][i])
        
        p_nonclass *= gaussian_distribution_probability(X_sample[i],
                                                        u_vector[1][i],
                                                        sigma_sqr_vector[1][i])
        
    if (p_class > p_nonclass):
        return 1
    
    return 0
        

# =============================================================================
# --------------------------------Task
# =============================================================================
X_train, T_train_labels = get_data(train_path, train_labels_path, 2400)
X_test, T_test_labels = get_data(test_path, test_labels_path, 200)

# TODO u_vectors all together; same for sigma_sqr_vectors
# TODO apply naive_bayes_predict on all test samples
all_weights = np.zeros((10, 10, X_train.shape[1]))
for i in range(0, 10):
    print("=> Begin training on handwritten digit '" + str(i) + "'")
    T_train = np.where(T_train_labels == i, 1, -1)
    u_vector, sigma_sqr_vector = train_naive_bayes_classifier(X_train, T_train)




# TODO CONFUSION MATRIX CALCULATION











