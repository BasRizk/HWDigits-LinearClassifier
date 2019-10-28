# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os

train_path = "Train"
test_path = "Test"
train_labels_path = "Train/Training Labels.txt"
test_labels_path = "Test/Test Labels.txt"


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

def create_confusion_matrix(T_predict, T_labels, label):
    T_predict = (T_predict > 0).astype(int)
    T_orig = np.mat(np.transpose(np.where(T_labels == label, 2, 0)))
    diff_vector = (T_orig - T_predict)
    # 2 - 1 = 1; 2-0 = 0
    T_true_true = np.count_nonzero(diff_vector == 1) 
    # 0 - 1 = -1
    T_false_true = np.count_nonzero(diff_vector < 0)
    # 2 - 0 = 2
    T_false_false = np.count_nonzero(diff_vector == 2)
    T_true_false = T_labels.shape[0] - T_true_true -\
                    T_false_true - T_false_false
#    TODO MAYBE CHANGE ORDER
    conf_mat = np.zeros([2,2])
    conf_mat[0][:] = T_true_true, T_false_true
    conf_mat[1][:] = T_false_false, T_true_false
    return conf_mat

# =============================================================================
# -----------------------ALGORITHM IMPLMENTATION
# =============================================================================  
X_train, T_train_labels = get_data(train_path, train_labels_path, 2400)
X_test, T_test_labels = get_data(test_path, test_labels_path, 200)

confusion_matrices = np.zeros([10,2,2])
for i in range(0, 10):
    T_train =  np.where(T_train_labels == i, 1, -1)
    X_train_t = np.transpose(X_train)
    weights = np.linalg.pinv(np.mat(X_train_t)*np.mat(X_train))\
                *np.mat(X_train_t)*np.mat(T_train)
                
    T_test = np.transpose(weights)*np.transpose(X_test)
    conf_mat = create_confusion_matrix(T_test, T_test_labels, i)
    confusion_matrices[i] = conf_mat

    
#    
#T_train_1 = np.where(T_train_labels == 1, 1, -1)
#X_train_t = np.transpose(X_train)
#weights_1 = np.linalg.pinv(np.mat(X_train_t)*np.mat(X_train))\
#                *np.mat(X_train_t)*np.mat(T_train_1)
#
#T_test_1 = np.transpose(weights_1)*np.transpose(X_test)
#
#conf_mat_1 = create_confusion_matrix(T_test_1, T_test_labels, 1)
#conf_mat_1
