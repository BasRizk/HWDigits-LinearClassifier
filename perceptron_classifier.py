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
confusion_dir_path = "perceptron_confusion"

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

def create_confusion_matrix(all_test, T_labels):
    confusion_matrix = np.zeros([10,10])
    for i in range(0, T_labels.shape[0]):
        maxValueIndex = 0
        maxValue = (all_test[i,0])
        for j in range(1,10):
            if(all_test[i,j] > maxValue):
                maxValue = all_test[i,j]
                maxValueIndex = j
        currentLabel = maxValueIndex
        confusion_matrix[T_labels[i,0].astype(int),currentLabel] += 1
    return confusion_matrix

def save_and_scale_confusion_matrix(confusion_matrix,
                                    img_filepath = "confusion.jpg"):
    image = np.where(confusion_matrix >= 0 ,
                     (confusion_matrix / 20) * 255,
                     (confusion_matrix / 20) * 255)
    scaledImage = np.zeros([1,1])
    
    #Scaling the 10x10 array to 500x500
    for i in range(0,10):
        for j in range(0,10):
            if(j == 0):
                row = np.full((50,50) , image[i,0])
            else:
                row = np.append( row, np.full( (50,50) , image[i,j] ) , 0 )
        if(i == 0):
            scaledImage = row
        else:
            scaledImage = np.append(scaledImage, row , 1)
            
    from PIL import Image
    
    new_p = Image.fromarray(np.transpose(image))
    new_p = new_p.convert("L")
    new_p.save(img_filepath)
    new_p_scaled = Image.fromarray(np.transpose(scaledImage))
    new_p_scaled = new_p_scaled.convert("L")
    new_p_scaled.save(img_filepath + "_scaled.jpg")

# =============================================================================
# -----------------------ALGORITHM IMPLMENTATION
# =============================================================================  

def activation_function(x):
    if(x > 0):
        return 1
    return -1


def train_perceptron(X_train, T_train, learning_rate):
    weights = np.zeros((X_train.shape[1],1))
    weights[0] = 1
    
    weights_updated = True
    for epoch in range(500):
        print_progress("@ Epoch", epoch)
        weights_updated = False
        for row in range(X_train.shape[0]):
            
            x_sample = X_train[row]
            t_sample = T_train[row]
            
            predicted_label =\
                activation_function(weights.T.dot(x_sample))
                
            if(predicted_label != t_sample):
                weights = weights +\
                    (learning_rate*x_sample*t_sample)
                weights_updated = True
                
        if not(weights_updated):
            break
        
    print()
    print("Finished training.")
    return weights

def perceptron_predict(X_test, weights, one_versus_one = False):
    predictions = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        x_sample = X_test[i]
        if(one_versus_one):
            predictions[i] =\
                activation_function(weights.T.dot(x_sample))
        else:
            predictions[i] =\
                weights.T.dot(x_sample)
    return predictions

# =============================================================================
# --------------------------------Task
# =============================================================================
X_train, T_train_labels = get_data(train_path, train_labels_path, 2400)
X_test, T_test_labels = get_data(test_path, test_labels_path, 200)
X_train = np.expand_dims(X_train, axis = 2)
X_test = np.expand_dims(X_test, axis = 2)

# Training Phase
all_weights = np.zeros((10, 10, X_train.shape[1]))
for i in range(0, 10):
    print("=> Begin training on handwritten digit '" + str(i) + "'")
    T_train =  np.where(T_train_labels == i, 1, -1)
    for learning_phase in range(0, 10):
        learning_rate = math.pow(10, -learning_phase)
        print("..with learning_rate = " + str(learning_rate))
        weights = train_perceptron(X_train, T_train, learning_rate)
        all_weights[i][learning_phase] = weights.reshape(weights.shape[0])


# Saving trained weights
if not(os.path.exists(weights_dir_path)):    
    os.makedirs(weights_dir_path)
    
    for i in range(all_weights.shape[0]):
        weights_filepath = os.path.join(weights_dir_path,
                                        "digit_" + str(i) + "_weights.csv")
        np.savetxt(weights_filepath, all_weights[i], delimiter=",")


# Testing Phase
all_predictions = np.zeros((10, 10, X_test.shape[0]))
# loop over the digits classes weights
for i in range(all_weights.shape[0]):
    # loop over the different learning_rate weights    
    for j in range(all_weights.shape[1]):
        target_weights = np.expand_dims(all_weights[i][j], axis = 1)
        predictions = perceptron_predict(X_test, target_weights)
        all_predictions[j][i] = predictions


# Calculating confusion matrices
if not(os.path.exists(confusion_dir_path)):
    os.makedirs(confusion_dir_path)
    for i in range(0, 10):
        confusion_matrix = create_confusion_matrix(all_predictions[i].T, T_test_labels)
        img_filepath = os.path.join(confusion_dir_path,
                                    "Confusion-" + str(i) + ".jpg")
        save_and_scale_confusion_matrix(confusion_matrix, img_filepath)
        











