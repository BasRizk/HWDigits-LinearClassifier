# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os

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
# -----------------------CONFUSION MATRIX RELATED
# =============================================================================
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
    
    