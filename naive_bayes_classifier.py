# -*- coding: utf-8 -*-

from utils import get_data
from utils import create_confusion_matrix, save_confusion_matrix_plot
import numpy as np
import os

# =============================================================================
# ------------------------------Pathes
# =============================================================================
train_path = "Train"
test_path = "Test"
train_labels_path = "Train/Training Labels.txt"
test_labels_path = "Test/Test Labels.txt"
weights_dir_path = "naive_bayes_trained_weights"
confusion_dir_path = "naive_bayes_confusion"

# =============================================================================
# -----------------------ALGORITHM IMPLMENTATION
# =============================================================================  

def calculate_u_and_sigma_sqr(X_column, T_train):
    X_class = X_column[(T_train == 1)]
    X_nonclass = X_column[(T_train == -1)]
    
    u_class = (1/X_class.shape[0]) * sum(X_class) 
    u_nonclass  = (1/X_nonclass.shape[0]) * sum(X_nonclass)
    
    sigma_sqr_class =\
        (1/X_class.shape[0]) * sum((X_class - u_class)**2) 
    sigma_sqr_nonclass =\
        (1/X_nonclass.shape[0]) * sum((X_nonclass - u_nonclass)**2) 
    
    if(sigma_sqr_class < 0.01):
        sigma_sqr_class = 0.01
    if(sigma_sqr_nonclass < 0.01):
        sigma_sqr_nonclass = 0.01
        
    return (u_class, u_nonclass), (sigma_sqr_class, sigma_sqr_nonclass)

    
def train_naive_bayes_classifier(X_train, T_train):
    u_vector = np.zeros((X_train.shape[1], 2))
    sigma_sqr_vector = np.zeros((X_train.shape[1], 2))
    
    X_train_t = X_train.T
    for column_num in range(X_train_t.shape[0]):
        u_vector[:][column_num], sigma_sqr_vector[:][column_num] =\
            calculate_u_and_sigma_sqr(X_train_t[column_num], T_train)
            
    print("Finished training.")
    return u_vector, sigma_sqr_vector

def gaussian_distribution_probability(x, u, sigma_sqr):
    result = 1.0
    result = result * (1/(np.sqrt(2*np.pi*sigma_sqr)))
    result = result * np.exp((-((x-u)**2)/(2*sigma_sqr)))
    return result

def naive_bayes_predict(X_sample, u_vector, sigma_sqr_vector,
                        one_against_one = False):
    p_class = 1.0
    p_nonclass = 1.0
    for i in range(X_sample.shape[0]):
        p_class *= gaussian_distribution_probability(X_sample[i],
                                                     u_vector[i][0],
                                                     sigma_sqr_vector[i][0])

        p_nonclass *= gaussian_distribution_probability(X_sample[i],
                                                        u_vector[i][1],
                                                        sigma_sqr_vector[i][1])
        
    probability = p_class / p_nonclass
    
    if not(one_against_one):
        return probability
    
    if (probability > 1):
        return 1
    else:
        return 0
        
def naive_bayes_predict_all(X_test, u_vector, sigma_sqr_vector):
    all_predictions = np.zeros((X_test.shape[0], 1))
    for i in range(X_test.shape[0]):
        all_predictions[i] = naive_bayes_predict(X_test[i],
                                                 u_vector,
                                                 sigma_sqr_vector)
        
    return all_predictions
        

# =============================================================================
# --------------------------------Task
# =============================================================================
X_train, T_train_labels = get_data(train_path, train_labels_path, 2400)
X_test, T_test_labels = get_data(test_path, test_labels_path, 200)

X_train = np.divide(X_train, 255)
X_test = np.divide(X_test, 255)

# Training and calculating means and variances
all_u_vectors = all_sigma_sqr_vectors = np.zeros((10, X_train.shape[1], 2))
for i in range(0, 10):
    print("=> Begin training on handwritten digit '" + str(i) + "'")
    T_train = np.where(T_train_labels == i, 1, -1)
    T_train = T_train.reshape(T_train.shape[0])
    all_u_vectors[i], all_sigma_sqr_vectors[i] =\
        train_naive_bayes_classifier(X_train, T_train)
        
# Testing using the natural gaussian distribution
all_predictions = np.zeros((10, X_test.shape[0], 1))
for i in range(0, 10):
    all_predictions[i] = naive_bayes_predict_all(X_test,
                                           all_u_vectors[i],
                                           all_sigma_sqr_vectors[i])

# Creating Confusion Matrix
if not(os.path.exists(confusion_dir_path)):
    os.makedirs(confusion_dir_path)
all_predictions_for_confusion = all_predictions.reshape(all_predictions.shape[:-1]).T
confusion_matrix = create_confusion_matrix(all_predictions_for_confusion, T_test_labels)
img_filepath = os.path.join(confusion_dir_path,
                                    "Confusion.jpg")
save_confusion_matrix_plot(confusion_matrix,"Confusion.jpg" ,  img_filepath)










