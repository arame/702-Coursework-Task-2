import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
from activation import Activation
from derivative import Derivative
from dropout import Dropout
from scipy.stats import truncnorm


class NeuralNetwork:
    # intialization method (constructor)
    def __init__(self, 
                 no_of_in_nodes, 
                 no_of_out_nodes, 
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate 
        self.create_weight_matrices()

# Code for training  

    def truncated_normal(self, mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, 
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)

    def create_weight_matrices(self):
        """ 
        A method to initialize the weight 
        matrices of the neural network
        """
        rad = 2 / np.sqrt(self.no_of_in_nodes)
        X = self.truncated_normal(mean=0, 
                             sd=1, 
                             low=-rad, 
                             upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes))
        rad = 2 / np.sqrt(self.no_of_hidden_nodes)
        X = self.truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_output = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))

      
    def train_single(self, input_vector, target_vector):
        """
        input_vector and target_vector can be tuple, 
        list or ndarray
        """
        
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        
        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        output_hidden = Activation.reLU(output_vector1)
        output_hidden *= Dropout.get_mask(output_vector1)
        output_vector2 = np.dot(self.weights_hidden_output, output_hidden)
        output_network = Activation.reLU(output_vector2)
        output_network *= Dropout.get_mask(output_vector2)
        output_errors = target_vector - output_network
        # update the weights:
        #tmp = output_errors * Derivative.sigmoid(output_network)
        try:
            tmp = output_errors * Derivative.reLU(output_network)
            tmp = self.learning_rate  * np.dot(tmp, output_hidden.T)
            self.weights_hidden_output += tmp 
        except:
            print("Something went wrong when writing to the file")

        # calculate hidden errors:
        try:
            hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)
        except:
            print("Something went wrong when writing to the file")
        # ----------------------------------------------------------------------
        # update the weights:
        tmp1 = Derivative.reLU(output_hidden)
        tmp = hidden_errors * tmp1
        # -----------------------------------------------------------------------
        self.weights_in_hidden += self.learning_rate * np.dot(tmp, input_vector.T)
        
    def train(self, data_array, 
              labels_one_hot_array,
              epochs=1,
              dropout_rate=0.5,
              intermediate_results=False):
        intermediate_weights = []
        Dropout.filter_percentage = dropout_rate
        for _ in range(epochs):  
            print("*", end="")
            for i in range(len(data_array)):
                self.train_single(data_array[i], labels_one_hot_array[i])
            if intermediate_results:
                intermediate_weights.append((self.weights_in_hidden.copy(), 
                                             self.weights_hidden_output.copy()))
        return intermediate_weights        

 # Code for testing       
    
    def predict(self, input_vector):
        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T
        # 1st layer
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = Activation.reLU(output_vector)
        # 2nd layer
        output_vector = np.dot(self.weights_hidden_output, output_vector)
        #output_vector = Activation.sigmoid(output_vector)
        output_vector = Activation.reLU(output_vector)
    
        return output_vector
            
    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.predict(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm    
    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        calc_value = confusion_matrix[label, label] / col.sum()
        return calc_value
    
    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        calc_value = confusion_matrix[label, label] / row.sum()
        return calc_value
    
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.predict(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs