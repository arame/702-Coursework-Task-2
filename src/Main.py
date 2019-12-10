#%matplotlib inline
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.stats import truncnorm
from files import Files
from local_time import Localtime
from neural_network import NeuralNetwork
import torch

print("*"*100)
print("** Start at ", Localtime.get())
print("*"*100)

print("Cuda is available; ", torch.cuda.is_available())
print("Torch version;     ", torch.__version__)

image_size = 28 # width and height
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
print("Load file at ", Localtime.get())
f = Files("pickled_mnist.pkl")
with open(os.path.join(".", f.file_path), "br") as fh:
    data = pickle.load(fh)
print("File successufully read at ", Localtime.get())
train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]
print("File successufully loaded at ", Localtime.get())
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size

lr = np.arange(no_of_different_labels)
# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)
#train_labels_one_hot = (train_labels).type(torch.FloatTensor)
#test_labels_one_hot = (test_labels).type(torch.FloatTensor)
# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99

#**************************************************
# To get the model to work, adjust these values
epochs = 10
learning_rate = 0.001
#**************************************************

for i in range(5):
    dropout_rate = (i + 1) / 10
    ANN = NeuralNetwork(no_of_in_nodes = image_pixels, 
                                no_of_out_nodes = 10, 
                                no_of_hidden_nodes = 100,
                                learning_rate = learning_rate)
        
    print("-"*100)
    print("Training for dropout rate", dropout_rate)
    print("-"*100)    
    
    weights = ANN.train(train_imgs, 
                        train_labels_one_hot, 
                        epochs=epochs, 
                        dropout_rate = dropout_rate,
                        intermediate_results=True)

    print("-"*100)
    print("Testing for dropout rate", dropout_rate)
    print("-"*100)
    for i in range(epochs):  
        ANN.weights_in_hidden = weights[i][0]
        ANN.weights_hidden_output = weights[i][1] 
        print("-"*100)
        print("epoch: ", i + 1)
    
        corrects, wrongs = ANN.evaluate(train_imgs, train_labels)
        print("accuracy train: ", corrects / ( corrects + wrongs))
        corrects, wrongs = ANN.evaluate(test_imgs, test_labels)
        print("accuracy: test", corrects / ( corrects + wrongs))

    print("Confusion Matrix:")
    cm = ANN.confusion_matrix(train_imgs, train_labels)
    print(cm)
    for i in range(epochs):
        print("digit: ", i, "precision: ", ANN.precision(i, cm), "recall: ", ANN.recall(i, cm))
    print("-"*100)
    print("** End for dropout rate ", dropout_rate)
    print("** At time ", Localtime.get())
    print("-"*100)
print("*"*100)
print("** End at ", Localtime.get())
print("*"*100)