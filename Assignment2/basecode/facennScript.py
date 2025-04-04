'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from math import sqrt
import time
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return 1/(1+np.exp(-z))
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    n = training_data.shape[0]
    training_data = np.append(training_data, np.ones((n, 1)), axis=1)
    h1 = sigmoid(np.dot(training_data, w1.T))
    h1 = np.append(h1, np.ones((h1.shape[0], 1)), axis=1)
    output = sigmoid(np.dot(h1, w2.T))

    y = np.zeros((n, n_class))
    
    for i, value in enumerate(training_label):
        digit = int(value)
        y[i][digit] = 1

    obj_val = -np.sum(y * np.log(output) + (1 - y) * np.log(1 - output)) / n + (lambdaval / (2 * n)) * (np.sum(w1[:, :-1] ** 2) + np.sum(w2[:, :-1] ** 2))

    delta_output = output - y
    delta_h1 = np.dot(delta_output, w2) * h1 * (1 - h1)
    delta_h1 = delta_h1[:, :-1]

    grad_w1 = (np.dot(delta_h1.T, training_data) + lambdaval * np.append(w1[:, :-1], np.zeros((n_hidden, 1)), axis=1)) / n
    grad_w2 = (np.dot(delta_output.T, h1) + lambdaval * np.append(w2[:, :-1], np.zeros((n_class, 1)), axis=1)) / n
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), axis=0)
    return (obj_val, obj_grad)
    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    data_bias = np.append(data, np.ones((data.shape[0], 1)), axis=1)
    h1 = sigmoid(np.dot(data_bias, w1.T))
    h1 = np.append(h1, np.ones((h1.shape[0], 1)), axis=1)
    output = sigmoid(np.dot(h1, w2.T))
    labels = np.argmax(output, axis=1)
    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('./Assignment2/basecode/face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def plot_results(results):
    """
    Plots the results of the tuning process, including accuracy and training time.
    Args:
    results (list of tuples): Each tuple contains (lambda, hidden_units, accuracy, train_time)
    """
    # Unzipping the results into separate lists
    lambdas, hidden_units, accuracies, train_times = zip(*results)

    # Converting arrays.
    lambdas = np.array(lambdas)
    hidden_units = np.array(hidden_units)
    accuracies = np.array(accuracies)
    train_times = np.array(train_times)

    # Plotting the Accuracy vs Lambda for each hidden unit
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    for n_hidden in np.unique(hidden_units):
        mask = hidden_units == n_hidden
        plt.plot(lambdas[mask], accuracies[mask], label=f"Hidden Units: {n_hidden}")
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Lambda for Different Hidden Units')
    plt.legend()

    # Plotting the training time vs Lambda for each hidden unit
    plt.subplot(1, 2, 2)
    for n_hidden in np.unique(hidden_units):
        mask = hidden_units == n_hidden
        plt.plot(lambdas[mask], train_times[mask], label=f"Hidden Units: {n_hidden}")
    plt.xlabel('Lambda')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs Lambda for Different Hidden Units')
    plt.legend()

    plt.tight_layout()
    plt.show()

def tune_hyperparameters(n_input, n_hidden_options, n_class, train_data, train_label, valid_data, valid_label, lambda_options):
    best_lambda = None
    best_hidden_units = None
    best_accuracy = 0
    results = []

    for lmbda in tqdm(lambda_options):
        for n_hidden in tqdm(n_hidden_options):
            
            # noting the starting time for training
            start_time = time.time()

            # Initializing the weights for this set of configuration
            initial_W1 = initializeWeights(n_input, n_hidden)
            initial_W2 = initializeWeights(n_hidden, n_class)
            
            # Combining the W1 and W2 into a single vector for better optimization
            initial_weights = np.concatenate((initial_W1.flatten(), initial_W2.flatten()), 0)
            
            # Now using this set, we are training the network
            args = (n_input, n_hidden, n_class, train_data, train_label, lmbda)
            opts = {'maxiter': 50} 

            trained_weights = minimize(nnObjFunction, initial_weights, jac=True, args=args, method='CG', options=opts)
            
            # noting the end time and saving the total training time for this set of hyperpaprameters
            end_time = time.time()
            train_time = end_time-start_time

            # Extracting the trained weights and then reshapping them
            W1 = trained_weights.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
            W2 = trained_weights.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
            
            # Performing prediction on the validation set
            predicted_labels = nnPredict(W1, W2, valid_data)
            accuracy = 100 * np.mean((predicted_labels == valid_label).astype(float))
            results.append((lmbda, n_hidden, accuracy, train_time))
            
            # If this batch of hyperparameters is better, then we are updating it.
            if accuracy > best_accuracy:
                best_lambda = lmbda
                best_hidden_units = n_hidden
                best_accuracy = accuracy

    plot_results(results)
    
    return best_lambda, best_hidden_units, best_accuracy, results

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2
lambda_options = [0, 10, 20, 30]
n_hidden_options = [128, 256, 512, 1024]
# Calling the function "tune_hyperparameter" to achieve the best set of hyperparameters possible.
best_lambda, best_hidden_units, best_accuracy, results = tune_hyperparameters(
    n_input, n_hidden_options, n_class, train_data, train_label, validation_data, validation_label, lambda_options
)
n_hidden = best_hidden_units

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# setting the regularization hyper-parameter as the best lambda that we got.
lambdaval = best_lambda
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
