import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return  1/(1+np.exp(-z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    n = 9
    split = 50000

    total_train = np.append(mat['train0'], np.zeros((mat['train0'].shape[0],1)), axis=1)
    test = np.append(mat['test0'], np.zeros((mat['test0'].shape[0],1)), axis=1)

    for i in range(1, n+1):
      total_train = np.append(total_train, np.append(mat[f'train{i}'], i*np.ones((mat[f'train{i}'].shape[0],1)), axis=1), axis=0)
      test = np.append(test, np.append(mat[f'test{i}'], i*np.ones((mat[f'test{i}'].shape[0],1)), axis=1), axis=0)

    indices = np.arange(len(total_train))
    np.random.shuffle(indices)

    train_indices = indices[:split]
    validation_indices = indices[split:]

    train = total_train[train_indices]
    validation = total_train[validation_indices]

    train_data = train[:,:-1] / 255.0
    train_label = train[:, -1].reshape((-1,1))

    validation_data = validation[:, :-1] / 255.0
    validation_label = validation[:, -1].reshape((-1,1))

    test_data = test[:, :-1] / 255.0
    test_label = test[:, -1].reshape((-1,1))
    
    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


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


def nnPredict(w1, w2, data):
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
    return labels.reshape((-1,1))


"""**************Neural Network Script Starts here********************************"""
if __name__ == "__main__":
    
        
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 50

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 0

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    # find the accuracy on Training Dataset

    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset

    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset

    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


