import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm, metrics
import matplotlib.pyplot as plt


def preprocess():
    """
    Input:
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
    """

    mat = loadmat("mnist_all.mat")  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation : (i + 1) * n_validation, :] = mat.get(
            "train" + str(i)
        )[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation : (i + 1) * n_validation, :] = i * np.ones(
            (n_validation, 1)
        )

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp : temp + size_i - n_validation, :] = mat.get("train" + str(i))[
            n_validation:size_i, :
        ]
        train_label[temp : temp + size_i - n_validation, :] = i * np.ones(
            (size_i - n_validation, 1)
        )
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp : temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp : temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if sigma[i] > 0.001:
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return (
        train_data,
        train_label,
        validation_data,
        validation_label,
        test_data,
        test_label,
    )


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    bias = np.ones((n_data, 1))
    X = np.hstack((bias, train_data))

    h_x = sigmoid(np.dot(X, initialWeights.reshape((n_features + 1, 1))))

    error = -np.sum(labeli * np.log(h_x) + (1 - labeli) * np.log(1 - h_x)) / n_data
    error_grad = np.dot(X.T, (h_x - labeli)) / n_data

    # print(error)
    # print(error_grad.shape)
    return error, error_grad


def blrObjWrapper(initialWeights, *args):
    error, error_grad = blrObjFunction(initialWeights, *args)
    return error, error_grad.flatten()


def blrPredict(W, data):
    """
    blrObjFunction predicts the label of data given the data and parameter W
    of Logistic Regression

    Input:
        W: the matrix of weight of size (D + 1) x 10. Each column is the weight
        vector of a Logistic Regression classifier.
        X: the data matrix of size N x D

    Output:
        label: vector of size N x 1 representing the predicted label of
        corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    bias = np.ones((data.shape[0], 1))
    X = np.hstack((bias, data))

    probabilities = sigmoid(np.dot(X, W))
    label = np.argmax(probabilities, axis=1).reshape((data.shape[0], 1))

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        Y: the label vector of size N x 10 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    n_class = 10
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    training_data = np.hstack((np.ones((n_data, 1)), train_data))
    logits = np.dot(training_data, params.reshape((n_feature + 1, n_class))) 
    expLogits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  
    probability_dist = expLogits / np.sum(expLogits, axis=1, keepdims=True)  
    error_grad = np.dot(training_data.T, (probability_dist - labeli)) / n_data  
    error =  -np.sum(labeli * np.log(probability_dist)) / n_data
    return error, error_grad.flatten()

def mlrPredict(W, data):
    """
    mlrObjFunction predicts the label of data given the data and parameter W
    of Logistic Regression

    Input:
        W: the matrix of weight of size (D + 1) x 10. Each column is the weight
        vector of a Logistic Regression classifier.
        X: the data matrix of size N x D

    Output:
        label: vector of size N x 1 representing the predicted label of
        corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    data = np.hstack((np.ones((data.shape[0], 1)), data))  
    logits = np.dot(data, W)  
    expLogits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  
    probs = expLogits / np.sum(expLogits, axis=1, keepdims=True)  
    label = np.argmax(probs, axis=1)  
    return label.reshape(-1,1)

if __name__ == "__main__":

    """
    Script for Logistic Regression
    """
    (
        train_data,
        train_label,
        validation_data,
        validation_label,
        test_data,
        test_label,
    ) = preprocess()

    # number of classes
    n_class = 10

    # number of training samples
    n_train = train_data.shape[0]

    # number of features
    n_feature = train_data.shape[1]

    Y = np.zeros((n_train, n_class))
    for i in range(n_class):
        Y[:, i] = (train_label == i).astype(int).ravel()

    # Logistic Regression with Gradient Descent
    W = np.zeros((n_feature + 1, n_class))
    initialWeights = np.zeros((n_feature + 1, 1))
    opts = {"maxiter": 100}
    for i in range(n_class):
        labeli = Y[:, i].reshape(n_train, 1)
        args = (train_data, labeli)
        nn_params = minimize(
            blrObjWrapper,
            initialWeights.flatten(),
            jac=True,
            args=args,
            method="CG",
            options=opts,
        )
        W[:, i] = nn_params.x.reshape((n_feature + 1,))

    # Find the accuracy on Training Dataset
    predicted_label = blrPredict(W, train_data)
    print(
        "\n Training set Accuracy:"
        + str(100 * np.mean((predicted_label == train_label).astype(float)))
        + "%"
    )

    # Find the accuracy on Validation Dataset
    predicted_label = blrPredict(W, validation_data)
    print(
        "\n Validation set Accuracy:"
        + str(100 * np.mean((predicted_label == validation_label).astype(float)))
        + "%"
    )

    # Find the accuracy on Testing Dataset
    predicted_label = blrPredict(W, test_data)
    print(
        "\n Testing set Accuracy:"
        + str(100 * np.mean((predicted_label == test_label).astype(float)))
        + "%"
    )

    """
    Script for Support Vector Machine
    """

    print("\n\n--------------SVM-------------------\n\n")
    ##################
    # YOUR CODE HERE #
    ##################
    sample_indexes = np.random.choice(train_data.shape[0], 10000, replace=False)
    sample_train_data = train_data[sample_indexes]
    sample_train_label = train_label[sample_indexes]

    model_linear = svm.SVC(kernel="linear")
    model_linear.fit(sample_train_data, sample_train_label)

    # labels, counts = np.unique(train_label, return_counts=True)
    # print("actual",labels, counts)

    # labels, counts = np.unique(sample_train_label, return_counts=True)
    # print("sample",labels, counts)

    predicted_label = model_linear.predict(train_data)
    # print(predicted_label)
    # print(train_label)
    print(
        "\n SVM linear Training set Accuracy:",
        metrics.accuracy_score(y_true=train_label, y_pred=predicted_label) * 100,
        "%",
    )

    predicted_label = model_linear.predict(validation_data)
    print(
        "\n SVM linear Validation set Accuracy:",
        metrics.accuracy_score(y_true=validation_label, y_pred=predicted_label) * 100,
        "%",
    )

    predicted_label = model_linear.predict(test_data)
    print(
        "\n SVM linear Testing set Accuracy:",
        metrics.accuracy_score(y_true=test_label, y_pred=predicted_label) * 100,
        "%",
    )

    # Test with radial basis function
    model_rbf_1 = svm.SVC(kernel="rbf", gamma=1)
    model_rbf_1.fit(sample_train_data, sample_train_label)

    predicted_label = model_rbf_1.predict(train_data)
    print(
        "\n SVM rbf and gamma=1 Training set Accuracy:",
        metrics.accuracy_score(y_true=train_label, y_pred=predicted_label) * 100,
        "%",
    )

    predicted_label = model_rbf_1.predict(validation_data)
    print(
        "\n SVM rbf and gamma=1 Validation set Accuracy:",
        metrics.accuracy_score(y_true=validation_label, y_pred=predicted_label) * 100,
        "%",
    )

    predicted_label = model_rbf_1.predict(test_data)
    print(
        "\n SVM rbf and gamma=1 Testing set Accuracy:",
        metrics.accuracy_score(y_true=test_label, y_pred=predicted_label) * 100,
        "%",
    )

    # Test with different values of c
    c_values = [1] + [10 * x for x in range(1, 11)]
    train_acc = []
    test_acc = []
    validation_acc = []
    for c in c_values:
        model = svm.SVC(C=c)
        model.fit(sample_train_data, sample_train_label)

        predicted_label = model.predict(train_data)
        accuracy = metrics.accuracy_score(y_true=train_label, y_pred=predicted_label) * 100
        train_acc.append(accuracy)
        print(
            f"\n SVM rbf with c:{c} Training set Accuracy:",
            accuracy,
            "%",
        )

        predicted_label = model.predict(validation_data)
        accuracy = metrics.accuracy_score(y_true=validation_label, y_pred=predicted_label) * 100
        validation_acc.append(accuracy)
        print(
            f"\n SVM  rbf with c:{c} Validation set Accuracy:",
            accuracy,
            "%",
        )

        predicted_label = model.predict(test_data)
        accuracy = metrics.accuracy_score(y_true=test_label, y_pred=predicted_label) * 100
        test_acc.append(accuracy)
        print(
            f"\n SVM rbf with c:{c}Testing set Accuracy:",
            accuracy,
            "%",
        )

    plt.figure(figsize=(10, 6))
    plt.plot(c_values, train_acc, label="Training Accuracy", marker='x', color="grey")
    plt.plot(c_values, validation_acc, label="Validation Accuracy", marker='^', color="yellow")
    plt.plot(c_values, test_acc, label="Testing Accuracy", marker='o', color="green")

    # Customize the plot
    plt.xlabel("C: Regularization parameter")
    plt.ylabel("Accuracy(%)")
    plt.xticks(c_values, labels=c_values)
    # plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # plt.legend(loc="best")
    plt.show()
    """
    Script for Extra Credit Part
    """
    # FOR EXTRA CREDIT ONLY
    W_b = np.zeros((n_feature + 1, n_class))
    initialWeights_b = np.zeros((n_feature + 1, n_class))
    opts_b = {"maxiter": 100}

    args_b = (train_data, Y)
    nn_params = minimize(
        mlrObjFunction,
        initialWeights_b.flatten(),
        jac=True,
        args=args_b,
        method="CG",
        options=opts_b,
    )
    W_b = nn_params.x.reshape((n_feature + 1, n_class))

    # Find the accuracy on Training Dataset
    predicted_label_b = mlrPredict(W_b, train_data)
    print(
        "\n Training set Accuracy:"
        + str(100 * np.mean((predicted_label_b == train_label).astype(float)))
        + "%"
    )

    # Find the accuracy on Validation Dataset
    predicted_label_b = mlrPredict(W_b, validation_data)
    print(
        "\n Validation set Accuracy:"
        + str(100 * np.mean((predicted_label_b == validation_label).astype(float)))
        + "%"
    )

    # Find the accuracy on Testing Dataset
    predicted_label_b = mlrPredict(W_b, test_data)
    print(
        "\n Testing set Accuracy:"
        + str(100 * np.mean((predicted_label_b == test_label).astype(float)))
        + "%"
    )
