import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import itertools
import math

## HYPERPARAMETERS ## (for specified training)
learning_rate = 0.001
lambd = 0.3
dropout_rate = 0.8
training_iterations = 1000
mini_batch_size = 64

## CURRENT BEST ##
# test accuracy = 93.777777%

## Range of values for hyperparameters (for finding best hyperparameters)
alpha_values = [.00001, .0001, .001, .01] # learning rate
lambda_values = [0.1, 0.4, 0.7, 1] # regularization strength
mini_batches = [16] # mini batch sizes
iteration_values = [1000] # number of iterations 


## Variables to store best parameters
best_accuracy = 0
best_hyperparams = {}

# load all data (training and testing)
data = pd.read_csv('data/train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:900].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255 # flatten data since they are images

data_train = data[1000:m].T # partition data into training and test set
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
_, m_train = X_train.shape

# load initial parameters
def init_params(): 
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# one (of many) activation functions
def ReLU(Z): 
    return np.maximum(0, Z)

def softmax(Z): 
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def one_hot(Y): 
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z): 
    return Z > 0

def apply_dropout(A, p): 
    dropout_mask = (np.random.rand(*A.shape) < p) / p
    A *= dropout_mask
    return A, dropout_mask

def forward_prop(W1, b1, W2, b2, X): 
    Z1 = W1.dot(X) + b1 ## dot product of W1 with bias (b1) added
    A1 = ReLU(Z1) ## activation function applied to Z1 

    Z2 = W2.dot(A1) + b2 ## dot product of Z2 with bias (b2) added
    A2 = softmax(Z2) ## creates an array of certainties of outputs (0 - 9) 

    return Z1, A1, Z2, A2

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y, lambd=0.01): 
    m = Y.shape[1]  # Use the second dimension for mini-batch size
    dZ2 = A2 - Y  # Y is now 2D
    dW2 = 1 / m * dZ2.dot(A1.T) + (lambd / m) * W2
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)

    dW1 = 1 / m * dZ1.dot(X.T) + (lambd / m) * W1
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha): 
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def get_predictions(A2): 
    return np.argmax(A2, 0) ## returns the index of the highest "certainty" value - this is what the neural network thinks

def get_accuracy(predictions, Y): 
    return (np.sum(predictions == np.argmax(Y, 0)) / Y.shape[1])  # Adjusted to handle 2D Y

def loss(X, Y, A2, W1, W2, lambd): 
    m = Y.shape[1]  # Use the second dimension for mini-batch size
    log_likelihood = -np.log(A2[Y.argmax(axis=0), range(m)])  # Y is now 2D
    loss = 1 / m * np.sum(log_likelihood)

    L2_regularization = lambd / (2 * m) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))  ## helps with overfitting

    total_loss = loss + L2_regularization
    return total_loss

def gradient_descent(X, Y, iterations, alpha, lambd): 
    W1, b1, W2, b2 = init_params()
    for i in range(iterations): 
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y, lambd)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 5000 == 0: 
            print("Iteration: ", i)
            print("Accuracy: ", f'{get_accuracy(get_predictions(A2), Y) * 100}%')
            print("Cost: ", loss(X, Y, A2, W1, W2, lambd))

    return W1, b1, W2, b2

## PREDICTIONS ##
def make_predictions(X, W1, b1, W2, b2): 
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2): 
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation="nearest")
    plt.show()

## save params to test on later ##
def save_params(W1, b1, W2, b2, filename='trained_params.pkl'): 
    with open(filename, 'wb') as f: 
        pickle.dump((W1, b1, W2, b2), f)
    print(f'Parameters saved to {filename}')

def load_saved_params(filename='trained_params.pkl'): 
    with open(filename, 'rb') as f: 
        W1, b1, W2, b2 = pickle.load(f)
    print(f'Parameters loaded from {filename}')
    return W1, b1, W2, b2
    
def find_best_params(best_accuracy): 
    for alpha, lambd, iterations, mini_batch_size in itertools.product(alpha_values, lambda_values, iteration_values, mini_batches): 
        print(f'\nTraining with learning rate: {alpha}, reg. strength={lambd}, iterations={iterations}, mini batch size={mini_batch_size}')

        ## Train model 
        W1, b1, W2, b2 = gradient_descent_with_mini_batches(X_train, Y_train, iterations=iterations, learning_rate=alpha, lambd=lambd, mini_batch_size=mini_batch_size)

        ## Evaluate on testing set (X_dev, Y_dev)
        Z1_val, A1_val, Z2_val, A2_val = forward_prop(W1, b1, W2, b2, X_dev)
        predictions_train = get_predictions(A2_val)
        accuracy = get_accuracy(predictions_train, Y_dev)

        print(f'Validation Accuracy: {accuracy * 100}%')

        if accuracy > best_accuracy: 
            best_accuracy = accuracy
            best_hyperparams['learning_rate'] = alpha
            best_hyperparams['lambd'] = lambd
            best_hyperparams['iterations'] = iterations
            best_hyperparams['mini batch size'] = mini_batch_size

    print(f'Best Hyperparameters: {best_hyperparams}')

## OPTIMIZERS ##

# MINI BATCH TRAINING #

## Divide dataset into mini-batches ##


## Greate reference Andrew NG: https://github.com/gaoisbest/Machine-Learning-and-Deep-Learning-basic-concepts-and-sample-codes/blob/master/Andrew%20Ng's%20DL%20-%20Class%202%20-%20Week%202.ipynb
def random_mini_batches(X, Y, mini_batch_size): 
    m = X.shape[1]
    mini_batches = []

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    #Partition
    num_complete_batches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_batches): 
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_batches*mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_batches*mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def gradient_descent_with_mini_batches(X, Y, iterations, learning_rate, lambd, mini_batch_size): 
    
    W1, b1, W2, b2 = init_params()

    for i in range(iterations): 

        minibatches = random_mini_batches(X, Y, mini_batch_size)

        for minibatch in minibatches: 
            (minibatch_x, minibatch_y) = minibatch

            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, minibatch_x)
            dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, minibatch_x, minibatch_y, lambd)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        if i % 500 == 0: 
            print("Iteration: ", i)
            print("Training accuracy: ", f'{get_accuracy(get_predictions(A2), minibatch_y) * 100}%')
            
            ## get the validation accuracy ## 
            Z1_val, A1_val, Z2_val, A2_val = forward_prop(W1, b1, W2, b2, X_dev)
            val_predictions = get_predictions(A2_val)
            val_accuracy = get_accuracy(val_predictions, Y_dev)

            print("Validation accuracy: ", f'{val_accuracy * 100}%')
            print("Cost: ", loss(minibatch_x, minibatch_y, A2, W1, W2, lambd)) 

    return W1, b1, W2, b2





if __name__ == "__main__": 
    # Convert labels to one-hot encoding
    Y_train = one_hot(Y_train)
    Y_dev = one_hot(Y_dev)

    print(f'Learning rate: {learning_rate}')
    print(f'Lambd: {lambd}')
    print(f'Dropout rate: {dropout_rate}')
    print(f'Training iterations: {training_iterations}\n')

    # Mini Batch Training #
    # W1, b1, W2, b2 = gradient_descent_with_mini_batches(X_train, Y_train, iterations=training_iterations, learning_rate=learning_rate, lambd=lambd, mini_batch_size=mini_batch_size)
    
    ## run this to find best params
    W1, b1, W2, b2 = find_best_params(best_accuracy)   

    # General Training with just L2 Regularization
    # W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations=training_iterations, alpha=learning_rate, lambd=lambd) ## train model on specific parameters

    ## TEST ACCURACY ## 
    # Z1_dev, A1_dev, Z2_val, A2_val = forward_prop(W1, b1, W2, b2, X_dev)
    # predictions_test = get_predictions(A2_val)
    # accuracy = get_accuracy(predictions_test, Y_dev)
    # print(f"Test accuracy: {accuracy * 100}%")
    

    # save trained model parameters
    save_params(W1, b1, W2, b2)
