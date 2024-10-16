from train import forward_prop, back_prop, loss, np, plt, get_accuracy, get_predictions

## Divide dataset into mini-batches ##
def create_mini_batches(X, Y, mini_batch_size): 
    m = X.shape[1]
    mini_batches = []

    #Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # Partition (Separate/shuffling ensures randomness)
    num_complete_batches = m // mini_batch_size
    for k in range(num_complete_batches): 
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    #If the last mini-batch is smaller than the rest (they need to be the same size)
    if m % mini_batch_size != 0: 
        mini_batch_X = shuffled_X[:, num_complete_batches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_batches * mini_batch_size:]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches

## Gradient descent using the minibatches ## 
def gradient_descent_with_mini_batches(X, Y, iterations, learning_rate, lambd, mini_batch_size): 
    m = X.shape[1]
    n_x = X.shape[0]

    W1 = np.random.randn(10, n_x) * 0.01
    b1 = np.random.randn(10, 1)
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros(10, 1)

    for i in range(iterations): 
        mini_batches = create_mini_batches(X, Y, mini_batch_size=mini_batch_size)

        for batch in mini_batches: 
            (mini_batch_X, mini_batch_Y) = batch

            #forward prop
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, mini_batch_X)

            #Back propt
            dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, mini_batch_X, mini_batch_Y, lambd)

            # Update parameters
            W1 = W1 - learning_rate * dW1
            b2 = b1 - learning_rate * db1
            W2 = W2 - learning_rate * dW2
            b2 = b2 - learning_rate * db2

        ## Print cost and accuracy of each iteration
        if i % 500 == 0: 
            print("Iteration: ", i)
            print("Accuracy: ", f'{get_accuracy(get_predictions(A2), Y) * 100}%')
            print("Cost: ", loss(X, Y, A2, W1, W2, lambd))

    return W1, b1, W2, b2 
