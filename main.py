import numpy as np
from sklearn.datasets import fetch_openml
import time

# Dataset MNIST------
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data      # shape (70000, 784) -- 70,000 images, 784 pixels each
y = mnist.target    # shape (70000,) -- the correct digit label for each image, as strings

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

X_train = X_train / 255.0
X_test = X_test / 255.0
#--------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_values = np.exp(x)
    return exp_values / np.sum(exp_values)

def cross_entropy_loss(predicted, true_label):
    predicted_clipped = np.clip(predicted, 1e-12, 1 - 1e-12)
    return -np.log(predicted_clipped[true_label])

def back_prop(input, true_label, outputNodes, hiddenNodes1, weights1, biases1, weightsOut, biasesOut, learning_rate=0.1):
    # Output Layer error
    one_hot = np.zeros(10)
    one_hot[true_label] = 1
    output_error = outputNodes - one_hot

    grad_weightsOut = np.outer(hiddenNodes1, output_error)
    grad_biasesOut = output_error

    # Hidden Layer error
    hidden_error_raw = np.dot(weightsOut, output_error)
    sigmoid_deriv = hiddenNodes1 * (1 - hiddenNodes1)
    hidden_error = hidden_error_raw * sigmoid_deriv

    grad_weights1 = np.outer(input, hidden_error)
    grad_biases1 = hidden_error

    # Update
    weightsOut = weightsOut - learning_rate * grad_weightsOut
    biasesOut = biasesOut - learning_rate * grad_biasesOut
    weights1 = weights1 - learning_rate * grad_weights1
    biases1 = biases1 - learning_rate * grad_biases1

    return weights1, biases1, weightsOut, biasesOut

def test_run(X_test, y_test, weights1, biases1, weightsOut, biasesOut):
    correct = 0

    for i in range(len(X_test)):
        single_input = X_test[i]
        true_label = int(y_test[i])

        hiddenNodes1 = sigmoid(np.dot(single_input, weights1) + biases1)
        outputNodes = softmax(np.dot(hiddenNodes1, weightsOut) + biasesOut)

        predicted_label = np.argmax(outputNodes)  # index of the highest probability

        if predicted_label == true_label:
            correct += 1

    accuracy = correct / len(X_test)
    return accuracy

epochs = 5

weights1 = np.random.randn(784, 128) * 0.01
biases1 = np.zeros(128)

weightsOut = np.random.randn(128, 10) * 0.01
biasesOut = np.zeros(10)

print(f"Accuracy before Training: {test_run(X_test, y_test, weights1, biases1, weightsOut, biasesOut) * 100}%")

print("")

print("Training:")
for epoch in range(epochs):
    start = time.time()
    total_loss = 0

    for i in range(len(X_train)):
        input = X_train[i]
        true_label = int(y_train[i])

        # Forward pass
        hiddenNodes1 = sigmoid(np.dot(input, weights1) + biases1)
        outputNodes = softmax(np.dot(hiddenNodes1, weightsOut) + biasesOut)

        total_loss += cross_entropy_loss(outputNodes, true_label)

        # Backward pass
        weights1, biases1, weightsOut, biasesOut = back_prop(
            input, true_label, outputNodes, hiddenNodes1,
            weights1, biases1, weightsOut, biasesOut
        )

    end = time.time()
    print(f"Epoch {epoch+1}, average loss: {total_loss / len(X_train)}, current Accuracy: {test_run(X_test, y_test, weights1, biases1, weightsOut, biasesOut) * 100}%, time taken: {end - start}")

print("")

print(f"Final accuracy: {test_run(X_test, y_test, weights1, biases1, weightsOut, biasesOut) * 100}%")