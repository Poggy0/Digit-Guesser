import time

import numpy as np
from sklearn.datasets import fetch_openml

import interface as inf


# Convert weighted sums into values between zero and one for hidden neurons.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Turn output scores into probabilities that add up to one.
def softmax(x):
    exp_values = np.exp(x - np.max(x))
    return exp_values / np.sum(exp_values)


# Measure the penalty for assigning a low probability to the correct digit.
def cross_entropy_loss(predicted, true_label):
    predicted_clipped = np.clip(predicted, 1e-12, 1 - 1e-12)
    return -np.log(predicted_clipped[true_label])


# Calculate gradients and update both layers' weights and biases for one image.
def back_prop(input, true_label, output_nodes, hidden_nodes, weights1, biases1, weights_out, biases_out, learning_rate=0.1):
    one_hot = np.zeros(10)
    one_hot[true_label] = 1
    output_error = output_nodes - one_hot

    # Propagate output error backward through the output and hidden layers.
    grad_weights_out = np.outer(hidden_nodes, output_error)
    grad_biases_out = output_error

    hidden_error_raw = np.dot(weights_out, output_error)
    sigmoid_deriv = hidden_nodes * (1 - hidden_nodes)
    hidden_error = hidden_error_raw * sigmoid_deriv

    grad_weights1 = np.outer(input, hidden_error)
    grad_biases1 = hidden_error

    weights_out = weights_out - learning_rate * grad_weights_out
    biases_out = biases_out - learning_rate * grad_biases_out
    weights1 = weights1 - learning_rate * grad_weights1
    biases1 = biases1 - learning_rate * grad_biases1

    return weights1, biases1, weights_out, biases_out


# Predict digits for a batch of test images and return the fraction correct.
def test_run(X_test, y_test, weights1, biases1, weights_out, biases_out):
    hidden_nodes = sigmoid(np.dot(X_test, weights1) + biases1)
    output_nodes = np.dot(hidden_nodes, weights_out) + biases_out
    predictions = np.argmax(output_nodes, axis=1)
    return np.mean(predictions == y_test.astype(int))


class Model:
    # Store the two MNIST pools and initialize the network's parameters.
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.weights1 = np.random.randn(784, 128) * 0.01
        self.biases1 = np.zeros(128)
        self.weightsOut = np.random.randn(128, 10) * 0.01
        self.biasesOut = np.zeros(10)

    # Train on the requested training subset and optionally report test metrics.
    def train(self, epochs, status_update, training_size=None, testing_size=None, metrics_update=None):
        training_size = len(self.X_train) if training_size is None else training_size
        testing_size = len(self.X_test) if testing_size is None else testing_size
        if not 1 <= training_size <= len(self.X_train):
            raise ValueError(f"Training image count must be between 1 and {len(self.X_train)}.")
        if not 1 <= testing_size <= len(self.X_test):
            raise ValueError(f"Testing image count must be between 1 and {len(self.X_test)}.")

        status_update("Starting training")
        for epoch in range(epochs):
            # Track loss and elapsed time for this full pass through the subset.
            start = time.time()
            total_loss = 0
            for i in range(training_size):
                input_image = self.X_train[i]
                true_label = int(self.y_train[i])

                # Make a prediction, measure its loss, then update parameters.
                hidden_nodes = sigmoid(np.dot(input_image, self.weights1) + self.biases1)
                output_nodes = softmax(np.dot(hidden_nodes, self.weightsOut) + self.biasesOut)

                total_loss += cross_entropy_loss(output_nodes, true_label)

                self.weights1, self.biases1, self.weightsOut, self.biasesOut = back_prop(
                    input_image, true_label, output_nodes, hidden_nodes,
                    self.weights1, self.biases1, self.weightsOut, self.biasesOut
                )

            # Send epoch metrics back to the interface when callbacks are provided.
            avg_loss = total_loss / training_size
            elapsed = time.time() - start
            if metrics_update is not None:
                accuracy = self.test(testing_size)
                metrics_update(epoch + 1, avg_loss, accuracy, elapsed)
            status_update(
                f"Epoch {epoch + 1}/{epochs}, loss: {avg_loss:.4f}, time: {elapsed:.2f}s"
            )

        status_update("Training finished")

    # Evaluate the model on the requested prefix of the separate test pool.
    def test(self, testing_size=None):
        testing_size = len(self.X_test) if testing_size is None else testing_size
        if not 1 <= testing_size <= len(self.X_test):
            raise ValueError(f"Testing image count must be between 1 and {len(self.X_test)}.")
        return test_run(
            self.X_test[:testing_size], self.y_test[:testing_size],
            self.weights1, self.biases1, self.weightsOut, self.biasesOut
        )


# Load and normalize MNIST, then launch the desktop interface.
def main():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    # Keep MNIST's official 60,000-image training and 10,000-image test split.
    X_train = mnist.data[:60000] / 255.0
    X_test = mnist.data[60000:] / 255.0
    y_train = mnist.target[:60000]
    y_test = mnist.target[60000:]

    model = Model(X_train, y_train, X_test, y_test)
    root = inf.init_interface()
    inf.init_buttons(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
