import time

import numpy as np
from sklearn.datasets import fetch_openml

import interface as inf

l2_lambda = 0.0001

# Convert weighted sums into values between zero and one for hidden neurons.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Turn output scores into probabilities that add up to one.
def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


# Measure the average penalty for assigning low probabilities to correct digits.
def cross_entropy_loss(predicted, true_labels, weights):
    predicted_clipped = np.clip(predicted, 1e-12, 1 - 1e-12)
    batch_indices = np.arange(len(true_labels))

    #Cross entropy loss
    ce = np.mean(-np.log(predicted_clipped[batch_indices, true_labels]))

    #L2 Regulation penalty: sum of squared weights across all layers
    l2 = l2_lambda / 2 * sum(np.sum(w**2) for w in weights)

    return ce + l2


# Average a batch's gradients and apply one update to the network parameters.
def back_prop(inputs, true_labels, output_nodes, hidden_nodes, weights1, biases1, weights_out, biases_out, learning_rate=0.1):
    batch_size = len(true_labels)
    one_hot = np.zeros_like(output_nodes)
    one_hot[np.arange(batch_size), true_labels] = 1
    output_error = output_nodes - one_hot

    # Propagate output error backward through the output and hidden layers.
    grad_weights_out = np.dot(hidden_nodes.T, output_error) / batch_size
    grad_biases_out = np.mean(output_error, axis=0)

    hidden_error_raw = np.dot(output_error, weights_out.T)
    sigmoid_deriv = hidden_nodes * (1 - hidden_nodes)
    hidden_error = hidden_error_raw * sigmoid_deriv

    grad_weights1 = np.dot(inputs.T, hidden_error) / batch_size
    grad_biases1 = np.mean(hidden_error, axis=0)

    grad_weights_out += l2_lambda * weights_out
    grad_weights1 += l2_lambda * weights1

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
    def train(
        self,
        epochs,
        status_update,
        training_size=None,
        testing_size=None,
        metrics_update=None,
        batch_size=32,
        cancel_event=None,
    ):
        training_size = len(self.X_train) if training_size is None else training_size
        testing_size = len(self.X_test) if testing_size is None else testing_size
        if not 1 <= training_size <= len(self.X_train):
            raise ValueError(f"Training image count must be between 1 and {len(self.X_train)}.")
        if not 1 <= testing_size <= len(self.X_test):
            raise ValueError(f"Testing image count must be between 1 and {len(self.X_test)}.")
        if not 1 <= batch_size <= len(self.X_train):
            raise ValueError(f"Batch size must be between 1 and {len(self.X_train)}.")

        status_update("Starting training")
        for epoch in range(epochs):
            # Track loss and elapsed time for this full pass through the subset.
            start = time.time()
            total_loss = 0
            # Shuffle the selected training images so batches vary in order each epoch.
            shuffled_indices = np.random.permutation(training_size)
            for start_index in range(0, training_size, batch_size):
                if cancel_event is not None and cancel_event.is_set():
                    status_update("Training cancelled")
                    return

                batch_indices = shuffled_indices[start_index:start_index + batch_size]
                input_images = self.X_train[batch_indices]
                true_labels = self.y_train[batch_indices].astype(int)

                # Make batch predictions and measure the average loss before updating.
                hidden_nodes = sigmoid(np.dot(input_images, self.weights1) + self.biases1)
                output_nodes = softmax(np.dot(hidden_nodes, self.weightsOut) + self.biasesOut)

                batch_loss = cross_entropy_loss(output_nodes, true_labels, weights = [self.weights1, self.weightsOut])
                total_loss += batch_loss * len(batch_indices)

                self.weights1, self.biases1, self.weightsOut, self.biasesOut = back_prop(
                    input_images, true_labels, output_nodes, hidden_nodes,
                    self.weights1, self.biases1, self.weightsOut, self.biasesOut
                )

                # Stop cleanly after the current batch has finished updating.
                if cancel_event is not None and cancel_event.is_set():
                    status_update("Training cancelled")
                    return

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
