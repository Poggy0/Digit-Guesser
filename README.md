# Digit Recognizer

A neural network that can guess handwritten digits. Built from scratch using only NumPy without PyTorch or Tensorflow. Currently it can only recognize numbers from the MNIST data set. I built this primarily to learn more about how AI works so it isn't the best or most efficient way to build this (I'm not an expert in this).

## How it works

- **Architecture:** 784 → 128 → 10 (sigmoid hidden layer, softmax output)
- **Forward pass:** input pixels → weighted sum + bias → sigmoid → weighted sum + bias → softmax
- **Loss:** cross-entropy
- **Learning:** backpropagation with mini-batch gradient descent
  (32 images per update by default; batch size is configurable in the interface)

## Interface

Run `python main.py` to open the desktop interface. Choose the number of epochs
and the number of training and testing images before starting. The image counts
can be adjusted independently within the separate MNIST pools: up to 60,000
training images and 10,000 test images. The test set is separate from the
training set. You can also set the batch size; the default is 32 images per
weight update, and the final batch may be smaller if the training count is not
divisible by the batch size.
During training, the interface reports the average training loss and plots
held-out test accuracy after each epoch. The test button evaluates the current
model using the selected number of test images. When cancelling the training, cancellation takes effect after the current
mini-batch finishes.

## Results

- Test accuracy: **97.78%** on 10,000 MNIST test images
- Accuracy peaked around epoch 40, and after that it overfit

## Setup

```bash
git clone https://github.com/Poggy0/Digit-Guesser.git
cd Digit-Guesser
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install numpy scikit-learn pandas
python main.py
```

The MNIST dataset is downloaded automatically on first run via scikit-learn.

## Possible improvements

- L2 regularization / early stopping to reduce overfitting
- ReLU activation
- Testing with own handwritten digits