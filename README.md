# Digit Recognizer

A neural network that can guess handwritten digits. Built from scratch using only NumPy without PyTorch or Tensorflow. Currently it can only recognize numbers from the MNIST data set (60000 for training, 10000 for testing). I built this primarily to learn more about how AI works so it isn't the best or most efficient way to build this (I'm not an expert in this).

## How it works

- Architecture: 784 → 128 → 10 (sigmoid hidden layer, softmax output)
- **Forward pass:** input pixels → weighted sum + bias → sigmoid → weighted sum + bias → softmax
- **Loss:** cross-entropy
- **Learning:** backpropagation with plain stochastic gradient descent
  (one image per update, learning rate 0.1)

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

- Mini-batch training for speed
- L2 regularization / early stopping to reduce overfitting
- ReLU activation
- Testing with own handwritten digits

## License

MIT