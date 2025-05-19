# Machine Learning Examples

This repository demonstrates a simple machine learning workflow using
scikit-learn. Run `train_model.py` to train a logistic regression model
on the Iris dataset.

A more advanced example can be found in `mnist_live_cnn.py`, which trains
a convolutional neural network on the MNIST digit dataset while
animating training and test accuracy in real time. The script accepts
CLI options such as `--epochs` to control training length, `--save` to
export the animation as a GIF, and `--checkpoint`/`--resume` to save and
reload model weights so the network can remember what it has learned.
