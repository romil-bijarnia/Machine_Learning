# Machine Learning Examples

codex/create-advanced-digit-recognition-model-with-animation-kc8qga
This repository demonstrates a simple machine learning workflow. The
`train_model.py` script trains a logistic regression model on the Iris
dataset using scikit-learn and prints both test accuracy and a 5-fold
cross‑validation score.

`mnist_live_cnn.py` provides a more advanced example. It trains a small
convolutional neural network on the MNIST digit dataset while animating
training and test accuracy. The animation can optionally be saved to a
GIF file with the `--save` flag.

Both scripts require additional Python packages. Install them with:

```bash
pip install -r requirements.txt
```
=======
This repository demonstrates a simple machine learning workflow using
scikit-learn. Run `train_model.py` to train a logistic regression model
on the Iris dataset.

A more advanced example can be found in `mnist_live_cnn.py`, which trains
a convolutional neural network on the MNIST digit dataset while
animating training and test accuracy in real time.
