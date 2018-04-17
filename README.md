# GANs
Generative Adversarial Network (GAN) variant implementations.

# Overview
Commented implementations and general comparison introductions for of minimax, non-saturating, wasserstein, wasserstein gradient penalty, least squares, deep regret analytic, bounded equilibrium, and GANs. Paper links are supplied at the beginning of each file. Jupyter notebooks for interactivity supplied.

# Note
The architecture chosen for both the generator (G) and discriminator (D) in each of these files is a simple multilayer perceptron. While this will give sensible output, it is recommended in practice to use deep convolutional architectures (i.e., DCGANs) instead. The main file changes can be see in the functions train_D and train_G of the trainer function, although changes are not limited to only these two areas (e.g. Wasserstein GAN clamps weight in the train function).
