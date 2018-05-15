# Overview
PyTorch version: 0.3.1

Commented / annotated implementations and comparative introductions for minimax, non-saturating, wasserstein, wasserstein gradient penalty, least squares, deep regret analytic, bounded equilibrium, and pack GANs. Paper links are supplied at the beginning of each file. See src folder for files to run via terminal, or notebooks folder for Jupyter notebooks.

The main file changes can be see in the train, train_D, and train_G of the Trainer class, although changes are not completely limited to only these two areas (e.g. Wasserstein GAN clamps weight in the train function, BEGAN gives multiple outputs from train_D).

# Note
The architecture chosen for both the generator (G) and discriminator (D) in most of these files is a simple two layer feedforward network with the occasional swap between sigmoid / relu activation function in the Discriminator class. While this will give sensible output, in practice it is recommended to use deep convolutional architectures (i.e. DCGANs) to get nicer visualizations. This can be done by editing the Generator and Discriminator classes.
