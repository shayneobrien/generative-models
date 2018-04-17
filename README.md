# Overview
Commented implementations and general comparison introductions for minimax, non-saturating, wasserstein, wasserstein gradient penalty, least squares, deep regret analytic, bounded equilibrium, and pack GANs. Paper links are supplied at the beginning of each file. Jupyter notebooks for interactivity supplied.

Note: The architecture chosen for both the generator (G) and discriminator (D) in each of these files is a simple two layer feedforward network. While this will give sensible output, in practice it is recommended to use deep convolutional architectures (i.e. DCGANs) instead. Given that the architecture for all GANs in this repo is the same, it can trivially be swapped out in favor of one that is more expressive by editing the Generator and Discriminator classes of the file. 

The main file changes can be see in the functions train_D and train_G of the trainer function, although changes are not completely limited to only these two areas (e.g. Wasserstein GAN clamps weight in the train function).
