# OVERVIEW
PyTorch version: 0.4.1 | Python 3.6.5

Annotated implementations with comparative introductions for minimax, non-saturating, wasserstein, wasserstein gradient penalty, least squares, deep regret analytic, bounded equilibrium, relativistic, f-divergence, Fisher, and information generative adversarial networks (GANs), and standard, variational, and bounded information rate variational autoencoders (VAEs). Paper links are supplied at the beginning of each file with a short summary of the paper. See src folder for files to run via terminal, or notebooks folder for Jupyter notebook visualizations via your local browser. All code in this repository operates in a generative, unsupervised manner on binary (black and white) MNIST. The architectures are compatible with all datatypes (1D, 2D, 3D) and plotting functions work with RGB images, too.

The main file changes can be see in the train, train_D, and train_G of the Trainer class, although changes are not completely limited to only these two areas (e.g. Wasserstein GAN clamps weight in the train function, BEGAN gives multiple outputs from train_D).

# NOTE
The architecture chosen in these implementations for both the generator (G) and discriminator (D) consists of a simple, two-layer feedforward network. While this will give sensible output for MNIST, in practice it is recommended to use deep convolutional architectures (i.e. DCGANs) to get nicer outputs. This can be done by editing the Generator and Discriminator classes for GANs, or the Encoder and Decoder classes for VAEs.

# TO ADD
Models: CVAE, denoising VAE | Bayesian GAN, Self-attention GAN, Primal-Dual Wasserstein GAN, adversarial autoencoder
Architectures: Add DCGAN option, double check activation fncs
Housekeeping: fill out notebooks, update notebooks to src content, src linebreaks
