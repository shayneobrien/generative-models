# OVERVIEW
PyTorch version: 0.4.1 | Python 3.6.5

Commented / annotated implementations and comparative introductions for minimax, non-saturating, wasserstein, wasserstein gradient penalty, least squares, deep regret analytic, bounded equilibrium generative adversarial networks, relativistic, f-divergence, Fisher, and information (GANs), a standard autoencoder, and a variational autoencoder (VAE). Paper links are supplied at the beginning of each file with a short summary of the paper. See src folder for files to run via terminal, or notebooks folder for Jupyter notebooks via your local browser. The Jupyter notebooks contain in-notebook visualizations. All code in this repository operates in a generative, unsupervised manner on binary MNIST.

The main file changes can be see in the train, train_D, and train_G of the Trainer class, although changes are not completely limited to only these two areas (e.g. Wasserstein GAN clamps weight in the train function, BEGAN gives multiple outputs from train_D).

# NOTE
The architecture chosen in these implementations for both the generator (G) and discriminator (D) consists of a simple, two-layer feedforward network. While this will give sensible output for MNIST, in practice it is recommended to use deep convolutional architectures (i.e. DCGANs) to get nicer visualizations. This can be done by editing the Generator and Discriminator classes for GANs, or the Encoder and Decoder classes for VAEs.

# TO ADD
Models: CVAE, denoising VAE | Bayesian GAN, Self-attention GAN, Primal-Dual Wasserstein GAN, adversarial autoencoder
Architectures: Add DCGAN option, double check activation fncs
Housekeeping: fill out notebooks, update notebooks to src content, src linebreaks
