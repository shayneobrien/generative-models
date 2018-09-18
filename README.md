# OVERVIEW
PyTorch version: 0.4.1 | Python 3.6.5

Annotated implementations with comparative introductions for minimax, non-saturating, wasserstein, wasserstein gradient penalty, least squares, deep regret analytic, bounded equilibrium, relativistic, f-divergence, Fisher, and information generative adversarial networks (GANs), and standard, variational, and bounded information rate variational autoencoders (VAEs).

Paper links are supplied at the beginning of each file with a short summary of the paper. See src folder for files to run via terminal, or notebooks folder for Jupyter notebook visualizations via your local browser. The main file changes can be see in the train, train_D, and train_G of the Trainer class, although changes are not completely limited to only these two areas (e.g. Wasserstein GAN clamps weight in the train function, BEGAN gives multiple outputs from train_D, fGAN has a slight modification in viz_loss function to indicate method used in title).

All code in this repository operates in a generative, unsupervised manner on binary (black and white) MNIST. The architectures are compatible with a variety of datatypes (1D, 2D, 3D) and plotting functions work with binary/RGB images too. If a GPU is detected, the models use it. Otherwise, they default to CPu.

# ARCHITECTURES
The architecture chosen in these implementations for both the generator (G) and discriminator (D) consists of a simple, two-layer feedforward network. While this will give sensible output for MNIST, in practice it is recommended to use deep convolutional architectures (i.e. DCGANs) to get nicer outputs. This can be done by editing the Generator and Discriminator classes for GANs, or the Encoder and Decoder classes for VAEs.

# VIZ | LOSS
*Model* | *Epoch 1* | *Epoch 25* | *GIF* | *Loss*
:---: | :---: | :---: | :---: | :---: |
[NSGAN](https://arxiv.org/abs/1406.2661) | <img src = 'viz/NSGAN/reconst_1.png' height = '200px'> | <img src = 'viz/NSGAN/reconst_25.png' height = '200px'> | <img src = 'viz/NSGAN/reconst_1.png' height = '200px'> | <img src = 'viz/losses/NSGAN_loss.png' height = '200px'>


<!-- LSGAN | <img src = 'assets/mnist_results/LSGAN_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/LSGAN_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/LSGAN_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/LSGAN_generate_animation.gif' height = '200px'>
WGAN | <img src = 'assets/mnist_results/WGAN_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/WGAN_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/WGAN_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/WGAN_generate_animation.gif' height = '200px'>
WGAN_GP | <img src = 'assets/mnist_results/WGAN_GP_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/WGAN_GP_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/WGAN_GP_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/WGAN_GP_generate_animation.gif' height = '200px'>
DRAGAN | <img src = 'assets/mnist_results/DRAGAN_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/DRAGAN_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/DRAGAN_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/DRAGAN_generate_animation.gif' height = '200px'>
EBGAN | <img src = 'assets/mnist_results/EBGAN_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/EBGAN_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/EBGAN_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/EBGAN_generate_animation.gif' height = '200px'>
BEGAN | <img src = 'assets/mnist_results/BEGAN_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/BEGAN_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/BEGAN_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/BEGAN_generate_animation.gif' height = '200px'> -->

# TODO
Models: CVAE, denoising VAE, adversarial autoencoder | Bayesian GAN, Self-attention GAN, Primal-Dual Wasserstein GAN  
Architectures: Add DCGAN option  
