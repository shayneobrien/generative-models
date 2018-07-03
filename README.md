# OVERVIEW
PyTorch version: 0.4.0 | Python 3

Commented / annotated implementations and comparative introductions for minimax, non-saturating, wasserstein, wasserstein gradient penalty, least squares, deep regret analytic, bounded equilibrium generative adversarial networks (GANs), and variational autoencoder (VAE). Paper links are supplied at the beginning of each file. See src folder for files to run via terminal, or notebooks folder for Jupyter notebooks which include in-notebook visualizations and compatibility with loss visualization. All code in this repository operates in a generative, unsupervised manner on binary MNIST.

The main file changes can be see in the train, train_D, and train_G of the Trainer class, although changes are not completely limited to only these two areas (e.g. Wasserstein GAN clamps weight in the train function, BEGAN gives multiple outputs from train_D).

# TODO

Fixes: discriminator activation function, change description, double check correctness on DRAGAN, fix BEGAN autoencoder issue, rename trainer classes, switch tqdm_notebook to tqdm, push src. Make the only difference between the trainers train_D and train_G (?)

Adding: CVAE, other VAE types? Relativistic GAN, Self-attention GAN, Primal-Dual Wasserstein GAN, InfoGAN (?), Fisher GAN (?), Fairness GAN (?), Bayesian GAN (?)

# CAUTIONARY NOTE
The architecture chosen for both the generator (G) and discriminator (D) in these implementations is a simple two layer feedforward network. While this will give sensible output, in practice it is recommended to use deep convolutional architectures (i.e. DCGANs) to get nicer visualizations. This can be done by editing the Generator and Discriminator classes for GANs, or the Encoder and Decoder classes for VAEs.
