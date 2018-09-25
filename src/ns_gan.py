""" (NS GAN) https://arxiv.org/abs/1406.2661
Non-saturating GAN.

From the abstract: 'We propose a new framework for estimating generative models
via an adversarial process, in which we simultaneously train two models: a
generative model G that captures the data distribution, and a discriminative
model D that estimates the probability that a sample came from the training data
rather than G. The training procedure for G is to maximize the probability of D
making a mistake.'

Compared to MM GAN, the only change is the generator's loss.

NS GAN: L(G) = -E[log(D(G(z)))]
MM GAN: L(G) =  E[log(1-D(G(z)))]

In both NS GAN and MM GAN, the output of G can be interpretted as a probability.
"""

import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import os
import matplotlib.pyplot as plt
import numpy as np

from itertools import product
from tqdm import tqdm

from .utils import *


class Generator(nn.Module):
    """ Generator. Input is noise, output is a generated image.
    """
    def __init__(self, image_size, hidden_dim, z_dim):
        super().__init__()
        self.linear = nn.Linear(z_dim, hidden_dim)
        self.generate = nn.Linear(hidden_dim, image_size)

    def forward(self, x):
        activated = F.relu(self.linear(x))
        generation = torch.sigmoid(self.generate(activated))
        return generation


class Discriminator(nn.Module):
    """ Discriminator. Input is an image (real or generated), output is P(generated).
    """
    def __init__(self, image_size, hidden_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(image_size, hidden_dim)
        self.discriminate = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        activated = F.relu(self.linear(x))
        discrimination = torch.sigmoid(self.discriminate(activated))
        return discrimination


class NSGAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """
    def __init__(self, image_size, hidden_dim, z_dim, output_dim=1):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(image_size, hidden_dim, z_dim)
        self.D = Discriminator(image_size, hidden_dim, output_dim)

        self.shape = int(image_size ** 0.5)


class NSGANTrainer:
    """ Object to hold data iterators, train a GAN variant
    """
    def __init__(self, model, train_iter, val_iter, test_iter, viz=False):
        self.model = to_cuda(model)
        self.name = model.__class__.__name__

        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        self.Glosses = []
        self.Dlosses = []

        self.viz = viz
        self.num_epochs = 0

    def train(self, num_epochs, G_lr=2e-4, D_lr=2e-4, D_steps=1):
        """ Train a vanilla GAN using non-saturating gradients loss.

            Logs progress using G loss, D loss, G(x), D(G(x)), visualizations
            of Generator output.

        Inputs:
            num_epochs: int, number of epochs to train for
            G_lr: float, learning rate for generator's Adam optimizer
            D_lr: float, learning rate for discriminator's Adam optimizer
            D_steps: int, ratio for how often to train D compared to G
        """
        # Initialize optimizers
        G_optimizer = optim.Adam(params=[p for p in self.model.G.parameters()
                                        if p.requires_grad], lr=G_lr)
        D_optimizer = optim.Adam(params=[p for p in self.model.D.parameters()
                                        if p.requires_grad], lr=D_lr)

        # Approximate steps/epoch given D_steps per epoch
        # --> roughly train in the same way as if D_step (1) == G_step (1)
        epoch_steps = int(np.ceil(len(self.train_iter) / (D_steps)))

        # Begin training
        for epoch in tqdm(range(1, num_epochs+1)):

            self.model.train()
            G_losses, D_losses = [], []

            for _ in range(epoch_steps):

                D_step_loss = []

                for _ in range(D_steps):

                    # Reshape images
                    images = self.process_batch(self.train_iter)

                    # TRAINING D: Zero out gradients for D
                    D_optimizer.zero_grad()

                    # Learn to discriminate between real and generated images
                    D_loss = self.train_D(images)

                    # Update parameters
                    D_loss.backward()
                    D_optimizer.step()

                    # Log results, backpropagate the discriminator network
                    D_step_loss.append(D_loss.item())

                # So that G_loss and D_loss have the same number of entries.
                D_losses.append(np.mean(D_step_loss))

                # TRAINING G: Zero out gradients for G
                G_optimizer.zero_grad()

                # Learn to generate images that fool the discriminator
                G_loss = self.train_G(images)

                # Log results, update parameters
                G_losses.append(G_loss.item())
                G_loss.backward()
                G_optimizer.step()

            # Save progress
            self.Glosses.extend(G_losses)
            self.Dlosses.extend(D_losses)

            # Progress logging
            print ("Epoch[%d/%d], G Loss: %.4f, D Loss: %.4f"
                   %(epoch, num_epochs, np.mean(G_losses), np.mean(D_losses)))
            self.num_epochs += 1

            # Visualize generator progress
            if self.viz:
                self.generate_images(epoch)
                plt.show()

    def train_D(self, images):
        """ Run 1 step of training for discriminator

        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: non-saturing loss for discriminator,
            -E[log(D(x))] - E[log(1 - D(G(z)))]
        """

        # Sample noise z, generate output G(z)
        noise = self.compute_noise(images.shape[0], self.model.z_dim)
        G_output = self.model.G(noise)

        # Classify the generated and real batch images
        DX_score = self.model.D(images) # D(x)
        DG_score = self.model.D(G_output) # D(G(z))

        # Compute vanilla (original paper) D loss
        D_loss = torch.sum(-torch.mean(torch.log(DX_score + 1e-8)
                            + torch.log(1 - DG_score + 1e-8)))

        return D_loss

    def train_G(self, images):
        """ Run 1 step of training for generator

        Input:
            images: batch of images reshaped to [batch_size, -1]
        Output:
            G_loss: non-saturating loss for how well G(z) fools D,
            -E[log(D(G(z)))]
        """

        # Get noise (denoted z), classify it using G, then classify the output
        # of G using D.
        noise = self.compute_noise(images.shape[0], self.model.z_dim) # (z)
        G_output = self.model.G(noise) # G(z)
        DG_score = self.model.D(G_output) # D(G(z))

        # Compute the non-saturating loss for how D did versus the generations
        # of G using sigmoid cross entropy
        G_loss = -torch.mean(torch.log(DG_score + 1e-8))

        return G_loss

    def compute_noise(self, batch_size, z_dim):
        """ Compute random noise for the generator to learn to make images from """
        return to_cuda(torch.randn(batch_size, z_dim))

    def process_batch(self, iterator):
        """ Generate a process batch to be input into the discriminator D """
        images, _ = next(iter(iterator))
        images = to_cuda(images.view(images.shape[0], -1))
        return images

    def generate_images(self, epoch, num_outputs=36, save=True):
        """ Visualize progress of generator learning """
        # Turn off any regularization
        self.model.eval()

        # Sample noise vector
        noise = self.compute_noise(num_outputs, self.model.z_dim)

        # Transform noise to image
        images = self.model.G(noise)

        # Reshape to square image size
        images = images.view(images.shape[0],
                             self.model.shape,
                             self.model.shape,
                             -1).squeeze()

        # Plot
        plt.close()
        grid_size, k = int(num_outputs**0.5), 0
        fig, ax = plt.subplots(grid_size, grid_size, figsize=(5, 5))
        for i, j in product(range(grid_size), range(grid_size)):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
            ax[i,j].imshow(images[k].data.numpy(), cmap='gray')
            k += 1

        # Save images if desired
        if save:
            outname = '../viz/' + self.name + '/'
            if not os.path.exists(outname):
                os.makedirs(outname)
            torchvision.utils.save_image(images.unsqueeze(1).data,
                                         outname + 'reconst_%d.png'
                                         %(epoch), nrow=grid_size)

    def viz_loss(self):
        """ Visualize loss for the generator, discriminator """
        # Set style, figure size
        plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (8,6)

        # Plot Discriminator loss in red, Generator loss in green
        plt.plot(np.linspace(1, self.num_epochs, len(self.Dlosses)),
                 self.Dlosses,
                 'r')
        plt.plot(np.linspace(1, self.num_epochs, len(self.Dlosses)),
                 self.Glosses,
                 'g')

        # Add legend, title
        plt.legend(['Discriminator', 'Generator'])
        plt.title(self.name)
        plt.show()

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath)

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)


if __name__ == "__main__":

    # Load in binarized MNIST data, separate into data loaders
    train_iter, val_iter, test_iter = get_data()

    # Init model
    model = NSGAN(image_size=784,
                  hidden_dim=400,
                  z_dim=20)

    # Init trainer
    trainer = NSGANTrainer(model=model,
                           train_iter=train_iter,
                           val_iter=val_iter,
                           test_iter=test_iter,
                           viz=False)

    # Train
    trainer.train(num_epochs=25,
                  G_lr=2e-4,
                  D_lr=2e-4,
                  D_steps=1)
