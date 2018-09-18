# TODO
""" (BayesGAN)

From the authors:
"
Bayesian GAN (Saatchi and Wilson, 2017) is a Bayesian formulation of Generative
Adversarial Networks (Goodfellow, 2014) where we learn the distributions of the
generator parameters $\theta_g$ and the discriminator parameters $\theta_d$
instead of optimizing for point estimates. The benefits of the Bayesian approach
include the flexibility to model multimodality in the parameter space, as well
as the ability to prevent mode collapse in the maximum likelihood (non-Bayesian)
case.

We learn Bayesian GAN via an approximate inference algorithm called Stochastic
Gradient Hamiltonian Monte Carlo (SGHMC) which is a gradient-based MCMC methods
whose samples approximate the true posterior distributions of $\theta_g$ and 
$\theta_d$. The Bayesian GAN training process starts from sampling noise $z$
from a fixed  distribution(typically standard d-dim normal). The noise is fed
to the generator where the parameters  $\theta_g$ are sampled from the posterior
distribution $p(\theta_g | D)$. The generated  image given the parameters
$\theta_g$ ($G(z|\theta_g)$) as well as the real data are presented to the
discriminator, whose parameters are sample from its posterior
distribution $p(\theta_d|D)$. We update the posteriors using the gradients
$\frac{\partial \log p(\theta_g|D) }{\partial \theta_g }$ and
$\frac{\partial \log p(\theta_d|D) }{\partial \theta_d }$ with
Stochastic Gradient Hamiltonian Monte Carlo (SGHMC)."

SGHMC is fancy for using point estimates (as in most GANs) to infer the
posteriors.

https://arxiv.org/pdf/1705.09558.pdf
"""
