#!/usr/bin/env python3


import math
import pyro
import pyro.distributions as dist
import torch

import math

import matplotlib.pyplot as plt

from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO



def model(x, obs=None):
  alpha = pyro.sample("alpha", dist.Normal(0, 1))
  beta = pyro.sample("beta", dist.Normal(0, 1))

  with pyro.plate("season", 25):
    kappa = pyro.sample("kappa", dist.Normal(0, 1))

  y_hat = alpha + beta * x + kappa.repeat(math.ceil(x.shape[0] / 25))[:x.shape[0]]

  return pyro.sample("obs", dist.Normal(y_hat, 1), obs=obs)



if __name__ == "__main__":
  
  x = torch.linspace(0, 20, 200) 
  y = x * 1.2 + torch.sin(4 * 2 * torch.pi *  x / 10) * 2 + torch.randn(x.shape[0]) * 0.4

  x_train = x[:100]
  y_train = y[:100]

  nuts_kernel = NUTS(model)
  mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200, num_chains=1)
  mcmc.run(x=x_train, obs=y_train)

  mcmc.summary()

  samples = mcmc.get_samples()

  alpha = samples["alpha"]
  alpha_mean = alpha.mean(dim=0)

  beta = samples["beta"]
  beta_mean = beta.mean(dim=0)

  kappa = samples["kappa"]
  kappa_means = kappa.mean(dim=0)

  y_pred = alpha_mean + beta_mean * x + kappa_means.repeat(math.ceil(x.shape[0] / 25))[:x.shape[0]]

  fig, ax = plt.subplots(1,3)

  plt.title("$y = \\alpha + \\beta x$")

  ax[0].scatter(x_train.numpy(), y_train.numpy())
  ax[0].plot(x.numpy(), y_pred.numpy())
  ax[0].set_ylabel("$y$")
  ax[0].set_xlabel("$x$")

  ax[1].hist(alpha.numpy(), bins=50)
  ax[1].set_title("$\\alpha$")

  ax[2].hist(beta.numpy(), bins=50)
  ax[2].set_title("$\\beta$")

  plt.show()


