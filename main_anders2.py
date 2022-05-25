#!/usr/bin/env python3

"""
  Inspired by this: https://github.com/pyro-ppl/pyro/blob/dev/examples/hmm.py
  The problem we have with constructing our models probably boils down to having categorical states (z),
  since we have to eliminate the variable.
"""

import os
import pyro
import pyro.distributions as dist
import torch
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from pyro import poutine
from pyro.optim import Adam
from pyro.infer import NUTS, MCMC, config_enumerate, Trace_ELBO, TraceEnum_ELBO, SVI
from pyro.infer.autoguide import AutoNormal, AutoDelta, AutoDiagonalNormal



def model(lambda_f, lambda_g, gamma, tau, sigma_alpha, sigma_beta, M, T, T_pred, h_dim, X_E, X_W, obs=None):

  X = torch.cat([X_E, X_W], dim=1)
  y_obs = None
  y_pred = None

  season_plate = pyro.plate("season", M)
  with season_plate:
    sigma = pyro.sample("sigma", dist.HalfCauchy(tau))
    R = pyro.sample("R", dist.HalfCauchy(torch.ones(h_dim) * gamma).to_event(1))
    theta_g = pyro.sample("theta_g", dist.Normal(torch.zeros(h_dim), torch.ones(h_dim) * lambda_g).to_event(1))
    theta_f = pyro.sample("theta_f", dist.Normal(torch.zeros((h_dim, X.shape[1])), torch.ones((h_dim, X.shape[1])) * lambda_f).to_event(2))
    alpha = pyro.sample("alpha", dist.Normal(torch.zeros(M), sigma_alpha * torch.ones(M)).to_event(1))
    beta = pyro.sample("beta", dist.Normal(torch.zeros(X_W.shape[1]), sigma_beta * torch.ones(X_W.shape[1])).to_event(1))
    h = pyro.sample("h_0", dist.Normal(torch.zeros(h_dim), torch.ones(h_dim)).to_event(1))
  
  z = pyro.sample("z_0", dist.Categorical(torch.ones(M) / M), infer={"enumerate": "parallel"})

  for t in pyro.markov(range(T + T_pred)):
    z_logits = X_W[t] @ beta.T + alpha[z]
    z = pyro.sample(f"z_{t+1}", dist.Categorical(logits=z_logits), infer={"enumerate": "parallel"})

    with season_plate:
      h_mean = h + theta_f @ X[t]
      h = pyro.sample(f"h_{t+1}", dist.Normal(h_mean, R).to_event(1))

    y_mean = h @ theta_g.T
    if t < T:
      y_obs = pyro.sample(f"y_obs_{t+1}", dist.Normal(y_mean[z], sigma[z]).to_event(1), obs=obs[t])
    else:
      y_pred = pyro.sample(f"y_pred_{t+1}", dist.Normal(y_mean[z], sigma[z]).to_event(1))


  return y_obs, y_pred

    #print("h_mean.shape", h_mean.shape)

    # h = pyro.sample(f"h_{t+1}", dist.Normal(h_mean, h_var).to_event(1))
    
    # y_in = h @ theta_g[z].T
    # if t < T:
    #   y_obs = pyro.sample(f"y_obs_{t+1}", dist.Normal(y_in, sigma[z]), obs=obs[t])
    # else:
    #   y_pred = pyro.sample(f"y_pred_{t+1}", dist.Normal(y_in, sigma[z]), obs=None)

  #return y_obs, y_pred


def get_weather_data(df):
  dfW = df[['temp', 'temp','temp_min','temp_max','pressure','humidity','wind_speed','wind_deg','rain_1h','rain_3h','snow_3h']].copy()

  # Set outliers to mean
  #dfW["pressure"] = dfW["pressure"].apply(lambda p: dfW["pressure"].mean() if )
  dfW.loc[(df["pressure"] > 1e4) | (df["pressure"] < 1e2), "pressure"] = df["pressure"].mean()

  # Normalize stuff
  dfW['temp'] = (dfW['temp'] - 273.15) / 50
  dfW['temp_min'] = (dfW['temp_min'] - 273.15) / 50
  dfW['temp_max'] = (dfW['temp_max'] - 273.15) / 50
  dfW['pressure'] = (dfW["pressure"] - 1013) / 1e3
  dfW['humidity'] = dfW["humidity"] / 100
  dfW['wind_speed'] = dfW["wind_speed"] / 50
  dfW['wind_deg'] = dfW["wind_deg"] / 360
  dfW['rain_1h'] = dfW["rain_1h"] / 1e3
  dfW['rain_3h'] = dfW["rain_3h"] / 1e3
  dfW['snow_3h'] = dfW["snow_3h"] / 1e3


  return dfW

def get_energy_data(df):
  dfE = df[[
    'generation biomass',
    'generation fossil',
    'generation hydro',
    'generation nuclear',
    'generation other',
    'generation other renewable',
    'generation solar',
    'generation total',
    'generation waste',
    'generation wind onshore'
  ]].copy()

  return dfE


def run(method: str):
  df = pd.read_csv("preprocessed_data/df.csv").iloc[:1000]
  dfW = get_weather_data(df)
  dfE = get_energy_data(df)

  X_W = torch.from_numpy(dfW.values).float()
  X_E = torch.from_numpy(dfE.values).float()

  obs = torch.from_numpy(df["price actual"].values).float()

  T = len(dfW) - 1
  T_pred = 0

  kwargs = {
    "lambda_f":   0.1,
    "lambda_g":   0.1,
    "gamma":      0.1,
    "tau":        0.1,
    "sigma_alpha":0.1,
    "sigma_beta": 0.1,
    "M":          8,
    "T":          T,
    "T_pred":     T_pred,
    "h_dim":      10,
    "X_W":        X_W,
    "X_E":        X_E, 
    "obs":        obs
  }


  if method == "MCMC":
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=800, warmup_steps=0, num_chains=4)
    mcmc.run(**kwargs)
    # Show summary of inference results
    mcmc.summary()
    
  elif method == "SVI":
    optim = Adam({ 'lr': 1e-3 })
    elbo = TraceEnum_ELBO(max_plate_nesting=1)
    guide = AutoNormal(poutine.block(model, hide=[f"z_{i}" for i in range(T + T_pred + 1)]))

    svi = SVI(model, guide, optim, elbo)

    tqdm_loop = tqdm(range(100))

    # Do actual dataloading here.
    loss_history = []
    for i in tqdm_loop:
      loss = svi.step(**kwargs)
      loss_history.append(loss)
      tqdm_loop.set_description(f"loss={loss:.2f}")

    plt.plot(loss_history)
    plt.show()


if __name__ == "__main__":
  run("SVI")
