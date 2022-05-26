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

  y_obss = []
  y_preds = []
  for t in pyro.markov(range(T + T_pred)):
    z_logits = X_W[t] @ beta.T + alpha[z]
    z = pyro.sample(f"z_{t+1}", dist.Categorical(logits=z_logits), infer={"enumerate": "sequential"})

    with season_plate:
      h_mean = h + theta_f @ X[t]
      h = pyro.sample(f"h_{t+1}", dist.Normal(h_mean, R).to_event(1))


    y_mean = h[z].squeeze() @ theta_g[z].squeeze().T
    y_std = sigma[z]
    if t < T:
      y_obs = pyro.sample(f"y_obs_{t+1}", dist.Normal(y_mean, y_std), obs=obs[t])
      y_obss.append(y_obs)
    else:
      y_pred = pyro.sample(f"y_pred_{t+1}", dist.Normal(y_mean, y_std))
      y_preds.append(y_pred)

  return y_obss, y_preds

    #print("h_mean.shape", h_mean.shape)

    # h = pyro.sample(f"h_{t+1}", dist.Normal(h_mean, h_var).to_event(1))
    
    # y_in = h @ theta_g[z].T
    # if t < T:
    #   y_obs = pyro.sample(f"y_obs_{t+1}", dist.Normal(y_in, sigma[z]), obs=obs[t])
    # else:
    #   y_pred = pyro.sample(f"y_pred_{t+1}", dist.Normal(y_in, sigma[z]), obs=None)

  #return y_obs, y_pred


def get_weather_data(df):
  dfW = df[['temp','temp_min','temp_max','pressure','humidity','wind_speed','wind_deg','rain_1h','rain_3h','snow_3h']].copy()

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

  for col in dfE.columns:
    dfE[col] = (dfE[col] - dfE[col].mean()) / dfE[col].std()

  return dfE


def run(mode: str, method: str):
  df = pd.read_csv("preprocessed_data/df.csv")[:5000]
  df["time_str"] = [d.split("+")[0] for d in df["time_str"]]
  df["time_str"] = pd.to_datetime(df["time_str"], infer_datetime_format=True)#'%Y-%m-%d %H:%M:%S.%f') # 2015-01-01 10:00:00+00:00
  df = df.groupby("time_str").mean()
  dfW = get_weather_data(df)
  dfE = get_energy_data(df)

  price_mean = df["price actual"].mean()
  price_std = df["price actual"].std()
  price = (df["price actual"] - price_mean) / price_std

  print(f"Price mean: {price_mean:2.4f}")
  print(f"Price std:  {price_std:2.4f}")

  X_W = torch.from_numpy(dfW.values).float()
  X_E = torch.from_numpy(dfE.values).float()

  obs = torch.from_numpy(price.values).float()

  #plt.plot(df.index, dfW["temp"])
  #plt.show()

  #return

  T = len(dfW) - 1
  T_pred = 0

  kwargs = {
    "lambda_f":   1,
    "lambda_g":   1,
    "gamma":      1,
    "tau":        1,
    "sigma_alpha":1,
    "sigma_beta": 1,
    "M":          8,
    "T":          T,
    "T_pred":     T_pred,
    "h_dim":      10,
    "X_W":        X_W,
    "X_E":        X_E, 
    "obs":        obs
  }

  if mode == "train":
    if method == "MCMC":
      nuts_kernel = NUTS(model)
      mcmc = MCMC(nuts_kernel, num_samples=800, warmup_steps=0, num_chains=4)
      mcmc.run(**kwargs)
      # Show summary of inference results
      mcmc.summary()
      
    elif method == "SVI":
      optim = Adam({ 'lr': 1e-1 })
      elbo = TraceEnum_ELBO(max_plate_nesting=1)
      guide = AutoNormal(poutine.block(model, hide=[f"z_{i}" for i in range(T + T_pred + 1)]))

      svi = SVI(model, guide, optim, elbo)

      tqdm_loop = tqdm(range(500))

      # Do actual dataloading here.
      loss_history = []
      for _ in tqdm_loop:
        loss = svi.step(**kwargs)
        loss_history.append(loss)
        tqdm_loop.set_description(f"loss={loss:.2f}")

      pyro.get_param_store().save("./params")
      plt.plot(loss_history)
      plt.show()

  elif mode == "test":
    pyro.get_param_store().load("./params")
    y_obs, y_pred = model(
      **{
        **kwargs,
        'T': len(dfW) - 500,
        'T_pred': 500
      }
    )

    # y_obs = torch.vstack(y_obs)
    # y_pred = torch.vstack(y_pred)[:, 0]
    # y_pred = (y_pred - y_pred.mean(dim=0)) / y_pred.std(dim=0)

    # plt.plot(dfW.index, obs)
    # plt.plot(dfW.index[:500], y_obs)
    # plt.plot(dfW.index[-500:], y_pred)

    plt.show()


if __name__ == "__main__":
  run("train", "SVI")
