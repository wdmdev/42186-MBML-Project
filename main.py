#!/usr/bin/env python3
import os
import pyro
import pyro.distributions as dist
import torch
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import jax

from tqdm import tqdm

# from pyro.infer import NUTS, MCMC

from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
from pyro.optim import Adam, ClippedAdam

# from datetime import datetime

# date_parser = lambda date: pd.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')


def f(carry, D_t):
  (theta_f, theta_g, sigma, R, h_prev) = carry

  X_E_t, X_W_t, z_t = D_t[:10], D_t[10:21], D_t[:-1]

  c = torch.concat((X_E_t, X_W_t, h_prev))[:, None]

  h = pyro.sample("h", dist.Normal((theta_f[z_t] @ c).squeeze(), R[z_t]))

  mean = theta_g[z_t] @ h
  var = sigma[z_t]

  return (theta_f, theta_g, sigma, R, h), (mean, var)


def model(lambda_f, lambda_g, gamma, tau, sigma_beta, M, T, T_pred, h_dim, X_E, X_W, obs=None):

  with pyro.plate("season", M):
    sigma = pyro.sample("sigma", dist.HalfCauchy(tau))
    R = pyro.sample("R", dist.HalfCauchy(gamma), sample_shape=(h_dim,)) 
    theta_g = pyro.sample("theta_g", dist.Normal(0, lambda_g), sample_shape=(h_dim,))
    theta_f = pyro.sample("theta_f", dist.Normal(0, lambda_f), sample_shape=(h_dim, X_E.shape[1] + X_W.shape[1] + h_dim))
    beta = pyro.sample("beta", dist.Normal(0, sigma_beta), sample_shape=(X_W.shape[1],))


  print(beta.shape)
  z = pyro.sample("z", dist.Categorical(logits=X_W @ beta))
  h = pyro.sample("h_0", dist.Normal(0, 1), sample_shape=(h_dim,))

  R = R.T
  theta_g = theta_g.T
  theta_f = theta_f.T
  
  T_obs = T-T_pred
  y_obs = torch.zeros(T_obs)
  y_pred = torch.zeros(T_pred)

  for t in tqdm(pyro.plate("time", T), total=T):
    # Draw season variable zt ∼ Multinomial(zt |Softmax(XtW , β1 , . . . , βM ))
    c = torch.concat((X_E[t], X_W[t], h)).unsqueeze(1)


    h_mean = (theta_f[z[t]].T @ c).squeeze()
    h_var = R[z[t]]
    h = pyro.sample(f"h_{t+1}", dist.Normal(h_mean, h_var))
    h = h.squeeze()

    if t < T - T_pred:
      y_in = theta_g[z[t]] @ h
      y_obs[t] = pyro.sample(f"y_obs_{t+1}", dist.Normal(y_in, sigma[z[t]]), obs=obs[t])
    else:
      y_pred[t-T_obs] = pyro.sample(f"y_pred_{t+1}", dist.Normal(theta_g[z[t]] @ h, sigma[z[t]]), obs=None)

  return y_obs, y_pred


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



def run(inference_method = None):
  df = pd.read_csv("preprocessed_data/df.csv").iloc[:1000]
  df["time_str"] = [d.split("+")[0] for d in df["time_str"]]
  print(df.head())
  df["time_str"] = pd.to_datetime(df["time_str"], infer_datetime_format=True)#'%Y-%m-%d %H:%M:%S.%f') # 2015-01-01 10:00:00+00:00
  dfW = get_weather_data(df)
  dfE = get_energy_data(df)

  X_W = torch.from_numpy(dfW.values).float()
  X_E = torch.from_numpy(dfE.values).float()

  obs = torch.from_numpy(df["price actual"].values).float()

  kwargs = {
    "lambda_f":   0.1,
    "lambda_g":   0.1,
    "gamma":      0.1,
    "tau":        0.1,
    "sigma_beta": 0.1,
    "M":          8,
    "T":          len(dfW),# - 100,
    "T_pred":     100,
    "h_dim":      10,
    "X_W":        X_W,
    "X_E":        X_E, 
    "obs":        obs
    }

  
  # Run inference in Pyro
  if inference_method == "MCMC":
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=800, warmup_steps=200, num_chains=1)
    mcmc.run(**kwargs)

    # Show summary of inference results
    mcmc.summary()
  elif inference_method == "SVI":
    # Define guide function
    guide = AutoDiagonalNormal(model)

    # Reset parameter values
    pyro.clear_param_store()

    # Define the number of optimization steps
    n_steps = 12000

    # Setup the optimizer
    adam_params = {"lr": 0.01}
    optimizer = ClippedAdam(adam_params)

    # Setup the inference algorithm
    elbo = Trace_ELBO(num_particles=3)
    svi = SVI(model, guide, optimizer, loss=elbo)

    # Do gradient steps
    for step in range(n_steps):
        elbo = svi.step(**kwargs)
        if step % 500 == 0:
            print("[%d] ELBO: %.1f" % (step, elbo))
  else:
    y_obs, y_pred = model(**kwargs)

    fig, ax = plt.subplots(1,2, figsize = (12,6))

    ax[0].plot(df.time_str[:len(y_obs)], obs[:len(y_obs)])
    plt.setp(ax[0].get_xticklabels(), rotation = 45)
    ax[0].set_title("From data")

    ax[1].plot(df.time_str[:len(y_obs)], y_obs)
    plt.setp(ax[1].get_xticklabels(), rotation = 45)
    ax[1].set_title("From model")


    plt.show()




if __name__ == "__main__":
  run("SVI")
