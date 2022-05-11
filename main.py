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

    with pyro.plate("hidden_dim", h_dim):
      theta_g = pyro.sample("theta_g", dist.Normal(0, lambda_g))
      R = pyro.sample("R", dist.HalfCauchy(gamma))

      with pyro.plate("total_features", X_E.shape[1] + X_W.shape[1] + h_dim):
        theta_f = pyro.sample("theta_f", dist.Normal(0, lambda_f))


    with pyro.plate("weather_features", X_W.shape[1]):
      beta = pyro.sample("beta", dist.Normal(0, sigma_beta))

  theta_f = theta_f.permute(2, 1, 0)
  theta_g = theta_g.permute(1, 0)
  R = R.T

  """
  z = pyro.sample("z", dist.Categorical(logits=X_W @ beta))

  

  carry = (theta_f, R, h)

  D = torch.concat((X_E, X_W, z), dim=1)
  carry_updated, outs = jax.lax.scan(f, carry, D, T+T_pred-1)
  
  outs = jnp.array(out)
  mean, var = outs
  """

  h = pyro.sample("h", dist.Normal(torch.zeros(h_dim), torch.ones(h_dim)))
  ys = []
  for t in tqdm(pyro.plate("time", T + T_pred), total=T + T_pred):
    # Draw season variable zt ∼ Multinomial(zt |Softmax(XtW , β1 , . . . , βM ))
    z = pyro.sample("z", dist.Categorical(logits=X_W[t] @ beta))
    c = torch.concat((X_E[t], X_W[t], h))[:, None]
    h = pyro.sample("h", dist.Normal((theta_f[z] @ c).squeeze(), R[z]))
    y = pyro.sample("y", dist.Normal(theta_g[z] @ h, sigma[z]), obs=obs[t] if t < T else None)
    ys.append(y)

  return torch.tensor(ys)


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



def run():
  df = pd.read_csv("preprocessed_data/df.csv").iloc[:1000]
  dfW = get_weather_data(df)
  dfE = get_energy_data(df)

  X_W = torch.from_numpy(dfW.values).float()
  X_E = torch.from_numpy(dfE.values).float()

  obs = torch.from_numpy(df["price actual"].values).float()

  y_pred = model(
    lambda_f=0.1,
    lambda_g=0.1,
    gamma=0.1,
    tau=0.1,
    sigma_beta=0.1,
    M=8,
    T=len(dfW) - 100,
    T_pred=100,
    h_dim=10,
    X_W=X_W,
    X_E=X_E, 
    obs=obs
  )
  

 

  # Show summary of inference results
  # mcmc.summary()

  fig, ax = plt.subplots(1,2)

  ax[0].plot(obs)
  ax[1].plot(y_pred)
  plt.show()




if __name__ == "__main__":
  run()
