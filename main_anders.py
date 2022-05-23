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

from pyro.infer import NUTS, MCMC, config_enumerate, Trace_ELBO, TraceEnum_ELBO

from pyro import poutine
from pyro.infer.autoguide import AutoNormal


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
    R = pyro.sample("R", dist.HalfCauchy(torch.ones(h_dim) * gamma).to_event(1))
    theta_g = pyro.sample("theta_g", dist.Normal(torch.zeros(h_dim), torch.ones(h_dim) * lambda_g).to_event(1))
    theta_f = pyro.sample("theta_f", dist.Normal(torch.zeros((h_dim, X_E.shape[1] + X_W.shape[1] + h_dim)), torch.ones((h_dim, X_E.shape[1] + X_W.shape[1] + h_dim)) * lambda_f).to_event(2))
    beta = pyro.sample("beta", dist.Normal(torch.zeros(X_W.shape[1]), sigma_beta * torch.ones(X_W.shape[1])).to_event(1))

  h = pyro.sample("h_0", dist.Normal(torch.zeros(M, h_dim), torch.ones(M, h_dim)).to_event(1))


  
  for t in tqdm(pyro.markov(range(T))):
    z = pyro.sample(f"z_{t}", dist.Categorical(logits=X_W[t] @ beta.T), infer={"enumerate": "parallel"}).squeeze()
    #print("z.shape", z.shape)
    # Draw season variable zt ∼ Multinomial(zt |Softmax(XtW , β1 , . . . , βM ))
    
    print("theta_f[z].shape", theta_f[z].shape)
    print("h.shape", h.shape)
    print("X_E.repeat", X_E[t].repeat(h.shape[0], 1).shape)
    print("X_W.repat", X_W[t].repeat(h.shape[0], 1).shape)


    c = torch.concat((X_E[t].repeat(h.shape[0], 1), X_W[t].repeat(h.shape[0], 1), h), dim=1)
    print("c.shape", c.shape)

    h_mean = (theta_f[z] @ c.T).T
    h_var = R[z]


    print("h_mean.shape", h_mean.shape)
    print("h_var.shape", h_var.shape)

    #print("h_mean.shape", h_mean.shape)

    h = pyro.sample(f"h_{t+1}", dist.Normal(h_mean, h_var).to_event(1))
    
    y_in = h @ theta_g[z].T
    if t < T:
      y_obs = pyro.sample(f"y_obs_{t+1}", dist.Normal(y_in, sigma[z]), obs=obs[t])
    else:
      y_pred = pyro.sample(f"y_pred_{t+1}", dist.Normal(y_in, sigma[z]), obs=None)

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

def guide(*args, **kwargs):
  pass

def run():
  df = pd.read_csv("preprocessed_data/df.csv").iloc[:1000]
  dfW = get_weather_data(df)
  dfE = get_energy_data(df)

  X_W = torch.from_numpy(dfW.values).float()
  X_E = torch.from_numpy(dfE.values).float()

  obs = torch.from_numpy(df["price actual"].values).float()

  # kwargs = {lambda_f=0.1,
  #   lambda_g=0.1,
  #   gamma=0.1,
  #   tau=0.1,
  #   sigma_beta=0.1,
  #   M=8,
  #   T=len(dfW) - 100,
  #   T_pred=100,
  #   h_dim=10,
  #   X_W=X_W,
  #   X_E=X_E, 
  #   obs=obs}

  

 

  # T = len(dfW) - 100

  # print(len(X_W), T)
  # guide = AutoNormal(poutine.block(model, hide=[f"z_{i}" for i in range(len(X_W))]))
  # #guide = AutoNormal(model)
  # pyro.clear_param_store()

  # elbo = TraceEnum_ELBO(max_plate_nesting=1)
  # elbo.loss(
  #   model, 
  #   guide, 
  #   lambda_f=0.1,
  #   lambda_g=0.1,
  #   gamma=0.1,
  #   tau=0.1,
  #   sigma_beta=0.1,
  #   M=8,
  #   T=T,
  #   T_pred=100,
  #   h_dim=10,
  #   X_W=X_W,
  #   X_E=X_E, 
  #   obs=obs
  # )

    # Run inference in Pyro
  nuts_kernel = NUTS(model)
  mcmc = MCMC(nuts_kernel, num_samples=800, warmup_steps=200, num_chains=1)
  mcmc.run(
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
  mcmc.summary()

  # fig, ax = plt.subplots(1,2)

  # ax[0].plot(obs)
  # ax[1].plot(y_pred)
  # plt.show()




if __name__ == "__main__":
  run()
