#!/usr/bin/env python3
import os
import pyro
import pyro.distributions as dist
import torch
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


def model(lambda_f, lambda_g, gamma, tau, sigma_beta, M, T, X_W, h_dim, obs=None):
  with pyro.plate("season", M):
    theta_f = pyro.sample("theta_f", dist.Normal(torch.zeros(10), torch.ones(10)))
    theta_g = pyro.sample("theta_g", dist.Normal(torch.zeros(10), torch.ones(10)))
    R = pyro.sample("R", dist.HalfCauchy(torch.zeros(h_dim), torch.ones(h_dim)))
    sigma = pyro.sample("sigma", dist.HalfCauchy(0, 1))

    with pyro.plate("features", X_W.shape[1]):
      beta = pyro.sample("beta", dist.Normal(0, 1))


  with pyro.plate("time", T) as t:
    # Draw season variable zt ∼ Multinomial(zt |Softmax(XtW , β1 , . . . , βM ))
    X_W_t = torch.from_numpy(X_W.iloc[t].values).float()
    z = pyro.sample("z", dist.Categorical(logits=X_W_t @ beta))
    #h = pyro.sample("h", dist.Normal(theta_f[z], sigma))

    pass


def get_weather_data(df):
  dfW = df[['temp', 'temp','temp_min','temp_max','pressure','humidity','wind_speed','wind_deg','rain_1h','rain_3h','snow_3h']]

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


def run():
  df = pd.read_csv("preprocessed_data/df.csv")
  dfW = get_weather_data(df)

  model(lambda_f=0.1, lambda_g=0.1, gamma=0.1, tau=0.1, sigma_beta=0.1, M=10, T=len(dfW), X_W=dfW)

  
  #print(data.head())

  






if __name__ == "__main__":
  run()
