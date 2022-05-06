#!/usr/bin/env python3
import os
import pyro
import pyro.distributions as dist
import torch
import numpy as np

import pandas as pd


def model(lambda_f, lambda_g, gamma, tau, sigma_beta, M, T, obs=None):
  theta_f, theta_g, R, sigma, beta = [], [], [], [], []

  with pyro.plate("season", M) as ind:
    theta_f = pyro.sample("theta_f", dist.Normal(0, 1))
    theta_g = pyro.sample("theta_g", dist.Normal(0, 1))
    R = pyro.sample("R", dist.Normal(0, 1))
    sigma = pyro.sample("sigma", dist.Normal(0, 1))
    beta = pyro.sample("beta", dist.Normal(0, 1))

  with pyro.plate("time", T) as ind:
    # Draw season variable zt ∼ Multinomial(zt |Softmax(XtW , β1 , . . . , βM ))
    


    z = pyro.sample("z", dist.Categorical())

    pass


def get_weather_data(df):
  dfW = df[['temp', 'temp','temp_min','temp_max','pressure','humidity','wind_speed','wind_deg','rain_1h','rain_3h','snow_3h','weather_id']]

  # Set outliers to mean
  dfW["pressure"][(dfW['pressure'] < 1e2) | (dfW['pressure'] > 1e4)] = dfW["pressure"].mean()


  # Normalize stuff
  dfW['temp'] = dfW['temp'] - 273.15 / 30
  dfW['temp_min'] = dfW['temp_min'] / 50
  dfW['temp_max'] = dfW['temp_max'] / 50
  dfW['pressure'] = dfW["pressure"] - 1013 / 1e3
  dfW['humidity'] = dfW["humidity"] / 100
  dfW['wind_speed'] = dfW["wind_speed"] / 10
  dfW['wind_deg'] = dfW["wind_deg"] / 360
  dfW['rain_1h'] = dfW["rain_1h"] / 1e3
  dfW['rain_3h'] = dfW["rain_3h"] / 1e3
  dfW['snow_3h'] = dfW["snow_3h"] / 1e3
  

  return dfW


def run():
  data = pd.read_csv("preprocessed_data/df.csv")
  data.info()
  

  #model(lambda_f=0.1, lambda_g=0.1, gamma=0.1, tau=0.1, sigma_beta=0.1, M=10, T=1)
  #print(data.head())

  






if __name__ == "__main__":
  run()
