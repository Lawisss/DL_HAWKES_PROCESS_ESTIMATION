#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stan evaluation module

File containing stan evaluation functions

"""

import pystan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

# Charger le modèle Stan
mcmc_fit = pystan.StanModel(file="stan_test.stan")

# Définir les données
data = {
    'p': latent_dim,
    'p1': intermediate_dim_2,
    'p2': intermediate_dim,
    'n': input_dim,
    'W1': W1,
    'B1': B1,
    'W2': W2,
    'B2': B2,
    'W3': W3,
    'B3': B3,
    'y': test_process[0,:]
}

# Effectuer l'ajustement MCMC
mcmc_result = mcmc_fit.sampling(data=data)

# Extraire les échantillons
latent_draws = mcmc_result['z']

# Charger le modèle du décodeur theta
theta_decoder = load_model('theta_decoder.h5')

# Prédire les valeurs à partir des échantillons
test = theta_decoder.predict(latent_draws)
data = pd.DataFrame({'eta': test[:, 0], 'mu': test[:, 1]})

# Tracer la densité
sns.set_style("whitegrid")
plt.figure(figsize=(8, 4))
sns.kdeplot(data=data, x='eta', y='mu', fill=True, cmap="viridis", levels=5)
plt.axhline(y=test_mu, color="sienna2", linewidth=1)
plt.axvline(x=test_eta, color="sienna2", linewidth=1)
plt.scatter(x=test_eta, y=test_mu, color="sienna2", s=25)
plt.text(0.15, 3.8, "True value", color="sienna2")
plt.xlabel('Eta')
plt.ylabel('Mu')
plt.savefig("lowlow_density.png", dpi=100)
plt.show()
