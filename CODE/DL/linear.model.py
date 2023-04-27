#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Linear model module

File containing Linear Aggregated/Binned Hawkes Process estimation (alpha/beta)

"""

import numpy as np

def linear_model(model, train_x, val_x, params, step_size = 0.05):

    val_y_pred = model.predict(val_x)

    val_eta = np.median(val_y_pred[:, 0])
    val_mu = np.median(val_y_pred[:, 1])

    min_eta = val_eta - 0.05
    max_eta = val_eta + 0.05
    eta = params[:, 0] / params[:, 1]

    similar_eta = train_x[(eta > min_eta) & (eta < max_eta), :]
    similar_eta_alpha = params[(eta > min_eta) & (eta < max_eta), 0]
    similar_eta_mu = params[(eta > min_eta) & (eta < max_eta), 2]

    max_alpha = int(np.ceil(np.max(params[:, 0])))

    min_v = np.arange(0, max_alpha, step_size)
    max_v = np.arange(step_size, (max_alpha + step_size), step_size)

    mask = np.logical_or.reduce([(similar_eta_alpha >= min_v) & (similar_eta_alpha < max_v) for min_v, max_v in zip(min_v, max_v)])
    stats = np.mean(np.max(similar_eta[mask], axis=1), axis=0)

    x = np.arange((step_size / 2), (max_alpha * step_size), step_size)[:len(stats)]
    a = np.vstack([x, np.ones(len(x))]).T

    coefs = np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, stats))
    slope, intercept = coefs[0], coefs[1]

    stat = np.median(np.max(val_x, axis=1))
    alpha_pred, beta_pred = slope * stat + intercept, alpha_pred / val_eta

    return {"alpha_pred": alpha_pred, "beta_pred": beta_pred, "val_mu": val_mu, "val_eta": val_eta}
