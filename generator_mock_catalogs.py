# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 20:03:16 2026

@author: jakub
"""

import os
import numpy as np
import pandas as pd


input_file = "Chen2019_161_strong_lenses.csv"
output_folder = "mock_catalogs"

n_mocks = 1
seed = 6767

np.random.seed(seed)

catalog = pd.read_csv(input_file)

sigma_original = catalog["sigma_ap_km_s"].to_numpy()
sigma_error = catalog["sigma_ap_err_km_s"].to_numpy()

os.makedirs(output_folder, exist_ok=True)


for i in range(n_mocks):

    
    sigma_mock = np.random.normal(
        sigma_original,
        sigma_error
    )

    
    lower_lim = sigma_original - 3 * sigma_error
    upper_lim = sigma_original + 3 * sigma_error

   
    outside = (
        (sigma_mock < lower_lim)
        | (sigma_mock > upper_lim)
    )

    
    while np.any(outside):

        sigma_mock[outside] = np.random.normal(
            sigma_original[outside],
            sigma_error[outside]
        )

        outside = (
            (sigma_mock < lower_lim)
            | (sigma_mock > upper_lim)
        )

    
    mock_catalog = catalog.copy()

    
    mock_catalog["sigma_ap_original_km_s"] = sigma_original

    
    mock_catalog["sigma_ap_km_s"] = sigma_mock

    
    mock_catalog["mock_id"] = i + 1

    
    output_file = os.path.join(
        output_folder,
        f"Chen_mock_{i + 1:04d}.csv"
    )

    mock_catalog.to_csv(
        output_file,
        index=False
    )


print("Gotowe")
