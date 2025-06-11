import numpy as np
import pandas as pd

def build_dynamic_profile(data_slice:pd.DataFrame, num_bins:int) -> tuple[np.ndarray, np.ndarray]:
    min_price = data_slice['low'].min() * 0.995
    max_price = data_slice['high'].max() * 1.005
    price_bins = np.linspace(min_price, max_price, num_bins)
    volume_profile = np.zeros(num_bins-1)

    for i in range(num_bins-1):
        in_bin = (data_slice['high'] >= price_bins[i]) & (data_slice['low'] <= price_bins[i+1])
        volume_profile[i] = data_slice.loc[in_bin, 'volume'].sum()

    condition_for_visibility = volume_profile > volume_profile[volume_profile.nonzero()].mean()

    price_bins = price_bins[:-1][condition_for_visibility]
    volume_profile = volume_profile[condition_for_visibility]
    return price_bins, volume_profile