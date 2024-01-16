import pandas as pd
import numpy as np

def make_series_summary(values):
    summary = {}
    if pd.isna(values).sum() > 0:
        summary["missingness"] = pd.isna(values).mean()
        values = values[~pd.isna(values)]
    num_unique = len(np.unique(values))
    try:
        is_binary = num_unique == 2 and set(np.unique(values).astype(int).tolist()) == set([0, 1])
    except:
        is_binary = False
    if is_binary:
        summary["type"] = "binary"
        summary["rate"] = values.mean().astype(float)
    elif pd.api.types.is_object_dtype(values.dtype) or num_unique <= 10:
        summary["type"] = "categorical"
        summary["counts"] = {str(k): int(v) for k, v in zip(*np.unique(values, return_counts=True))}
    else:
        summary["type"] = "continuous"
        summary["mean"] = np.mean(values.astype(float))
        summary["std"] = np.std(values.astype(float))
        
        min_val = values.min()
        max_val = values.max()
        data_range = max_val - min_val
        bin_scale = np.floor(np.log10(data_range))
        if data_range / (10 ** bin_scale) < 2.5:
            bin_scale -= 1 # Make sure there aren't only 2-3 bins
        upper_tol = 2 if (np.ceil(max_val / (10 ** bin_scale))) * (10 ** bin_scale) == max_val else 1
        hist_bins = np.arange(np.floor(min_val / (10 ** bin_scale)) * (10 ** bin_scale),
                                (np.ceil(max_val / (10 ** bin_scale)) + upper_tol) * (10 ** bin_scale),
                                10 ** bin_scale)
        
        summary["hist"] = {"counts": np.histogram(values, bins=hist_bins)[0].astype(int).tolist(), 
                        "bins": hist_bins.astype(float).tolist()}
    return summary