#!/usr/bin/env python3
"""Generate reference feature extraction outputs from TVB/NumPy.

Runs a short simulation, then extracts features using Python implementations
matching hyburn's feature extractors, and saves them for comparison.

Saves:
  - ref/features/{name}_{feature_set}_features.npy — feature vectors

Feature sets: classic, fc, spectral, temporal

Usage:
    ref/.venv/bin/python ref/generate_features.py
"""

import os
import sys
import warnings

import numpy as np
from scipy.signal import welch
from scipy.fftpack import fft

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from configs import SMALL_CONFIGS


# ---------------------------------------------------------------------------
# Feature extraction functions (matching hyburn implementations)
# ---------------------------------------------------------------------------

EEG_BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 80.0),
]


def classic_features(series, fs):
    """Extract mean, var, lag-1 autocorrelation — matching hyburn classic features."""
    mean = np.mean(series)
    var = np.var(series)
    # Lag-1 autocorrelation
    if var > 1e-12:
        ac1 = np.corrcoef(series[:-1], series[1:])[0, 1]
    else:
        ac1 = 0.0
    return np.array([mean, var, ac1], dtype=np.float32)


def fc_features(timeseries_matrix):
    """Extract FC-based features from [n_steps, nnodes] matrix.

    Returns: fc_mean, fc_std, fc_min, fc_max, fc_median,
             fc_homotopic_mean, fc_homotopic_std
    """
    # Correlation matrix
    n_steps, nnodes = timeseries_matrix.shape
    if nnodes < 2 or n_steps < 10:
        return np.full(7, np.nan, dtype=np.float32)

    corr = np.corrcoef(timeseries_matrix.T)  # [nnodes, nnodes]
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    upper = corr[mask]

    # Homotopic: diagonal of off-diagonal block for even/odd pairing
    n_half = nnodes // 2
    if n_half > 0:
        homotopic = np.diag(corr[:n_half, n_half:2*n_half]) if 2*n_half <= nnodes else np.array([])
    else:
        homotopic = np.array([])

    features = [
        np.nanmean(upper),
        np.nanstd(upper),
        np.nanmin(upper),
        np.nanmax(upper),
        np.nanmedian(upper),
        np.nanmean(homotopic) if len(homotopic) > 0 else np.nan,
        np.nanstd(homotopic) if len(homotopic) > 0 else np.nan,
    ]
    return np.array(features, dtype=np.float32)


def spectral_features(series, fs):
    """Extract spectral features matching hyburn's implementation.

    Returns: 5 band-power fractions + centroid + spread + skewness + kurtosis = 9 features.
    """
    if len(series) < 16:
        return np.full(9, np.nan, dtype=np.float32)

    nperseg = min(256, max(16, len(series) // 2))
    freqs, psd = welch(series, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    total_power = np.sum(psd)
    if total_power <= 0 or not np.isfinite(total_power):
        return np.full(9, np.nan, dtype=np.float32)

    # Band powers
    features = []
    for _, f_low, f_high in EEG_BANDS:
        mask = (freqs >= f_low) & (freqs < f_high)
        band_power = np.sum(psd[mask])
        features.append(band_power / total_power)

    # Spectral moments
    centroid = np.sum(psd * freqs) / total_power
    spread = np.sqrt(np.sum(psd * (freqs - centroid)**2) / total_power)

    skewness = 0.0
    kurtosis = 0.0
    if spread > 1e-12:
        skewness = np.sum(psd * (freqs - centroid)**3) / total_power / spread**3
        kurtosis = np.sum(psd * (freqs - centroid)**4) / total_power / spread**4 - 3.0

    features.extend([centroid, spread, skewness, kurtosis])
    return np.array(features, dtype=np.float32)


def temporal_features(series):
    """Extract temporal statistics matching hyburn's implementation.

    Returns: mean, std, min, max, median, skewness, kurtosis = 7 features.
    """
    from scipy.stats import skew, kurtosis as kurt
    return np.array([
        np.mean(series),
        np.std(series),
        np.min(series),
        np.max(series),
        np.median(series),
        skew(series) if np.std(series) > 1e-12 else 0.0,
        kurt(series) if np.std(series) > 1e-12 else 0.0,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def run_feature_generation(name, cfg, output_dir):
    """Run simulation, extract features, save."""
    from tvb.simulator.hybrid.subnetwork import Subnetwork

    model = cfg["model_factory"]()
    scheme_factory = cfg.get("integrator", "heun")
    if scheme_factory == "heun":
        from tvb.simulator.integrators import HeunDeterministic
        scheme = HeunDeterministic(dt=cfg["dt"])
    else:
        from tvb.simulator.integrators import EulerDeterministic
        scheme = EulerDeterministic(dt=cfg["dt"])

    nnodes = cfg["nnodes"]
    nvar = model.nvar
    ncvar = len(model.cvar)

    sn = Subnetwork(name=name, model=model, scheme=scheme, nnodes=nnodes)

    # Zero IC
    ic = np.zeros((nvar, nnodes, 1), dtype=np.float32)
    if nvar >= 2:
        ic[1, :, :] = 0.5
    sn.state = ic.copy()
    sn.configure(simulation_length=cfg["sim_length"])

    # Run
    n_steps = int(cfg["sim_length"] / cfg["dt"])
    fs = 1000.0 / cfg["dt"]  # Hz

    c = np.zeros((ncvar, nnodes, 1), dtype=np.float32)
    # Collect trajectory: [n_steps+1, nvar, nnodes]
    trajectory = np.zeros((n_steps + 1, nvar, nnodes), dtype=np.float32)
    trajectory[0] = sn.state[:, :, 0]

    for step in range(1, n_steps + 1):
        sn.state = sn.step(step, sn.state, c)
        trajectory[step] = sn.state[:, :, 0]

    os.makedirs(output_dir, exist_ok=True)

    # Extract features for each variable, averaged over nodes
    for var in range(nvar):
        var_series = trajectory[:, var, :].mean(axis=1)  # mean over nodes

        # Classic
        feats_classic = classic_features(var_series, fs)
        path = os.path.join(output_dir, f"{name}_v{var}_classic_features.npy")
        np.save(path, feats_classic)
        print(f"  {name}_v{var}_classic: shape={feats_classic.shape}")

        # Spectral
        feats_spectral = spectral_features(var_series, fs)
        path = os.path.join(output_dir, f"{name}_v{var}_spectral_features.npy")
        np.save(path, feats_spectral)
        print(f"  {name}_v{var}_spectral: shape={feats_spectral.shape}")

        # Temporal
        feats_temporal = temporal_features(var_series)
        path = os.path.join(output_dir, f"{name}_v{var}_temporal_features.npy")
        np.save(path, feats_temporal)
        print(f"  {name}_v{var}_temporal: shape={feats_temporal.shape}")

    # FC features (need multi-node timeseries)
    if nnodes >= 2:
        for var in range(nvar):
            ts_matrix = trajectory[:, var, :]  # [n_steps, nnodes]
            feats_fc = fc_features(ts_matrix)
            path = os.path.join(output_dir, f"{name}_v{var}_fc_features.npy")
            np.save(path, feats_fc)
            print(f"  {name}_v{var}_fc: shape={feats_fc.shape}")


def main():
    output_dir = os.path.join(REPO_ROOT, "ref", "features")

    print(f"Generating feature extraction reference outputs...")
    for name, cfg in SMALL_CONFIGS.items():
        run_feature_generation(name, cfg, output_dir)

    print(f"\nDone. Reference outputs saved to ref/features/")


if __name__ == "__main__":
    main()
