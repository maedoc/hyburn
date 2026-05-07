#!/usr/bin/env python3
"""
Validation script: vbjax CrossCoder on a synthetic multi-view connectome cohort.

Compares vbjax (Python/JAX) CrossCoder behaviour against the ground-truth
latent structure used to generate the data.

Outputs are written to tests/validate_output/.
"""

import os
import sys
import numpy as np
import json

# Ensure vbjax is importable
from vbjax.crosscoder import CrossCoder


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "validate_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_synthetic_cohort(
    n_subjects: int = 50,
    latent_dim: int = 4,
    view1_nodes: int = 10,
    view2_nodes: int = 15,
    noise_std: float = 0.1,
    seed: int = 42,
):
    """Generate a synthetic cohort with a known shared latent structure.

    Returns
    -------
    Z : ndarray, shape (n_subjects, latent_dim)
        Ground-truth shared latent codes.
    X1 : ndarray, shape (n_subjects, n_upper_tri_1)
        View-1 flat upper-triangular connectomes.
    X2 : ndarray, shape (n_subjects, n_upper_tri_2)
        View-2 flat upper-triangular connectomes.
    A1 : ndarray, shape (latent_dim, n_upper_tri_1)
        Ground-truth loading matrix for view 1.
    A2 : ndarray, shape (latent_dim, n_upper_tri_2)
        Ground-truth loading matrix for view 2.
    """
    rng = np.random.default_rng(seed)

    n_upper_tri_1 = view1_nodes * (view1_nodes - 1) // 2
    n_upper_tri_2 = view2_nodes * (view2_nodes - 1) // 2

    # Shared latent
    Z = rng.standard_normal((n_subjects, latent_dim))

    # Loading matrices
    A1 = rng.standard_normal((latent_dim, n_upper_tri_1))
    A2 = rng.standard_normal((latent_dim, n_upper_tri_2))

    # Observed views with additive Gaussian noise
    X1 = Z @ A1 + rng.normal(0.0, noise_std, (n_subjects, n_upper_tri_1))
    X2 = Z @ A2 + rng.normal(0.0, noise_std, (n_subjects, n_upper_tri_2))

    return Z, X1, X2, A1, A2


def compute_reconstruction_mse(cc: CrossCoder, arch: int, tts: int | None = None):
    """Compute per-view MSE on the *normalized* connectomes.

    For each view i, we:
      1. Encode view i deterministically -> z_i
      2. Decode z_i back to every view j -> recon_j
      3. Compute MSE(recon_j, conn_j)

    Returns a dict keyed by (encode_view, decode_view).
    """
    if tts is None:
        tts = cc.tts
    mses = {}
    for i, parc_i in enumerate(cc.parcs):
        z_i = cc.encode(arch, parc_i, tts=tts, sample=False)
        for j, parc_j in enumerate(cc.parcs):
            recon_j = cc.decode(arch, parc_j, z_i, raw=True)
            conn_j = cc.conns[j][tts:]
            mse = float(np.mean(np.square(np.asarray(recon_j) - np.asarray(conn_j))))
            mses[f"{parc_i}_to_{parc_j}"] = mse
    return mses


def main():
    print("=" * 70)
    print("vbjax CrossCoder Validation Script")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Synthetic cohort
    # ------------------------------------------------------------------
    N_SUBJECTS = 50
    LATENT_DIM = 4
    TTS = 40  # 40 train, 10 test
    NOISE_STD = 0.1

    Z_true, X1, X2, A1, A2 = generate_synthetic_cohort(
        n_subjects=N_SUBJECTS,
        latent_dim=LATENT_DIM,
        view1_nodes=10,
        view2_nodes=15,
        noise_std=NOISE_STD,
        seed=42,
    )

    print(f"\n1. Synthetic cohort generated")
    print(f"   Z_true shape  : {Z_true.shape}")
    print(f"   X1 shape    : {X1.shape}")
    print(f"   X2 shape    : {X2.shape}")
    print(f"   Train/test split: {TTS}/{N_SUBJECTS - TTS}")

    np.save(os.path.join(OUTPUT_DIR, "cohort_view1.npy"), X1)
    np.save(os.path.join(OUTPUT_DIR, "cohort_view2.npy"), X2)
    print(f"   Saved cohort_view1.npy, cohort_view2.npy")

    # ------------------------------------------------------------------
    # 2. Train vbjax CrossCoder (variational)
    # ------------------------------------------------------------------
    cc = CrossCoder(variational=True, chunked_training=True)
    cc.add_view(X1, "view1", normalize="zscore", nonneg=False)
    cc.add_view(X2, "view2", normalize="zscore", nonneg=False)
    cc.tts = TTS

    print(f"\n2. Training vbjax CrossCoder (variational=True)")
    print(f"   Latent dim   : {LATENT_DIM}")
    print(f"   Learning rate: 3e-4")
    print(f"   Iterations   : 2000")
    print(f"   Batch size   : 64")
    print(f"   β anneal     : 0.0 -> 1e-3 over 1500 steps")

    trace, wbs, cr = cc.train(
        nlat=LATENT_DIM,
        lr=3e-4,
        niter=2000,
        mb=64,
        tts=TTS,
        beta_start=0.0,
        beta_end=1e-3,
        anneal_steps=1500,
    )

    print(f"   Training complete.")
    print(f"   Final trace entry length: {len(trace[-1])}")

    # ------------------------------------------------------------------
    # 3. Extract & save latent codes (consensus = average over views)
    # ------------------------------------------------------------------
    z_all = cc.encode_all(arch=LATENT_DIM, tts=None, sample=False)
    # Average deterministic latents across both views for consensus
    z_consensus = np.mean([np.asarray(z_all[p]) for p in cc.parcs], axis=0)
    print(f"\n3. Consensus latent shape: {z_consensus.shape}")
    np.save(os.path.join(OUTPUT_DIR, "vbjax_latents.npy"), z_consensus)
    print(f"   Saved vbjax_latents.npy")

    # Also save per-view latents for reference
    for parc, z_p in z_all.items():
        np.save(os.path.join(OUTPUT_DIR, f"vbjax_latents_{parc}.npy"), np.asarray(z_p))

    # ------------------------------------------------------------------
    # 4. Extract & save decoder / encoder weights
    # ------------------------------------------------------------------
    ta = cc._get_arch(LATENT_DIM)

    for iv, (enc, dec) in enumerate(ta.wbs):
        parc = cc.parcs[iv]
        # Decoder: (w_dec, b_dec)  w_dec shape (nlat, n_features)
        w_dec, b_dec = dec
        np.save(os.path.join(OUTPUT_DIR, f"vbjax_decoder_{parc}.npy"), np.asarray(w_dec))

        # Encoder (variational): ((w_mu, b_mu), (w_lv, b_lv))
        if cc.variational:
            (w_mu, b_mu), (w_lv, b_lv) = enc
            # Concatenate mu and logvar weights along last axis: (n_features, 2*nlat)
            w_enc = np.concatenate([np.asarray(w_mu), np.asarray(w_lv)], axis=1)
            np.save(os.path.join(OUTPUT_DIR, f"vbjax_encoder_{parc}.npy"), w_enc)
        else:
            (w_mu, b_mu) = enc
            np.save(os.path.join(OUTPUT_DIR, f"vbjax_encoder_{parc}.npy"), np.asarray(w_mu))

    print(f"\n4. Saved decoder/encoder weights for {len(ta.wbs)} views")

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    # Confusion rate (returned by train)
    confusion_rate = float(cr)

    # Also compute confusion matrix (full cross-view matrix)
    conf_mat = cc.confusion_matrix(arch=LATENT_DIM, tts=None, n_samples=0)
    conf_mat = np.asarray(conf_mat)

    # Reconstruction MSE per view
    mses = compute_reconstruction_mse(cc, arch=LATENT_DIM, tts=None)

    # Final loss values from trace
    # trace entries for variational: [l_tr, l_te, r_tr, kl_tr, ...r_det...]
    final_trace = trace[-1]
    final_train_loss = float(final_trace[0])
    final_test_loss = float(final_trace[1])
    final_recon = float(final_trace[2])
    final_kl = float(final_trace[3])

    # Latent mean / covariance via calc_mvn
    mvn = cc.calc_mvn(arch=LATENT_DIM, tts=None)
    latent_mean = np.asarray(mvn.mean)
    latent_cov = np.asarray(mvn.cov)

    # Correlation between true latent and inferred consensus
    # Standardise both before comparing
    Z_true_std = (Z_true - Z_true.mean(axis=0)) / (Z_true.std(axis=0) + 1e-9)
    z_cons_std = (z_consensus - z_consensus.mean(axis=0)) / (z_consensus.std(axis=0) + 1e-9)
    # Procrustes-like: best linear alignment via SVD of cross-covariance
    R = np.linalg.svd(z_cons_std.T @ Z_true_std, full_matrices=False)
    aligned = z_cons_std @ (R[0] @ R[2])
    rmse_vs_true = float(np.sqrt(np.mean((aligned - Z_true_std) ** 2)))

    print(f"\n5. Metrics")
    print(f"   Confusion rate          : {confusion_rate:.4f}")
    print(f"   Confusion matrix        :\n{conf_mat}")
    for k, v in mses.items():
        print(f"   Recon MSE ({k}) : {v:.6f}")
    print(f"   Final train loss        : {final_train_loss:.6f}")
    print(f"   Final test loss         : {final_test_loss:.6f}")
    print(f"   Final recon term        : {final_recon:.6f}")
    print(f"   Final KL term           : {final_kl:.6f}")
    print(f"   Latent mean (consensus) : {latent_mean}")
    print(f"   Latent cov diagonal     : {np.diag(latent_cov)}")
    print(f"   RMSE vs true latent     : {rmse_vs_true:.4f}")

    # Save confusion rate to text
    with open(os.path.join(OUTPUT_DIR, "vbjax_confusion_rate.txt"), "w") as f:
        f.write(f"{confusion_rate}\n")

    # Save summary JSON
    summary = {
        "confusion_rate": confusion_rate,
        "confusion_matrix": conf_mat.tolist(),
        "reconstruction_mse": mses,
        "latent_dimension": LATENT_DIM,
        "training_iterations": 2000,
        "train_test_split": TTS,
        "n_subjects": N_SUBJECTS,
        "final_train_loss": final_train_loss,
        "final_test_loss": final_test_loss,
        "final_recon_term": final_recon,
        "final_kl_term": final_kl,
        "latent_mean": latent_mean.tolist(),
        "latent_cov": latent_cov.tolist(),
        "rmse_vs_true_latent": rmse_vs_true,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n6. All outputs written to {OUTPUT_DIR}")
    print(f"   vbjax_confusion_rate.txt")
    print(f"   summary.json")
    print("=" * 70)
    print("Validation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
