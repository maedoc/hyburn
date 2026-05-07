#!/usr/bin/env python3
"""
API discovery script for vbjax CrossCoder.

Instantiates a toy multi-view CrossCoder, trains for a few iterations,
and prints available method signatures.
"""

import numpy as np
import inspect

# vbjax should be installed
from vbjax.crosscoder import CrossCoder


def main():
    print("=" * 60)
    print("vbjax CrossCoder API Discovery")
    print("=" * 60)

    # 1. Instantiate
    cc = CrossCoder(variational=True, chunked_training=False)
    print(f"\n1. Instantiated CrossCoder(variational=True)")
    print(f"   Attributes: variational={cc.variational}, chunked_training={cc.chunked_training}")

    # 2. Add toy views
    np.random.seed(0)
    toy_view1 = np.random.randn(10, 10).astype(np.float32)
    toy_view2 = np.random.randn(10, 20).astype(np.float32)

    cc.add_view(toy_view1, "toy10", normalize="zscore", nonneg=False)
    cc.add_view(toy_view2, "toy20", normalize="zscore", nonneg=False)
    print(f"\n2. Added 2 toy views: toy10 (10 feats), toy20 (20 feats)")
    print(f"   parcs = {cc.parcs}")
    print(f"   conns shapes = {[c.shape for c in cc.conns]}")

    # 3. Mini training run (deterministic mode for speed)
    cc_det = CrossCoder(variational=False, chunked_training=False)
    cc_det.add_view(toy_view1, "toy10", normalize="zscore", nonneg=False)
    cc_det.add_view(toy_view2, "toy20", normalize="zscore", nonneg=False)
    cc_det.tts = 5

    print(f"\n3. Training deterministic CrossCoder for 10 iterations...")
    trace, wbs, cr = cc_det.train(nlat=2, lr=1e-2, niter=10, mb=5, tts=5)
    print(f"   trace length = {len(trace)}")
    print(f"   confusion rate = {cr}")
    print(f"   archs = {[a.nlat for a in cc_det.archs]}")

    # 4. Inspect trained weights structure
    print(f"\n4. Trained weights (wbs) structure:")
    for iv, (enc, dec) in enumerate(wbs):
        ew, eb = enc
        dw, db = dec
        print(f"   View {iv}: encoder_w={ew.shape}, encoder_b={eb.shape}, "
              f"decoder_w={dw.shape}, decoder_b={db.shape}")

    # 5. Encode / decode smoke test
    z = cc_det.encode(arch=2, parc="toy10", sample=False)
    print(f"\n5. Encoded toy10 -> latent shape = {np.asarray(z).shape}")

    z_all = cc_det.encode_all(arch=2, sample=False)
    for k, v in z_all.items():
        print(f"   encode_all '{k}' -> shape = {np.asarray(v).shape}")

    rec = cc_det.decode(arch=2, parc="toy10", z=z, raw=False)
    print(f"   Decoded latent -> rec shape = {np.asarray(rec).shape}")

    # 6. List all methods on CrossCoder
    print(f"\n6. CrossCoder methods:")
    for name, method in inspect.getmembers(CrossCoder, predicate=inspect.isfunction):
        if not name.startswith("_"):
            sig = inspect.signature(method)
            print(f"   {name}{sig}")

    print(f"\n7. Private methods of interest:")
    for name in ["_train_var", "_train_det", "_get_arch", "_conf_rates_var", "_conf_rates_det"]:
        if hasattr(CrossCoder, name):
            sig = inspect.signature(getattr(CrossCoder, name))
            print(f"   {name}{sig}")

    print("\n" + "=" * 60)
    print("API discovery complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
