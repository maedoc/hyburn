# Reference Datasets

Python-based reference outputs generated from the [TVB hybrid simulator](https://github.com/the-virtual-brain/tvb-root) (`hybrid-model` branch) for numerical validation of hyburn.

## Setup

```bash
./ref/setup.sh          # create venv + install tvb-root
./ref/setup.sh --check  # verify installation
```

Requires: [uv](https://docs.astral.sh/uv/), `~/src/tvb-root` on `hybrid-model` branch.

## Generate Reference Data

```bash
# Single simulations (small configs: 2 nodes, fast)
ref/.venv/bin/python ref/generate_single_sim.py

# Full 74-node simulation
ref/.venv/bin/python ref/generate_single_sim.py --config g2do_74

# Parameter sweeps (128-point I_ext sweep)
ref/.venv/bin/python ref/generate_sweep.py --config g2do_I_ext_sweep

# BOLD monitor outputs
ref/.venv/bin/python ref/generate_bold.py --config g2do_bold

# Feature extraction
ref/.venv/bin/python ref/generate_features.py
```

## Regenerate All

```bash
ref/.venv/bin/python ref/generate_single_sim.py
ref/.venv/bin/python ref/generate_single_sim.py --full
ref/.venv/bin/python ref/generate_sweep.py
ref/.venv/bin/python ref/generate_bold.py
ref/.venv/bin/python ref/generate_features.py
```

## Directory Layout

```
ref/
├── setup.sh                  # venv setup
├── configs.py                # parameter sets matching hyburn examples
├── generate_single_sim.py    # single simulation reference
├── generate_sweep.py         # sweep reference
├── generate_bold.py          # BOLD monitor reference
├── generate_features.py      # feature extraction reference
├── .venv/                    # Python venv (git-ignored)
├── single_sim/               # .npy reference outputs (git LFS)
├── sweep/                    # .npy reference outputs (git LFS)
├── bold/                     # .npy reference outputs (git LFS)
└── features/                 # .npy reference outputs (git LFS)
```

## Numerical Tolerance

Hyburn and TVB use different float operation ordering, so exact bit-for-bit
match is not expected. Valid tolerance ranges:

| Metric | Tolerance |
|--------|-----------|
| Single sim final state | relative error < 1e-3 |
| Sweep mean trajectory | relative error < 5e-3 |
| BOLD signal | relative error < 1e-2 |
| Feature vectors | relative error < 1e-2 |
