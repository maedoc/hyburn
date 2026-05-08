#!/usr/bin/env python3
"""Generate embedded WASM preset data from example TOML configs and NPY files.

Reads all SimConfig TOML files from examples/, resolves NPY initial_state
references to inline float arrays, serializes as JSON, and writes a Rust
source file with const preset data for the WASM module.

Run from the project root: python3 scripts/gen_presets.py
Output: src/presets.rs
"""

import json
import os
import struct
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
EXAMPLES_DIR = os.path.join(PROJECT_DIR, "examples")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "src", "presets.rs")

# Only include SimConfig files (not sweep/pipeline configs)
SIM_CONFIGS = [
    "demo.toml",
    "g2do_simple.toml",
    "g2do_bold.toml",
    "g2do_sweep.toml",
    "gpu_wgpu.toml",
    "crosscoder_sim.toml",
    "two_population_coupled.toml",
    "stimulus_patterns.toml",
    "jansen_rit_evoked.toml",
    "mpr_sweep.toml",
    "whole_brain_multi_model.toml",
    "wilson_cowan_ring.toml",
]

# Human-readable names and descriptions
PRESET_META = {
    "demo.toml": ("G2DO Demo (74 nodes)", "Standard 74-node Generic2dOscillator with all-to-all coupling"),
    "g2do_simple.toml": ("G2DO Simple", "Minimal 74-node G2DO with no projections"),
    "g2do_bold.toml": ("G2DO + BOLD Monitor", "74-node G2DO with Balloon-Windkessel BOLD signal"),
    "g2do_sweep.toml": ("G2DO Sweep", "74-node G2DO with inline initial state for parameter sweeps"),
    "gpu_wgpu.toml": ("G2DO GPU/WGPU", "74-node G2DO configured for GPU acceleration"),
    "crosscoder_sim.toml": ("CrossCoder Simulation", "74-node G2DO with CrossCoder projection weights"),
    "two_population_coupled.toml": ("Two-Population Coupled", "Two 74-node G2DO populations with cross-projections"),
    "stimulus_patterns.toml": ("Stimulus Patterns", "4 sub-populations with different stimulus waveforms"),
    "jansen_rit_evoked.toml": ("Jansen-Rit Evoked", "74-node JansenRit with impulse stimulus"),
    "mpr_sweep.toml": ("MPR Sweep", "74-node MontbrioPazoRoxin with inline initial state"),
    "whole_brain_multi_model.toml": ("Whole Brain Multi-Model", "G2DO + Kuramoto multi-model network"),
    "wilson_cowan_ring.toml": ("Wilson-Cowan Ring", "74-node WilsonCowan ring network"),
}


def read_npy(path: str) -> tuple[list[float], list[int]]:
    """Read an NPY file and return (flat_data, shape)."""
    arr = np.load(path)
    return arr.flatten().tolist(), list(arr.shape)


def parse_toml_simple(path: str) -> dict:
    """Minimal TOML parser for SimConfig files.

    Only handles the subset used by hyburn configs. Uses tomllib if available.
    """
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            pass

    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_npy_path(config: dict, examples_dir: str) -> dict:
    """Replace NpyPath initial_state references with inline data.

    Modifies the config dict in-place, replacing string initial_state values
    with inline float arrays. Also adds a '_shape' key for Memory variant.
    """
    import copy
    config = copy.deepcopy(config)

    network = config.get("network", {})
    subnetworks = network.get("subnetworks", [])

    for sub in subnetworks:
        initial = sub.get("initial_state")
        if isinstance(initial, str) and initial.endswith(".npy"):
            # Resolve relative to examples dir
            npy_path = os.path.join(examples_dir, os.path.basename(initial))
            if os.path.exists(npy_path):
                data, shape = read_npy(npy_path)
                # Convert to inline array with shape metadata
                sub["initial_state"] = data
                sub["_initial_state_shape"] = shape
            else:
                print(f"WARNING: NPY file not found: {npy_path}, using zeros", file=sys.stderr)
                model = sub.get("model", "Generic2dOscillator")
                nnodes = sub.get("nnodes", 74)
                nmodes = sub.get("nmodes", 1)
                # Guess nvar from model
                nvar_map = {
                    "Generic2dOscillator": 2,
                    "MontbrioPazoRoxin": 2,
                    "ReducedWongWang": 1,
                    "Kuramoto": 1,
                    "JansenRit": 6,
                    "WilsonCowan": 2,
                }
                nvar = nvar_map.get(model, 2)
                sub["initial_state"] = [0.0] * (nvar * nnodes * nmodes)
                sub["_initial_state_shape"] = [nvar, nnodes, nmodes]

    return config


def config_to_json(config: dict) -> str:
    """Convert a resolved config dict to a JSON string suitable for SimConfig::from_json_str."""
    # Build the JSON-serializable structure matching SimConfig's serde schema
    network = config.get("network", {})
    subnetworks = network.get("subnetworks", [])

    json_subs = []
    for sub in subnetworks:
        json_sub = {
            "model": sub.get("model"),
            "nnodes": sub.get("nnodes", 1),
            "nmodes": sub.get("nmodes", 1),
            "params": sub.get("params", []),
            "initial_state": sub.get("initial_state", []),
        }
        json_subs.append(json_sub)

    projections = network.get("projections", [])
    json_projs = []
    for proj in projections:
        json_proj = {
            "src": proj.get("src", 0),
            "tgt": proj.get("tgt", 0),
            "conn_type": proj.get("conn_type", "all_to_all"),
            "coupling_fn": proj.get("coupling_fn", "Linear"),
            "coupling_params": proj.get("coupling_params", []),
            "cvar_map": proj.get("cvar_map", "0:0"),
            "delays": proj.get("delays", []),
        }
        # Handle weights (can be scalar, dense, or CSR)
        weights = proj.get("weights")
        if isinstance(weights, (int, float)):
            json_proj["weights"] = weights
        elif isinstance(weights, list):
            json_proj["weights"] = weights
        elif isinstance(weights, dict):
            json_proj["weights"] = weights
        json_projs.append(json_proj)

    # Build monitors
    monitors = config.get("monitors", [])
    json_monitors = []
    for mon in monitors:
        json_mon = {
            "monitor_type": mon.get("monitor_type"),
        }
        if "period" in mon:
            json_mon["period"] = mon["period"]
        if "tr" in mon:
            json_mon["tr"] = mon["tr"]
        if "bold_period" in mon:
            json_mon["bold_period"] = mon["bold_period"]
        json_monitors.append(json_mon)

    # Build stimuli
    stimuli = config.get("stimuli", [])
    json_stimuli = []
    for stim in stimuli:
        json_stim = {
            "target": stim.get("target", 0),
            "temporal": stim.get("temporal", "Impulse"),
            "params": stim.get("params", []),
        }
        json_stimuli.append(json_stim)

    json_config = {
        "sim_length": config.get("sim_length", 1000.0),
        "dt": config.get("dt", 0.1),
        "integrator": config.get("integrator", "heun"),
        "nsig": config.get("nsig", 0.0),
        "network": {
            "subnetworks": json_subs,
            "projections": json_projs,
        },
    }
    if json_monitors:
        json_config["monitors"] = json_monitors
    if json_stimuli:
        json_config["stimuli"] = json_stimuli

    return json.dumps(json_config, separators=(',', ':'))


def escape_rust_string(s: str) -> str:
    """Escape a string for embedding in Rust source code."""
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')


def generate_rust_source(presets: list[tuple[str, str, str, str]]) -> str:
    """Generate Rust source code with embedded preset data.

    Args:
        presets: list of (id, name, description, json_config) tuples
    """
    lines = []
    lines.append("//! Auto-generated preset configurations for the WASM module.")
    lines.append("//!")
    lines.append("//! Generated by scripts/gen_presets.py — DO NOT EDIT MANUALLY.")
    lines.append("//!")
    lines.append("//! Each preset is a pre-resolved JSON SimConfig with inline initial_state data")
    lines.append("//! (NPY file references replaced with float arrays at build time).")
    lines.append("")
    lines.append("use serde::Deserialize;")
    lines.append("")
    lines.append("/// Metadata for a preset example.")
    lines.append("#[derive(Debug, Clone, Deserialize)]")
    lines.append("pub struct PresetMeta {")
    lines.append("    /// Identifier (filename stem).")
    lines.append("    pub id: &'static str,")
    lines.append("    /// Human-readable name.")
    lines.append("    pub name: &'static str,")
    lines.append("    /// One-line description.")
    lines.append("    pub description: &'static str,")
    lines.append("}")
    lines.append("")
    lines.append("/// All available presets.")
    lines.append("pub const PRESETS: &[PresetMeta] = &[")

    for pid, name, desc, _ in presets:
        lines.append(f'    PresetMeta {{ id: "{pid}", name: "{escape_rust_string(name)}", description: "{escape_rust_string(desc)}" }},')

    lines.append("];")
    lines.append("")

    # Now generate the JSON config strings
    for pid, _, _, json_cfg in presets:
        var_name = pid.replace("-", "_").replace(".", "_")
        # Split long strings into chunks for readability
        lines.append(f'/// JSON config for "{pid}" preset.')
        lines.append(f'pub const PRESET_{var_name}_JSON: &str = r#"{json_cfg}"#;')
        lines.append("")

    # Generate lookup function
    lines.append("/// Look up a preset's JSON config by ID.")
    lines.append("/// Returns None if the ID is not found.")
    lines.append("pub fn get_preset_json(id: &str) -> Option<&'static str> {")
    lines.append("    match id {")
    for pid, _, _, _ in presets:
        var_name = pid.replace("-", "_").replace(".", "_")
        lines.append(f'        "{pid}" => Some(PRESET_{var_name}_JSON),')
    lines.append('        _ => None,')
    lines.append("    }")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def main():
    presets = []

    for filename in SIM_CONFIGS:
        filepath = os.path.join(EXAMPLES_DIR, filename)
        if not os.path.exists(filepath):
            print(f"WARNING: {filepath} not found, skipping", file=sys.stderr)
            continue

        # Parse TOML
        config = parse_toml_simple(filepath)

        # Resolve NPY paths to inline data
        resolved = resolve_npy_path(config, EXAMPLES_DIR)

        # Convert to JSON
        json_str = config_to_json(resolved)

        # Get metadata
        pid = filename.replace(".toml", "")
        name, desc = PRESET_META.get(filename, (pid, ""))

        presets.append((pid, name, desc, json_str))
        print(f"  Processed: {filename} -> {len(json_str)} bytes JSON")

    # Generate Rust source
    rust_source = generate_rust_source(presets)

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(rust_source)

    print(f"\nGenerated {OUTPUT_PATH} with {len(presets)} presets")


if __name__ == "__main__":
    main()
