/* tslint:disable */
/* eslint-disable */

/**
 * Metadata about a constructed engine, returned to JS after creation.
 */
export class EngineInfo {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly dt: number;
    readonly n_bold_monitors: number;
    readonly n_subnetworks: number;
    readonly nmodes: number;
    readonly nnodes: number;
    readonly nvar: number;
    readonly total_steps: number;
}

/**
 * Web-accessible simulation engine.
 *
 * Wraps `HybridEngine<NdArray<f32>>` with a JS-friendly API.
 * Construct from a JSON config string, then call `step()` or `step_n()`
 * to advance the simulation, and `trajectory()` / `bold_signal()` to
 * retrieve data for visualization.
 */
export class WebEngine {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get the current state of all subnetworks as a Float32Array.
     */
    all_states(): Float32Array;
    /**
     * Get the BOLD monitor signal as a Float32Array.
     *
     * Returns data for all BOLD monitors concatenated.
     * Each monitor's data has shape `[n_bold_volumes, nnodes]`.
     */
    bold_signal(): Float32Array;
    /**
     * Number of BOLD volumes recorded so far.
     */
    bold_volumes(): number;
    /**
     * Get the current state of the first subnetwork as a Float32Array.
     *
     * Shape: `[nvar, nnodes, nmodes]`
     */
    current_state(): Float32Array;
    /**
     * Current step number.
     */
    current_step(): number;
    /**
     * Get the integration time step.
     */
    dt(): number;
    /**
     * Create a new engine from a JSON config string.
     *
     * The JSON must conform to the `SimConfig` schema. Example:
     * ```json
     * {
     *   "sim_length": 1000.0,
     *   "dt": 0.1,
     *   "network": {
     *     "subnetworks": [{
     *       "model": "Generic2dOscillator",
     *       "nnodes": 2,
     *       "nmodes": 1,
     *       "params": [1.0, 0.0, -2.0, -10.0, 0.0, 0.02, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0],
     *       "initial_state": [0.0, 0.5, 0.0, 0.5]
     *     }],
     *     "projections": []
     *   }
     * }
     * ```
     */
    constructor(json: string);
    /**
     * Create a new engine from a TOML config string.
     */
    static from_toml(toml: string): WebEngine;
    /**
     * Get engine metadata (dimensions, dt, etc.).
     */
    info(): EngineInfo;
    /**
     * Get the integrator kind as a string ("heun", "euler", "euler_stochastic", "heun_stochastic").
     */
    integrator(): string;
    /**
     * Get the number of subnetworks.
     */
    n_subnetworks(): number;
    /**
     * Get the noise amplitude (nsig).
     */
    nsig(): number;
    /**
     * Advance the simulation by one step.
     */
    step(): void;
    /**
     * Advance the simulation by `n` steps.
     */
    step_n(n: number): void;
    /**
     * Number of steps run so far.
     */
    steps_run(): number;
    /**
     * Get the nmodes for a subnetwork.
     */
    subnetwork_nmodes(idx: number): number;
    /**
     * Get the nnodes for a subnetwork.
     */
    subnetwork_nnodes(idx: number): number;
    /**
     * Get the nvar for a subnetwork.
     */
    subnetwork_nvar(idx: number): number;
    /**
     * Get the raw trajectory data as a Float32Array (zero-copy).
     *
     * The trajectory is a flat array of f32 values with layout:
     * `[step0_var0_node0_mode0, step0_var0_node0_mode1, ..., step0_var0_node1_mode0, ...]`
     *
     * For a single subnetwork with `nvar` variables, `nnodes` nodes,
     * `nmodes` modes, and `n_steps` recorded steps, the shape is
     * `[n_steps, nvar, nnodes, nmodes]`.
     *
     * For multiple subnetworks, the data is concatenated per step.
     */
    trajectory(): Float32Array;
    /**
     * Get the trajectory length (number of f32 values).
     */
    trajectory_len(): number;
}

/**
 * Get a preset's JSON config by its ID.
 *
 * Returns the full SimConfig JSON with inline initial_state data
 * (NPY files already resolved to float arrays at build time).
 *
 * Returns an empty string if the ID is not found.
 */
export function get_preset(id: string): string;

/**
 * Initialize the console logger for WASM.
 * Call this once from JS before using any simulation functions.
 */
export function init_logger(): void;

/**
 * Get the list of available preset examples as a JSON string.
 *
 * Returns an array of `{id, name, description}` objects.
 */
export function list_presets(): string;

/**
 * Get the default parameters for a model as a JSON string.
 */
export function model_default_params(model_name: string): string;

/**
 * Get the model registry as a JSON string.
 * Returns an array of {name, nvar, ncvar, nparams} objects.
 */
export function model_registry_json(): string;

/**
 * Run a small SBI pipeline and return results as a JSON string.
 *
 * This is intended for small demo problems (few nodes, short simulation).
 * For realistic-scale SBI, use a server-side pipeline.
 *
 * # Arguments
 * * `config_json` - JSON string matching `SimConfig` schema
 * * `n_sweep` - Number of parameter sweep points
 * * `n_steps` - Simulation steps per sweep point
 * * `n_epochs` - MAF training epochs
 * * `batch_size` - MAF training batch size
 * * `n_post_samples` - Number of posterior samples per test point
 * * `param_idx` - Parameter index to sweep (default: 1 = I_ext for G2DO)
 */
export function run_sbi_json(config_json: string, n_sweep: number, n_steps: number, n_epochs: number, batch_size: number, n_post_samples: number, param_idx: number): string;

/**
 * Validate a JSON config string without creating an engine.
 * Returns an error message if the config is invalid, or empty string if valid.
 */
export function validate_config_json(json: string): string;

/**
 * Validate a TOML config string without creating an engine.
 */
export function validate_config_toml(toml: string): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_engineinfo_free: (a: number, b: number) => void;
    readonly __wbg_webengine_free: (a: number, b: number) => void;
    readonly engineinfo_dt: (a: number) => number;
    readonly engineinfo_n_bold_monitors: (a: number) => number;
    readonly engineinfo_n_subnetworks: (a: number) => number;
    readonly engineinfo_nmodes: (a: number) => number;
    readonly engineinfo_nnodes: (a: number) => number;
    readonly engineinfo_nvar: (a: number) => number;
    readonly engineinfo_total_steps: (a: number) => number;
    readonly get_preset: (a: number, b: number) => [number, number];
    readonly list_presets: () => [number, number];
    readonly model_default_params: (a: number, b: number) => [number, number, number, number];
    readonly model_registry_json: () => [number, number];
    readonly run_sbi_json: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number, number];
    readonly validate_config_json: (a: number, b: number) => [number, number];
    readonly validate_config_toml: (a: number, b: number) => [number, number];
    readonly webengine_all_states: (a: number) => any;
    readonly webengine_bold_signal: (a: number) => any;
    readonly webengine_bold_volumes: (a: number) => number;
    readonly webengine_current_state: (a: number) => any;
    readonly webengine_current_step: (a: number) => number;
    readonly webengine_dt: (a: number) => number;
    readonly webengine_from_json: (a: number, b: number) => [number, number, number];
    readonly webengine_from_toml: (a: number, b: number) => [number, number, number];
    readonly webengine_info: (a: number) => number;
    readonly webengine_integrator: (a: number) => [number, number];
    readonly webengine_n_subnetworks: (a: number) => number;
    readonly webengine_nsig: (a: number) => number;
    readonly webengine_step: (a: number) => void;
    readonly webengine_step_n: (a: number, b: number) => void;
    readonly webengine_steps_run: (a: number) => number;
    readonly webengine_subnetwork_nmodes: (a: number, b: number) => number;
    readonly webengine_subnetwork_nnodes: (a: number, b: number) => number;
    readonly webengine_subnetwork_nvar: (a: number, b: number) => number;
    readonly webengine_trajectory: (a: number) => any;
    readonly webengine_trajectory_len: (a: number) => number;
    readonly init_logger: () => void;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
