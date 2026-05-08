/* @ts-self-types="./hyburn.d.ts" */

/**
 * Metadata about a constructed engine, returned to JS after creation.
 */
export class EngineInfo {
    static __wrap(ptr) {
        const obj = Object.create(EngineInfo.prototype);
        obj.__wbg_ptr = ptr;
        EngineInfoFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EngineInfoFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_engineinfo_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get dt() {
        const ret = wasm.engineinfo_dt(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get n_bold_monitors() {
        const ret = wasm.engineinfo_n_bold_monitors(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get n_subnetworks() {
        const ret = wasm.engineinfo_n_subnetworks(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get nmodes() {
        const ret = wasm.engineinfo_nmodes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get nnodes() {
        const ret = wasm.engineinfo_nnodes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get nvar() {
        const ret = wasm.engineinfo_nvar(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get total_steps() {
        const ret = wasm.engineinfo_total_steps(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) EngineInfo.prototype[Symbol.dispose] = EngineInfo.prototype.free;

/**
 * Web-accessible simulation engine.
 *
 * Wraps `HybridEngine<NdArray<f32>>` with a JS-friendly API.
 * Construct from a JSON config string, then call `step()` or `step_n()`
 * to advance the simulation, and `trajectory()` / `bold_signal()` to
 * retrieve data for visualization.
 */
export class WebEngine {
    static __wrap(ptr) {
        const obj = Object.create(WebEngine.prototype);
        obj.__wbg_ptr = ptr;
        WebEngineFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WebEngineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_webengine_free(ptr, 0);
    }
    /**
     * Get the current state of all subnetworks as a Float32Array.
     * @returns {Float32Array}
     */
    all_states() {
        const ret = wasm.webengine_all_states(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the BOLD monitor signal as a Float32Array.
     *
     * Returns data for all BOLD monitors concatenated.
     * Each monitor's data has shape `[n_bold_volumes, nnodes]`.
     * @returns {Float32Array}
     */
    bold_signal() {
        const ret = wasm.webengine_bold_signal(this.__wbg_ptr);
        return ret;
    }
    /**
     * Number of BOLD volumes recorded so far.
     * @returns {number}
     */
    bold_volumes() {
        const ret = wasm.webengine_bold_volumes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the current state of the first subnetwork as a Float32Array.
     *
     * Shape: `[nvar, nnodes, nmodes]`
     * @returns {Float32Array}
     */
    current_state() {
        const ret = wasm.webengine_current_state(this.__wbg_ptr);
        return ret;
    }
    /**
     * Current step number.
     * @returns {number}
     */
    current_step() {
        const ret = wasm.webengine_current_step(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the integration time step.
     * @returns {number}
     */
    dt() {
        const ret = wasm.webengine_dt(this.__wbg_ptr);
        return ret;
    }
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
     * @param {string} json
     */
    constructor(json) {
        const ptr0 = passStringToWasm0(json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.webengine_from_json(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        WebEngineFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create a new engine from a TOML config string.
     * @param {string} toml
     * @returns {WebEngine}
     */
    static from_toml(toml) {
        const ptr0 = passStringToWasm0(toml, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.webengine_from_toml(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WebEngine.__wrap(ret[0]);
    }
    /**
     * Get engine metadata (dimensions, dt, etc.).
     * @returns {EngineInfo}
     */
    info() {
        const ret = wasm.webengine_info(this.__wbg_ptr);
        return EngineInfo.__wrap(ret);
    }
    /**
     * Get the integrator kind as a string ("heun", "euler", "euler_stochastic", "heun_stochastic").
     * @returns {string}
     */
    integrator() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.webengine_integrator(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get the number of subnetworks.
     * @returns {number}
     */
    n_subnetworks() {
        const ret = wasm.webengine_n_subnetworks(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the noise amplitude (nsig).
     * @returns {number}
     */
    nsig() {
        const ret = wasm.webengine_nsig(this.__wbg_ptr);
        return ret;
    }
    /**
     * Advance the simulation by one step.
     */
    step() {
        wasm.webengine_step(this.__wbg_ptr);
    }
    /**
     * Advance the simulation by `n` steps.
     * @param {number} n
     */
    step_n(n) {
        wasm.webengine_step_n(this.__wbg_ptr, n);
    }
    /**
     * Number of steps run so far.
     * @returns {number}
     */
    steps_run() {
        const ret = wasm.webengine_steps_run(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the nmodes for a subnetwork.
     * @param {number} idx
     * @returns {number}
     */
    subnetwork_nmodes(idx) {
        const ret = wasm.webengine_subnetwork_nmodes(this.__wbg_ptr, idx);
        return ret >>> 0;
    }
    /**
     * Get the nnodes for a subnetwork.
     * @param {number} idx
     * @returns {number}
     */
    subnetwork_nnodes(idx) {
        const ret = wasm.webengine_subnetwork_nnodes(this.__wbg_ptr, idx);
        return ret >>> 0;
    }
    /**
     * Get the nvar for a subnetwork.
     * @param {number} idx
     * @returns {number}
     */
    subnetwork_nvar(idx) {
        const ret = wasm.webengine_subnetwork_nvar(this.__wbg_ptr, idx);
        return ret >>> 0;
    }
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
     * @returns {Float32Array}
     */
    trajectory() {
        const ret = wasm.webengine_trajectory(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the trajectory length (number of f32 values).
     * @returns {number}
     */
    trajectory_len() {
        const ret = wasm.webengine_trajectory_len(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) WebEngine.prototype[Symbol.dispose] = WebEngine.prototype.free;

/**
 * Get a preset's JSON config by its ID.
 *
 * Returns the full SimConfig JSON with inline initial_state data
 * (NPY files already resolved to float arrays at build time).
 *
 * Returns an empty string if the ID is not found.
 * @param {string} id
 * @returns {string}
 */
export function get_preset(id) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.get_preset(ptr0, len0);
        deferred2_0 = ret[0];
        deferred2_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * Initialize the console logger for WASM.
 * Call this once from JS before using any simulation functions.
 */
export function init_logger() {
    wasm.init_logger();
}

/**
 * Get the list of available preset examples as a JSON string.
 *
 * Returns an array of `{id, name, description}` objects.
 * @returns {string}
 */
export function list_presets() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.list_presets();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Get the default parameters for a model as a JSON string.
 * @param {string} model_name
 * @returns {string}
 */
export function model_default_params(model_name) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(model_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.model_default_params(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Get the model registry as a JSON string.
 * Returns an array of {name, nvar, ncvar, nparams} objects.
 * @returns {string}
 */
export function model_registry_json() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.model_registry_json();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Run a small SBI pipeline (legacy wrapper — calls `run_sbi_json_cfg`
 * with default range [-0.5, 0.5] and Classic feature set).
 * @param {string} config_json
 * @param {number} n_sweep
 * @param {number} n_steps
 * @param {number} n_epochs
 * @param {number} batch_size
 * @param {number} n_post_samples
 * @param {number} param_idx
 * @returns {string}
 */
export function run_sbi_json(config_json, n_sweep, n_steps, n_epochs, batch_size, n_post_samples, param_idx) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.run_sbi_json(ptr0, len0, n_sweep, n_steps, n_epochs, batch_size, n_post_samples, param_idx);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Run a full SBI pipeline in the browser and return results as JSON.
 *
 * Uses `BatchHybridEngine` for the sweep (all points in one batch-dim
 * engine) instead of the old per-point `HybridEngine` loop. Removes
 * hardcoded G2DO/Euler/dt=0.1 assumptions — reads model, integrator,
 * dt, and initial state from the config.
 *
 * # Arguments
 * * `config_json` - SimConfig JSON string
 * * `n_sweep` - number of sweep points
 * * `n_steps` - simulation steps per point
 * * `n_epochs` - MAF training epochs
 * * `batch_size` - MAF training batch size
 * * `n_post_samples` - posterior samples per test point
 * * `param_name` - sweep parameter name like "I_ext" or "subnetworks[0].params[1]"
 *   (or numeric param_idx like "1" for backward compat)
 * * `range_min`, `range_max` - sweep value range
 * * `feature_set` - "classic" (3 stats) or "catch22" (22 features)
 * @param {string} config_json
 * @param {number} n_sweep
 * @param {number} n_steps
 * @param {number} n_epochs
 * @param {number} batch_size
 * @param {number} n_post_samples
 * @param {string} param_name
 * @param {number} range_min
 * @param {number} range_max
 * @param {string} feature_set
 * @returns {string}
 */
export function run_sbi_json_cfg(config_json, n_sweep, n_steps, n_epochs, batch_size, n_post_samples, param_name, range_min, range_max, feature_set) {
    let deferred5_0;
    let deferred5_1;
    try {
        const ptr0 = passStringToWasm0(config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(param_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(feature_set, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.run_sbi_json_cfg(ptr0, len0, n_sweep, n_steps, n_epochs, batch_size, n_post_samples, ptr1, len1, range_min, range_max, ptr2, len2);
        var ptr4 = ret[0];
        var len4 = ret[1];
        if (ret[3]) {
            ptr4 = 0; len4 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred5_0 = ptr4;
        deferred5_1 = len4;
        return getStringFromWasm0(ptr4, len4);
    } finally {
        wasm.__wbindgen_free(deferred5_0, deferred5_1, 1);
    }
}

/**
 * Validate a JSON config string without creating an engine.
 * Returns an error message if the config is invalid, or empty string if valid.
 * @param {string} json
 * @returns {string}
 */
export function validate_config_json(json) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ptr0 = passStringToWasm0(json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.validate_config_json(ptr0, len0);
        deferred2_0 = ret[0];
        deferred2_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}

/**
 * Validate a TOML config string without creating an engine.
 * @param {string} toml
 * @returns {string}
 */
export function validate_config_toml(toml) {
    let deferred2_0;
    let deferred2_1;
    try {
        const ptr0 = passStringToWasm0(toml, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.validate_config_toml(ptr0, len0);
        deferred2_0 = ret[0];
        deferred2_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
    }
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_is_function_2f0fd7ceb86e64c5: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_object_5b22ff2418063a9c: function(arg0) {
            const val = arg0;
            const ret = typeof(val) === 'object' && val !== null;
            return ret;
        },
        __wbg___wbindgen_is_string_eddc07a3efad52e6: function(arg0) {
            const ret = typeof(arg0) === 'string';
            return ret;
        },
        __wbg___wbindgen_is_undefined_244a92c34d3b6ec0: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_throw_9c75d47bf9e7731e: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_call_a41d6421b30a32c5: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_crypto_38df2bab126b63dc: function(arg0) {
            const ret = arg0.crypto;
            return ret;
        },
        __wbg_debug_37240d2c1d0ce2bb: function(arg0) {
            console.debug(arg0);
        },
        __wbg_error_48655ee7e4756f8b: function(arg0) {
            console.error(arg0);
        },
        __wbg_getRandomValues_c44a50d8cfdaebeb: function() { return handleError(function (arg0, arg1) {
            arg0.getRandomValues(arg1);
        }, arguments); },
        __wbg_info_092aeeab8cd06a0b: function(arg0) {
            console.info(arg0);
        },
        __wbg_length_ba3c032602efe310: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_log_72d22df918dcc232: function(arg0) {
            console.log(arg0);
        },
        __wbg_msCrypto_bd5a034af96bcba6: function(arg0) {
            const ret = arg0.msCrypto;
            return ret;
        },
        __wbg_new_from_slice_0f99167330d1143b: function(arg0, arg1) {
            const ret = new Float32Array(getArrayF32FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_with_length_9011f5da794bf5d9: function(arg0) {
            const ret = new Uint8Array(arg0 >>> 0);
            return ret;
        },
        __wbg_new_with_length_d360e1480e55002f: function(arg0) {
            const ret = new Float32Array(arg0 >>> 0);
            return ret;
        },
        __wbg_node_84ea875411254db1: function(arg0) {
            const ret = arg0.node;
            return ret;
        },
        __wbg_process_44c7a14e11e9f69e: function(arg0) {
            const ret = arg0.process;
            return ret;
        },
        __wbg_prototypesetcall_fd4050e806e1d519: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_randomFillSync_6c25eac9869eb53c: function() { return handleError(function (arg0, arg1) {
            arg0.randomFillSync(arg1);
        }, arguments); },
        __wbg_require_b4edbdcf3e2a1ef0: function() { return handleError(function () {
            const ret = module.require;
            return ret;
        }, arguments); },
        __wbg_static_accessor_GLOBAL_THIS_1c7f1bd6c6941fdb: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_e039bc914f83e74e: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_8bf8c48c28420ad5: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_6aeee9b51652ee0f: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_subarray_fbe3cef290e1fa43: function(arg0, arg1, arg2) {
            const ret = arg0.subarray(arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_versions_276b2795b1c6a219: function(arg0) {
            const ret = arg0.versions;
            return ret;
        },
        __wbg_warn_1f9b94806da61fbb: function(arg0) {
            console.warn(arg0);
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./hyburn_bg.js": import0,
    };
}

const EngineInfoFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_engineinfo_free(ptr, 1));
const WebEngineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_webengine_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('hyburn_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
