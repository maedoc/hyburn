import { test, expect } from "@playwright/test";

const G2DO_CONFIG = JSON.stringify({
  sim_length: 100,
  dt: 0.1,
  integrator: "euler",
  nsig: 0,
  network: {
    subnetworks: [{
      model: "Generic2dOscillator",
      nnodes: 2,
      nmodes: 1,
      params: [1, 0, -2, -10, 0, 0.02, 3, 1, 0, 1, 1, 1],
      initial_state: [0, 0.5, 0, 0.5],
    }],
    projections: [],
  },
});

async function withWasm(page: any, fn: string, ...args: any[]) {
  return page.evaluate(async ([fnBody, fnArgs]) => {
    const mod = await import("./pkg/hyburn.js");
    await mod.default();
    const fn = new Function("mod", ...Object.keys(fnArgs), fnBody);
    return fn(mod, ...Object.values(fnArgs));
  }, [fn, args.reduce((o, v, i) => ({ ...o, [`a${i}`]: v }), {})]);
}

test.describe("hyburn WASM API", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/test-harness.html");
  });

  test("WASM module exports expected API", async ({ page }) => {
    const exports = await page.evaluate(async () => {
      const mod = await import("./pkg/hyburn.js");
      await mod.default();
      return Object.keys(mod).filter(k => k !== "default").sort();
    });
    expect(exports).toContain("WebEngine");
    expect(exports).toContain("validate_config_json");
    expect(exports).toContain("list_presets");
    expect(exports).toContain("get_preset");
    expect(exports).toContain("init_logger");
  });

  test("WebEngine constructor accepts JSON string", async ({ page }) => {
    const result = await page.evaluate(async (cfg) => {
      const mod = await import("./pkg/hyburn.js");
      await mod.default();
      try {
        const engine = new mod.WebEngine(cfg);
        return { ok: true, nvar: engine.info().nvar, nnodes: engine.info().nnodes };
      } catch (e: any) {
        return { ok: false, error: String(e) };
      }
    }, G2DO_CONFIG);
    expect(result.ok).toBe(true);
    expect(result.nvar).toBe(2);
    expect(result.nnodes).toBe(2);
  });

  test("WebEngine.from_json does NOT exist (static method is from_toml)", async ({ page }) => {
    const result = await page.evaluate(async () => {
      const mod = await import("./pkg/hyburn.js");
      await mod.default();
      return {
        hasFromJson: typeof (mod.WebEngine as any).from_json,
        hasFromToml: typeof (mod.WebEngine as any).from_toml,
      };
    });
    expect(result.hasFromJson).toBe("undefined");
    expect(result.hasFromToml).toBe("function");
  });

  test("new WebEngine.from_json() throws TypeError (catches the original bug)", async ({ page }) => {
    const result = await page.evaluate(async (cfg) => {
      const mod = await import("./pkg/hyburn.js");
      await mod.default();
      try {
        const engine = new (mod.WebEngine as any).from_json(cfg);
        return { ok: true };
      } catch (e: any) {
        return { ok: false, error: String(e) };
      }
    }, G2DO_CONFIG);
    expect(result.ok).toBe(false);
    expect(result.error).toContain("is not a constructor");
  });

  test("WebEngine.from_toml creates engine from TOML", async ({ page }) => {
    const toml = `
sim_length = 100.0
dt = 0.1
integrator = "euler"
nsig = 0.0

[network]

[[network.subnetworks]]
model = "Generic2dOscillator"
nnodes = 2
nmodes = 1
params = [1.0, 0.0, -2.0, -10.0, 0.0, 0.02, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0]
initial_state = [0.0, 0.5, 0.0, 0.5]

[[network.projections]]
src = 0
tgt = 0
conn_type = "all_to_all"
coupling_fn = "Linear"
coupling_params = [0.01]
weights = 0.01
cvar_map = "0:0"
delays = []
`;
    const result = await page.evaluate(async (toml) => {
      const mod = await import("./pkg/hyburn.js");
      await mod.default();
      try {
        const engine = mod.WebEngine.from_toml(toml);
        return { ok: true, nvar: engine.info().nvar };
      } catch (e: any) {
        return { ok: false, error: String(e) };
      }
    }, toml);
    expect(result.ok).toBe(true);
    expect(result.nvar).toBe(2);
  });

  test("step() and step_n() advance simulation", async ({ page }) => {
    const result = await page.evaluate(async (cfg) => {
      const mod = await import("./pkg/hyburn.js");
      await mod.default();
      const engine = new mod.WebEngine(cfg);
      engine.step();
      const s1 = engine.current_step();
      engine.step_n(99);
      const s100 = engine.current_step();
      return { s1, s100, trajLen: engine.trajectory_len() };
    }, G2DO_CONFIG);
    expect(result.s1).toBe(1);
    expect(result.s100).toBe(100);
    expect(result.trajLen).toBeGreaterThan(0);
  });

  test("trajectory() returns Float32Array with correct size", async ({ page }) => {
    const result = await page.evaluate(async (cfg) => {
      const mod = await import("./pkg/hyburn.js");
      await mod.default();
      const engine = new mod.WebEngine(cfg);
      engine.step_n(10);
      const traj = engine.trajectory();
      return { length: traj.length, isFloat32: traj instanceof Float32Array };
    }, G2DO_CONFIG);
    expect(result.isFloat32).toBe(true);
    expect(result.length).toBe(40);
  });

  test("validate_config_json returns empty for valid config", async ({ page }) => {
    const err = await page.evaluate(async (cfg) => {
      const mod = await import("./pkg/hyburn.js");
      await mod.default();
      return mod.validate_config_json(cfg);
    }, G2DO_CONFIG);
    expect(err).toBe("");
  });

  test("validate_config_json returns error for invalid config", async ({ page }) => {
    const err = await page.evaluate(async () => {
      const mod = await import("./pkg/hyburn.js");
      await mod.default();
      return mod.validate_config_json(JSON.stringify({ sim_length: 100, dt: -1, network: { subnetworks: [{ model: "Bad", nnodes: 2, nmodes: 1, params: [1], initial_state: [0, 0] }], projections: [] } }));
    });
    expect(err).not.toBe("");
  });

  test("list_presets returns non-empty array", async ({ page }) => {
    const count = await page.evaluate(async () => {
      const mod = await import("./pkg/hyburn.js");
      await mod.default();
      return JSON.parse(mod.list_presets()).length;
    });
    expect(count).toBeGreaterThan(0);
  });

  test("all presets validate successfully", async ({ page }) => {
    const errors = await page.evaluate(async () => {
      const mod = await import("./pkg/hyburn.js");
      await mod.default();
      const presets = JSON.parse(mod.list_presets());
      const bad: string[] = [];
      for (const p of presets) {
        const cfg = mod.get_preset(p.id);
        const err = mod.validate_config_json(cfg);
        if (err !== "") bad.push(`${p.id}: ${err}`);
      }
      return bad;
    });
    expect(errors).toEqual([]);
  });

  test("each preset creates a working engine", async ({ page }) => {
    const errors = await page.evaluate(async () => {
      const mod = await import("./pkg/hyburn.js");
      await mod.default();
      const presets = JSON.parse(mod.list_presets());
      const bad: string[] = [];
      for (const p of presets) {
        try {
          const cfg = mod.get_preset(p.id);
          const engine = new mod.WebEngine(cfg);
          engine.step_n(5);
          if (engine.current_step() !== 5) bad.push(`${p.id}: step count wrong`);
        } catch (e: any) {
          bad.push(`${p.id}: ${String(e)}`);
        }
      }
      return bad;
    });
    expect(errors).toEqual([]);
  });

  test("invalid JSON in constructor throws, not crashes", async ({ page }) => {
    const result = await page.evaluate(async () => {
      const mod = await import("./pkg/hyburn.js");
      await mod.default();
      try {
        new mod.WebEngine("{ not json }");
        return { ok: true };
      } catch (e: any) {
        return { ok: false, error: String(e) };
      }
    });
    expect(result.ok).toBe(false);
  });

  test("unknown model in constructor throws", async ({ page }) => {
    const result = await page.evaluate(async () => {
      const mod = await import("./pkg/hyburn.js");
      await mod.default();
      try {
        new mod.WebEngine(JSON.stringify({
          sim_length: 100, dt: 0.1,
          network: { subnetworks: [{ model: "Nonexistent", nnodes: 2, nmodes: 1, params: [1], initial_state: [0, 0] }], projections: [] },
        }));
        return { ok: true };
      } catch (e: any) {
        return { ok: false, error: String(e) };
      }
    });
    expect(result.ok).toBe(false);
  });
});
