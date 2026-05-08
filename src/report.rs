//! SBI report generation: self-contained HTML with embedded SVG plots.
//!
//! Generates a comprehensive diagnostic report for SBI pipeline runs,
//! including reproducibility metadata, training loss curves, posterior
//! validation plots, and z-score/shrinkage diagnostics.

use plotters::drawing::IntoDrawingArea;
use plotters::prelude::*;

use burn::backend::autodiff::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::tensor::{Tensor, TensorData};

use crate::engine::{EngineModel, HybridEngine, IntegratorKind};
use crate::model::g2do::g2do_default_params;
use crate::sbi::{extract_features, train_maf_with_data_and_log, MafConfig, SbiDiagnostics};
use crate::sbi::priors::ParamPrior;

type AD = Autodiff<NdArray<f32>>;
type B = NdArray<f32>;

/// Configuration for the SBI report pipeline.
pub struct ReportConfig {
    /// Path to TOML simulation config (for display)
    pub config_path: String,
    /// Compute backend used
    pub backend: String,
    /// Number of sweep points
    pub n_sweep: usize,
    /// Number of simulation steps per sweep point
    pub n_steps: usize,
    /// Number of nodes
    pub nnodes: usize,
    /// Number of posterior samples per test point
    pub n_post_samples: usize,
    /// MAF training epochs
    pub n_epochs: usize,
    /// MAF training batch size
    pub batch_size: usize,
    /// Output HTML path
    pub output_path: String,
    /// Parameter index to sweep (default: 1 = I_ext)
    pub param_idx: usize,
    /// Prior bounds per parameter.
    pub priors: Vec<ParamPrior>,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            config_path: String::new(),
            backend: "ndarray".to_string(),
            n_sweep: 20,
            n_steps: 500,
            nnodes: 2,
            n_post_samples: 100,
            n_epochs: 200,
            batch_size: 5,
            output_path: "sbi-report.html".to_string(),
            param_idx: 1,
            priors: vec![ParamPrior::new("I_ext", -0.5, 0.5)],
        }
    }
}

/// Results from the SBI report pipeline.
pub struct ReportData {
    /// Command line invocation
    pub command_line: String,
    /// Working directory
    pub working_dir: String,
    /// Date/time stamp
    pub timestamp: String,
    /// Backend info
    pub rust_info: String,
    /// Sweep parameter values
    pub sweep_values: Vec<f32>,
    /// Training loss history (epoch, loss)
    pub loss_history: Vec<(usize, f32)>,
    /// SBI diagnostics
    pub diagnostics: SbiDiagnostics,
    /// Per-test-point posterior data: (true_param, posterior_mean, posterior_std)
    pub posterior_stats: Vec<(f32, f32, f32)>,
    /// Feature dimension
    pub feature_dim: usize,
    /// MAF config used
    pub maf_config: MafConfig,
    /// Config file contents (if available)
    pub config_contents: Option<String>,
}

/// Run the full SBI report pipeline and generate an HTML report.
pub fn generate_report(cfg: ReportConfig) -> anyhow::Result<String> {
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let device_ad: <AD as burn::tensor::backend::Backend>::Device = Default::default();

    let nmodes = 1;
    let nvar = 2;

    // 1. Parameter sweep
    let mut all_params: Vec<f32> = Vec::with_capacity(cfg.n_sweep);
    let mut all_features: Vec<f32> = Vec::new();
    let mut sweep_values: Vec<f32> = Vec::with_capacity(cfg.n_sweep);

    let (range_min, range_max) = {
        let sweep_prior = cfg.priors.get(cfg.param_idx)
            .or_else(|| cfg.priors.first())
            .cloned()
            .unwrap_or_else(|| ParamPrior::new("I_ext", -0.5, 0.5));
        (sweep_prior.min, sweep_prior.max)
    };
    for i in 0..cfg.n_sweep {
        let param_val = range_min
            + i as f32 * ((range_max - range_min) / (cfg.n_sweep - 1).max(1) as f32);

        let mut params = g2do_default_params();
        params[cfg.param_idx] = param_val;

        let initial_data = vec![0.0f32; nvar * cfg.nnodes * nmodes];
        let state = Tensor::<B, 3>::from_data(
            TensorData::new::<f32, Vec<usize>>(initial_data, vec![nvar, cfg.nnodes, nmodes]),
            &device,
        );

        let model = EngineModel::<B>::G2do { params };
        let mut engine =
            HybridEngine::new(state, model, IntegratorKind::Euler, 0.1, 1, device);
        engine.run(cfg.n_steps);

        let features = extract_features(
            &engine.trajectory,
            &[cfg.n_steps, nvar, cfg.nnodes, nmodes],
        );

        all_params.push(param_val);
        all_features.extend_from_slice(&features);
        sweep_values.push(param_val);

        log::info!(
            "Sweep point {}/{}: param[{}]={:.4}, features dim={}",
            i + 1,
            cfg.n_sweep,
            cfg.param_idx,
            param_val,
            features.len()
        );
    }

    let feature_dim = all_features.len() / cfg.n_sweep;

    // 2. Train MAF with loss logging
    let maf_config = MafConfig {
        param_dim: 1,
        feature_dim,
        hidden_units: 16,
        n_flows: 2,
        learning_rate: 1e-2,
        feature_set: "classic".to_string(),
    };

    let (maf, loss_history) = train_maf_with_data_and_log(
        &maf_config,
        all_params.clone(),
        all_features.clone(),
        cfg.n_epochs,
        cfg.batch_size,
    );

    // 3. Posterior inference and diagnostics
    let (prior_mean, prior_std) = {
        let sweep_prior = cfg.priors.get(cfg.param_idx)
            .or_else(|| cfg.priors.first())
            .cloned()
            .unwrap_or_else(|| ParamPrior::new("I_ext", -0.5, 0.5));
        (sweep_prior.mean(), sweep_prior.std())
    };

    let mut posterior_stats: Vec<(f32, f32, f32)> = Vec::with_capacity(cfg.n_sweep);
    let mut all_posterior_samples: Vec<f32> = Vec::new();
    let mut all_true_params: Vec<f32> = Vec::new();

    for (i, &true_param) in all_params.iter().enumerate() {
        let f_start = i * feature_dim;
        let features_slice = &all_features[f_start..f_start + feature_dim];

        let context = Tensor::<AD, 2>::from_data(
            TensorData::new::<f32, Vec<usize>>(features_slice.to_vec(), vec![1, feature_dim]),
            &device_ad,
        );

        let samples = maf.inverse_sample(context, cfg.n_post_samples);
        let data = samples.into_data();
        let slice = data.as_slice::<f32>().unwrap();

        let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
        let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / slice.len() as f32;
        let std = var.sqrt();

        posterior_stats.push((true_param, mean, std));
        all_posterior_samples.extend_from_slice(slice);
        all_true_params.push(true_param);
    }

    let diagnostics = SbiDiagnostics::from_samples(
        &all_posterior_samples,
        &all_true_params,
        &[prior_mean],
        &[prior_std],
        cfg.n_post_samples,
        1,
    );

    // 4. Try to read config file contents
    let config_contents = {
        #[cfg(not(target_arch = "wasm32"))]
        { std::fs::read_to_string(&cfg.config_path).ok() }
        #[cfg(target_arch = "wasm32")]
        { None }
    };

    // 5. Build the HTML report
    let report_data = ReportData {
        command_line: {
            #[cfg(not(target_arch = "wasm32"))]
            { std::env::args().collect::<Vec<_>>().join(" ") }
            #[cfg(target_arch = "wasm32")]
            { "(wasm)".to_string() }
        },
        working_dir: {
            #[cfg(not(target_arch = "wasm32"))]
            { std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| "<unknown>".to_string()) }
            #[cfg(target_arch = "wasm32")]
            { "(browser)".to_string() }
        },
        timestamp: chrono_now(),
        rust_info: format!("Rust (Burn {} backend)", cfg.backend),
        sweep_values,
        loss_history,
        diagnostics,
        posterior_stats,
        feature_dim,
        maf_config: maf_config.clone(),
        config_contents,
    };

    let html = render_report(&report_data, &cfg);

    // 6. Write to file
    #[cfg(not(target_arch = "wasm32"))]
    std::fs::write(&cfg.output_path, &html)?;

    log::info!("SBI report written to {}", cfg.output_path);
    Ok(html)
}

fn chrono_now() -> String {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let output = std::process::Command::new("date")
            .arg("-u")
            .arg("+%Y-%m-%d %H:%M:%S UTC")
            .output();
        match output {
            Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
            Err(_) => format!("{:?}", std::time::SystemTime::now()),
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        // In WASM, just use SystemTime
        format!("{:?}", std::time::SystemTime::now())
    }
}

fn render_report(data: &ReportData, cfg: &ReportConfig) -> String {
    let mut html = String::with_capacity(50_000);

    html.push_str(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SBI Diagnostic Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }
  h1 { color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 8px; }
  h2 { color: #16213e; margin-top: 32px; }
  section { background: white; border-radius: 8px; padding: 16px 20px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  pre { background: #1a1a2e; color: #e0e0e0; padding: 12px 16px; border-radius: 6px; overflow-x: auto; font-size: 13px; line-height: 1.5; }
  .metric { display: inline-block; background: #f0f4f8; border-radius: 6px; padding: 8px 16px; margin: 4px; font-size: 14px; }
  .metric-good { background: #d4edda; color: #155724; }
  .metric-bad { background: #f8d7da; color: #721c24; }
  .metric-moderate { background: #fff3cd; color: #856404; }
  .plot-container { text-align: center; margin: 16px 0; }
  .plot-container svg { max-width: 100%; height: auto; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0; }
  th, td { border: 1px solid #ddd; padding: 6px 12px; text-align: left; font-size: 14px; }
  th { background: #16213e; color: white; }
  tr:nth-child(even) { background: #f8f9fa; }
</style>
</head>
<body>
<h1>&#128300; SBI Diagnostic Report</h1>
"#,
    );

    // ---- Reproducibility section ----
    html.push_str(
        r#"<section id="repro">
<h2>&#128203; Reproducibility</h2>
<table>
"#,
    );
    html.push_str(&format!(
        "<tr><td><strong>Command</strong></td><td><code>{}</code></td></tr>\n",
        html_escape(&data.command_line)
    ));
    html.push_str(&format!(
        "<tr><td><strong>Working dir</strong></td><td><code>{}</code></td></tr>\n",
        html_escape(&data.working_dir)
    ));
    html.push_str(&format!(
        "<tr><td><strong>Date</strong></td><td>{}</td></tr>\n",
        data.timestamp
    ));
    html.push_str(&format!(
        "<tr><td><strong>Backend</strong></td><td>{}</td></tr>\n",
        data.rust_info
    ));
    html.push_str(&format!(
        "<tr><td><strong>Sweep points</strong></td><td>{}</td></tr>\n",
        data.sweep_values.len()
    ));
    html.push_str(&format!(
        "<tr><td><strong>Simulation steps</strong></td><td>{}</td></tr>\n",
        cfg.n_steps
    ));
    html.push_str(&format!(
        "<tr><td><strong>Posterior samples</strong></td><td>{}</td></tr>\n",
        cfg.n_post_samples
    ));
    html.push_str(&format!(
        "<tr><td><strong>Feature dim</strong></td><td>{}</td></tr>\n",
        data.feature_dim
    ));
    html.push_str("</table>\n</section>\n");

    // ---- Config section ----
    if let Some(ref contents) = data.config_contents {
        html.push_str(
            r#"<section id="config">
<h2>&#9881;&#65039; Configuration</h2>
<pre>"#,
        );
        html.push_str(&html_escape(contents));
        html.push_str("</pre>\n</section>\n");
    }

    // ---- MAF config ----
    html.push_str(
        r#"<section id="maf-config">
<h2>&#129504; MAF Configuration</h2>
<table>
"#,
    );
    html.push_str(&format!(
        "<tr><td>param_dim</td><td>{}</td></tr>\n",
        data.maf_config.param_dim
    ));
    html.push_str(&format!(
        "<tr><td>feature_dim</td><td>{}</td></tr>\n",
        data.maf_config.feature_dim
    ));
    html.push_str(&format!(
        "<tr><td>hidden_units</td><td>{}</td></tr>\n",
        data.maf_config.hidden_units
    ));
    html.push_str(&format!(
        "<tr><td>n_flows</td><td>{}</td></tr>\n",
        data.maf_config.n_flows
    ));
    html.push_str(&format!(
        "<tr><td>learning_rate</td><td>{:.6}</td></tr>\n",
        data.maf_config.learning_rate
    ));
    html.push_str("</table>\n</section>\n");

    // ---- Training loss curve ----
    html.push_str(
        r#"<section id="training">
<h2>&#128201; Training Loss</h2>
<div class="plot-container">
"#,
    );
    html.push_str(&plot_loss_curve(&data.loss_history));
    html.push_str("\n</div>\n</section>\n");

    // ---- Diagnostics ----
    html.push_str(
        r#"<section id="diagnostics">
<h2>&#128202; Diagnostics</h2>
"#,
    );

    let diag = &data.diagnostics;

    let z_class = if diag.mean_z_score < 1.0 {
        "metric-good"
    } else if diag.mean_z_score < 2.0 {
        "metric-moderate"
    } else {
        "metric-bad"
    };
    let s_class = if diag.mean_shrinkage > 0.5 {
        "metric-good"
    } else if diag.mean_shrinkage > 0.0 {
        "metric-moderate"
    } else {
        "metric-bad"
    };

    html.push_str(&format!(
        "<div class=\"metric {z_class}\"><strong>Mean Z-score:</strong> {:.4}</div>\n",
        diag.mean_z_score,
    ));
    html.push_str(&format!(
        "<div class=\"metric {s_class}\"><strong>Mean Shrinkage:</strong> {:.4}</div>\n",
        diag.mean_shrinkage,
    ));
    html.push_str(&format!(
        "<div class=\"metric {}\"><strong>Well-calibrated:</strong> {}</div>\n",
        if diag.is_well_calibrated() {
            "metric-good"
        } else {
            "metric-bad"
        },
        if diag.is_well_calibrated() {
            "YES &#x2713;"
        } else {
            "NO &#x2717;"
        },
    ));

    // Detailed table
    html.push_str("<table>\n<tr><th>Parameter</th><th>Z-score</th><th>Shrinkage</th><th>Post. Mean</th><th>Post. Std</th><th>Quality</th></tr>\n");
    for (d, z) in diag.z_scores.iter().enumerate() {
        let sh = diag.shrinkages[d];
        let quality = if sh > 0.5 {
            "excellent"
        } else if sh > 0.0 {
            "moderate"
        } else {
            "FAILED"
        };
        html.push_str(&format!(
            "<tr><td>&#952;[{}]</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{}</td></tr>\n",
            d, z, sh, diag.posterior_means[d], diag.posterior_stds[d], quality
        ));
    }
    html.push_str("</table>\n");

    // Z-score bar chart
    html.push_str("<div class=\"plot-container\">\n");
    html.push_str(&plot_z_scores(&diag.z_scores));
    html.push_str("\n</div>\n");

    // Shrinkage bar chart
    html.push_str("<div class=\"plot-container\">\n");
    html.push_str(&plot_shrinkage(&diag.shrinkages));
    html.push_str("\n</div>\n");

    html.push_str("</section>\n");

    // ---- Posterior validation scatter ----
    html.push_str(
        r#"<section id="validation">
<h2>&#127919; Posterior vs True Parameters</h2>
<div class="plot-container">
"#,
    );
    html.push_str(&plot_posterior_scatter(&data.posterior_stats));
    html.push_str("\n</div>\n</section>\n");

    // ---- Footer ----
    html.push_str(
        r#"<section id="footer">
<p><em>Generated by hyburn SBI report</em></p>
</section>

</body>
</html>"#,
    );

    html
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ---- SVG Plot Generation ----

fn plot_loss_curve(loss_history: &[(usize, f32)]) -> String {
    let mut svg = String::with_capacity(8_000);
    let (w, h) = (800u32, 400u32);
    let root = SVGBackend::with_string(&mut svg, (w, h)).into_drawing_area();

    if let Err(e) = draw_loss_curve(&root, loss_history) {
        drop(root);
        return format!(
            "<!-- Loss plot error: {} -->\n<pre>Error rendering loss curve: {}</pre>",
            e, e
        );
    }
    drop(root);
    svg
}

fn draw_loss_curve(
    root: &plotters::drawing::DrawingArea<SVGBackend<'_>, plotters::coord::Shift>,
    loss_history: &[(usize, f32)],
) -> Result<(), Box<dyn std::error::Error>> {
    if loss_history.is_empty() {
        root.draw(&Text::new(
            "No loss data",
            (100, 200),
            ("sans-serif", 20).into_font(),
        ))?;
        return Ok(());
    }

    let max_loss = loss_history
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_loss = loss_history
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::INFINITY, f32::min);
    let max_epoch = loss_history.last().map(|(e, _)| *e).unwrap_or(1);

    let mut chart = ChartBuilder::on(root)
        .caption("Training Loss", ("sans-serif", 20))
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0usize..max_epoch.max(1), min_loss..max_loss)?;

    chart.configure_mesh().x_desc("Epoch").y_desc("Loss (NLL)").draw()?;

    chart.draw_series(LineSeries::new(
        loss_history.iter().map(|&(e, l)| (e, l)),
        &BLUE,
    ))?;

    Ok(())
}

fn plot_z_scores(z_scores: &[f32]) -> String {
    let mut svg = String::with_capacity(6_000);
    let (w, h) = (800u32, 400u32);
    let root = SVGBackend::with_string(&mut svg, (w, h)).into_drawing_area();

    if let Err(e) = draw_bar_chart(
        &root,
        "Z-scores by Parameter",
        "Parameter",
        "Z-score",
        z_scores,
        2.0f32,
    ) {
        drop(root);
        return format!("<!-- Z-score plot error: {} -->", e);
    }
    drop(root);
    svg
}

fn plot_shrinkage(shrinkages: &[f32]) -> String {
    let mut svg = String::with_capacity(6_000);
    let (w, h) = (800u32, 400u32);
    let root = SVGBackend::with_string(&mut svg, (w, h)).into_drawing_area();

    if let Err(e) = draw_bar_chart(
        &root,
        "Shrinkage by Parameter",
        "Parameter",
        "Shrinkage",
        shrinkages,
        1.0f32,
    ) {
        drop(root);
        return format!("<!-- Shrinkage plot error: {} -->", e);
    }
    drop(root);
    svg
}

fn draw_bar_chart(
    root: &plotters::drawing::DrawingArea<SVGBackend<'_>, plotters::coord::Shift>,
    title: &str,
    x_label: &str,
    y_label: &str,
    values: &[f32],
    reference_line: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if values.is_empty() {
        root.draw(&Text::new(
            "No data",
            (100, 200),
            ("sans-serif", 20).into_font(),
        ))?;
        return Ok(());
    }

    let max_val = values
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(0.0)
        * 1.2;
    let min_val = values
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min)
        .min(0.0)
        * 1.2;
    let n = values.len();

    let mut chart = ChartBuilder::on(root)
        .caption(title, ("sans-serif", 20))
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0usize..n.max(1), min_val..max_val)?;

    chart.configure_mesh().x_desc(x_label).y_desc(y_label).draw()?;

    // Draw bars
    let bar_style = RGBColor(70, 130, 180);
    chart.draw_series(values.iter().enumerate().map(|(i, &v)| {
        let color = if v > reference_line { RED } else { bar_style };
        Rectangle::new([(i, 0.0f32), (i + 1, v)], color.filled())
    }))?;

    // Reference line
    chart.draw_series(LineSeries::new(
        (0..=n.max(1)).map(|x| (x, reference_line)),
        &RED,
    ))?;

    Ok(())
}

fn plot_posterior_scatter(stats: &[(f32, f32, f32)]) -> String {
    let mut svg = String::with_capacity(8_000);
    let (w, h) = (800u32, 500u32);
    let root = SVGBackend::with_string(&mut svg, (w, h)).into_drawing_area();

    if let Err(e) = draw_posterior_scatter(&root, stats) {
        drop(root);
        return format!("<!-- Posterior scatter error: {} -->", e);
    }
    drop(root);
    svg
}

fn draw_posterior_scatter(
    root: &plotters::drawing::DrawingArea<SVGBackend<'_>, plotters::coord::Shift>,
    stats: &[(f32, f32, f32)],
) -> Result<(), Box<dyn std::error::Error>> {
    if stats.is_empty() {
        root.draw(&Text::new(
            "No data",
            (100, 200),
            ("sans-serif", 20).into_font(),
        ))?;
        return Ok(());
    }

    let all_vals: Vec<f32> = stats
        .iter()
        .flat_map(|(t, m, s)| [*t, *m + s, *m - s].into_iter())
        .collect();
    let x_min = all_vals.iter().cloned().fold(f32::INFINITY, f32::min) - 0.1;
    let x_max = all_vals
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max)
        + 0.1;
    let y_min = x_min;
    let y_max = x_max;

    let mut chart = ChartBuilder::on(root)
        .caption("Posterior Mean vs True Parameter", ("sans-serif", 20))
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("True parameter")
        .y_desc("Posterior mean")
        .draw()?;

    // Identity line (perfect prediction)
    chart.draw_series(LineSeries::new(
        (0..=20).map(|i| {
            let v = x_min + (x_max - x_min) * i as f32 / 20.0;
            (v, v)
        }),
        &BLACK,
    ))?;

    // Data points
    chart.draw_series(stats.iter().map(|&(true_val, mean, _std)| {
        Circle::new((true_val, mean), 4, BLUE.filled())
    }))?;

    // Error bars (vertical lines from mean-std to mean+std)
    for &(true_val, mean, std) in stats {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(true_val, mean - std), (true_val, mean + std)],
            BLUE,
        )))?;
    }

    Ok(())
}