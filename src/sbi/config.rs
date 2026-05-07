use serde::{Deserialize, Serialize};
use crate::sbi::features::FeatureSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MafConfig {
    pub param_dim: usize,
    pub feature_dim: usize,
    pub hidden_units: usize,
    pub n_flows: usize,
    pub learning_rate: f64,
    /// Feature extraction method: "classic", "catch22", or "catch24"
    #[serde(default = "default_feature_set")]
    pub feature_set: String,
}

fn default_feature_set() -> String {
    "classic".to_string()
}

impl Default for MafConfig {
    fn default() -> Self {
        Self {
            param_dim: 2,
            feature_dim: 1,
            hidden_units: 64,
            n_flows: 4,
            learning_rate: 1e-3,
            feature_set: default_feature_set(),
        }
    }
}

impl MafConfig {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    /// Resolve the feature set string to a FeatureSet enum.
    ///
    /// Supports combined feature sets via the syntax:
    /// `combined:classic,fc,spectral`
    pub fn resolve_feature_set(&self) -> FeatureSet {
        let s = self.feature_set.trim();
        if let Some(rest) = s.strip_prefix("combined:") {
            let parts: Vec<FeatureSet> = rest
                .split(',')
                .map(|p| Self::parse_single_feature_set(p.trim()))
                .collect();
            if parts.is_empty() {
                FeatureSet::Classic
            } else {
                FeatureSet::Combined(parts)
            }
        } else {
            Self::parse_single_feature_set(s)
        }
    }

    fn parse_single_feature_set(s: &str) -> FeatureSet {
        match s {
            "catch22" => FeatureSet::Catch22,
            "catch24" => FeatureSet::Catch24,
            "fc" => FeatureSet::Fc,
            "spectral" => FeatureSet::Spectral,
            "temporalstat" | "temporal" => FeatureSet::TemporalStat,
            _ => FeatureSet::Classic,
        }
    }
}