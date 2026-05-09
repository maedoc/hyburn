pub mod config;
pub mod crosscoder;
pub mod crosscoder_cohort;
pub mod crosscoder_pipeline;
pub mod crosscoder_train;
pub mod crosscoder_validate;
pub mod dataset;
pub mod diagnostics;
pub mod features;
pub mod made;
pub mod maf;
pub mod priors;
pub mod train;

pub use config::MafConfig;
pub use crosscoder::{CrossCoder, CrossCoderConfig, CrossCoderView, load_crosscoder, CROSSCODER_CKPT_EXT};
pub use crosscoder_cohort::{encode_cohort, fit_mvn_over_latents, MvnPrior};
#[cfg(not(target_arch = "wasm32"))]
pub use crosscoder_cohort::load_cohort_from_npy;
pub use crosscoder_pipeline::{
    build_sim_config_with_sc, generate_synthetic_sc_matrices,
    run_crosscoder_simulation_pipeline,
};
pub use crosscoder_train::train_crosscoder;
pub use dataset::SbiDataset;
pub use diagnostics::SbiDiagnostics;
pub use features::{
    apply_normalization, extract_features, extract_features_with, normalize_features,
    FeatureDomain, FeatureSet, parse_feature_set,
};
pub use made::MADE;
pub use maf::MAF;
pub use priors::{ParamPrior, PriorConfig, PriorDistribution};
pub use train::{infer_maf, infer_maf_to_file, train_maf, train_maf_with_data, train_maf_with_data_and_log, train_maf_with_output};
