//! Engine construction — structs, enums, constructors (new, from_config).
//!
//! Defines EngineModel, Projection, ProgressReporter, HybridEngine,
//! and their construction-logic methods.

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use crate::io::ndarray_to_tensor;
#[cfg(not(target_arch = "wasm32"))]
use crate::io::read_npy_f32;
use crate::config::{InitialStateConfig, SimConfig};
use crate::model::{NeuralMassModel, g2do::Generic2dOscillator, mpr::MontbrioPazoRoxin, rww::ReducedWongWang, kuramoto_model::Kuramoto, jansen_rit::JansenRit, wilson_cowan::WilsonCowan, linear::Linear, sup_hopf::SupHopf, hopfield::Hopfield, coombes_byrne2d::CoombesByrne2D, coombes_byrne::CoombesByrne, gast_schmidt_knosche_sd::GastSchmidtKnoscheSD, gast_schmidt_knosche_sf::GastSchmidtKnoscheSF, larter_breakspear::LarterBreakspear, epileptor2d::Epileptor2D, epileptor::Epileptor, rww_exc_inh::ReducedWongWangExcInh, deco_balanced_exc_inh::DecoBalancedExcInh, epileptor_codim3::EpileptorCodim3, epileptor_codim3_slowmod::EpileptorCodim3SlowMod, epileptor_rs::EpileptorRestingState, zetterberg_jansen::ZetterbergJansen, reduced_fhn::ReducedSetFitzHughNagumo, reduced_hr::ReducedSetHindmarshRose, dumont_gutkin::DumontGutkin, zerlaut_first::ZerlautAdaptationFirstOrder, zerlaut_second::ZerlautAdaptationSecondOrder, kionex::KIonEx};
use crate::error::{Result, SimulationError};

use super::coupling::CouplingFnConfig;
use super::stimulus::StimulusApplier;
pub use super::integrator::{IntegratorKind, euler_step, euler_stochastic_step, heun_step, heun_stochastic_step, rk4_step, rk4_stochastic_step};
use super::subnetwork::Subnetwork;

/// Supported engine models (dispatches to concrete `NeuralMassModel` impls).
#[derive(Clone)]
pub enum EngineModel<B: Backend> {
    G2do { params: Vec<f32> },
    Mpr { params: Vec<f32> },
    Rww { params: Vec<f32> },
    Kuramoto { params: Vec<f32> },
    JansenRit { params: Vec<f32> },
    WilsonCowan { params: Vec<f32> },
    Linear { params: Vec<f32> },
    SupHopf { params: Vec<f32> },
    Hopfield { params: Vec<f32> },
    CoombesByrne2D { params: Vec<f32> },
    CoombesByrne { params: Vec<f32> },
    GastSD { params: Vec<f32> },
    GastSF { params: Vec<f32> },
    LarterBreakspear { params: Vec<f32> },
    Epileptor2D { params: Vec<f32> },
    Epileptor { params: Vec<f32> },
    RwwExcInh { params: Vec<f32> },
    DecoBalancedExcInh { params: Vec<f32> },
    EpileptorCodim3 { params: Vec<f32> },
    EpileptorCodim3SlowMod { params: Vec<f32> },
    EpileptorRS { params: Vec<f32> },
    ZetterbergJansen { params: Vec<f32> },
    ReducedFHN { params: Vec<f32> },
    ReducedHR { params: Vec<f32> },
    DumontGutkin { params: Vec<f32> },
    ZerlautFirst { params: Vec<f32> },
    ZerlautSecond { params: Vec<f32> },
    KIonEx { params: Vec<f32> },
    #[doc(hidden)]
    #[allow(dead_code)]
    _Phantom(std::marker::PhantomData<B>),
}

impl<B: Backend> EngineModel<B> {
    pub fn from_config(model_name: &str, params: Vec<f32>) -> Result<Self> {
        match model_name {
            "Generic2dOscillator" => Ok(EngineModel::G2do { params }),
            "MontbrioPazoRoxin" => Ok(EngineModel::Mpr { params }),
            "ReducedWongWang" => Ok(EngineModel::Rww { params }),
            "Kuramoto" => Ok(EngineModel::Kuramoto { params }),
            "JansenRit" => Ok(EngineModel::JansenRit { params }),
            "WilsonCowan" => Ok(EngineModel::WilsonCowan { params }),
            "Linear" => Ok(EngineModel::Linear { params }),
            "SupHopf" => Ok(EngineModel::SupHopf { params }),
            "Hopfield" => Ok(EngineModel::Hopfield { params }),
            "CoombesByrne2D" => Ok(EngineModel::CoombesByrne2D { params }),
            "CoombesByrne" => Ok(EngineModel::CoombesByrne { params }),
            "GastSchmidtKnoscheSD" => Ok(EngineModel::GastSD { params }),
            "GastSchmidtKnoscheSF" => Ok(EngineModel::GastSF { params }),
            "LarterBreakspear" => Ok(EngineModel::LarterBreakspear { params }),
            "Epileptor2D" => Ok(EngineModel::Epileptor2D { params }),
            "Epileptor" => Ok(EngineModel::Epileptor { params }),
            "ReducedWongWangExcInh" => Ok(EngineModel::RwwExcInh { params }),
            "DecoBalancedExcInh" => Ok(EngineModel::DecoBalancedExcInh { params }),
            "EpileptorCodim3" => Ok(EngineModel::EpileptorCodim3 { params }),
            "EpileptorCodim3SlowMod" => Ok(EngineModel::EpileptorCodim3SlowMod { params }),
            "EpileptorRestingState" => Ok(EngineModel::EpileptorRS { params }),
            "ZetterbergJansen" => Ok(EngineModel::ZetterbergJansen { params }),
            "ReducedSetFitzHughNagumo" => Ok(EngineModel::ReducedFHN { params }),
            "ReducedSetHindmarshRose" => Ok(EngineModel::ReducedHR { params }),
            "DumontGutkin" => Ok(EngineModel::DumontGutkin { params }),
            "ZerlautAdaptationFirstOrder" => Ok(EngineModel::ZerlautFirst { params }),
            "ZerlautAdaptationSecondOrder" => Ok(EngineModel::ZerlautSecond { params }),
            "KIonEx" => Ok(EngineModel::KIonEx { params }),
            _ => Err(SimulationError::InvalidConfig(format!("Unknown model: {}", model_name))),
        }
    }

    pub fn nvar(&self) -> usize {
        match self {
            EngineModel::G2do { .. } => <Generic2dOscillator as NeuralMassModel<B>>::NVAR,
            EngineModel::Mpr { .. } => <MontbrioPazoRoxin as NeuralMassModel<B>>::NVAR,
            EngineModel::Rww { .. } => <ReducedWongWang as NeuralMassModel<B>>::NVAR,
            EngineModel::Kuramoto { .. } => <Kuramoto as NeuralMassModel<B>>::NVAR,
            EngineModel::JansenRit { .. } => <JansenRit as NeuralMassModel<B>>::NVAR,
            EngineModel::WilsonCowan { .. } => <WilsonCowan as NeuralMassModel<B>>::NVAR,
            EngineModel::Linear { .. } => <Linear as NeuralMassModel<B>>::NVAR,
            EngineModel::SupHopf { .. } => <SupHopf as NeuralMassModel<B>>::NVAR,
            EngineModel::Hopfield { .. } => <Hopfield as NeuralMassModel<B>>::NVAR,
            EngineModel::CoombesByrne2D { .. } => <CoombesByrne2D as NeuralMassModel<B>>::NVAR,
            EngineModel::CoombesByrne { .. } => <CoombesByrne as NeuralMassModel<B>>::NVAR,
            EngineModel::GastSD { .. } => <GastSchmidtKnoscheSD as NeuralMassModel<B>>::NVAR,
            EngineModel::GastSF { .. } => <GastSchmidtKnoscheSF as NeuralMassModel<B>>::NVAR,
            EngineModel::LarterBreakspear { .. } => <LarterBreakspear as NeuralMassModel<B>>::NVAR,
            EngineModel::Epileptor2D { .. } => <Epileptor2D as NeuralMassModel<B>>::NVAR,
            EngineModel::Epileptor { .. } => <Epileptor as NeuralMassModel<B>>::NVAR,
            EngineModel::RwwExcInh { .. } => <ReducedWongWangExcInh as NeuralMassModel<B>>::NVAR,
            EngineModel::DecoBalancedExcInh { .. } => <DecoBalancedExcInh as NeuralMassModel<B>>::NVAR,
            EngineModel::EpileptorCodim3 { .. } => <EpileptorCodim3 as NeuralMassModel<B>>::NVAR,
            EngineModel::EpileptorCodim3SlowMod { .. } => <EpileptorCodim3SlowMod as NeuralMassModel<B>>::NVAR,
            EngineModel::EpileptorRS { .. } => <EpileptorRestingState as NeuralMassModel<B>>::NVAR,
            EngineModel::ZetterbergJansen { .. } => <ZetterbergJansen as NeuralMassModel<B>>::NVAR,
            EngineModel::ReducedFHN { .. } => <ReducedSetFitzHughNagumo as NeuralMassModel<B>>::NVAR,
            EngineModel::ReducedHR { .. } => <ReducedSetHindmarshRose as NeuralMassModel<B>>::NVAR,
            EngineModel::DumontGutkin { .. } => <DumontGutkin as NeuralMassModel<B>>::NVAR,
            EngineModel::ZerlautFirst { .. } => <ZerlautAdaptationFirstOrder as NeuralMassModel<B>>::NVAR,
            EngineModel::ZerlautSecond { .. } => <ZerlautAdaptationSecondOrder as NeuralMassModel<B>>::NVAR,
            EngineModel::KIonEx { .. } => <KIonEx as NeuralMassModel<B>>::NVAR,
            _ => unreachable!(),
        }
    }

    pub fn ncvar(&self) -> usize {
        match self {
            EngineModel::G2do { .. } => <Generic2dOscillator as NeuralMassModel<B>>::NCVAR,
            EngineModel::Mpr { .. } => <MontbrioPazoRoxin as NeuralMassModel<B>>::NCVAR,
            EngineModel::Rww { .. } => <ReducedWongWang as NeuralMassModel<B>>::NCVAR,
            EngineModel::Kuramoto { .. } => <Kuramoto as NeuralMassModel<B>>::NCVAR,
            EngineModel::JansenRit { .. } => <JansenRit as NeuralMassModel<B>>::NCVAR,
            EngineModel::WilsonCowan { .. } => <WilsonCowan as NeuralMassModel<B>>::NCVAR,
            EngineModel::Linear { .. } => <Linear as NeuralMassModel<B>>::NCVAR,
            EngineModel::SupHopf { .. } => <SupHopf as NeuralMassModel<B>>::NCVAR,
            EngineModel::Hopfield { .. } => <Hopfield as NeuralMassModel<B>>::NCVAR,
            EngineModel::CoombesByrne2D { .. } => <CoombesByrne2D as NeuralMassModel<B>>::NCVAR,
            EngineModel::CoombesByrne { .. } => <CoombesByrne as NeuralMassModel<B>>::NCVAR,
            EngineModel::GastSD { .. } => <GastSchmidtKnoscheSD as NeuralMassModel<B>>::NCVAR,
            EngineModel::GastSF { .. } => <GastSchmidtKnoscheSF as NeuralMassModel<B>>::NCVAR,
            EngineModel::LarterBreakspear { .. } => <LarterBreakspear as NeuralMassModel<B>>::NCVAR,
            EngineModel::Epileptor2D { .. } => <Epileptor2D as NeuralMassModel<B>>::NCVAR,
            EngineModel::Epileptor { .. } => <Epileptor as NeuralMassModel<B>>::NCVAR,
            EngineModel::RwwExcInh { .. } => <ReducedWongWangExcInh as NeuralMassModel<B>>::NCVAR,
            EngineModel::DecoBalancedExcInh { .. } => <DecoBalancedExcInh as NeuralMassModel<B>>::NCVAR,
            EngineModel::EpileptorCodim3 { .. } => <EpileptorCodim3 as NeuralMassModel<B>>::NCVAR,
            EngineModel::EpileptorCodim3SlowMod { .. } => <EpileptorCodim3SlowMod as NeuralMassModel<B>>::NCVAR,
            EngineModel::EpileptorRS { .. } => <EpileptorRestingState as NeuralMassModel<B>>::NCVAR,
            EngineModel::ZetterbergJansen { .. } => <ZetterbergJansen as NeuralMassModel<B>>::NCVAR,
            EngineModel::ReducedFHN { .. } => <ReducedSetFitzHughNagumo as NeuralMassModel<B>>::NCVAR,
            EngineModel::ReducedHR { .. } => <ReducedSetHindmarshRose as NeuralMassModel<B>>::NCVAR,
            EngineModel::DumontGutkin { .. } => <DumontGutkin as NeuralMassModel<B>>::NCVAR,
            EngineModel::ZerlautFirst { .. } => <ZerlautAdaptationFirstOrder as NeuralMassModel<B>>::NCVAR,
            EngineModel::ZerlautSecond { .. } => <ZerlautAdaptationSecondOrder as NeuralMassModel<B>>::NCVAR,
            EngineModel::KIonEx { .. } => <KIonEx as NeuralMassModel<B>>::NCVAR,
            _ => unreachable!(),
        }
    }

    pub fn dfun(&self, state: Tensor<B, 2>, coupling: Tensor<B, 2>) -> Tensor<B, 2> {
        use crate::engine::batch_engine::dfun::{dfun_batch, model_param_slice};

        let params = model_param_slice(self);
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = dfun_batch(self, state3, coupling3, &params, None);
        result3.squeeze::<2>(0)
    }

    pub fn clamp(&self, state: &mut Tensor<B, 2>) {
        use crate::engine::batch_engine::dfun::clamp_batch;

        let mut state3 = state.clone().unsqueeze_dim::<3>(0);
        clamp_batch(self, &mut state3);
        *state = state3.squeeze::<2>(0);
    }
}

/// A coupling projection between two subnetworks.
pub struct Projection<B: Backend> {
    pub src: usize,
    pub tgt: usize,
    pub weights: Tensor<B, 2>,
    pub delays: Vec<u32>,
    pub coupling_cfg: CouplingFnConfig,
    pub csr_data: Option<Vec<f32>>,
    pub csr_indices: Option<Vec<usize>>,
    pub csr_indptr: Option<Vec<usize>>,
    pub csr_idelays: Option<Vec<u32>>,
    pub is_sparse: bool,
    /// Cvar mapping: pairs of (src_cvar_idx, tgt_cvar_idx).
    /// Parsed from config "src:tgt" format.
    pub cvar_map: Vec<(usize, usize)>,
}

/// Parse a cvar_map string like "0:0" or "0:0,1:1" into pairs of (src, tgt) indices.
/// Default is "0:0" meaning source cvar 0 → target cvar 0.
pub fn parse_cvar_map(s: &str) -> Vec<(usize, usize)> {
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            let parts: Vec<&str> = p.trim().split(':').collect();
            if parts.len() != 2 {
                (0, 0) // fallback
            } else {
                (parts[0].parse().unwrap_or(0), parts[1].parse().unwrap_or(0))
            }
        })
        .collect()
}

/// Progress reporter that logs simulation progress at regular intervals.
pub struct ProgressReporter {
    pub interval: usize,
}

impl ProgressReporter {
    pub fn new(interval: usize) -> Self {
        assert!(interval > 0, "progress interval must be > 0");
        Self { interval }
    }

    /// Log progress if the current step matches the interval.
    pub fn report(&self, step: usize) {
        if step == 0 || step.is_multiple_of(self.interval) {
            log::info!("Simulation step {}", step);
        }
    }
}

/// The main hybrid simulation engine.
pub struct HybridEngine<B: Backend> {
    pub subnetworks: Vec<Subnetwork<B>>,
    pub states: Vec<Tensor<B, 3>>,
    pub histories: Vec<Tensor<B, 4>>,
    pub step: usize,
    pub dt: f64,
    pub integrator: IntegratorKind,
    pub device: B::Device,
    pub projections: Vec<Projection<B>>,
    /// Stimulus appliers per configured stimulus.
    pub stimuli: Vec<StimulusApplier>,
    /// Flat state trajectory recorded every step (all subnetworks concatenated).
    pub trajectory: Vec<f32>,
    /// Noise amplitude per variable for stochastic integration.
    pub nsig: Vec<f32>,
    /// Optional progress reporter.
    pub progress: Option<ProgressReporter>,
    /// Active BOLD monitors.
    pub bold_monitors: Vec<crate::engine::bold_monitor::BoldMonitor>,
    /// GPU-side accumulators for BOLD neural input: one per monitor.
    /// Each is a 1D tensor of shape [nnodes] that accumulates var0 mean-over-modes.
    pub bold_accumulators: Vec<Option<Tensor<B, 1>>>,
    /// Per-monitor accumulation counters: how many neural steps have been
    /// accumulated for each BOLD monitor. Indices match bold_monitors.
    pub bold_accumulator_counts: Vec<usize>,
    /// Active sensor projection monitors (EEG/MEG/iEEG).
    pub sensor_monitors: Vec<crate::engine::monitor::SensorProjectionMonitor>,
    /// Target subnetwork index for each sensor monitor.
    pub sensor_monitor_targets: Vec<usize>,
    /// Active spatial average monitors.
    pub spatial_monitors: Vec<crate::engine::monitor::SpatialAverageMonitor>,
    /// Target subnetwork index for each spatial average monitor.
    pub spatial_monitor_targets: Vec<usize>,
}

/// Checkpoint constants.
pub const CKPT_MAGIC: &[u8; 8] = b"HYBURNCK";
pub const CKPT_VERSION: u64 = 2;

impl<B: Backend> HybridEngine<B> {
    /// Build a single-subnetwork engine directly (backward-compatible with Phase 0).
    pub fn new(
        initial_state: Tensor<B, 3>,
        model: EngineModel<B>,
        integrator: IntegratorKind,
        dt: f64,
        horizon: usize,
        device: B::Device,
    ) -> Self {
        let shape = initial_state.shape();
        let dims = shape.dims;
        let nvar = dims[0];
        let nnodes = dims[1];
        let nmodes = dims[2];

        let model_name = match &model {
            EngineModel::G2do { .. } => "Generic2dOscillator".to_string(),
            EngineModel::Mpr { .. } => "MontbrioPazoRoxin".to_string(),
            EngineModel::Rww { .. } => "ReducedWongWang".to_string(),
            EngineModel::Kuramoto { .. } => "Kuramoto".to_string(),
            EngineModel::JansenRit { .. } => "JansenRit".to_string(),
            EngineModel::WilsonCowan { .. } => "WilsonCowan".to_string(),
            EngineModel::Linear { .. } => "Linear".to_string(),
            EngineModel::SupHopf { .. } => "SupHopf".to_string(),
            EngineModel::Hopfield { .. } => "Hopfield".to_string(),
            EngineModel::CoombesByrne2D { .. } => "CoombesByrne2D".to_string(),
            EngineModel::CoombesByrne { .. } => "CoombesByrne".to_string(),
            EngineModel::GastSD { .. } => "GastSchmidtKnoscheSD".to_string(),
            EngineModel::GastSF { .. } => "GastSchmidtKnoscheSF".to_string(),
            EngineModel::LarterBreakspear { .. } => "LarterBreakspear".to_string(),
            EngineModel::Epileptor2D { .. } => "Epileptor2D".to_string(),
            EngineModel::Epileptor { .. } => "Epileptor".to_string(),
            EngineModel::RwwExcInh { .. } => "ReducedWongWangExcInh".to_string(),
            EngineModel::DecoBalancedExcInh { .. } => "DecoBalancedExcInh".to_string(),
            EngineModel::EpileptorCodim3 { .. } => "EpileptorCodim3".to_string(),
            EngineModel::EpileptorCodim3SlowMod { .. } => "EpileptorCodim3SlowMod".to_string(),
            EngineModel::EpileptorRS { .. } => "EpileptorRestingState".to_string(),
            EngineModel::ZetterbergJansen { .. } => "ZetterbergJansen".to_string(),
            EngineModel::ReducedFHN { .. } => "ReducedSetFitzHughNagumo".to_string(),
            EngineModel::ReducedHR { .. } => "ReducedSetHindmarshRose".to_string(),
            EngineModel::DumontGutkin { .. } => "DumontGutkin".to_string(),
            EngineModel::ZerlautFirst { .. } => "ZerlautAdaptationFirstOrder".to_string(),
            EngineModel::ZerlautSecond { .. } => "ZerlautAdaptationSecondOrder".to_string(),
            EngineModel::KIonEx { .. } => "KIonEx".to_string(),
            _ => unreachable!(),
        };
        let params = match &model {
            EngineModel::G2do { params } => params.clone(),
            EngineModel::Mpr { params } => params.clone(),
            EngineModel::Rww { params } => params.clone(),
            EngineModel::Kuramoto { params } => params.clone(),
            EngineModel::JansenRit { params } => params.clone(),
            EngineModel::WilsonCowan { params } => params.clone(),
            EngineModel::Linear { params } => params.clone(),
            EngineModel::SupHopf { params } => params.clone(),
            EngineModel::Hopfield { params } => params.clone(),
            EngineModel::CoombesByrne2D { params } => params.clone(),
            EngineModel::CoombesByrne { params } => params.clone(),
            EngineModel::GastSD { params } => params.clone(),
            EngineModel::GastSF { params } => params.clone(),
            EngineModel::LarterBreakspear { params } => params.clone(),
            EngineModel::Epileptor2D { params } => params.clone(),
            EngineModel::Epileptor { params } => params.clone(),
            EngineModel::RwwExcInh { params } => params.clone(),
            EngineModel::DecoBalancedExcInh { params } => params.clone(),
            EngineModel::EpileptorCodim3 { params } => params.clone(),
            EngineModel::EpileptorCodim3SlowMod { params } => params.clone(),
            EngineModel::EpileptorRS { params } => params.clone(),
            EngineModel::ZetterbergJansen { params } => params.clone(),
            EngineModel::ReducedFHN { params } => params.clone(),
            EngineModel::ReducedHR { params } => params.clone(),
            EngineModel::DumontGutkin { params } => params.clone(),
            EngineModel::ZerlautFirst { params } => params.clone(),
            EngineModel::ZerlautSecond { params } => params.clone(),
            EngineModel::KIonEx { params } => params.clone(),
            _ => unreachable!(),
        };

        let sub = Subnetwork {
            model_name,
            params,
            nnodes,
            nmodes,
            nvar,
            ncvar: model.ncvar(),
            state_offset: 0,
            state_len: nvar * nnodes * nmodes,
            _phantom: std::marker::PhantomData,
        };

        let history = if horizon > 0 {
            let state_slice = initial_state.clone().unsqueeze_dim::<4>(3);
            let h = if horizon > 1 {
                let rest = Tensor::<B, 4>::zeros([nvar, nnodes, nmodes, horizon - 1], &device);
                Tensor::cat(vec![state_slice, rest], 3)
            } else {
                state_slice
            };
            vec![h]
        } else {
            vec![]
        };

        Self {
            subnetworks: vec![sub],
            states: vec![initial_state],
            histories: history,
            step: 0,
            dt,
            integrator,
            device,
            projections: vec![],
            stimuli: vec![],
            trajectory: Vec::new(),
            nsig: vec![0.0],
progress: None,
            bold_monitors: vec![],
            bold_accumulators: vec![],
            bold_accumulator_counts: vec![],
            sensor_monitors: vec![],
            sensor_monitor_targets: vec![],
            spatial_monitors: vec![],
            spatial_monitor_targets: vec![],
        }
    }

    /// Build an engine from a `SimConfig`.
    #[track_caller]
    pub fn from_config(cfg: SimConfig, device: B::Device) -> Result<Self> {
        let mut subnetworks = Vec::new();
        let mut states = Vec::new();

        for sub_cfg in &cfg.network.subnetworks {
            let sub = Subnetwork::<B>::new(
                sub_cfg.model.clone(),
                sub_cfg.params.clone(),
                sub_cfg.nnodes,
                sub_cfg.nmodes,
                0,
            )?;

            let state = match &sub_cfg.initial_state {
                InitialStateConfig::Inline(vals) => {
                    Tensor::<B, 3>::from_floats(
                        TensorData::new::<f32, Vec<usize>>(
                            vals.clone(),
                            vec![sub.nvar, sub_cfg.nnodes, sub_cfg.nmodes],
                        ),
                        &device,
                    )
                }
                InitialStateConfig::NpyPath(path) => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let (data, shape) = read_npy_f32(path)?;
                        if shape.len() == 3 {
                            ndarray_to_tensor::<B, 3>(data, shape, &device)
                        } else if shape.len() == 2 && sub_cfg.nmodes == 1 {
                            ndarray_to_tensor::<B, 3>(data, vec![shape[0], shape[1], 1], &device)
                        } else {
                            return Err(SimulationError::InvalidState(format!(
                                "Initial state NPY has {} dims, expected 3 (or 2 when nmodes=1)",
                                shape.len()
                            )));
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        let _ = path;
                        return Err(SimulationError::InvalidState(
                            "NPY file loading not supported in WASM. Use Inline or Memory initial_state.".into()
                        ));
                    }
                }
                InitialStateConfig::Memory { data, shape } => {
                    ndarray_to_tensor::<B, 3>(data.clone(), shape.clone(), &device)
                }
            };

            subnetworks.push(sub);
            states.push(state);
        }

        // Determine history horizon from projections (max delay + 1).
        let max_delay = cfg.network.projections.iter()
            .map(|p| p.delays.iter().copied().max().unwrap_or(0) as usize)
            .max()
            .unwrap_or(0);
        let horizon = if max_delay == 0 { 1 } else { max_delay + 1 };

        let mut histories = Vec::new();
        for (i, sub) in subnetworks.iter().enumerate() {
            let state_slice = states[i].clone().unsqueeze_dim::<4>(3);
            let h = if horizon > 1 {
                let rest = Tensor::<B, 4>::zeros([sub.nvar, sub.nnodes, sub.nmodes, horizon - 1], &device);
                Tensor::cat(vec![state_slice, rest], 3)
            } else {
                state_slice
            };
            histories.push(h);
        }

        // Parse projections into engine-side structures.
        let mut projections = Vec::new();
        for proj_cfg in &cfg.network.projections {
            let mut is_sparse = false;
            let mut csr_data: Option<Vec<f32>> = None;
            let mut csr_indices: Option<Vec<usize>> = None;
            let mut csr_indptr: Option<Vec<usize>> = None;
            let mut csr_idelays: Option<Vec<u32>> = None;

            let weights_tensor = match &proj_cfg.weights {
                crate::config::WeightsConfig::Dense(mat) => {
                    if mat.is_empty() {
                        return Err(SimulationError::InvalidConfig("Empty weight matrix".into()));
                    }
                    let nrows = mat.len();
                    let ncols = mat[0].len();
                    let flat: Vec<f32> = mat.iter().flatten().copied().collect();
                    Tensor::<B, 2>::from_floats(
                        TensorData::new::<f32, Vec<usize>>(flat, vec![nrows, ncols]),
                        &device,
                    )
                }
                crate::config::WeightsConfig::Scalar(s) => {
                    let src_nnodes = subnetworks[proj_cfg.src].nnodes;
                    let tgt_nnodes = subnetworks[proj_cfg.tgt].nnodes;
                    if src_nnodes != tgt_nnodes {
                        return Err(SimulationError::InvalidConfig(
                            "Scalar weights require src and tgt nnodes to match".into()
                        ));
                    }
                    let mut flat = vec![0.0f32; src_nnodes * tgt_nnodes];
                    for i in 0..src_nnodes {
                        flat[i * tgt_nnodes + i] = *s;
                    }
                    Tensor::<B, 2>::from_floats(
                        TensorData::new::<f32, Vec<usize>>(flat, vec![tgt_nnodes, src_nnodes]),
                        &device,
                    )
                }
                crate::config::WeightsConfig::Csr { data, indices, indptr } => {
                    is_sparse = true;
                    let src_nnodes = subnetworks[proj_cfg.src].nnodes;
                    let tgt_nnodes = subnetworks[proj_cfg.tgt].nnodes;
                    csr_data = Some(data.clone());
                    csr_indices = Some(indices.iter().map(|&x| x as usize).collect());
                    csr_indptr = Some(indptr.iter().map(|&x| x as usize).collect());
                    if !proj_cfg.delays.is_empty() {
                        csr_idelays = Some(proj_cfg.delays.clone());
                    }
                    // Dummy weights matrix so existing dense code doesn't crash.
                    let flat = vec![0.0f32; tgt_nnodes * src_nnodes];
                    Tensor::<B, 2>::from_floats(
                        TensorData::new::<f32, Vec<usize>>(flat, vec![tgt_nnodes, src_nnodes]),
                        &device,
                    )
                }
            };

            let coupling_cfg = match proj_cfg.coupling_fn.as_str() {
                "Linear" => {
                    let a = proj_cfg.coupling_params.first().copied().unwrap_or(1.0);
                    CouplingFnConfig::Linear { a }
                }
                "Sigmoidal" => {
                    let cmax = proj_cfg.coupling_params.first().copied().unwrap_or(1.0);
                    let midpoint = proj_cfg.coupling_params.get(1).copied().unwrap_or(0.0);
                    let steepness = proj_cfg.coupling_params.get(2).copied().unwrap_or(1.0);
                    CouplingFnConfig::Sigmoidal { cmax, midpoint, steepness }
                }
                "Difference" => {
                    let a = proj_cfg.coupling_params.first().copied().unwrap_or(1.0);
                    CouplingFnConfig::Difference { a }
                }
                "Kuramoto" => {
                    let a = proj_cfg.coupling_params.first().copied().unwrap_or(1.0);
                    CouplingFnConfig::Kuramoto { a }
                }
                _ => {
                    return Err(SimulationError::InvalidConfig(
                        format!("Unknown coupling function: {}", proj_cfg.coupling_fn)
                    ));
                }
            };

            let cvar_map_parsed = parse_cvar_map(&proj_cfg.cvar_map);
            // Validate cvar_map against model ncvar
            let src_ncvar = subnetworks[proj_cfg.src].ncvar;
            let tgt_ncvar = subnetworks[proj_cfg.tgt].ncvar;
            // Validate coupling function ncvar requirements
            let min_ncvar = coupling_cfg.min_src_ncvar();
            if src_ncvar < min_ncvar {
                return Err(SimulationError::InvalidConfig(
                    format!(
                        "Projection {}: {} coupling requires ncvar >= {} but source model has ncvar {}",
                        projections.len(), proj_cfg.coupling_fn, min_ncvar, src_ncvar
                    )
                ));
            }
            for &(s, t) in &cvar_map_parsed {
                if s >= src_ncvar {
                    return Err(SimulationError::InvalidConfig(
                        format!("Projection {} cvar_map src index {} >= src_ncvar {}", projections.len(), s, src_ncvar)
                    ));
                }
                if t >= tgt_ncvar {
                    return Err(SimulationError::InvalidConfig(
                        format!("Projection {} cvar_map tgt index {} >= tgt_ncvar {}", projections.len(), t, tgt_ncvar)
                    ));
                }
            }

            projections.push(Projection {
                src: proj_cfg.src,
                tgt: proj_cfg.tgt,
                weights: weights_tensor,
                delays: proj_cfg.delays.clone(),
                coupling_cfg,
                csr_data,
                csr_indices,
                csr_indptr,
                csr_idelays,
                is_sparse,
                cvar_map: cvar_map_parsed,
            });
        }

        // Parse BOLD monitors from config.
        let mut bold_monitors = Vec::new();
        let mut sensor_monitors = Vec::new();
        let mut sensor_monitor_targets = Vec::new();
        let mut spatial_monitors = Vec::new();
        let mut spatial_monitor_targets = Vec::new();

        for mon_cfg in &cfg.monitors {
            let mon_type = mon_cfg.monitor_type.to_ascii_lowercase();
            if mon_type == "bold" {
                let target = 0usize;
                if target >= subnetworks.len() {
                    log::warn!("BOLD monitor target {} out of range ({} subnets); skipping", target, subnetworks.len());
                    continue;
                }
                let bold_period = mon_cfg.bold_period.unwrap_or_else(|| {
                    let period_ms = mon_cfg.period.unwrap_or(2000.0);
                    (period_ms / cfg.dt).max(1.0).round() as usize
                });
                let tr = mon_cfg.tr.unwrap_or(2.0);
                let nnodes = subnetworks.get(target).map(|s| s.nnodes).unwrap_or(0);
                if nnodes == 0 {
                    continue;
                }
                let bm = crate::engine::bold_monitor::BoldMonitor::new(
                    target,
                    nnodes,
                    bold_period,
                    tr,
                    cfg.dt,
                    None,
                );
                bold_monitors.push(bm);
            } else if mon_type == "eeg" || mon_type == "meg" || mon_type == "ieeg" {
                let target = 0usize;
                if target >= subnetworks.len() {
                    log::warn!("Sensor monitor target {} out of range; skipping", target);
                    continue;
                }
                let sub = &subnetworks[target];
                let nvar = sub.nvar;
                let nnodes = sub.nnodes;
                let nmodes = sub.nmodes;

                let gain = if let Some(ref g) = mon_cfg.gain {
                    g.clone()
                } else if let Some(ref path) = mon_cfg.gain_path {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        match crate::io::read_npy_f32(path) {
                            Ok((data, shape)) => {
                                if shape.len() != 2 {
                                    log::warn!("Gain matrix NPY must be 2D, got {}D; skipping monitor", shape.len());
                                    continue;
                                }
                                let n_sensors = shape[0];
                                let n_regions = shape[1];
                                let mut gain = vec![vec![0.0f32; n_regions]; n_sensors];
                                for (i, row) in gain.iter_mut().enumerate() {
                                    for (j, val) in row.iter_mut().enumerate() {
                                        *val = data[i * n_regions + j];
                                    }
                                }
                                gain
                            }
                            Err(e) => {
                                log::warn!("Failed to read gain matrix from {}: {:?}; skipping monitor", path, e);
                                continue;
                            }
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        log::warn!("gain_path not supported in WASM; skipping sensor monitor");
                        continue;
                    }
                } else {
                    log::warn!("Sensor projection monitor requires gain or gain_path; skipping");
                    continue;
                };

                let voi = mon_cfg.voi.clone().unwrap_or_else(|| vec![0]);
                let period_steps = mon_cfg.period
                    .map(|p_ms| (p_ms / cfg.dt).max(1.0).round() as usize)
                    .unwrap_or(1);

                let sm = crate::engine::monitor::SensorProjectionMonitor::new(
                    gain, voi, period_steps, nvar, nnodes, nmodes,
                );
                sensor_monitors.push(sm);
                sensor_monitor_targets.push(target);
            } else if mon_type == "spatialaverage" {
                let target = 0usize;
                if target >= subnetworks.len() {
                    log::warn!("Spatial average monitor target {} out of range; skipping", target);
                    continue;
                }
                let sub = &subnetworks[target];
                let nvar = sub.nvar;
                let nnodes = sub.nnodes;
                let nmodes = sub.nmodes;

                let mask = mon_cfg.spatial_mask.clone().unwrap_or_else(|| vec![1.0; nnodes]);
                if mask.len() != nnodes {
                    log::warn!("Spatial mask length {} != nnodes {}; skipping monitor", mask.len(), nnodes);
                    continue;
                }

                let period_steps = mon_cfg.period
                    .map(|p_ms| (p_ms / cfg.dt).max(1.0).round() as usize)
                    .unwrap_or(1);

                let sm = crate::engine::monitor::SpatialAverageMonitor::new(
                    mask, period_steps, nvar, nnodes, nmodes,
                );
                spatial_monitors.push(sm);
                spatial_monitor_targets.push(target);
            }
        }

        // Resolve nsig to per-variable vec using first subnetwork's nvar
        let first_nvar = subnetworks.first().map(|s| s.nvar).unwrap_or(1);
        let nsig_vec = cfg.nsig.to_vec(first_nvar);

        Ok(Self {
            subnetworks,
            states,
            histories,
            step: 0,
            dt: cfg.dt,
            integrator: cfg.integrator,
            device,
            projections,
            stimuli: cfg.stimuli.iter()
                .map(StimulusApplier::from_config)
                .collect::<Result<Vec<_>>>()?,
            trajectory: Vec::new(),
            nsig: nsig_vec,
            progress: None,
            bold_accumulators: vec![],
            bold_accumulator_counts: vec![0; bold_monitors.len()],
            bold_monitors,
            sensor_monitors,
            sensor_monitor_targets,
            spatial_monitors,
            spatial_monitor_targets,
        })
    }
}
