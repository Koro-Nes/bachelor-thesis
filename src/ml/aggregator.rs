use std::fmt::Display;

use tch::Tensor;

use crate::network::mixing_matrix::MixingMatrix;

#[derive(Debug, Clone, Copy)]
pub enum AggregatorType {
    DFedAvgM,
    ClippedMean,
    Balance,
}

impl Display for AggregatorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregatorType::Balance => f.write_str("Balance"),
            AggregatorType::DFedAvgM => f.write_str("DFedAvgM"),
            AggregatorType::ClippedMean => f.write_str("ClippedMean"),
        }
    }
}

pub trait Aggregator {
    /// returns the resulting model parameters and if the model has converged
    /// `neighbor_models` are expected to be the models received via the communication layer.
    fn aggregate(
        &mut self,
        self_model: &Tensor,
        neighbor_models: &[(u32, Tensor)],
    ) -> AggregationResult;
}

#[derive(Debug)]
pub struct AggregationResult {
    pub model: Tensor,
    pub used_neighbor_ids: Vec<u32>,
}

#[derive(Debug)]
pub struct DFedAvgMAggregator {
    beta: f64,
    mixing_matrix: MixingMatrix,
    momentum: Option<Tensor>,
}

impl DFedAvgMAggregator {
    pub fn new(beta: f64, mixing_matrix: MixingMatrix) -> Self {
        DFedAvgMAggregator {
            beta,
            mixing_matrix,
            momentum: None,
        }
    }
}

impl Aggregator for DFedAvgMAggregator {
    fn aggregate(&mut self, self_model: &Tensor, neighbors: &[(u32, Tensor)]) -> AggregationResult {
        let mut w_neighbor_sum = 0.0;
        for (id, _) in neighbors.iter() {
            w_neighbor_sum += self.mixing_matrix.get_by_id(id);
        }

        let w_self = 1.0 - w_neighbor_sum;

        let mut agg_model = self_model.shallow_clone() * w_self;

        for (id, neighbor_model) in neighbors.iter() {
            let w_neighbor = self.mixing_matrix.get_by_id(id);
            agg_model += neighbor_model * w_neighbor;
        }

        let update = &agg_model - self_model;

        let m = match &self.momentum {
            Some(prev_m) => prev_m.shallow_clone() * self.beta + &update,
            None => update.shallow_clone(),
        };
        self.momentum = Some(m.shallow_clone());

        AggregationResult {
            model: self_model + &m,
            used_neighbor_ids: neighbors.iter().map(|(id, _)| *id).collect(),
        }
    }
}

#[derive(Debug)]
pub struct ClippedMeanAggregator {
    /// Slack applied to the median update norm for clipping. Should be >= 0.
    beta: f64,
}

impl ClippedMeanAggregator {
    pub fn new(beta: f64) -> Self {
        ClippedMeanAggregator { beta }
    }
}

impl Aggregator for ClippedMeanAggregator {
    fn aggregate(
        &mut self,
        self_model: &Tensor,
        neighbor_models: &[(u32, Tensor)],
    ) -> AggregationResult {
        let mut updates = Vec::with_capacity(neighbor_models.len());
        let mut update_norms = Vec::with_capacity(neighbor_models.len() + 1);

        for (_, neighbor_model) in neighbor_models.iter() {
            let update = neighbor_model - self_model;
            let norm = update.norm().double_value(&[]);
            updates.push((update, norm));
            update_norms.push(norm);
        }

        // Include self update (=0) to stabilize the median with tiny neighbor counts.
        update_norms.push(0.0);

        let median_norm = median(&mut update_norms);
        let clip_norm = (1.0 + self.beta).max(0.0) * median_norm;

        let mut sum = Tensor::zeros_like(self_model);
        let eps = 1e-12_f64;

        for (update, norm) in updates.iter() {
            let scale = if *norm > clip_norm && clip_norm > 0.0 {
                clip_norm / (norm + eps)
            } else {
                1.0
            };
            sum += update * scale;
        }

        let denom = (neighbor_models.len() + 1) as f64;
        AggregationResult {
            model: self_model + (sum / denom),
            used_neighbor_ids: neighbor_models.iter().map(|(id, _)| *id).collect(),
        }
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) * 0.5
    }
}

#[derive(Debug)]
#[allow(non_snake_case)]
pub struct BalanceAggregator {
    alpha: f64,
    gamma: f64,
    kappa: f64,
    t: u64,
    T: u64,
}

impl BalanceAggregator {
    #[allow(non_snake_case)]
    pub fn new(alpha: f64, gamma: f64, kappa: f64, T: u64) -> Self {
        Self {
            alpha,
            gamma,
            kappa,
            t: 0,
            T,
        }
    }
}

impl Aggregator for BalanceAggregator {
    fn aggregate(
        &mut self,
        self_model: &Tensor,
        neighbor_models: &[(u32, Tensor)],
    ) -> AggregationResult {
        let lambda_t = self.t as f64 / self.T as f64;
        let rhs =
            (self.gamma * (-self.kappa * lambda_t).exp()) * self_model.norm().double_value(&[]);

        let mut used_neighbor_ids = Vec::with_capacity(neighbor_models.len());
        let mut accepted_count = 0_usize;
        let mut sum = Tensor::zeros_like(self_model);
        for (neighbor_id, neighbor_model) in neighbor_models.iter() {
            let lhs = (self_model - neighbor_model).norm().double_value(&[]);
            if lhs <= rhs {
                accepted_count += 1;
                used_neighbor_ids.push(*neighbor_id);
                sum += neighbor_model;
            }
        }

        let final_model = if accepted_count == 0 {
            self_model.shallow_clone()
        } else {
            self.alpha * self_model + (1.0 - self.alpha) * (sum / accepted_count as f64)
        };

        self.t += 1;
        AggregationResult {
            model: final_model,
            used_neighbor_ids,
        }
    }
}
