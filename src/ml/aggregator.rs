use std::{fmt::Display, ops::Add};

use tch::{Kind, Tensor};

use crate::network::mixing_matrix::MixingMatrix;

#[derive(Debug, Clone, Copy)]
pub enum AggregatorType {
    DFedAvgM,
    TrimmedMean,
    Balance,
}

impl Display for AggregatorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregatorType::Balance => f.write_str("Balance"),
            AggregatorType::DFedAvgM => f.write_str("DFedAvgM"),
            AggregatorType::TrimmedMean => f.write_str("TrimmedMean"),
        }
    }
}

pub trait Aggregator {
    /// returns the resulting model parameters and if the model has converged
    fn aggregate(
        &mut self,
        self_model: &Tensor,
        neighbor_models: &[(u32, Tensor)],
    ) -> Tensor;
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
    fn aggregate(&mut self, self_model: &Tensor, neighbors: &[(u32, Tensor)]) -> Tensor {
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

        self_model + &m
    }
}

#[derive(Debug)]
pub struct TrimmedMeanAggregator {
    beta: f64,
    history: Vec<f64>,
    prev_result: Option<Tensor>,
}

impl TrimmedMeanAggregator {
    pub fn new(beta: f64) -> Self {
        TrimmedMeanAggregator {
            beta,
            history: Vec::new(),
            prev_result: None,
        }
    }
}

impl Aggregator for TrimmedMeanAggregator {
    fn aggregate(
        &mut self,
        self_model: &Tensor,
        neighbor_models: &[(u32, Tensor)],
    ) -> Tensor {
        let mut all_models = vec![self_model.shallow_clone()];
        all_models.extend(
            neighbor_models
                .iter()
                .map(|m| m.1.shallow_clone())
                .collect::<Vec<Tensor>>(),
        );

        let stacked = Tensor::stack(&all_models, 0);
        let sorted = stacked.sort(0, false).0;

        let num_models = sorted.size()[0];
        let k = (self.beta * num_models as f64).floor() as i64;
        let dims: &[i64] = &[0];

        let result = if 2 * k >= num_models {
            stacked.mean_dim(dims, false, Kind::Float)
        } else {
            let trimmed = sorted.narrow(0, k, num_models - 2 * k);
            trimmed.mean_dim(dims, false, Kind::Float)
        };
        let current_drift = if let Some(prev) = &self.prev_result {
            (&result - prev).abs().mean(Kind::Float).double_value(&[])
        } else {
            0.0
        };

        self.prev_result = Some(result.shallow_clone());

        self.history.push(current_drift);
        if self.history.len() > 5 {
            self.history.remove(0);
        }

        result
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
    ) -> Tensor {
        let mut s = Vec::with_capacity(neighbor_models.len());
        let lambda_t = self.t as f64 / self.T as f64;

        for i in 0..neighbor_models.len() {
            let lhs = (self_model - &neighbor_models[i].1).norm();

            let rhs = (self.gamma * (-self.kappa * lambda_t).exp()) * self_model.norm();

            if lhs.double_value(&[]) <= rhs.double_value(&[]) {
                s.push(i);
            }
        }
        let final_model = if s.is_empty() {
            self_model.shallow_clone()
        } else {
            let mut sum = Tensor::zeros_like(self_model);
            for i in &s {
                sum = sum.add(&neighbor_models[*i].1);
            }
            let s_len = s.len() as f64;
            self.alpha * self_model + (1.0 - self.alpha) * (sum / s_len)
        };

        self.t += 1;
        final_model
    }
}
