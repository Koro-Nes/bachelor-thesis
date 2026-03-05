use std::{collections::HashMap, ops::Add};

use tch::{Kind, Tensor};

pub trait Aggregator {
    fn aggregate(&mut self, self_model: &Tensor, neighbor_models: &[Tensor]) -> Tensor;
}

#[derive(Debug)]
struct DFedAvgMAggregator {
    beta: f64,
    mixing_matrix: Vec<f64>,
    neighbor_ids: Vec<usize>,
    momentum: Option<Tensor>,
}

impl DFedAvgMAggregator {
    pub fn new(beta: f64, mixing_matrix: Vec<f64>, neighbor_ids: Vec<usize>) -> Self {
        DFedAvgMAggregator {
            beta,
            mixing_matrix,
            neighbor_ids,
            momentum: None,
        }
    }

    pub fn update_neighbor_data(&mut self, (neighbor_ids, mixing_matrix): (Vec<usize>, Vec<f64>)) {
        self.neighbor_ids = neighbor_ids;
        self.mixing_matrix = mixing_matrix;
    }
}

impl Aggregator for DFedAvgMAggregator {
    fn aggregate(&mut self, self_model: &Tensor, neighbor_models: &[Tensor]) -> Tensor {
        if neighbor_models.len() != neighbor_models.len() {
            panic!(
                "do not forget to update neighbor matrix (and keep the order of the neighbors the same)"
            )
        }

        let mut agg_model = self_model.shallow_clone() * self.mixing_matrix[0];

        for (neighbor_model, i) in neighbor_models.iter().zip(self.neighbor_ids.clone()) {
            agg_model += neighbor_model * self.mixing_matrix[i];
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
struct TrimmedMeanAggregator {
    beta: f64,
}

impl TrimmedMeanAggregator {
    pub fn new(beta: f64) -> Self {
        TrimmedMeanAggregator { beta }
    }
}

impl Aggregator for TrimmedMeanAggregator {
    fn aggregate(&mut self, self_model: &Tensor, neighbor_models: &[Tensor]) -> Tensor {
        let mut all_models = vec![self_model.shallow_clone()];
        all_models.extend(
            neighbor_models
                .iter()
                .map(|m| m.shallow_clone())
                .collect::<Vec<Tensor>>(),
        );

        let stacked = Tensor::stack(&all_models, 0);
        let sorted = stacked.sort(0, false).0;

        let num_models = sorted.size()[0];
        let k = (self.beta * num_models as f64).floor() as i64;
        let dims: &[i64] = &[0];
        if 2 * k >= num_models {
            stacked.mean_dim(dims, false, Kind::Float)
        } else {
            let trimmed = sorted.narrow(0, k, num_models - 2 * k);
            trimmed.mean_dim(dims, false, Kind::Float)
        }
    }
}

#[derive(Debug)]
pub struct BalanceAggregator {
    alpha: f64,
    gamma: f64,
    kappa: f64,
    t: u64,
    T: u64,
}

impl BalanceAggregator {
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
    fn aggregate(&mut self, self_model: &Tensor, neighbor_models: &[Tensor]) -> Tensor {
        let mut s = Vec::with_capacity(neighbor_models.len() + 1);
        let lambda_t = self.t / self.T;

        for i in 0..neighbor_models.len() {
            let lhs = (self_model.add(neighbor_models[i].shallow_clone())).norm();

            let rhs = (self.gamma * (-self.kappa * lambda_t as f64)) * self_model.norm();

            if lhs.double_value(&[]) <= rhs.double_value(&[]) {
                s.push(i);
            }
        }
        let mut sum = Tensor::new();
        for i in &s {
            sum += neighbor_models[i.to_owned()].shallow_clone();
        }

        self.alpha * self_model + (1.0 - self.alpha) * (1.0 / s.len() as f64) * sum
    }
}
