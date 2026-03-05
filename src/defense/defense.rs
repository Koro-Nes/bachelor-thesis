use tch::{Kind, Tensor};

use crate::{config::config::CONFIG, network::reputation::ReputationTable};

pub trait DefenseMechanism {
    fn filter(
        &mut self,
        self_model: &Tensor,
        neighbor_models: &[(u32, Tensor)],
        reputation_table: &mut ReputationTable,
        round: usize,
    ) -> Vec<(u32, Tensor)>;
}

#[derive(Debug)]
pub struct NoDefense {

}

impl DefenseMechanism for NoDefense {
    fn filter(
        &mut self,
        _self_model: &Tensor,
        neighbor_models: &[(u32, Tensor)],
        _reputation_table: &mut ReputationTable,
        _round: usize,
    ) -> Vec<(u32, Tensor)> {
        let mut res = Vec::with_capacity(neighbor_models.len());
        for (idx, model) in neighbor_models {
            res.push((idx.to_owned(), model.shallow_clone()))
        }
        res
    }
}

#[derive(Debug)]
pub struct Reputation {
    id: u32,
}

impl Reputation {
    pub fn new(id: u32, neighbors: Vec<u32>, rep: &mut ReputationTable) -> Self {
        for n in neighbors {
            rep.put((id, n), 1.0);
        }
        Reputation {
            id,
        }
    }
}

impl DefenseMechanism for Reputation {
    fn filter(
        &mut self,
        self_model: &Tensor,
        neighbor_models: &[(u32, Tensor)],
        reputation_table: &mut ReputationTable,
        _round: usize,
    ) -> Vec<(u32, Tensor)> {

        let mut res = Vec::with_capacity(neighbor_models.len());

        let alpha = CONFIG.network.reputation_weight_alpha as f64;
        let beta = CONFIG.network.reputation_weight_beta as f64;
        let gamma = CONFIG.network.reputation_weight_gamma as f64;

        for (outer_idx, model) in neighbor_models {
            let score_direct = alpha * reputation_table.get(&(self.id, outer_idx.to_owned())) + (1.0 - alpha) * (cos_sim(self_model, model) / 2.0);

            let mut sum_scores_weighted = 0.0;
            let mut sum_scores = 0.0;
            for (inner_idx, _model) in neighbor_models {
                if !inner_idx != *outer_idx {
                    sum_scores_weighted += reputation_table.get(&(self.id, *inner_idx)) * reputation_table.get(&(*inner_idx, *outer_idx));
                    sum_scores += reputation_table.get(&(self.id, *inner_idx));
                } 
            }

            let score_indirect_full = sum_scores_weighted / sum_scores;
            let score_indirect = if score_indirect_full.abs() <= score_direct.abs() {
                score_indirect_full
            } else {
                1.1 * score_direct
            };
            let score_combined = beta * score_direct + (1.0 - beta) * score_indirect;
            
            let score_decayed = (1.0 - gamma) * score_combined + gamma * 0.5;
            reputation_table.put((self.id, *outer_idx), score_decayed);

            if (score_decayed >= CONFIG.network.reputation_threshold as f64) {
                res.push((*outer_idx, model.shallow_clone()));
            }
        }
        res
    }
}

fn cos_sim(a: &Tensor, b: &Tensor) -> f64 {
    a.dot(b).double_value(&[0i64]) / (mag(a) * mag(b))
}

fn mag(a: &Tensor) -> f64 {
    a.sum(Kind::Float).sqrt().double_value(&[0i64])
}