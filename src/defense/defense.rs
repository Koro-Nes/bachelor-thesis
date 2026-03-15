use std::{cell::RefCell, collections::HashSet, fmt::Display, rc::Rc};

use tch::{Kind, Tensor};

use crate::{
    config::config::CONFIG,
    network::reputation::ReputationTable,
    node::stats::{NeighborScores, NodeStats, ReputationStats, RoundStats},
};

#[derive(Clone, Copy, Debug)]
pub enum DefenseType {
    NoDefense,
    Reputation,
}

impl Display for DefenseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DefenseType::NoDefense => f.write_str("NoD"),
            DefenseType::Reputation => f.write_str("Rep"),
        }
    }
}

pub trait DefenseMechanism {
    fn filter(
        &mut self,
        self_model: &Tensor,
        neighbor_models: &[(u32, Tensor)],
        reputation_table: &mut ReputationTable,
        round: usize,
        adversarial_set: Option<&HashSet<u32>>,
        stats: &mut NodeStats,
        round_stats: &mut RoundStats,
    ) -> Vec<(u32, Tensor)>;
}

#[derive(Debug)]
pub struct NoDefense {}

impl NoDefense {
    pub fn new() -> Self {
        Self {}
    }
}

impl DefenseMechanism for NoDefense {
    fn filter(
        &mut self,
        _self_model: &Tensor,
        neighbor_models: &[(u32, Tensor)],
        _reputation_table: &mut ReputationTable,
        round: usize,
        adversarial_set: Option<&HashSet<u32>>,
        stats: &mut NodeStats,
        _round_stats: &mut RoundStats,
    ) -> Vec<(u32, Tensor)> {
        let mut res = Vec::with_capacity(neighbor_models.len());
        for (idx, model) in neighbor_models {
            match adversarial_set {
                Some(m) => {
                    if m.contains(idx) {
                        stats.set_adversarial_include_round(round);
                    }
                }
                None => (),
            }
            res.push((idx.to_owned(), model.shallow_clone()))
        }
        res
    }
}

#[derive(Debug)]
pub struct Reputation {
    id: u32,
    colluding_set: Option<HashSet<u32>>,
}

impl Reputation {
    pub fn new(id: u32, neighbors: Vec<u32>, rep: Rc<RefCell<ReputationTable>>) -> Self {
        for n in neighbors {
            rep.borrow_mut().put((id, n), 1.0);
        }
        Reputation {
            id,
            colluding_set: None,
        }
    }

    pub fn set_colluding(&mut self, colluding_set: HashSet<u32>) {
        self.colluding_set = Some(colluding_set.clone());
    }
}

impl DefenseMechanism for Reputation {
    fn filter(
        &mut self,
        self_model: &Tensor,
        neighbor_models: &[(u32, Tensor)],
        reputation_table: &mut ReputationTable,
        round: usize,
        adversarial_set: Option<&HashSet<u32>>,
        stats: &mut NodeStats,
        round_stats: &mut RoundStats,
    ) -> Vec<(u32, Tensor)> {
        let mut res = Vec::with_capacity(neighbor_models.len());
        let mut reputation_stats = ReputationStats::default();
        let mut correctly_blocked = 0;
        let mut incorrectly_blocked = 0;
        let mut incorrectly_not_blocked = 0;

        let alpha = CONFIG.network.reputation_weight_alpha as f64;
        let beta = CONFIG.network.reputation_weight_beta as f64;
        let gamma = CONFIG.network.reputation_weight_gamma as f64;

        for (outer_idx, model) in neighbor_models {
            let is_adversarial = adversarial_set.map_or(false, |s| s.contains(outer_idx));

            // if colluding, only apply exponential decay
            if self.colluding_set.is_some() {
                let curr_score = reputation_table.get(&(self.id, *outer_idx));
                let score_decayed = (1.0 - gamma) * curr_score + gamma * 0.5;
                reputation_table.put((self.id, *outer_idx), score_decayed);

                reputation_stats
                    .neighbor_scores
                    .push(NeighborScores {
                        neighbor_id: *outer_idx,
                        score: score_decayed,
                        adversarial: is_adversarial,
                    });
                continue;
            }

            let score_direct = alpha * reputation_table.get(&(self.id, outer_idx.to_owned()))
                + (1.0 - alpha) * (cos_sim(self_model, model) / 2.0);
            let mut sum_scores_weighted = 0.0;
            let mut sum_scores = 0.0;
            for (inner_idx, _model) in neighbor_models {
                if *inner_idx != *outer_idx {
                    let neigh_score = reputation_table.get_optional(&(*inner_idx, *outer_idx));
                    if neigh_score.is_none() {
                        continue;
                    }
                    sum_scores_weighted +=
                        neigh_score.unwrap() * reputation_table.get(&(self.id, *outer_idx));
                    sum_scores += reputation_table.get(&(self.id, *inner_idx));
                }
            }

            let score_indirect_full = if sum_scores_weighted == 0.0 {
                score_direct
            } else {
                sum_scores_weighted / sum_scores
            };
            let score_indirect = if score_indirect_full.abs() <= score_direct.abs() {
                score_indirect_full
            } else {
                1.1 * score_direct
            };
            let score_combined = beta * score_direct + (1.0 - beta) * score_indirect;

            let score_decayed = (1.0 - gamma) * score_combined + gamma * 0.5;
            reputation_table.put((self.id, *outer_idx), score_decayed);

            reputation_stats
                .neighbor_scores
                .push(NeighborScores {
                    neighbor_id: *outer_idx,
                    score: score_decayed,
                    adversarial: is_adversarial,
                });

            if score_decayed >= CONFIG.network.reputation_threshold as f64 {
                if is_adversarial {
                    incorrectly_not_blocked += 1;
                    stats.set_adversarial_include_round(round);
                }
                res.push((*outer_idx, model.shallow_clone()));
            } else {
                if is_adversarial {
                    correctly_blocked += 1;
                } else {
                    incorrectly_blocked += 1;
                }
            }
        }
        if let Some(adversarial_set) = adversarial_set {
            let total_adversarial = adversarial_set.len();
            let total_benign = neighbor_models.len() - total_adversarial;

            if total_adversarial > 0 {
                reputation_stats.blocking_stats.srba =
                    correctly_blocked as f64 / total_adversarial as f64;
                reputation_stats.blocking_stats.frba =
                    incorrectly_not_blocked as f64 / total_adversarial as f64;
            }
            if total_benign > 0 {
                reputation_stats.blocking_stats.fpbr =
                    incorrectly_blocked as f64 / total_benign as f64;
            }
        }
        round_stats.reputation_stats = Some(reputation_stats);
        res
    }
}

fn cos_sim(a: &Tensor, b: &Tensor) -> f64 {
    a.dot(b).double_value(&[]) / (mag(a) * mag(b))
}

fn mag(a: &Tensor) -> f64 {
    a.sum(Kind::Float).sqrt().double_value(&[])
}
