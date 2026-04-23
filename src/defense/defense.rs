use std::{cell::RefCell, collections::HashSet, fmt::Display, rc::Rc};

use tch::Tensor;

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
        received_scores: &[(u32, Vec<(u32, f64)>)],
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
        _received_scores: &[(u32, Vec<(u32, f64)>)],
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
        received_scores: &[(u32, Vec<(u32, f64)>)],
    ) -> Vec<(u32, Tensor)> {
        if !received_scores.is_empty() {
            for (neighbor_id, scores) in received_scores {
                for (shared_id, score) in scores {
                    reputation_table.put((*neighbor_id, *shared_id), *score);
                }
            }
        }

        let mut res = Vec::with_capacity(neighbor_models.len());
        let mut reputation_stats = ReputationStats::default();
        let mut correctly_blocked = 0;
        let mut incorrectly_blocked = 0;
        let mut incorrectly_not_blocked = 0;

        let config = CONFIG.get_network_config();
        let alpha = config.reputation_weight_alpha as f64;
        let beta = config.reputation_weight_beta as f64;
        let gamma = config.reputation_weight_gamma as f64;

        let is_colluding = self.colluding_set.is_some();
        let self_mag = self_model.norm().double_value(&[]);
        let self_neighbor_scores: Vec<f64> = neighbor_models
            .iter()
            .map(|(neighbor_id, _)| reputation_table.get(&(self.id, *neighbor_id)))
            .collect();

        for (outer_pos, (outer_idx, model)) in neighbor_models.iter().enumerate() {
            let is_adversarial = adversarial_set.map_or(false, |s| s.contains(outer_idx));

            // if colluding, only apply exponential decay
            if is_colluding {
                let curr_score = self_neighbor_scores[outer_pos];
                let score_decayed = (1.0 - gamma) * curr_score + gamma * 0.5;
                reputation_table.put((self.id, *outer_idx), score_decayed);

                reputation_stats.neighbor_scores.push(NeighborScores {
                    neighbor_id: *outer_idx,
                    score: score_decayed,
                    adversarial: is_adversarial,
                });
                continue;
            }

            let self_to_outer_score = self_neighbor_scores[outer_pos];
            let score_direct = alpha * self_to_outer_score
                + (1.0 - alpha) * (cos_sim(self_model, self_mag, model) / 2.0);
            let mut sum_scores_weighted = 0.0;
            let mut sum_scores = 0.0;
            for (inner_pos, (inner_idx, _)) in neighbor_models.iter().enumerate() {
                if *inner_idx != *outer_idx {
                    if let Some(neigh_score) =
                        reputation_table.get_optional(&(*inner_idx, *outer_idx))
                    {
                        sum_scores_weighted += neigh_score * self_to_outer_score;
                        sum_scores += self_neighbor_scores[inner_pos];
                    }
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

            reputation_stats.neighbor_scores.push(NeighborScores {
                neighbor_id: *outer_idx,
                score: score_decayed,
                adversarial: is_adversarial,
            });

            if score_decayed >= config.reputation_threshold as f64 {
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
            let total_adversarial = neighbor_models
                .iter()
                .filter(|(neighbor_id, _)| adversarial_set.contains(neighbor_id))
                .count();
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

fn cos_sim(a: &Tensor, mag_a: f64, b: &Tensor) -> f64 {
    let mag_b = b.norm().double_value(&[]);

    if mag_a < 1e-9 || mag_b < 1e-9 {
        return 0.0;
    }

    a.dot(b).double_value(&[]) / (mag_a * mag_b)
}
