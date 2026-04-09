use std::fmt::Write;

use crate::{attack::attack::AttackType, defense::defense::DefenseType, node::node::NodeKind};

#[derive(Debug)]
pub struct GlobalStats {
    pub experiment_name: String,
    pub seed: usize,
    pub final_global_accuracy_min: f64,
    pub final_global_accuracy_max: f64,
    pub final_global_accuracy_avg: f64,
    pub final_global_accuracy_median: f64,
    pub bytes_sent_min: u128,
    pub bytes_sent_max: u128,
    pub bytes_sent_median: u128,
    pub bytes_received_min: u128,
    pub bytes_received_max: u128,
    pub bytes_received_median: u128,
    pub attack_success_rate: Option<f64>,
    pub accuracy_on_non_target_classes: Option<f64>,
    pub biggest_collusion_group_node_count: usize,
    pub rounds_until_model_below_80_percent_baseline: Option<u32>,
    pub number_of_regular_nodes_that_integrated_malicious_updates: usize,
    pub rounds_until_50_percent_nodes_integrated_malicious: Option<u32>,
    pub node_stats: Vec<NodeStats>,
}

#[derive(Debug, Clone)]
pub struct NodeStats {
    pub id: u32,
    pub kind: NodeKind,
    pub defense_type: DefenseType,
    pub attack_type: Option<AttackType>,
    pub neighbors_count: usize,
    /// First round in which this node integrated an update originating from an adversarial node.
    pub included_adversarial_in_round: Option<usize>,
    /// First round in which this node integrated a malicious update, including indirect contamination
    /// via already-contaminated benign nodes.
    pub included_malicious_update_in_round: Option<usize>,
    pub round_stats: Vec<RoundStats>,
    pub final_accuracy: f64,
}

impl NodeStats {
    pub fn new(
        id: u32,
        kind: NodeKind,
        defense_type: DefenseType,
        attack_type: Option<AttackType>,
        neighbors_count: usize,
    ) -> Self {
        NodeStats {
            id,
            kind,
            defense_type,
            attack_type,
            neighbors_count,
            included_adversarial_in_round: None,
            included_malicious_update_in_round: None,
            round_stats: Vec::new(),
            final_accuracy: 0.0,
        }
    }

    pub fn add_round(&mut self, round_stats: RoundStats) {
        self.round_stats.push(round_stats);
    }

    pub fn set_adversarial_include_round(&mut self, round: usize) {
        if self.included_adversarial_in_round.is_none() {
            self.included_adversarial_in_round = Some(round);
        }
        self.set_malicious_include_round(round);
    }

    pub fn set_malicious_include_round(&mut self, round: usize) {
        if self.included_malicious_update_in_round.is_none() {
            self.included_malicious_update_in_round = Some(round);
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct RoundStats {
    pub loss: f64,
    pub accuracy: f64,
    pub memory_usage: Option<u128>,
    pub cpu_usage: Option<f64>,
    pub bytes_sent: u128,
    pub bytes_received: u128,
    pub reputation_stats: Option<ReputationStats>,
}

#[derive(Debug, Default, Clone)]
pub struct ReputationStats {
    pub blocking_stats: BlockingStats,
    pub neighbor_scores: Vec<NeighborScores>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NeighborScores {
    pub neighbor_id: u32,
    pub score: f64,
    pub adversarial: bool,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BlockingStats {
    pub srba: f64,
    pub frba: f64,
    pub fpbr: f64,
}

impl GlobalStats {
    pub fn to_human_readable(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "Experiment: {}", self.experiment_name);
        let _ = writeln!(out, "Seed: {}", self.seed);
        let _ = writeln!(out, "Final Global Accuracy:");
        let _ = writeln!(out, "  Min: {:.6}", self.final_global_accuracy_min);
        let _ = writeln!(out, "  Max: {:.6}", self.final_global_accuracy_max);
        let _ = writeln!(out, "  Avg: {:.6}", self.final_global_accuracy_avg);
        let _ = writeln!(out, "  Median: {:.6}", self.final_global_accuracy_median);
        let _ = writeln!(out, "Bytes Sent (per round, all nodes):");
        let _ = writeln!(out, "  Min: {}", self.bytes_sent_min);
        let _ = writeln!(out, "  Max: {}", self.bytes_sent_max);
        let _ = writeln!(out, "  Median: {}", self.bytes_sent_median);
        let _ = writeln!(out, "Bytes Received (per round, all nodes):");
        let _ = writeln!(out, "  Min: {}", self.bytes_received_min);
        let _ = writeln!(out, "  Max: {}", self.bytes_received_max);
        let _ = writeln!(out, "  Median: {}", self.bytes_received_median);
        let _ = writeln!(
            out,
            "Attack Success Rate: {}",
            format_optional_f64(self.attack_success_rate)
        );
        let _ = writeln!(
            out,
            "Accuracy On Non-Target Classes: {}",
            format_optional_f64(self.accuracy_on_non_target_classes)
        );
        let _ = writeln!(
            out,
            "Biggest Collusion Group Node Count: {}",
            self.biggest_collusion_group_node_count
        );
        let _ = writeln!(
            out,
            "Rounds Until Model Below 80 Percent Baseline: {}",
            format_optional_u32(self.rounds_until_model_below_80_percent_baseline)
        );
        let _ = writeln!(
            out,
            "Number Of Regular Nodes That Integrated Malicious Updates: {}",
            self.number_of_regular_nodes_that_integrated_malicious_updates
        );
        let _ = writeln!(
            out,
            "Rounds Until 50 Percent Nodes Integrated Malicious: {}",
            format_optional_u32(self.rounds_until_50_percent_nodes_integrated_malicious)
        );
        let _ = writeln!(out, "Node Count: {}", self.node_stats.len());
        out
    }
}

impl NodeStats {
    pub fn to_human_readable(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "Node: {}", self.id);
        let _ = writeln!(out, "Kind: {}", format_node_kind(&self.kind));
        let _ = writeln!(out, "Defense Type: {}", self.defense_type);
        let _ = writeln!(
            out,
            "Attack Type: {}",
            self.attack_type
                .map(|a| a.to_string())
                .unwrap_or_else(|| "None".to_string())
        );
        let _ = writeln!(out, "Neighbors Count: {}", self.neighbors_count);
        let _ = writeln!(
            out,
            "Included Direct Adversarial In Round: {}",
            self.included_adversarial_in_round
                .map(|r| r.to_string())
                .unwrap_or_else(|| "None".to_string())
        );
        let _ = writeln!(
            out,
            "Included Malicious Update In Round: {}",
            self.included_malicious_update_in_round
                .map(|r| r.to_string())
                .unwrap_or_else(|| "None".to_string())
        );
        let _ = writeln!(out, "Final Accuracy: {:.6}", self.final_accuracy);
        let _ = writeln!(out, "Rounds: {}", self.round_stats.len());
        for (idx, round) in self.round_stats.iter().enumerate() {
            let _ = writeln!(out, "Round {}:", idx);
            let _ = writeln!(out, "  Loss: {:.6}", round.loss);
            let _ = writeln!(out, "  Accuracy: {:.6}", round.accuracy);
            let _ = writeln!(
                out,
                "  CPU Usage: {}",
                round
                    .cpu_usage
                    .map(|v| format!("{:.6}", v))
                    .unwrap_or_else(|| "None".to_string())
            );
            let _ = writeln!(
                out,
                "  Memory Usage: {}",
                round
                    .memory_usage
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "None".to_string())
            );
            let _ = writeln!(out, "  Bytes Sent: {}", round.bytes_sent);
            let _ = writeln!(out, "  Bytes Received: {}", round.bytes_received);
            match &round.reputation_stats {
                Some(rep) => {
                    let _ = writeln!(out, "  Reputation Stats:");
                    let _ = writeln!(
                        out,
                        "    Blocking: srba={:.6}, frba={:.6}, fpbr={:.6}",
                        rep.blocking_stats.srba, rep.blocking_stats.frba, rep.blocking_stats.fpbr
                    );
                    if rep.neighbor_scores.is_empty() {
                        let _ = writeln!(out, "    Neighbor Scores: None");
                    } else {
                        let _ = writeln!(out, "    Neighbor Scores:");
                        for score in &rep.neighbor_scores {
                            let _ = writeln!(
                                out,
                                "      Neighbor {}: score={:.6}, adversarial={}",
                                score.neighbor_id, score.score, score.adversarial
                            );
                        }
                    }
                }
                None => {
                    let _ = writeln!(out, "  Reputation Stats: None");
                }
            }
        }
        out
    }
}

fn format_optional_f64(value: Option<f64>) -> String {
    value
        .map(|v| format!("{:.6}", v))
        .unwrap_or_else(|| "None".to_string())
}

fn format_optional_u32(value: Option<u32>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "None".to_string())
}

fn format_node_kind(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Benign => "Benign".to_string(),
        NodeKind::Adversarial {
            colluding,
            attack_type,
        } => format!(
            "Adversarial (colluding: {}, attack: {})",
            colluding, attack_type
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_adversarial_integration_implies_malicious_integration() {
        let mut stats = NodeStats::new(0, NodeKind::Benign, DefenseType::NoDefense, None, 0);
        stats.set_adversarial_include_round(5);
        assert_eq!(stats.included_adversarial_in_round, Some(5));
        assert_eq!(stats.included_malicious_update_in_round, Some(5));
    }

    #[test]
    fn indirect_contamination_sets_only_malicious_round() {
        let mut stats = NodeStats::new(0, NodeKind::Benign, DefenseType::NoDefense, None, 0);
        stats.set_malicious_include_round(2);
        assert_eq!(stats.included_adversarial_in_round, None);
        assert_eq!(stats.included_malicious_update_in_round, Some(2));

        // Later direct integration should not overwrite the first malicious-round timestamp.
        stats.set_adversarial_include_round(7);
        assert_eq!(stats.included_adversarial_in_round, Some(7));
        assert_eq!(stats.included_malicious_update_in_round, Some(2));
    }
}
