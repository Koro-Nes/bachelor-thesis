use std::collections::HashSet;

use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use tch::nn::{self};

use crate::{
    attack::attack::AttackStrategy,
    defense::defense::DefenseMechanism,
    ml::{aggregator::Aggregator, dataset::LocalDataSet, model::CNNModel},
    network::{graph::Graph, reputation::ReputationTable},
    node::stats::NodeStats,
};

pub struct Node {
    pub id: u32,
    pub model: CNNModel,
    pub vs: nn::VarStore,
    pub dataset: LocalDataSet,
    pub neighbors: Vec<u32>,
    pub kind: NodeKind,

    pub attack: Option<Box<dyn AttackStrategy>>,
    pub defense: Box<dyn DefenseMechanism>,
    pub aggregator: Box<dyn Aggregator>,
    pub reputation: ReputationTable,

    pub stats: NodeStats,
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("id", &self.id)
            .field("model", &self.model)
            .field("vs", &"varstore")
            .field("dataset", &self.dataset)
            .field("neighbors", &self.neighbors)
            .field("kind", &self.kind)
            .field("attack", &"<dyn AttackStrategy")
            .field("defense", &"<dyn DefenseMechansim>")
            .field("aggregator", &"<dyn Aggregator>")
            .field("reputation", &self.reputation)
            .field("stats", &self.stats)
            .finish()
    }
}

#[derive(Debug)]
pub enum NodeKind {
    Benign,
    Adversarial { colluding: bool },
}

impl Node {
    pub fn generate_nodes_from_graph(
        graph: &Graph,
        byzantine_fraction: f32,
        collusion_fraction: f32,
        vs: nn::VarStore,
        seed: usize,
    ) -> Vec<Node> {
        let n = graph.adjacency.len();

        let mut rng = StdRng::seed_from_u64(seed as u64);

        let n_adversarial = ((n as f32) * byzantine_fraction).floor() as usize;
        let n_collusion = (n_adversarial as f32 * collusion_fraction).ceil() as usize;
        let n_benign = n - n_adversarial;

        let mut res = Vec::with_capacity(n);

        let mut ids: Vec<u32> = graph.adjacency.keys().copied().collect();
        ids.shuffle(&mut rng);

        let benign_ids = &ids[..n_benign];
        let adversarial_ids = &ids[n_benign..];
        let mut colluding_set = HashSet::new();
        let mut stack = vec![adversarial_ids[0]];

        while colluding_set.len() < n_collusion {
            let node = stack.pop().unwrap();
            colluding_set.insert(node);
            for neighbor in &graph.adjacency[&node] {
                if adversarial_ids.contains(&neighbor) && !colluding_set.contains(&neighbor) {
                    stack.push(*neighbor);
                }
            }
        }

        for id in &ids {
            let kind = if colluding_set.contains(&id) {
                NodeKind::Adversarial { colluding: true }
            } else if adversarial_ids.contains(&id) {
                NodeKind::Adversarial { colluding: false }
            } else {
                NodeKind::Benign
            };

            match kind {
                NodeKind::Benign => {}
                NodeKind::Adversarial { colluding } => match colluding {
                    true => {}
                    false => {}
                },
            }
        }
        //res.sort();
        res
    }
}
