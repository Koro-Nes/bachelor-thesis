use std::{
    cell::RefCell,
    collections::{HashSet, VecDeque},
    rc::Rc,
    u32,
};

use rand::{seq::SliceRandom, SeedableRng, rngs::StdRng};

use crate::{
    attack::attack::{
        AttackStrategy, AttackType, BackDoorTriggerAttack, LabelFlippingAttack, SignFlipAttack,
    },
    config::config::CONFIG,
    defense::defense::{DefenseMechanism, DefenseType, NoDefense, Reputation},
    ml::{
        aggregator::{
            Aggregator, AggregatorType, BalanceAggregator, DFedAvgMAggregator,
            TrimmedMeanAggregator,
        },
        dataset::{DataSet, LocalDataSet},
        model::Model,
    },
    network::{graph::Graph, mixing_matrix::MixingMatrix, reputation::ReputationTable},
    node::stats::NodeStats,
};

pub struct Node {
    pub id: u32,
    pub model: Model,
    pub dataset: LocalDataSet,
    pub neighbors: Vec<u32>,
    pub kind: NodeKind,

    pub attack: Option<Box<dyn AttackStrategy>>,
    pub defense: Box<dyn DefenseMechanism>,
    pub aggregator: Box<dyn Aggregator>,
    pub reputation: Rc<RefCell<ReputationTable>>,

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

#[derive(Debug, Clone, Copy)]
pub enum NodeKind {
    Benign,
    Adversarial {
        colluding: bool,
        attack_type: AttackType,
    },
}

impl Node {
    pub fn generate_nodes_from_graph(
        graph: &Graph,
        mut data: DataSet,
        attack_type: AttackType,
        defense_type: DefenseType,
        aggregator_type: AggregatorType,
        byzantine_fraction: f32,
        seed: usize,
    ) -> (Vec<Node>, usize) {
        let n = graph.adjacency.len();
        let collusion_fraction = CONFIG.network.collusion_fraction;

        let mut rng = StdRng::seed_from_u64(seed as u64);

        data.shuffle(&mut rng);

        let n_adversarial = ((n as f32) * byzantine_fraction).floor() as usize;
        let n_collusion = (n_adversarial as f32 * collusion_fraction).ceil() as usize;
        let n_benign = n - n_adversarial;
        let reputation_table = Rc::new(RefCell::new(ReputationTable::new()));
        let initial_params = Model::new(CONFIG.training.learning_rate.into()).model();

        let mut res = Vec::with_capacity(n);

        let mut ids: Vec<u32> = graph.adjacency.keys().copied().collect();
        ids.sort(); // ensures deterministic base order
        ids.shuffle(&mut rng);

        let adversarial_ids = &ids[n_benign..];
        let adversarial_set: HashSet<u32> = adversarial_ids.iter().copied().collect();

        let mut colluding_set = HashSet::new();
        let mut queue = VecDeque::new();

        for &start in adversarial_ids {
            if colluding_set.len() >= n_collusion {
                break;
            }

            if colluding_set.contains(&start) {
                continue;
            }

            queue.push_back(start);

            while let Some(node) = queue.pop_front() {
                if colluding_set.len() >= n_collusion {
                    break;
                }

                if !colluding_set.insert(node) {
                    continue;
                }

                for &neighbor in &graph.adjacency[&node] {
                    if adversarial_set.contains(&neighbor) && !colluding_set.contains(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        for id in &ids {
            let mut node_model = Model::new(CONFIG.training.learning_rate.into());
            node_model.update_after_aggregation(&initial_params);
            let kind = if colluding_set.contains(&id) {
                NodeKind::Adversarial {
                    colluding: true,
                    attack_type: attack_type.clone(),
                }
            } else if adversarial_ids.contains(&id) {
                NodeKind::Adversarial {
                    colluding: false,
                    attack_type: attack_type.clone(),
                }
            } else {
                NodeKind::Benign
            };

            let (x, y) = data.pop();
            let data_set = LocalDataSet::new(
                x,
                y,
                CONFIG.training.batch_size as usize,
                CONFIG.training.test_ratio as f64,
                CONFIG.device,
            );

            let neighbors = graph.adjacency.get(id).unwrap().to_vec();
            let mixing_matrix = MixingMatrix::new(*id, graph);
            let node_stats = NodeStats::new(
                *id,
                kind,
                defense_type,
                match kind {
                    NodeKind::Adversarial { attack_type, .. } => Some(attack_type),
                    _ => None,
                },
                neighbors.len(),
            );

            match kind {
                NodeKind::Benign => {
                    res.push(Node {
                        id: *id,
                        model: node_model,
                        dataset: data_set,
                        neighbors: neighbors.clone(),
                        kind,
                        attack: None,
                        defense: match defense_type {
                            DefenseType::NoDefense => Box::new(NoDefense {}),
                            DefenseType::Reputation => Box::new(Reputation::new(
                                *id,
                                neighbors.clone(),
                                Rc::clone(&reputation_table),
                            )),
                        },
                        aggregator: match aggregator_type {
                            AggregatorType::Balance => Box::new(BalanceAggregator::new(
                                CONFIG.balance.alpha,
                                CONFIG.balance.gamma,
                                CONFIG.balance.kappa,
                                CONFIG.balance.T,
                            )),
                            AggregatorType::DFedAvgM => Box::new(DFedAvgMAggregator::new(
                                CONFIG.dfedavgm.beta,
                                mixing_matrix,
                            )),
                            AggregatorType::TrimmedMean => Box::new(TrimmedMeanAggregator::new(
                                CONFIG.trimmedmean.beta,
                            )),
                        },
                        reputation: Rc::clone(&reputation_table),
                        stats: node_stats,
                    });
                }
                NodeKind::Adversarial {
                    colluding: _,
                    attack_type,
                } => {
                    res.push(Node {
                        id: *id,
                        model: node_model,
                        dataset: data_set,
                        neighbors: neighbors.clone(),
                        kind,
                        attack: match attack_type {
                            AttackType::LabelFlipping => Some(Box::new(LabelFlippingAttack::new())),
                            AttackType::SignFlipping => Some(Box::new(SignFlipAttack::new())),
                            AttackType::BackdoorTrigger => {
                                Some(Box::new(BackDoorTriggerAttack::new()))
                            }
                            AttackType::NoAttack => None,
                        },
                        defense: match defense_type {
                            DefenseType::Reputation => {
                                let mut rep = Reputation::new(
                                    *id,
                                    neighbors.clone(),
                                    Rc::clone(&reputation_table),
                                );
                                if colluding_set.contains(id) {
                                    rep.set_colluding(colluding_set.clone());
                                }
                                Box::new(rep)
                            }
                            DefenseType::NoDefense => Box::new(NoDefense::new()),
                        },
                        aggregator: match aggregator_type {
                            AggregatorType::Balance => Box::new(BalanceAggregator::new(
                                CONFIG.balance.alpha,
                                CONFIG.balance.gamma,
                                CONFIG.balance.kappa,
                                CONFIG.balance.T,
                            )),
                            AggregatorType::DFedAvgM => Box::new(DFedAvgMAggregator::new(
                                CONFIG.dfedavgm.beta,
                                mixing_matrix,
                            )),
                            AggregatorType::TrimmedMean => Box::new(TrimmedMeanAggregator::new(
                                CONFIG.trimmedmean.beta,
                            )),
                        },
                        reputation: Rc::clone(&reputation_table),
                        stats: node_stats,
                    });
                }
            }
        }
        res.iter_mut()
            .filter(|n| {
                if n.attack.as_ref().is_some() {
                    matches!(
                        n.attack.as_ref().unwrap().kind(),
                        AttackType::BackdoorTrigger
                    ) || matches!(n.attack.as_ref().unwrap().kind(), AttackType::LabelFlipping)
                } else {
                    false
                }
            })
            .for_each(|n| {
                n.perform_attack();
            });
        (res, colluding_set.len())
    }

    pub fn train_local(&mut self) -> f64 {
        let train_len = self.dataset.train_len();
        let batch_size = self.dataset.batch_size;
        let steps_per_epoch = train_len.div_ceil(batch_size);
        let mut avg_loss = 0.0;

        for _ in 0..CONFIG.training.local_epochs {
            self.dataset.shuffle_train();
            for _ in 0..steps_per_epoch {
                let (xs, ys) = self.dataset.next_train_batch().unwrap();
                avg_loss += self.model.train_batch(&xs, &ys);
            }
        }
        avg_loss / (CONFIG.training.local_epochs * steps_per_epoch as u32) as f64
    }

    pub fn perform_attack(&mut self) {
        let attack = self.attack.as_mut().unwrap();
        match attack.kind() {
            AttackType::LabelFlipping => attack.manipulate_dataset(&mut self.dataset),
            AttackType::SignFlipping => {
                let new_model = attack.manipulate_model(&self.model.model());
                self.model.update_after_aggregation(&new_model);
            }
            AttackType::BackdoorTrigger => attack.manipulate_dataset(&mut self.dataset),
            AttackType::NoAttack => (),
        }
    }
}
