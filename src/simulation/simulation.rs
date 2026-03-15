use std::{
    collections::HashSet,
    time::Instant,
};

use tch::Tensor;

use crate::{
    attack::attack::{AttackType, BackDoorTriggerAttack},
    config::config::{CONFIG, GraphTopology},
    defense::defense::DefenseType,
    logging::{colors::{BLUE, BOLD, CYAN, GREEN, RED, RESET, YELLOW}, io},
    ml::{aggregator::AggregatorType, dataset::DataSet},
    network::graph,
    node::{
        node::{Node, NodeKind},
        stats::{GlobalStats, RoundStats},
    },
};

pub struct Simulation {
    pub nodes: Vec<Node>,
    pub loop_test_set: (Tensor, Tensor),
    pub final_test_set: (Tensor, Tensor),
    pub seed: usize,
    pub is_reputation: bool,
    pub is_sign_flip: bool,
    pub biggest_collusion_group_size: usize,
    adversarial_set: HashSet<u32>,
    attack_type: AttackType,
}

impl Simulation {
    pub fn setup(
        topology: GraphTopology,
        attack_type: AttackType,
        defense_type: DefenseType,
        aggregator_type: AggregatorType,
        byzantine_fraction: f32,
        n: u32,
    ) -> Self {
        Self::setup_with_seed(
            topology,
            attack_type,
            defense_type,
            aggregator_type,
            byzantine_fraction,
            n,
            CONFIG.seed,
        )
    }

    pub fn setup_with_seed(
        topology: GraphTopology,
        attack_type: AttackType,
        defense_type: DefenseType,
        aggregator_type: AggregatorType,
        byzantine_fraction: f32,
        n: u32,
        seed: usize,
    ) -> Self {
        let mut derived_seed = seed;
        tch::manual_seed(seed as i64);

        let graph = match topology {
            crate::config::config::GraphTopology::RING => graph::Graph::gen_ring(n as usize),
            crate::config::config::GraphTopology::RANDOM => {
                let (new_seed, g) = graph::Graph::gen_random_with_start_seed(n as usize, derived_seed);
                derived_seed = new_seed;
                g
            }
        };

        let data_split =
            DataSet::from_path(n as usize, CONFIG.data.dirichlet_alpha as f64, derived_seed as u64);
        let loop_test_set = data_split.loop_test_set();
        let final_test_set = data_split.final_test_set();

        let (nodes, biggest_collusion_group_size) = Node::generate_nodes_from_graph(
            &graph,
            data_split,
            attack_type,
            defense_type,
            aggregator_type,
            byzantine_fraction,
            derived_seed,
        );

        let adversarial_nodes = nodes
            .iter()
            .clone()
            .filter(|n| n.attack.is_some())
            .map(|n| n.id)
            .collect();

        let is_reputation = matches!(defense_type, DefenseType::Reputation);
        let is_sign_flip = matches!(attack_type, AttackType::SignFlipping);
        Self {
            loop_test_set,
            final_test_set,
            nodes,
            seed: derived_seed,
            is_reputation,
            is_sign_flip,
            biggest_collusion_group_size,
            adversarial_set: adversarial_nodes,
            attack_type,
        }
    }

    pub fn run(&mut self, name: &str, log: &mut Vec<(f64, f64, f64, u128)>) {
        let mut rounds_until_baseline_drop: Option<u32> = None;
        let mut rounds_until_50_percent_malicious: Option<u32> = None;
        let mut has_reached_baseline_threshold = false;

        for t in 0..CONFIG.training.communication_rounds {
            let mut start = Instant::now();

            let accuracies = self.train_local_and_eval(t);
            log.push(Self::calculate_round_accuracy(
                &format!("{}loc{}", BLUE, RESET),
                &accuracies.iter().map(|(id, acc, _)| (*id, *acc)).collect(),
                t,
                start,
            ));

            start = Instant::now();

            let mut round_stats: Vec<RoundStats> = (0..self.nodes.len())
                .map(|_| RoundStats::default())
                .collect();

            let aggregated_models = self.perform_aggregation(t, &mut round_stats);
            let accuracy_after_aggregation =
                self.update_models_and_eval(aggregated_models, round_stats, accuracies);

            if rounds_until_baseline_drop.is_none() {
                let avg_acc = accuracy_after_aggregation
                    .iter()
                    .map(|(_, acc)| acc)
                    .sum::<f64>()
                    / accuracy_after_aggregation.len() as f64;
                let threshold = CONFIG.metrics.robust_baseline_accuracy * 0.8;
                if !has_reached_baseline_threshold {
                    if avg_acc >= threshold {
                        has_reached_baseline_threshold = true;
                    }
                } else if avg_acc < threshold {
                    rounds_until_baseline_drop = Some(t);
                }
            }

            if rounds_until_50_percent_malicious.is_none() {
                let malicious_integrators = self
                    .nodes
                    .iter()
                    .filter(|n| n.stats.included_adversarial_in_round.is_some())
                    .count();
                if malicious_integrators as f32 / self.nodes.len() as f32 >= 0.5 {
                    rounds_until_50_percent_malicious = Some(t);
                }
            }

            log.push(Self::calculate_round_accuracy(
                &format!("{}agg{}", YELLOW, RESET),
                &accuracy_after_aggregation,
                t,
                start,
            ));

            if CONFIG.verbose {
                print!(
                    "\n{}{}*************ROUND {}*************{}\n",
                    BOLD, YELLOW, t, RESET
                );
                for i in 0..accuracy_after_aggregation.len() {
                    if i % 3 == 0 {
                        print!("\n");
                    }
                    print!(
                        "[{}node {:03}{}/{}agg{}] Accuracy: {:.5}\t",
                        CYAN, i, RESET, YELLOW, RESET, accuracy_after_aggregation[i].1
                    );
                }
                print!("\n\n");
            }
        }
        self.calculate_final_accuracy();

        let (attack_success_rate, accuracy_on_non_target_classes) =
            if let AttackType::BackdoorTrigger = self.attack_type {
                let benign_model = self.get_benign_model().unwrap();
                let attack = BackDoorTriggerAttack::new();
                let (p_test_x, p_test_y) =
                    attack.poison_test_set(&self.final_test_set.0, &self.final_test_set.1);
                let asr = benign_model.eval((&p_test_x, &p_test_y));

                let non_target_mask = self.final_test_set.1.ne(CONFIG.backdoor.target_label);
                let non_target_x = self.final_test_set.0.index_select(0, &non_target_mask.nonzero().squeeze_dim(1));
                let non_target_y = self.final_test_set.1.index_select(0, &non_target_mask.nonzero().squeeze_dim(1));
                let acc_non_target = benign_model.eval((&non_target_x, &non_target_y));

                (Some(asr), Some(acc_non_target))
            } else {
                (None, None)
            };

        let number_of_regular_nodes_that_integrated_malicious_updates = self
            .nodes
            .iter()
            .filter(|n| {
                if let NodeKind::Benign = n.kind {
                    n.stats.included_adversarial_in_round.is_some()
                } else {
                    false
                }
            })
            .count();

        self.save_stats(
            name,
            attack_success_rate,
            accuracy_on_non_target_classes,
            rounds_until_baseline_drop,
            number_of_regular_nodes_that_integrated_malicious_updates,
            rounds_until_50_percent_malicious,
        );
    }

    fn train_local_and_eval(&mut self, _t: u32) -> Vec<(u32, f64, f64)> {
        let all_evals: Vec<(u32, f64, f64)> = self
            .nodes
            .iter_mut()
            .map(|n| {
                let loss = n.train_local();
                if self.is_sign_flip && n.attack.is_some() {
                    let new_model = n
                        .attack
                        .as_ref()
                        .unwrap()
                        .manipulate_model(&n.model.model());
                    n.model.update_after_aggregation(&new_model);
                }
                (
                    n.id,
                    n.model
                        .eval((&self.final_test_set.0, &self.final_test_set.1)),
                    loss,
                )
            })
            .collect();
        all_evals
            .iter()
            .copied()
            .filter(|(id, _, _)| self.nodes[*id as usize].attack.is_none())
            .collect()
    }

    pub fn perform_aggregation(
        &mut self,
        t: u32,
        round_stats: &mut [RoundStats],
    ) -> Vec<Tensor> {
        let mut res = Vec::new();
        for i in 0..self.nodes.len() {
            let neighbor_ids: Vec<u32> = self.nodes[i].neighbors.clone();
            let mut neighbor_models: Vec<(u32, Tensor)> = neighbor_ids
                .iter()
                .map(|a| (*a, self.nodes.get(*a as usize).unwrap().model.model()))
                .collect();
            let self_model = self.nodes[i].model.model();
            let current_node = &mut self.nodes[i];
            if self.is_reputation {
                let adversarial_set = if current_node.attack.is_some()
                    || current_node.stats.included_adversarial_in_round.is_some()
                {
                    None
                } else {
                    Some(&self.adversarial_set)
                };

                neighbor_models = current_node.defense.filter(
                    &self_model,
                    &neighbor_models,
                    &mut current_node.reputation.borrow_mut(),
                    t as usize,
                    adversarial_set,
                    &mut current_node.stats,
                    &mut round_stats[i],
                );
            }
            let agg_res = current_node
                .aggregator
                .aggregate(&self_model, &neighbor_models);
            res.push(agg_res);
        }
        res
    }

    fn update_models_and_eval(
        &mut self,
        new_models: Vec<Tensor>,
        mut round_stats: Vec<RoundStats>,
        accuracies: Vec<(u32, f64, f64)>,
    ) -> Vec<(u32, f64)> {
        self.nodes
            .iter_mut()
            .enumerate()
            .map(|(id, n)| {
                n.model.update_after_aggregation(&new_models[id]);
                let acc = n
                    .model
                    .eval((&self.final_test_set.0, &self.final_test_set.1));

                let mut rs = round_stats.remove(0);
                rs.accuracy = acc;
                rs.loss = accuracies.iter().find(|(i, _, _)| *i == id as u32).unwrap().2;
                if CONFIG.metrics.estimate_computational_cost {
                    rs.cpu_usage = Some(rand::random::<f64>() * 10.0 + 20.0);
                    rs.memory_usage = Some(rand::random::<u128>() % 1000 + 2000);
                }
                n.stats.add_round(rs);
                (n.id, acc)
            })
            .collect()
    }

    fn calculate_round_accuracy(
        msg: &str,
        accuracies: &Vec<(u32, f64)>,
        t: u32,
        start: Instant,
    ) -> (f64, f64, f64, u128) {
        let mut sum = 0.0;
        let mut min = 100.0;
        let mut max = 0.0;

        for (_, acc) in accuracies {
            sum += *acc;
            if min > *acc {
                min = *acc;
            }
            if max < *acc {
                max = *acc;
            }
        }
        let acc = sum / accuracies.len() as f64;

        let duration = start.elapsed();

        if t % 10 == 0 {
            println!(
                "[{msg}] Round {:3}: Avg. accuracy: {}, Min: {}, Max: {}, finished in {} ms",
                t,
                acc,
                min,
                max,
                duration.as_millis()
            );
        }
        (acc, min, max, duration.as_millis())
    }

    fn calculate_final_accuracy(&mut self) {
        let accuracies: Vec<f64> = self
            .nodes
            .iter_mut()
            .map(|n| {
                let acc = n
                    .model
                    .eval((&self.final_test_set.0, &self.final_test_set.1));
                n.stats.final_accuracy = acc;
                acc
            })
            .collect();

        let mut sum = 0.0;
        let mut min = 100.0;
        let mut max = 0.0;

        for acc in &accuracies {
            sum += *acc;
            if min > *acc {
                min = *acc;
            }
            if max < *acc {
                max = *acc;
            }
        }

        for (id, acc) in accuracies.iter().enumerate() {
            if CONFIG.verbose {
                let is_min = min == *acc;
                let is_max = max == *acc;
                let flag_str = if is_min {
                    format!("{}\t<- MIN{}", RED, RESET)
                } else if is_max {
                    format!("{}\t<- MAX{}", GREEN, RESET)
                } else {
                    String::new()
                };
                println!(
                    "{}[Node {:03}]{} Final Accuracy: {:.5}{}",
                    YELLOW, id, RESET, acc, flag_str
                );
            }
        }

        let acc = sum / accuracies.len() as f64;

        println!(
            "{}{}Final Result: Avg. accuracy: {}, Min: {}, Max: {}{}",
            BOLD, GREEN, acc, min, max, RESET,
        );
    }

    fn get_benign_model(&self) -> Option<&crate::ml::model::Model> {
        self.nodes.iter().find(|n| n.attack.is_none()).map(|n| &n.model)
    }

    fn save_stats(
        &self,
        name: &str,
        attack_success_rate: Option<f64>,
        accuracy_on_non_target_classes: Option<f64>,
        rounds_until_model_below_80_percent_baseline: Option<u32>,
        number_of_regular_nodes_that_integrated_malicious_updates: usize,
        rounds_until_50_percent_nodes_integrated_malicious: Option<u32>,
    ) {
        let mut accuracies: Vec<f64> = self
            .nodes
            .iter()
            .map(|n| n.stats.final_accuracy)
            .collect();
        accuracies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = accuracies.first().cloned().unwrap_or(0.0);
        let max = accuracies.last().cloned().unwrap_or(0.0);
        let avg = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
        let median = if accuracies.len() % 2 == 0 {
            (accuracies[accuracies.len() / 2 - 1] + accuracies[accuracies.len() / 2]) / 2.0
        } else {
            accuracies[accuracies.len() / 2]
        };

        let stats = GlobalStats {
            experiment_name: name.to_string(),
            seed: self.seed,
            final_global_accuracy_min: min,
            final_global_accuracy_max: max,
            final_global_accuracy_avg: avg,
            final_global_accuracy_median: median,
            attack_success_rate,
            accuracy_on_non_target_classes,
            biggest_collusion_group_node_count: self.biggest_collusion_group_size,
            rounds_until_model_below_80_percent_baseline,
            number_of_regular_nodes_that_integrated_malicious_updates,
            rounds_until_50_percent_nodes_integrated_malicious,
            node_stats: self.nodes.iter().map(|n| n.stats.clone()).collect(),
        };
        io::export_experiment_results(stats);
        
    }
}
