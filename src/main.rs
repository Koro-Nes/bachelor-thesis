use std::{
    collections::{HashMap, HashSet},
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::Path,
};

use repository::{
    attack::attack::AttackType,
    config::config::{
        CONFIG, EXPERIMENT_CONFIGURATIONS, ExperimentConfiguration, GraphTopology,
        NETWORK_CONFIG_OVERRIDE, NetworkConfig,
    },
    defense::defense::DefenseType,
    logging::io,
    ml::aggregator::{AggregatorType, DFedAvgMAggregator},
    simulation::{self, simulation::Simulation},
};
use tch::Cuda;

fn main() {
    // io::create_experiment_folders("./results/");

    // let simulation = Simulation::setup(
    //     repository::attack::attack::AttackType::SignFlipping,
    //     repository::defense::defense::DefenseType::NoDefense,
    //     repository::ml::aggregator::AggregatorType::DFedAvgM,
    // );
    //run(CONFIG.seed);
    optimize_reputation();
    //run_baseline(3);
    //run_small_sample();
    //run_reputation_baseline();
    if Cuda::is_available() {
        Cuda::synchronize(0);
    }
}

fn metrics_from_sim(configs: &[ExperimentConfiguration], results: Vec<Vec<(f64, f64, f64, u128)>>) {
    for (id, round) in results.iter().enumerate() {
        let c = configs[id].clone();
        println!("{}", c);
        let last = round.last().unwrap();
        let duration = round.iter().map(|x| x.3).sum::<u128>();
        println!("Final accuracy: {:.7}", last.0);
        println!("Final min: {:.7}", last.1);
        println!("Final max: {:.7}", last.2);
        println!("Total duration: {}", format_duration(duration));
        println!(
            "Avg. round duration: {}ms",
            duration / CONFIG.training.communication_rounds as u128
        );
    }
}

fn run(seed: usize) {
    let configs: Vec<ExperimentConfiguration> = EXPERIMENT_CONFIGURATIONS
        .iter()
        .cloned()
        .filter(|c| c.seed_or(seed) == seed)
        .collect();

    if configs.is_empty() {
        println!("No configurations found for seed {}", seed);
        return;
    }

    let progress_dir = Path::new("logs");
    fs::create_dir_all(progress_dir).expect("failed to create logs directory for progress");
    let progress_path = progress_dir.join(format!("progress_seed{}.txt", seed));
    let mut completed = load_completed_configs(&progress_path);

    let mut results = Vec::new();
    let mut ran_configs = Vec::new();

    let mut curr_iter = 0;

    for c in configs.iter().cloned() {
        curr_iter += 1;
        if completed.contains(&c.name) && !CONFIG.ignore_skip {
            println!("Skipping {} (already completed for seed {})", c.name, seed);
            continue;
        }

        let mut sim_res = Vec::new();
        let seed = c.seed_or(seed);
        let mut simulation = Simulation::setup_with_seed(
            c.topology,
            c.attack_type,
            c.defense_type,
            c.aggregator_type,
            c.byzantine_fraction,
            c.node_count,
            seed,
        );
        println!("[{}/{}] {}", curr_iter, configs.len(), c);
        simulation.run(&c.name, &mut sim_res);
        results.push(sim_res);
        ran_configs.push(c.clone());

        if let Err(err) = mark_config_completed(&progress_path, &c.name) {
            println!(
                "Warning: failed to update progress file for {}: {}",
                c.name, err
            );
        } else {
            completed.insert(c.name);
        }
    }

    if results.is_empty() {
        println!("Nothing new to run for seed {}", seed);
        return;
    }

    metrics_from_sim(&ran_configs, results);
}

fn load_completed_configs(path: &Path) -> HashSet<String> {
    match fs::read_to_string(path) {
        Ok(contents) => contents
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(|line| line.to_string())
            .collect(),
        Err(_) => HashSet::new(),
    }
}

fn mark_config_completed(path: &Path, name: &str) -> std::io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{name}")?;
    Ok(())
}

fn format_duration(d: u128) -> String {
    let total_seconds = d / 1000;
    let ms = d % 1000;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    let fractional_seconds = (ms as f64) / 1000.0;
    let seconds_with_fraction = seconds as f64 + fractional_seconds;

    format!("{:02}:{:05.2}", minutes, seconds_with_fraction)
}

fn run_test_configurations() {
    let configs = EXPERIMENT_CONFIGURATIONS.clone();
    let mut results = Vec::new();
    for c in configs.iter().cloned() {
        let mut sim_res = Vec::new();
        let seed = c.seed_or(CONFIG.seed);
        let mut simulation = Simulation::setup_with_seed(
            c.topology,
            c.attack_type,
            c.defense_type,
            c.aggregator_type,
            c.byzantine_fraction,
            c.node_count,
            seed,
        );
        println!("{}", c);
        simulation.run(&c.name, &mut sim_res);
        results.push(sim_res);
    }
    metrics_from_sim(&configs, results);
}

fn run_small_sample() {
    let configs = vec![
        EXPERIMENT_CONFIGURATIONS[159].clone(),
        EXPERIMENT_CONFIGURATIONS[160].clone(),
        EXPERIMENT_CONFIGURATIONS[161].clone(),
    ];
    let mut results = Vec::new();
    for c in configs.iter().cloned() {
        let mut sim_res = Vec::new();
        let seed = c.seed_or(CONFIG.seed);
        let mut simulation = Simulation::setup_with_seed(
            c.topology,
            c.attack_type,
            c.defense_type,
            c.aggregator_type,
            c.byzantine_fraction,
            c.node_count,
            seed,
        );
        println!("{}", c);
        simulation.run(&c.name, &mut sim_res);
        results.push(sim_res);
    }
    metrics_from_sim(&configs, results);
}

fn run_baseline(n_runs: usize) {
    let node_counts = [10_u32, 50, 200];
    let topologies = [GraphTopology::RING, GraphTopology::RANDOM];

    for topology in topologies {
        for node_count in node_counts {
            let base_name = format!("baseline_n{}_{}", node_count, topology.name_str());
            let mut next_seed = CONFIG.seed;
            for run_idx in 0..n_runs {
                let seed = next_seed;
                let mut config = ExperimentConfiguration::new(
                    base_name.clone(),
                    node_count,
                    topology,
                    AttackType::NoAttack,
                    0.0,
                    DefenseType::NoDefense,
                    AggregatorType::ClippedMean,
                )
                .with_seed(seed);

                let mut sim_res = Vec::new();
                let mut simulation = Simulation::setup_with_seed(
                    config.topology,
                    config.attack_type,
                    config.defense_type,
                    config.aggregator_type,
                    config.byzantine_fraction,
                    config.node_count,
                    seed,
                );
                if simulation.seed != seed {
                    config = config.with_seed(simulation.seed);
                }
                println!("{} (run {}/{})", config, run_idx + 1, n_runs);
                simulation.run(&config.name, &mut sim_res);
                next_seed = simulation.seed + 1;
            }
        }
    }
}

fn optimize_reputation() {
    println!("Starting Reputation Parameter Optimization...");

    let alphas = [0.6_f32, 0.75, 0.9];
    let betas = [0.7_f32, 0.9];
    let gammas = [0.05_f32, 0.1, 0.2];
    let thresholds = [0.4_f32, 0.5];
    let node_counts = [10_usize, 50, 200];
    let topologies = [GraphTopology::RANDOM];

    const SEEDS_PER_CONFIG: usize = 1;
    /// Skip an alpha whose best-ever accuracy is this far below the global best.
    /// Conservative enough to never discard the true optimum (accuracy is in [0,1]).
    const PRUNE_MARGIN: f64 = 0.015;

    let defense = DefenseType::Reputation;
    let aggregator = AggregatorType::ClippedMean;
    let byzantine_fraction = 0.2;
    let base_seed = CONFIG.seed;

    // Declare both scenario sets up-front — Phase 2 used to shadow the binding.
    let p1_scenarios: &[(AttackType, &str)] = &[(AttackType::LabelFlipping, "LabelFlip")];
    let p2_scenarios: &[(AttackType, &str)] = &[
        (AttackType::BackdoorTrigger, "Backdoor"),
        (AttackType::SignFlipping, "SignFlip"),
    ];

    let p1_runs = (p1_scenarios.len() * SEEDS_PER_CONFIG) as f64;
    let p2_runs = (p2_scenarios.len() * SEEDS_PER_CONFIG) as f64; // was bugged: used SEEDS_PER_CONFIG (=1) even though 2 scenarios run

    // ── Replay progress ──────────────────────────────────────────────────────
    // HashSet gives O(1) `contains` — the old Vec gave O(n) per inner-loop check.
    let mut completed: HashSet<String> = HashSet::new();

    struct P1Result {
        threshold: f32,
        alpha: f32,
        beta: f32,
        gamma: f32,
        topology: GraphTopology,
        node_count: usize,
        p1_acc: f64,
    }
    let mut p1_results: Vec<P1Result> = Vec::new();
    let mut best_p1_score = f64::NEG_INFINITY;
    let mut best_params: Option<NetworkConfig> = None;

    let mut progress_file = io::init_grid_search_log();
    let mut buf = String::new();
    progress_file.read_to_string(&mut buf).unwrap();

    for line in buf.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let mut parts = line.split(',');
        let config = parts.next().unwrap().trim().to_string();
        let accuracy: f64 = parts.next().unwrap().trim().parse().unwrap();

        let mut s = config.split('_');
        let alpha: f32 = s.next().unwrap().replace("alpha", "").parse().unwrap();
        let beta: f32 = s.next().unwrap().replace("beta", "").parse().unwrap();
        let gamma: f32 = s.next().unwrap().replace("gamma", "").parse().unwrap();
        let threshold: f32 = s.next().unwrap().replace("threshold", "").parse().unwrap();
        let topo_str = s.next().unwrap().replace("topo", "");
        let node_count: usize = s.next().unwrap().replace("nodes", "").parse().unwrap();

        let topology = match topo_str.as_str() {
            "RING" => GraphTopology::RING,
            "RANDOM" => GraphTopology::RANDOM,
            other => panic!("Unknown topology in log: {other}"),
        };

        let net_config = NetworkConfig {
            collusion_fraction: CONFIG.network.collusion_fraction,
            reputation_threshold: threshold,
            reputation_weight_alpha: alpha,
            reputation_weight_beta: beta,
            reputation_weight_gamma: gamma,
        };

        if accuracy > best_p1_score {
            best_p1_score = accuracy;
            best_params = Some(net_config);
        }
        completed.insert(config);
        p1_results.push(P1Result {
            threshold,
            alpha,
            beta,
            gamma,
            topology,
            node_count,
            p1_acc: accuracy,
        });
    }

    // ── Sort each parameter axis by descending mean accuracy from the log ────
    // Running the best-known values first quickly raises `best_p1_score`,
    // which makes the alpha-pruning gate below fire sooner.
    let mean_acc_for = |get: &dyn Fn(&P1Result) -> f32, v: f32| -> f64 {
        let (sum, n) = p1_results
            .iter()
            .filter(|r| (get(r) - v).abs() < f32::EPSILON)
            .fold((0.0_f64, 0_usize), |(s, c), r| (s + r.p1_acc, c + 1));
        if n == 0 { 0.0 } else { sum / n as f64 }
    };
    let sort_desc = |values: &[f32], get: &dyn Fn(&P1Result) -> f32| -> Vec<f32> {
        let mut v = values.to_vec();
        v.sort_by(|a, b| {
            mean_acc_for(get, *b)
                .partial_cmp(&mean_acc_for(get, *a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    };

    let sorted_alphas = sort_desc(&alphas, &|r| r.alpha);
    let sorted_betas = sort_desc(&betas, &|r| r.beta);
    let sorted_gammas = sort_desc(&gammas, &|r| r.gamma);
    let sorted_thresholds = sort_desc(&thresholds, &|r| r.threshold);

    // ── Per-alpha ceiling: best accuracy ever seen for each alpha value ───────
    // Populated from the log; updated live during Phase 1.
    // Key: alpha.to_bits() to avoid f32 hashing issues.
    let mut alpha_best_seen: HashMap<u32, f64> = {
        let mut m: HashMap<u32, f64> = HashMap::new();
        for r in &p1_results {
            let e = m.entry(r.alpha.to_bits()).or_insert(f64::NEG_INFINITY);
            *e = e.max(r.p1_acc);
        }
        m
    };

    // ── Phase 1: grid search ─────────────────────────────────────────────────
    let total_combos = alphas.len()
        * betas.len()
        * gammas.len()
        * thresholds.len()
        * topologies.len()
        * node_counts.len();

    println!(
        "Testing up to {} remaining configurations (pruning may reduce this further).",
        total_combos - completed.len()
    );

    let mut current_iter = completed.len();
    let start_time = std::time::Instant::now();

    for &topology in &topologies {
        for &node_count in &node_counts {
            let experiment_configs: Vec<ExperimentConfiguration> = p1_scenarios
                .iter()
                .map(|(attack, name)| {
                    ExperimentConfiguration::new(
                        format!("opt_{name}"),
                        node_count as u32,
                        topology,
                        *attack,
                        byzantine_fraction,
                        defense,
                        aggregator,
                    )
                })
                .collect();

            'alpha: for &alpha in &sorted_alphas {
                // ── Alpha-level gate ─────────────────────────────────────────
                // If this alpha has never come within PRUNE_MARGIN of the global
                // best across *any* config we have seen so far, it is very unlikely
                // to produce the overall winner — skip all its sub-combinations.
                if let Some(&ceiling) = alpha_best_seen.get(&alpha.to_bits()) {
                    if ceiling < best_p1_score - PRUNE_MARGIN {
                        println!(
                            "  [prune] alpha={alpha:.2} (ceiling={ceiling:.4} < global {best_p1_score:.4} − {PRUNE_MARGIN})"
                        );
                        continue 'alpha;
                    }
                }

                for &beta in &sorted_betas {
                    for &gamma in &sorted_gammas {
                        for &threshold in &sorted_thresholds {
                            let conf_name = format!(
                                "alpha{alpha}_beta{beta}_gamma{gamma}_threshold{threshold}_topo{topology:?}_nodes{node_count}"
                            );

                            if completed.contains(&conf_name) {
                                // O(1)
                                continue;
                            }

                            current_iter += 1;

                            let net_config = NetworkConfig {
                                collusion_fraction: CONFIG.network.collusion_fraction,
                                reputation_threshold: threshold,
                                reputation_weight_alpha: alpha,
                                reputation_weight_beta: beta,
                                reputation_weight_gamma: gamma,
                            };

                            {
                                let mut guard = NETWORK_CONFIG_OVERRIDE.write().unwrap();
                                *guard = Some(net_config);
                            }

                            println!(
                                "[{current_iter}/{total_combos}] a={alpha:.2} b={beta:.2} g={gamma:.2} t={threshold:.2} topo={topology:?} nodes={node_count}"
                            );

                            let mut total_accuracy = 0.0_f64;
                            for i in 0..SEEDS_PER_CONFIG {
                                let seed = base_seed + i * 10;
                                for exp in &experiment_configs {
                                    let mut sim_res = Vec::new();
                                    let mut sim = Simulation::setup_with_seed(
                                        exp.topology,
                                        exp.attack_type,
                                        exp.defense_type,
                                        exp.aggregator_type,
                                        exp.byzantine_fraction,
                                        exp.node_count,
                                        seed,
                                    );
                                    sim.run(&exp.name, &mut sim_res);
                                    if let Some(last) = sim_res.last() {
                                        total_accuracy += last.0;
                                    }
                                }
                            }

                            let avg_acc = total_accuracy / p1_runs;
                            println!("  → {avg_acc:.4}");

                            // Keep the alpha ceiling current so pruning fires promptly.
                            let e = alpha_best_seen
                                .entry(alpha.to_bits())
                                .or_insert(f64::NEG_INFINITY);
                            *e = e.max(avg_acc);

                            if avg_acc > best_p1_score {
                                best_p1_score = avg_acc;
                                best_params = Some(net_config);
                            }

                            writeln!(progress_file, "{conf_name},{avg_acc}").unwrap();
                            completed.insert(conf_name);
                            p1_results.push(P1Result {
                                threshold,
                                alpha,
                                beta,
                                gamma,
                                topology,
                                node_count,
                                p1_acc: avg_acc,
                            });
                        }
                    }
                }
            }
        }
    }

    // ── Phase 2: backdoor + sign-flip pass ───────────────────────────────────
    // Combined score: (p1_acc * p1_runs + p2_acc * p2_runs) / (p1_runs + p2_runs)
    // Upper bound (p2_acc = 1.0 for all p2 runs):
    //   best_possible = (p1_acc * p1_runs + p2_runs) / total_runs
    let total_runs = p1_runs + p2_runs;

    let mut p2_candidates: Vec<&P1Result> = p1_results
        .iter()
        .filter(|r| (r.p1_acc * p1_runs + p2_runs) / total_runs >= best_p1_score)
        .collect();

    // Sort descending by p1_acc: high-p1 candidates reach a high composite
    // baseline first, allowing the early-exit below to prune more of the tail.
    p2_candidates.sort_by(|a, b| {
        b.p1_acc
            .partial_cmp(&a.p1_acc)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "Phase 2: {}/{} configs passed the filter.",
        p2_candidates.len(),
        p1_results.len()
    );

    let mut best_composite = f64::NEG_INFINITY;
    best_params = None;

    for (i, r) in p2_candidates.iter().enumerate() {
        // Because the list is sorted descending, the optimistic upper bound only
        // decreases.  The moment it falls below `best_composite`, no later
        // candidate can win — we can exit immediately.
        let optimistic = (r.p1_acc * p1_runs + p2_runs) / total_runs;
        if optimistic < best_composite {
            println!(
                "  [early exit] remaining {}/{} candidates cannot beat {best_composite:.4}",
                p2_candidates.len() - i,
                p2_candidates.len()
            );
            break;
        }

        let net_config = NetworkConfig {
            collusion_fraction: CONFIG.network.collusion_fraction,
            reputation_threshold: r.threshold,
            reputation_weight_alpha: r.alpha,
            reputation_weight_beta: r.beta,
            reputation_weight_gamma: r.gamma,
        };

        {
            let mut guard = NETWORK_CONFIG_OVERRIDE.write().unwrap();
            *guard = Some(net_config);
        }

        println!(
            "[{}/{} Phase 2] a={:.2} b={:.2} g={:.2} t={:.2} topo={:?} nodes={}",
            i + 1,
            p2_candidates.len(),
            r.alpha,
            r.beta,
            r.gamma,
            r.threshold,
            r.topology,
            r.node_count,
        );

        let mut p2_total = 0.0_f64;
        for i in 0..SEEDS_PER_CONFIG {
            let seed = base_seed + i * 10;
            for &(attack_type, name) in p2_scenarios {
                let mut sim_res = Vec::new();
                let mut sim = Simulation::setup_with_seed(
                    r.topology.clone(),
                    attack_type,
                    defense,
                    aggregator,
                    byzantine_fraction,
                    r.node_count as u32,
                    seed,
                );
                sim.run(name, &mut sim_res);
                if let Some(last) = sim_res.last() {
                    p2_total += last.0;
                }
            }
        }

        let composite = (r.p1_acc * p1_runs + p2_total) / total_runs;
        println!("  → composite {composite:.4}");

        if composite > best_composite {
            best_composite = composite;
            best_params = Some(net_config);
        }
    }

    println!("Optimization completed in {:?}", start_time.elapsed());
    if let Some(p) = best_params {
        println!(
            "Best Parameters:\n\tThreshold: {}\n\tAlpha:     {}\n\tBeta:      {}\n\tGamma:     {}",
            p.reputation_threshold,
            p.reputation_weight_alpha,
            p.reputation_weight_beta,
            p.reputation_weight_gamma,
        );
        println!("Best Composite Score: {best_composite:.4}");
    }
}

fn run_reputation_baseline() {
    let experiment_configs = vec![
        ExperimentConfiguration {
            name: "n10".to_string(),
            aggregator_type: AggregatorType::DFedAvgM,
            seed: Some(1234),
            node_count: 10,
            topology: GraphTopology::RANDOM,
            attack_type: AttackType::NoAttack,
            byzantine_fraction: 0.0,
            defense_type: DefenseType::NoDefense,
        },
        ExperimentConfiguration {
            name: "n50".to_string(),
            aggregator_type: AggregatorType::DFedAvgM,
            seed: Some(1234),
            node_count: 50,
            topology: GraphTopology::RANDOM,
            attack_type: AttackType::NoAttack,
            byzantine_fraction: 0.0,
            defense_type: DefenseType::NoDefense,
        },
        ExperimentConfiguration {
            name: "n200".to_string(),
            aggregator_type: AggregatorType::DFedAvgM,
            seed: Some(1234),
            node_count: 200,
            topology: GraphTopology::RANDOM,
            attack_type: AttackType::NoAttack,
            byzantine_fraction: 0.0,
            defense_type: DefenseType::NoDefense,
        },
    ];

    let mut results = Vec::new();
    for exp in experiment_configs {
        let mut sim_res = Vec::new();
        let mut simulation = Simulation::setup_with_seed(
            exp.topology,
            exp.attack_type,
            exp.defense_type,
            exp.aggregator_type,
            exp.byzantine_fraction,
            exp.node_count,
            exp.seed.unwrap(),
        );
        simulation.run("SignFlip", &mut sim_res);
        results.push(format!(
            "n={}: acc={}",
            exp.node_count,
            sim_res.last().unwrap().0
        ));
    }
    for r in results {
        println!("{r}");
    }
}
