use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::Path,
};

use repository::{
    attack::attack::AttackType, config::config::{CONFIG, EXPERIMENT_CONFIGURATIONS, ExperimentConfiguration, GraphTopology, NETWORK_CONFIG_OVERRIDE, NetworkConfig}, defense::defense::DefenseType, logging::io, ml::aggregator::{AggregatorType, DFedAvgMAggregator}, simulation::{self, simulation::Simulation}
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
    //optimize_reputation();
    //run_baseline(3);
    //run_small_sample();
    run_reputation_baseline();
    if Cuda::is_available() {
        Cuda::synchronize(0);
    }
}

fn metrics_from_sim(
    configs: &[ExperimentConfiguration],
    results: Vec<Vec<(f64, f64, f64, u128)>>,
) {
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


    let alphas = [0.6, 0.7, 0.8, 0.9];
    let betas = [0.7, 0.9];
    let gammas = [0.05, 0.1, 0.2];
    let thresholds = [0.4, 0.45, 0.5];

    const SEEDS_PER_CONFIG: usize = 3;
    let node_count = 20;
    let topology = GraphTopology::RANDOM;
    let defense = DefenseType::Reputation;
    let aggregator = AggregatorType::DFedAvgM;
    let byzantine_fraction = 0.2;
    let base_seed = CONFIG.seed;

    let scenarios = vec![
        (AttackType::LabelFlipping, "LabelFlip"),
        (AttackType::BackdoorTrigger, "Backdoor"),
        (AttackType::NoAttack, "Baseline"),
    ];

    let mut experiment_configs = Vec::new();
    for (attack, name) in &scenarios {
        experiment_configs.push(ExperimentConfiguration::new(
            format!("opt_{}", name),
            node_count,
            topology,
            *attack,
            byzantine_fraction,
            defense,
            aggregator
        ));
    }

    let mut best_score = f64::NEG_INFINITY;
    let mut best_params = None;

    let mut progress_file = io::init_grid_search_log();
    
    let mut buf = String::new();

    progress_file.read_to_string(&mut buf).unwrap();

    let mut already_completed = Vec::new();
    let mut completed_for_filtering = Vec::new();

    for line in buf.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split(',');
        let config = parts.next().unwrap().trim();
        let accuracy: f64 = parts.next().unwrap().trim().parse().unwrap();

        already_completed.push(config);


        let mut config_split = config.split('_');
        let alpha: f32 = config_split.next().unwrap().replace("alpha", "").parse().unwrap();
        let beta: f32 = config_split.next().unwrap().replace("beta", "").parse().unwrap();
        let gamma: f32 = config_split.next().unwrap().replace("gamma", "").parse().unwrap();
        let threshold: f32 = config_split.next().unwrap().replace("threshold", "").parse().unwrap();
        let net_config = NetworkConfig {
            collusion_fraction: CONFIG.network.collusion_fraction,
            reputation_threshold: threshold,
            reputation_weight_alpha: alpha,
            reputation_weight_beta: beta,
            reputation_weight_gamma: gamma,
        };

        completed_for_filtering.push((threshold, alpha, beta, gamma, accuracy));

        if accuracy > best_score {
            best_score = accuracy;

            best_params = Some(net_config);
        }
    }

    let total_iterations = alphas.len() * betas.len() * gammas.len() * thresholds.len();
    println!("Testing {} out of {total_iterations} iterations.", total_iterations - already_completed.len());
    let mut current_iter = already_completed.len();
    let start_time = std::time::Instant::now();

    for &alpha in &alphas {
        for &beta in &betas {
            for &gamma in &gammas {
                for &threshold in &thresholds {

                    let conf_name = format!("alpha{}_beta{}_gamma{}_threshold{}", alpha, beta, gamma, threshold);
                    if already_completed.contains(&conf_name.as_str()) {
                        println!("Skipping config {conf_name}");
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

                    println!("[{}/{}] Testing a={:.2} b={:.2} g={:.2} t={:.2}", 
                           current_iter, total_iterations, alpha, beta, gamma, threshold);

                    let mut total_accuracy = 0.0;
                    for i in 0..SEEDS_PER_CONFIG {
                        let current_seed = base_seed + i * 10;
                        
                        for exp in &experiment_configs {
                            let mut sim_res = Vec::new();
                            let mut simulation = Simulation::setup_with_seed(
                                exp.topology, exp.attack_type, exp.defense_type, exp.aggregator_type,
                                exp.byzantine_fraction, exp.node_count, current_seed,
                            );
                            simulation.run(&exp.name, &mut sim_res);
                            if let Some(last) = sim_res.last() {
                                total_accuracy += last.0;
                            }
                        }
                    }

                    let avg_acc = total_accuracy / (experiment_configs.len() * SEEDS_PER_CONFIG) as f64;
                    println!(" Avg Acc: {:.4}", avg_acc);

                    if avg_acc > best_score {
                        best_score = avg_acc;
                        best_params = Some(net_config);
                    }
                    writeln!(progress_file, "{},{}", conf_name, avg_acc).unwrap();
                    completed_for_filtering.push((threshold, alpha, beta, gamma, avg_acc));
                }
            }
        }
    }

    // lots of configs have no way to beat best score, so filter out before running sign flip configs for performance reasons
    let successful_configs: Vec<&(f32, f32, f32, f32, f64)> = completed_for_filtering
        .iter()
        .filter(|(_, _, _, _, acc)| {
            let current_runs = (experiment_configs.len() * SEEDS_PER_CONFIG) as f64;
            let extra_runs = SEEDS_PER_CONFIG as f64;

            let best_possible_avg =
                (acc * current_runs + extra_runs) / (current_runs + extra_runs);

            best_possible_avg >= best_score
        })
    .collect();

    println!("Found {} configs that pass the threshold", successful_configs.len());

    current_iter = 0;

    best_score = f64::NEG_INFINITY;
    best_params = None;

    for (threshold, alpha, beta, gamma, old_acc) in &successful_configs {
        current_iter += 1;

        let net_config = NetworkConfig {
            collusion_fraction: CONFIG.network.collusion_fraction,
            reputation_threshold: *threshold,
            reputation_weight_alpha: *alpha,
            reputation_weight_beta: *beta,
            reputation_weight_gamma: *gamma,
        };

        {
            let mut guard = NETWORK_CONFIG_OVERRIDE.write().unwrap();
            *guard = Some(net_config);
        }


        println!("[{}/{}] Testing a={:.2} b={:.2} g={:.2} t={:.2}", 
                current_iter, successful_configs.len(), alpha, beta, gamma, threshold);

        let mut total_accuracy = 0.0;
        for i in 0..SEEDS_PER_CONFIG {
            let current_seed = base_seed + i * 10;
            
            let mut sim_res = Vec::new();
            let mut simulation = Simulation::setup_with_seed(
                topology, AttackType::SignFlipping, defense, aggregator,
                byzantine_fraction, node_count, current_seed,
            );
            simulation.run("SignFlip", &mut sim_res);
            if let Some(last) = sim_res.last() {
                total_accuracy += last.0;
            }
        }

        let old_runs = (experiment_configs.len() * SEEDS_PER_CONFIG) as f64;
        let extra_runs = SEEDS_PER_CONFIG as f64;

        let avg_acc =(old_acc * old_runs + total_accuracy) / (old_runs + extra_runs);

        println!(" Avg Acc: {:.4}", avg_acc);

        if avg_acc > best_score {
            best_score = avg_acc;
            best_params = Some(net_config);
        }
    }

    println!("Optimization Completed in {:?}", start_time.elapsed());
    if let Some(p) = best_params {
        println!("Best Parameters:\n\tThreshold: {}\n\tAlpha: {}\n\tBeta: {}\n\tGamma: {}", 
            p.reputation_threshold, p.reputation_weight_alpha, p.reputation_weight_beta, p.reputation_weight_gamma);
        println!("Best Score: {:.4}", best_score);
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
            exp.topology, exp.attack_type, exp.defense_type, exp.aggregator_type,
            exp.byzantine_fraction, exp.node_count, exp.seed.unwrap(),
        );
        simulation.run("SignFlip", &mut sim_res);
        results.push(format!("n={}: acc={}", exp.node_count, sim_res.last().unwrap().0));
    }
    for r in results {
        println!("{r}");
    }
}
