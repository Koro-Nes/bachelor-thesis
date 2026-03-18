use repository::{
    attack::attack::AttackType,
    config::config::{CONFIG, EXPERIMENT_CONFIGURATIONS, ExperimentConfiguration, GraphTopology},
    defense::defense::DefenseType,
    ml::aggregator::AggregatorType,
    simulation::simulation::Simulation,
};
use tch::Cuda;

fn main() {
    // io::create_experiment_folders("./results/");

    // let simulation = Simulation::setup(
    //     repository::attack::attack::AttackType::SignFlipping,
    //     repository::defense::defense::DefenseType::NoDefense,
    //     repository::ml::aggregator::AggregatorType::DFedAvgM,
    // );
    //run_test_configurations();
    run_baseline(5);
    //run_small_sample();
    if Cuda::is_available() {
        Cuda::synchronize(0);
    }
}

fn metrics_from_sim(results: Vec<Vec<(f64, f64, f64, u128)>>) {
    for (id, round) in results.iter().enumerate() {
        let c = EXPERIMENT_CONFIGURATIONS[id].clone();
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
    let configs = &EXPERIMENT_CONFIGURATIONS
        .to_vec()[0..3];
    let mut results = Vec::new();
    for c in configs {
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
    metrics_from_sim(results);
}

fn run_small_sample() {
    let configs = vec![
        //EXPERIMENT_CONFIGURATIONS[38].clone(),
        //EXPERIMENT_CONFIGURATIONS[24].clone(),
        //EXPERIMENT_CONFIGURATIONS[28].clone(),
        EXPERIMENT_CONFIGURATIONS[132].clone()
    ];
    let mut results = Vec::new();
    for c in configs {
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
    metrics_from_sim(results);
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
