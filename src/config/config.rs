use std::fmt::Display;

use once_cell::sync::Lazy;
use tch::Device;

use crate::{
    args::args::Args,
    attack::attack::AttackType,
    defense::defense::DefenseType,
    logging::colors::{BLUE, CYAN, GREEN, MAGENTA, RED, RESET, WHITE, YELLOW},
    ml::aggregator::AggregatorType,
};

#[derive(Debug)]
pub struct Config {
    pub verbose: bool,
    pub ignore_skip: bool,

    pub seed: usize,

    pub device: Device,

    pub network: NetworkConfig,

    pub data: DataConfig,
    pub training: TrainingConfig,
    pub metrics: MetricsConfig,

    pub dfedavgm: DFedAvgMConfig,
    pub clippedmean: ClippedMeanConfig,
    pub balance: BalanceConfig,

    pub labelflipping: LabelFlippingConfig,
    pub backdoor: BackdoorTriggerConfig,
}

impl Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let basic_info = format!(
            "{}-VERBOSE: {}\n-SEED: {}\n-DEVICE: {:?}{}",
            WHITE, self.verbose, self.seed, self.device, RESET
        );

        f.write_fmt(format_args!(
            "{basic_info}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n",
            self.network,
            self.data,
            self.training,
            self.metrics,
            self.dfedavgm,
            self.clippedmean,
            self.balance,
            self.labelflipping,
            self.backdoor
        ))
    }
}

#[derive(Debug)]
pub struct MetricsConfig {
    pub estimate_computational_cost: bool,
    pub robust_baseline_accuracy: f64,
}

impl Display for MetricsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}Metrics Configuration:{}\n\tEstimate Computational Cost: {}\n\tRobust Baseline Accuracy: {:<.2}{}",
            WHITE, YELLOW, self.estimate_computational_cost, self.robust_baseline_accuracy, RESET
        ))
    }
}

#[derive(Debug)]
pub struct TrainingConfig {
    pub communication_rounds: u32,
    pub local_epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f32,
    pub test_ratio: f32,
    pub loop_test_size: u32,
    pub final_test_size: u32,
}

impl Display for TrainingConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}Training Configuration:{}\n\tCommunication Rounds: {}\n\tLocal Epochs: {}\n\tBatch Size: {}\n\tLearning Rate: {:<.2}\n\tTest Ratio: {:<.2}{}", 
            WHITE, BLUE, self.communication_rounds, self.local_epochs, self.batch_size, self.learning_rate, self.test_ratio, RESET))
    }
}

#[derive(Debug)]
pub struct DataConfig {
    pub path: String,
    pub dirichlet_alpha: f32,
}

impl Display for DataConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}Data Configuration:\n\t{}Data Path: {}\n\tDirichlet Alpha: {:<.2}{}",
            WHITE, MAGENTA, self.path, self.dirichlet_alpha, RESET
        ))
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GraphTopology {
    RING,
    RANDOM,
}

impl GraphTopology {
    pub fn name_str(&self) -> &str {
        match self {
            GraphTopology::RING => "ring",
            GraphTopology::RANDOM => "random",
        }
    }
}

impl Display for GraphTopology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphTopology::RING => f.write_str("Ring Toplogy"),
            GraphTopology::RANDOM => f.write_str("Random Topology"),
        }
    }
}

#[derive(Debug)]
pub struct NetworkConfig {
    pub collusion_fraction: f32,
    pub reputation_threshold: f32,
    pub reputation_weight_alpha: f32,
    pub reputation_weight_beta: f32,
    pub reputation_weight_gamma: f32,
}

impl Display for NetworkConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}Network Configuration:\n\t{}Malicious Config:\n\t\tCollusion Fraction: {:<.2}{}\n\t{}Reputation Config:\n\t\tThreshold: {:<.2}\n\t\tAlpha: {:<.2}\n\t\tBeta: {:<.2}\n\t\tGamma: {:<.2}{}",
            WHITE, RED, self.collusion_fraction, RESET, GREEN, self.reputation_threshold, self.reputation_weight_alpha, self.reputation_weight_beta, self.reputation_weight_gamma, RESET
        ))
    }
}

#[derive(Debug)]
pub struct DFedAvgMConfig {
    pub beta: f64,
}

impl Display for DFedAvgMConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}DFedAvgM\n\t{}Beta: {:<.2}{}",
            WHITE, CYAN, self.beta, RESET,
        ))
    }
}

#[derive(Debug)]
pub struct ClippedMeanConfig {
    pub beta: f64,
}

impl Display for ClippedMeanConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}Clipped Mean{}\n\tBeta: {:<.2}{}",
            WHITE, CYAN, self.beta, RESET,
        ))
    }
}

#[derive(Debug)]
#[allow(non_snake_case)]
pub struct BalanceConfig {
    pub alpha: f64,
    pub gamma: f64,
    pub kappa: f64,
    pub T: u64,
}

impl Display for BalanceConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}BALANCE{}\n\tAlpha: {:<.2}\n\tGamma: {:<.2}\n\tKappa: {:<.2}{}",
            WHITE, CYAN, self.alpha, self.gamma, self.kappa, RESET,
        ))
    }
}

#[derive(Debug)]
pub struct LabelFlippingConfig {
    pub to: Vec<u64>,
    pub from: Vec<u64>,
    pub flip_fraction: f64,
}

impl Display for LabelFlippingConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}Attack Strategy: Label Flipping{}\n\tTo: {:?}\n\tFrom: {:?}\n\tFlip Fraction: {:<.2}{}",
            WHITE, RED, self.to, self.from, self.flip_fraction, RESET
        ))
    }
}

#[derive(Debug)]
pub struct BackdoorTriggerConfig {
    pub target_label: i64,
    pub poison_fraction: f64,
    pub trigger_size: i64,
    pub trigger_value: f64,
}

impl Display for BackdoorTriggerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}Attack Strategy: Backdoor Trigger{}\n\tTarget Label: {}\n\tPoison Fraction: {:<.2}\n\tTrigger Size: {}\n\tTrigger Value: {:<.2}{}",
            WHITE, RED, self.target_label, self.poison_fraction, self.trigger_size, self.trigger_value, RESET
        ))
    }
}

#[derive(Clone)]
pub struct ExperimentConfiguration {
    pub name: String,
    pub seed: Option<usize>,
    pub node_count: u32,
    pub topology: GraphTopology,
    pub attack_type: AttackType,
    pub byzantine_fraction: f32,
    pub defense_type: DefenseType,
    pub aggregator_type: AggregatorType,
}

impl Display for ExperimentConfiguration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let seed_str = self
            .seed
            .map(|s| s.to_string())
            .unwrap_or_else(|| "default".to_string());
        f.write_fmt(format_args!(
            "{}\n\tSeed: {}\n\tN: {}\n\tTopology: {}\n\tAttack: {}, Byzantine Fraction: {}\n\tDefense: {}\n\tAggregator: {}",
            self.name,
            seed_str,
            self.node_count,
            self.topology,
            self.attack_type,
            self.byzantine_fraction,
            self.defense_type,
            self.aggregator_type
        ))
    }
}

impl ExperimentConfiguration {
    pub fn new(
        name: String,
        node_count: u32,
        topology: GraphTopology,
        attack_type: AttackType,
        byzantine_fraction: f32,
        defense_type: DefenseType,
        aggregator_type: AggregatorType,
    ) -> Self {
        Self {
            name,
            seed: None,
            node_count,
            topology,
            attack_type,
            byzantine_fraction,
            defense_type,
            aggregator_type,
        }
    }

    pub fn with_seed(mut self, seed: usize) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn seed_or(&self, default_seed: usize) -> usize {
        self.seed.unwrap_or(default_seed)
    }
}

pub static EXPERIMENT_CONFIGURATIONS: Lazy<Vec<ExperimentConfiguration>> = Lazy::new(|| {
    let node_counts = vec![10, 50, 200];
    let topologies = vec![GraphTopology::RING, GraphTopology::RANDOM];
    let attack_types = vec![
        AttackType::LabelFlipping,
        AttackType::SignFlipping,
        AttackType::BackdoorTrigger,
    ];
    let byzantine_fractions = vec![0.1, 0.2, 0.3];
    let defensive_mechanisms = vec![DefenseType::NoDefense, DefenseType::Reputation];
    let no_defense_aggregators = vec![AggregatorType::Balance, AggregatorType::ClippedMean];
    let reputation_aggregators = vec![AggregatorType::DFedAvgM];

    let mut res = Vec::new();

    for n in &node_counts {
        for t in &topologies {
            for a in &attack_types {
                for b in &byzantine_fractions {
                    for d in &defensive_mechanisms {
                        match d {
                            DefenseType::NoDefense => {
                                for agg in &no_defense_aggregators {
                                    res.push(ExperimentConfiguration::new(
                                        format!(
                                            "n{}_{}_{}_bf{}_{}_{}",
                                            n,
                                            t.name_str(),
                                            a,
                                            b,
                                            d,
                                            agg
                                        ),
                                        *n,
                                        *t,
                                        *a,
                                        *b,
                                        *d,
                                        *agg,
                                    ))
                                }
                            }
                            DefenseType::Reputation => {
                                for agg in &reputation_aggregators {
                                    res.push(ExperimentConfiguration::new(
                                        format!(
                                            "n{}_{}_{}_bf{}_{}_{}",
                                            n,
                                            t.name_str(),
                                            a,
                                            b,
                                            d,
                                            agg
                                        ),
                                        *n,
                                        *t,
                                        *a,
                                        *b,
                                        *d,
                                        *agg,
                                    ))
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    res
});

pub static TEST_CONFIGURATIONS: Lazy<Vec<ExperimentConfiguration>> = Lazy::new(|| {
    vec![ExperimentConfiguration::new(
        "test".to_string(),
        10,
        GraphTopology::RANDOM,
        AttackType::LabelFlipping,
        0.2,
        DefenseType::NoDefense,
        AggregatorType::ClippedMean,
    )]
});

pub static CONFIG: Lazy<Config> = Lazy::new(|| {
    let args = Args::get();

    let verbose = args.verbose;

    let seed = args.seed;

    let network = NetworkConfig {
        collusion_fraction: args.collusion_fraction,
        reputation_threshold: args.reputation_threshold,
        reputation_weight_alpha: args.reputation_weight_alpha,
        reputation_weight_beta: args.reputation_weight_beta,
        reputation_weight_gamma: args.reputation_weight_gamma,
    };

    let data = DataConfig {
        dirichlet_alpha: args.dirichlet_alpha,
        path: args.data_path,
    };

    let training = TrainingConfig {
        communication_rounds: args.communication_rounds,
        local_epochs: args.local_epochs,
        batch_size: args.batch_size,
        learning_rate: args.learning_rate,
        test_ratio: 0.2,
        loop_test_size: 500,
        final_test_size: 2000,
    };

    let mfedavgd = DFedAvgMConfig {
        beta: 0.9,
    };

    let clippedmean = ClippedMeanConfig {
        beta: args.clipped_mean_beta as f64,
    };

    let balance = BalanceConfig {
        alpha: 0.5,
        gamma: 0.1,
        kappa: 0.5,
        T: args.communication_rounds as u64,
    };

    let labelflipping = LabelFlippingConfig {
        from: vec![0, 3, 5, 8],
        to: vec![5, 9, 1, 3],
        flip_fraction: 0.2,
    };

    let backdoor = BackdoorTriggerConfig {
        poison_fraction: 0.1,
        target_label: 3,
        trigger_size: 3,
        trigger_value: 1.0, //white
    };

    let metrics = MetricsConfig {
        estimate_computational_cost: args.estimate_computational_cost,
        robust_baseline_accuracy: args.robust_baseline_accuracy,
    };

    let device = if tch::Cuda::is_available() {
        println!("{}CUDA/ROCm is available. Using GPU.{}", GREEN, RESET);
        Device::Cuda(0)
    } else {
        println!("count: {}", tch::Cuda::device_count());
        println!("{}", tch::Cuda::cudnn_is_available());

        println!("{}CUDA/ROCm is not available. Using CPU.{}", RED, RESET);
        Device::Cpu
    };

    let res = Config {
        ignore_skip: args.ignore_skip,
        verbose,
        seed,
        device,
        network,
        data,
        training,
        metrics,
        dfedavgm: mfedavgd,
        clippedmean,
        balance,
        labelflipping,
        backdoor,
    };
    println!("Started with config:\n{}", res);
    res
});
