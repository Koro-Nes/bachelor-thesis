use once_cell::sync::Lazy;

use crate::args::args::Args;

#[derive(Debug)]
pub struct Config {
    pub verbose: bool,

    pub graph: GraphConfig,
    pub network: NetworkConfig,

    pub data: DataConfig,
    pub training: TrainingConfig,

    pub mfedavgd: MFedAvgDConfig,
    pub trimmedmean: TrimmedMeanConfig,
    pub balance: BalanceConfig,

    pub labelflipping: LabelFlippingConfig,
    pub backdoor: BackdoorTriggerConfig,
}

#[derive(Debug)]
pub struct TrainingConfig {
    pub communication_rounds: u32,
    pub local_epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f32,
}

#[derive(Debug)]
pub struct DataConfig {
    pub path: String,
    pub dirichlet_alpha: f32,
}

#[derive(Debug)]
pub enum GraphTopology {
    CIRCLE,
    RANDOM,
}

#[derive(Debug)]
pub struct GraphConfig {
    pub topology: GraphTopology,
    pub n: u32,
}

#[derive(Debug)]
pub struct NetworkConfig {
    pub byzantine_fraction: f32,
    pub reputation_threshold: f32,
    pub reputation_weight_alpha: f32,
    pub reputation_weight_beta: f32,
    pub reputation_weight_gamma: f32,
}

#[derive(Debug)]
pub struct MFedAvgDConfig {
    pub beta: f64,
}

#[derive(Debug)]
pub struct TrimmedMeanConfig {
    pub beta: f64,
}

#[derive(Debug)]
pub struct BalanceConfig {
    pub alpha: f64,
    pub gamma: f64,
    pub kappa: f64,
    pub T: u64,
}

#[derive(Debug)]
pub struct LabelFlippingConfig {
    pub to: Vec<u64>,
    pub from: Vec<u64>,
    pub flip_fraction: f64,
}

#[derive(Debug)]
pub struct BackdoorTriggerConfig {
    pub target_label: i64,
    pub poison_fraction: f64,
    pub trigger_size: i64,
    pub trigger_value: f64,
}

pub static CONFIG: Lazy<Config> = Lazy::new(|| {
    let args = Args::get();

    let verbose = args.verbose;

    let graph = GraphConfig {
        n: args.n,
        topology: match args.topology {
            'c' => GraphTopology::CIRCLE,
            'r' => GraphTopology::RANDOM,
            default => panic!("invalid topology supplied"),
        },
    };

    let network = NetworkConfig {
        byzantine_fraction: args.byzantine_fraction,
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
    };

    let mfedavgd = MFedAvgDConfig { beta: 0.7 };

    let trimmedmean = TrimmedMeanConfig {
        beta: args.trimmed_mean_beta as f64,
    };

    let balance = BalanceConfig {
        alpha: 1.5,
        gamma: 0.75,
        kappa: 1.25,
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

    Config {
        verbose,
        graph,
        network,
        data,
        training,
        mfedavgd,
        trimmedmean,
        balance,
        labelflipping,
        backdoor,
    }
});
