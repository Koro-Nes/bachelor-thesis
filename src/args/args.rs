use std::process::exit;

use clap::Parser;

use crate::config::config::EXPERIMENT_CONFIGURATIONS;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    #[arg(short = 'l', long = "list", default_value_t = false)]
    pub list_configs: bool,

    #[arg(short = 'v', long)]
    pub verbose: bool,

    #[arg(short = 'd', long, default_value_t = String::from("data"))]
    pub data_path: String,

    #[arg(short = 's', long, default_value_t = 123456789)]
    pub seed: usize,

    #[arg(long = "cf", default_value_t = 0.3)]
    pub collusion_fraction: f32,

    /// number of communication rounds with default value 50
    #[arg(long = "cr", default_value_t = 50)]
    pub communication_rounds: u32,

    /// number of local epochs
    #[arg(long = "le", default_value_t = 1)]
    pub local_epochs: u32,

    /// batch size during local training
    #[arg(long = "bs", default_value_t = 32)]
    pub batch_size: u32,

    /// learning rate of the model
    #[arg(long = "lr", default_value_t = 0.01)]
    pub learning_rate: f32,

    /// controls non-IID-ness of data split
    #[arg(long = "da", default_value_t = 0.5)]
    pub dirichlet_alpha: f32,

    /// threshold where reputation <= are ignored
    #[arg(long = "rept", default_value_t = 0.7)]
    pub reputation_threshold: f32,

    /// assumed to be equal to the byzantine fraction; can be changed here
    #[arg(long = "tm", default_value_t = 0.2)]
    pub trimmed_mean_beta: f32,

    /// weight of last round's reputation
    #[arg(long = "rwa", default_value_t = 0.85)]
    pub reputation_weight_alpha: f32,

    /// weight of the indirect reputation (from neighboring nodes)
    #[arg(long = "rwb", default_value_t = 0.75)]
    pub reputation_weight_beta: f32,

    /// reputation decay
    #[arg(long = "rwg", default_value_t = 0.01)]
    pub reputation_weight_gamma: f32,

    /// estimate computational cost
    #[arg(long = "ecc", default_value_t = false)]
    pub estimate_computational_cost: bool,

    /// robust baseline accuracy
    #[arg(long = "rba", default_value_t = 0.939445)]
    pub robust_baseline_accuracy: f64,
}

impl Args {
    pub fn get() -> Self {
        let res = Args::parse();
        if res.list_configs {
            for (id, c) in EXPERIMENT_CONFIGURATIONS.clone().iter().enumerate() {
                println!("index: {id}\n{}\n", c);
            }
            exit(0);
        }

        res
    }
}
