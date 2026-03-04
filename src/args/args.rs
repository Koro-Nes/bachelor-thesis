use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// either 'c' for circle topology or 'r' for random topology
    #[arg(short = 't', long)]
    pub topology: char,

    /// number of nodes in the system
    #[arg(short = 'n')]
    pub n: u32,

    /// byzantine fraction (e.g, 0.3 for 30%)
    #[arg(long = "bf")]
    pub byzantine_fraction: f32,

    /// number of communication rounds with default value 50
    #[arg(long = "cr", default_value_t = 50)]
    pub communication_rounds: u32,

    /// number of local epochs
    #[arg(long = "le", default_value_t = 1)]
    pub local_epochs: u32,

    /// batch size during local training
    #[arg(long = "bs", default_value_t = 1)]
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
    #[arg(long = "tm", default_value_t = -1.0)]
    pub trimmed_mean_beta:f32,

    /// weight of last round's reputation
    #[arg(long = "rwa", default_value_t = 0.85)]
    pub reputation_weight_alpha: f32,

    /// weight of the indirect reputation (from neighboring nodes)
    #[arg(long = "rwb", default_value_t = 0.75)]
    pub reputation_weight_beta: f32,

    /// reputation decay
    #[arg(long = "rwg", default_value_t = 0.01)]
    pub reputation_weight_gamma: f32,
}

impl Args {
    pub fn get() -> Self {
        let mut res = Args::parse();

        if res.trimmed_mean_beta == -1.0 {
            res.trimmed_mean_beta = res.byzantine_fraction;
        }
        res
    }
}
