use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::OnceLock,
};

use crate::{
    config::config::ExperimentConfiguration,
    node::{stats::GlobalStats, visualization::Visualization},
};

const DEFAULT_RESULTS_PATH: &str = "./logs";

static RESULTS_BASE_PATH: OnceLock<String> = OnceLock::new();
static ROOT_FOLDER_NAME: OnceLock<String> = OnceLock::new();
static GRID_SEARCH_FILE_NAME: OnceLock<String> = OnceLock::new();

pub fn create_experiment_folders(
    path: &str,
    configurations: Vec<ExperimentConfiguration>,
) -> String {
    let base_path = RESULTS_BASE_PATH.get_or_init(|| path.to_string()).clone();
    let root_folder_name = ROOT_FOLDER_NAME
        .get_or_init(|| {
            let now = chrono::offset::Local::now();
            now.format("%Y%m%d_%H-%M-%S").to_string()
        })
        .clone();

    let root_path = Path::new(&base_path).join(&root_folder_name);
    let logs_folder = root_path.join("logs");
    let mut dir_builder = fs::DirBuilder::new();
    dir_builder.recursive(true).create(&logs_folder).unwrap();

    for c in configurations {
        let node_folder = root_path.join(&c.name).join("nodes");
        dir_builder.create(node_folder).unwrap();
    }

    root_folder_name
}

pub fn export_experiment_results(stats: GlobalStats) {
    let base_path = results_base_path();
    let root_folder_name = ensure_root_folder();
    let root_path = Path::new(&base_path).join(&root_folder_name);
    let seed_suffix = format!("seed{}", stats.seed);
    let experiment_root = root_path.join(&stats.experiment_name);
    let node_folder = experiment_root.join(format!("nodes_{}", seed_suffix));

    let mut dir_builder = fs::DirBuilder::new();
    dir_builder.recursive(true).create(&node_folder).unwrap();

    let global_file_name = format!("global_stats_{}_{}.txt", stats.experiment_name, seed_suffix);
    let global_file_path = experiment_root.join(global_file_name);
    fs::write(global_file_path, stats.to_human_readable()).unwrap();

    for node in &stats.node_stats {
        let node_file_name = format!("node_{:03}.txt", node.id);
        let node_file_path = node_folder.join(node_file_name);
        fs::write(node_file_path, node.to_human_readable()).unwrap();
    }
}

pub fn export_graph_to_json(file_name: &str, graph: Visualization) {
    let base_path = results_base_path();
    let root_folder_name = ensure_root_folder();

    let root_path = Path::new(&base_path).join(&root_folder_name);
    let json_folder = root_path.join("graph_jsons");

    let mut dir_builder = fs::DirBuilder::new();
    dir_builder.recursive(true).create(&json_folder).unwrap();

    let file_path = json_folder.join(format!("{}.json", file_name));

    let json = serde_json::to_string_pretty(&graph.nodes).unwrap();
    fs::write(file_path, json).unwrap();
}

fn results_base_path() -> String {
    RESULTS_BASE_PATH
        .get_or_init(|| DEFAULT_RESULTS_PATH.to_string())
        .clone()
}

fn ensure_root_folder() -> String {
    ROOT_FOLDER_NAME
        .get_or_init(|| {
            let now = chrono::offset::Local::now();
            let root_folder_name = now.format("%Y%m%d_%H-%M-%S").to_string();
            let base_path = results_base_path();
            let logs_folder = PathBuf::from(&base_path)
                .join(&root_folder_name)
                .join("logs");
            let mut dir_builder = fs::DirBuilder::new();
            dir_builder.recursive(true).create(logs_folder).unwrap();
            root_folder_name
        })
        .clone()
}

pub fn init_grid_search_log() -> File {
    let path = format!("{}", results_base_path());
    let grid_file_path = GRID_SEARCH_FILE_NAME.get_or_init(|| {
        let out_path = format!("{}/grid_search_progress.txt", path);
        out_path
    });
    File::options()
        .read(true)
        .append(true)
        .create(true)
        .open(grid_file_path)
        .unwrap()
}
