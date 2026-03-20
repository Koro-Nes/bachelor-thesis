#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


CONFIG_PATTERN = re.compile(
    r"^n(?P<n>\d+)_(?P<topology>[^_]+)_(?P<attack>[^_]+)_bf(?P<bf>\d+\.\d+)_(?P<defense>[^_]+)_(?P<aggregator>[^_]+)$"
)

ROUND_PATTERN = re.compile(r"Round (\d+):")
ACC_PATTERN = re.compile(r"Accuracy:\s*([0-9.]+)")


# -------------------------------------------------------
# Read accuracy per round for ONE node
# -------------------------------------------------------

def read_node_file(path: Path):
    accuracies = []
    current_round = None

    for line in path.read_text().splitlines():

        r = ROUND_PATTERN.search(line)
        if r:
            current_round = int(r.group(1))
            continue

        a = ACC_PATTERN.search(line)
        if a and current_round is not None:
            accuracies.append(float(a.group(1)))

    return accuracies


# -------------------------------------------------------
# Read all nodes of one config
# -------------------------------------------------------

def read_config_round_stats(config_folder: Path):

    node_files = list(config_folder.rglob("node_*.txt"))

    all_nodes = []

    for f in node_files:
        acc = read_node_file(f)
        if acc:
            all_nodes.append(acc)

    # Convert to numpy array (nodes x rounds)
    arr = np.array(all_nodes)

    median = np.median(arr, axis=0)
    minimum = np.min(arr, axis=0)
    maximum = np.max(arr, axis=0)

    return median, minimum, maximum


# -------------------------------------------------------
# Load all configs
# -------------------------------------------------------

def load_all_configs(base_folder: Path):

    experiments = {}

    for cfg in base_folder.iterdir():

        if not cfg.is_dir():
            continue

        m = CONFIG_PATTERN.match(cfg.name)
        if not m:
            continue

        p = m.groupdict()

        key = (
            p["n"],
            p["topology"],
            p["attack"],
            p["bf"]
        )

        method = f'{p["defense"]}+{p["aggregator"]}'

        median, minimum, maximum = read_config_round_stats(cfg)

        experiments.setdefault(key, {})[method] = {
            "median": median,
            "min": minimum,
            "max": maximum
        }

    return experiments


# -------------------------------------------------------
# Plot
# -------------------------------------------------------

def plot_all(experiments, out_folder: Path):

    out_folder.mkdir(parents=True, exist_ok=True)

    for key, methods in experiments.items():

        n, topology, attack, bf = key

        plt.figure(figsize=(10, 6))

        for method, stats in methods.items():

            rounds = range(len(stats["median"]))

            plt.plot(rounds, stats["median"], label=method)
            plt.fill_between(rounds, stats["min"], stats["max"], alpha=0.2)

        plt.title(f"n={n}, topology={topology}, attack={attack}, bf={bf}")
        plt.xlabel("Round")
        plt.ylabel("Node Accuracy")
        plt.legend()
        plt.grid(True)

        filename = f"n{n}_{topology}_{attack}_bf{bf}.png"
        plt.savefig(out_folder / filename, dpi=200)
        plt.close()


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-folder", required=True, type=Path)
    parser.add_argument("--out-folder", default="plots", type=Path)

    args = parser.parse_args()

    print("Reading experiments...")
    experiments = load_all_configs(args.base_folder)

    print(f"Found {len(experiments)} base configs")

    plot_all(experiments, args.out_folder)

    print("Done.")


if __name__ == "__main__":
    main()