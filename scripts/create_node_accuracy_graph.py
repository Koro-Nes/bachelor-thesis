import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

ROUND_REGEX = re.compile(r"Round (\d+):")
ACCURACY_REGEX = re.compile(r"Accuracy:\s*([0-9.]+)")

def parse_file(filepath):
    """Parse a node file and return {round_number: accuracy}"""
    round_accuracies = {}
    current_round = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            round_match = ROUND_REGEX.match(line)
            if round_match:
                current_round = int(round_match.group(1))
                continue

            if current_round is not None:
                acc_match = ACCURACY_REGEX.match(line)
                if acc_match:
                    acc = float(acc_match.group(1))
                    round_accuracies[current_round] = acc
                    current_round = None
    return round_accuracies

def collect_node_files(base_folder):
    """
    Recursively collect all node files under base_folder/n*/nodes_seed*/node_*.txt
    """
    node_files = []

    # Iterate over n<config> folders
    for config_name in os.listdir(base_folder):
        config_path = os.path.join(base_folder, config_name)
        if not os.path.isdir(config_path) or not config_name.startswith("n"):
            continue

        # Iterate over nodes_seed* folders
        for seed_name in os.listdir(config_path):
            seed_path = os.path.join(config_path, seed_name)
            if not os.path.isdir(seed_path) or not seed_name.startswith("nodes_seed"):
                continue

            # Collect all node_*.txt files
            for file_name in os.listdir(seed_path):
                if file_name.startswith("node_") and file_name.endswith(".txt"):
                    node_files.append(os.path.join(seed_path, file_name))

    return node_files

def aggregate(node_files):
    """Return dict {round: [acc1, acc2, ...]} and list of per-node accuracies"""
    data = defaultdict(list)
    node_accuracies = []

    for filepath in node_files:
        rounds = parse_file(filepath)
        node_accuracies.append(rounds)
        for r, acc in rounds.items():
            data[r].append(acc)

    return data, node_accuracies

def plot_all_nodes(node_accuracies, aggregated):
    rounds = sorted(aggregated.keys())
    all_acc_matrix = []

    # Build matrix: rows=nodes, cols=rounds
    for node in node_accuracies:
        acc_row = [node.get(r, np.nan) for r in rounds]
        all_acc_matrix.append(acc_row)
    all_acc_matrix = np.array(all_acc_matrix)

    avg = np.nanmean(all_acc_matrix, axis=0)
    min_vals = np.nanmin(all_acc_matrix, axis=0)
    max_vals = np.nanmax(all_acc_matrix, axis=0)
    p25 = np.nanpercentile(all_acc_matrix, 25, axis=0)
    p75 = np.nanpercentile(all_acc_matrix, 75, axis=0)

    plt.figure(figsize=(12, 6))

    # Plot all nodes in thin transparent lines
    for i in range(all_acc_matrix.shape[0]):
        plt.plot(rounds, all_acc_matrix[i], color="blue", alpha=0.2, linewidth=0.8)

    # Overlay percentiles and average
    plt.fill_between(rounds, p25, p75, color="orange", alpha=0.3, label="25-75 percentile")
    plt.plot(rounds, avg, color="red", linewidth=2, label="Average Accuracy")
    plt.plot(rounds, min_vals, linestyle="--", color="gray", linewidth=1, label="Min/Max Accuracy")
    plt.plot(rounds, max_vals, linestyle="--", color="gray", linewidth=1)

    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across All Nodes Over Rounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot all nodes accuracy")
    parser.add_argument("input_folder", help="Base input folder (logs/)")
    args = parser.parse_args()

    node_files = collect_node_files(args.input_folder)
    if not node_files:
        print("No node files found. Check folder structure.")
        return

    print(f"Found {len(node_files)} node files.")
    aggregated, node_accuracies = aggregate(node_files)
    plot_all_nodes(node_accuracies, aggregated)

if __name__ == "__main__":
    main()