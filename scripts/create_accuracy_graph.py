import os
import re
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict


ROUND_REGEX = re.compile(r"Round (\d+):")
ACCURACY_REGEX = re.compile(r"Accuracy:\s*([0-9.]+)")


def parse_file(filepath):
    """
    Parses a stats file and returns:
    {round_number: accuracy}
    """
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
                acc_match = ACCURACY_REGEX.match(line.strip())
                if acc_match:
                    acc = float(acc_match.group(1))
                    round_accuracies[current_round] = acc
                    current_round = None  # reset until next round block

    return round_accuracies


def collect_node_files(base_path, node_id=None):
    """
    Finds all node files in the structure.
    If node_id is given, only collect that node.
    """
    node_files = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if not file.startswith("node_") or not file.endswith(".txt"):
                continue

            if node_id is not None:
                expected = f"node_{node_id:03d}.txt"
                if file != expected:
                    continue

            node_files.append(os.path.join(root, file))

    return node_files


def aggregate(node_files):
    """
    Aggregates accuracies across nodes:
    returns dict: {round: [acc1, acc2, ...]}
    """
    data = defaultdict(list)

    for filepath in node_files:
        rounds = parse_file(filepath)
        for r, acc in rounds.items():
            data[r].append(acc)

    return data


def compute_stats(aggregated):
    """
    Returns sorted rounds and avg/min/max lists
    """
    rounds = sorted(aggregated.keys())

    avg = []
    min_vals = []
    max_vals = []

    for r in rounds:
        values = aggregated[r]
        avg.append(sum(values) / len(values))
        min_vals.append(min(values))
        max_vals.append(max(values))

    return rounds, avg, min_vals, max_vals


def plot(rounds, avg, min_vals, max_vals, title):
    plt.figure(figsize=(10, 6))

    plt.plot(rounds, avg, label="Average Accuracy")
    plt.plot(rounds, min_vals, linestyle="--", label="Min Accuracy")
    plt.plot(rounds, max_vals, linestyle="--", label="Max Accuracy")

    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy stats")
    parser.add_argument("input_folder", help="Base input folder")
    parser.add_argument("-n", "--node", type=int, help="Node ID (optional)")

    args = parser.parse_args()

    node_files = collect_node_files(args.input_folder, args.node)

    if not node_files:
        print("No matching node files found.")
        return

    aggregated = aggregate(node_files)
    rounds, avg, min_vals, max_vals = compute_stats(aggregated)

    title = (
        f"Node {args.node} Accuracy Over Rounds"
        if args.node is not None
        else "All Nodes Accuracy Over Rounds"
    )

    plot(rounds, avg, min_vals, max_vals, title)


if __name__ == "__main__":
    main()