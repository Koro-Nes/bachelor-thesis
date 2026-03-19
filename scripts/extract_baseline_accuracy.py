import os
import re
import argparse


def calculate(baseline_dir):
    avg_values = []

    for root, dirs, files in os.walk(baseline_dir):
        for file in files:
            if file.startswith("global_stats_") and file.endswith(".txt"):
                filepath = os.path.join(root, file)

                with open(filepath, "r") as f:
                    content = f.read()

                match = re.search(r"Avg:\s*([0-9.]+)", content)
                if match:
                    avg_values.append(float(match.group(1)))

    if not avg_values:
        print("No Avg values found.")
    else:
        mean_avg_accuracy = sum(avg_values) / len(avg_values)
        print(f"Files processed: {len(avg_values)}")
        print(f"Mean Average Final Global Accuracy: {mean_avg_accuracy:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Plot all nodes accuracy")
    parser.add_argument("input_folder", help="Base input folder (logs/)")
    args = parser.parse_args()
    file_path = args.input_folder
    calculate(file_path)


if __name__ == "__main__":
    main()