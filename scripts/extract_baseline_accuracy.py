import os
import re

baseline_dir = "./results/baseline"

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