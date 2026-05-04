#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Optional


DEFAULT_MARGIN_ERROR = 0.02
DEFAULT_Z_0975 = 1.96
DEFAULT_FAMILY_FIELDS = ("network_node_count", "topology", "attack")
DEFAULT_METHOD_FIELDS = ("defense", "aggregator")
CONFIG_FIELDS = (
    "network_node_count",
    "topology",
    "attack",
    "byzantine_fraction",
    "defense",
    "aggregator",
)

CONFIG_PATTERN = re.compile(
    r"^n(?P<node_count>\d+)_(?P<topology>[^_]+)_(?P<attack>[^_]+)_"
    r"bf(?P<byzantine_fraction>\d+(?:\.\d+)?)_(?P<defense>[^_]+)_(?P<aggregator>[^_]+)$"
)
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
BLOCKING_PATTERN = re.compile(
    rf"Blocking:\s*srba=(?P<srba>{FLOAT_PATTERN}),\s*"
    rf"frba=(?P<frba>{FLOAT_PATTERN}),\s*fpbr=(?P<fpbr>{FLOAT_PATTERN})"
)
NEIGHBOR_PATTERN = re.compile(r"Neighbor\s+\d+:\s+score=.*adversarial=(true|false)")
ROUND_PATTERN = re.compile(r"^Round\s+(\d+):")


T_0975_TABLE = {
    1: 12.7062047364,
    2: 4.3026527297,
    3: 3.1824463053,
    4: 2.7764451052,
    5: 2.5705818356,
    6: 2.4469118488,
    7: 2.3646242510,
    8: 2.3060041352,
    9: 2.2621571627,
    10: 2.2281388520,
    11: 2.2009851601,
    12: 2.1788128297,
    13: 2.1603686565,
    14: 2.1447866879,
    15: 2.1314495456,
    16: 2.1199052992,
    17: 2.1098155778,
    18: 2.1009220402,
    19: 2.0930240544,
    20: 2.0859634473,
    21: 2.0796138447,
    22: 2.0738730679,
    23: 2.0686576104,
    24: 2.0638985616,
    25: 2.0595385528,
    26: 2.0555294386,
    27: 2.0518305165,
    28: 2.0484071418,
    29: 2.0452296421,
    30: 2.0422724563,
}


@dataclass
class GlobalRun:
    seed_folder: str
    seed_id: str
    experiment: str
    config_folder: Path
    metrics: dict[str, Optional[float]]
    rounds: Optional[int] = None

    @property
    def cycle_id(self) -> str:
        if self.seed_folder == f"seed_{self.seed_id}":
            return self.seed_folder
        return f"{self.seed_folder}:seed{self.seed_id}"


@dataclass
class ReputationCounts:
    malicious_blocked: float = 0.0
    malicious_not_blocked: float = 0.0
    malicious_total: float = 0.0
    regular_blocked: float = 0.0
    regular_total: float = 0.0
    update_total: float = 0.0

    def add(self, other: "ReputationCounts") -> None:
        self.malicious_blocked += other.malicious_blocked
        self.malicious_not_blocked += other.malicious_not_blocked
        self.malicious_total += other.malicious_total
        self.regular_blocked += other.regular_blocked
        self.regular_total += other.regular_total
        self.update_total += other.update_total

    def rates(self, fpbr_denominator: str) -> dict[str, Optional[float]]:
        fpbr_total = self.update_total if fpbr_denominator == "all_updates" else self.regular_total
        return {
            "srba": safe_div(self.malicious_blocked, self.malicious_total),
            "frba": safe_div(self.malicious_not_blocked, self.malicious_total),
            "fpbr": safe_div(self.regular_blocked, fpbr_total),
        }


@dataclass
class RoundRepState:
    round_id: int
    srba: Optional[float] = None
    frba: Optional[float] = None
    fpbr: Optional[float] = None
    adversarial_neighbors: int = 0
    benign_neighbors: int = 0


@dataclass
class NodeHeader:
    is_benign: bool = False
    included_direct_round: Optional[int] = None
    rounds: Optional[int] = None


def parse_optional_float(raw: str) -> Optional[float]:
    raw = raw.strip()
    if raw == "None":
        return None
    return float(raw)


def parse_optional_int(raw: str) -> Optional[int]:
    raw = raw.strip()
    if raw == "None":
        return None
    return int(raw)


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def t_critical_0975(df: int) -> float:
    if df < 1:
        raise ValueError("Student's t critical value requires df >= 1")

    try:
        from scipy.stats import t  # type: ignore

        return float(t.ppf(0.975, df))
    except Exception:
        pass

    if df in T_0975_TABLE:
        return T_0975_TABLE[df]

    # Cornish-Fisher expansion around the standard normal 0.975 quantile.
    z = 1.959963984540054
    v = float(df)
    return (
        z
        + (z**3 + z) / (4.0 * v)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * v**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z)
        / (384.0 * v**3)
    )


def parse_config_parts(experiment: str) -> dict[str, str]:
    match = CONFIG_PATTERN.match(experiment)
    if not match:
        return {
            "network_node_count": "",
            "topology": "",
            "attack": "",
            "byzantine_fraction": "",
            "defense": "",
            "aggregator": "",
        }

    parts = match.groupdict()
    return {
        "network_node_count": parts["node_count"],
        "topology": parts["topology"],
        "attack": parts["attack"],
        "byzantine_fraction": parts["byzantine_fraction"],
        "defense": parts["defense"],
        "aggregator": parts["aggregator"],
    }


def parse_field_list(raw: str, default: tuple[str, ...]) -> tuple[str, ...]:
    normalized = raw.strip().lower()
    if normalized == "":
        return default
    if normalized in {"none", "pooled"}:
        return ()
    if normalized == "all":
        return CONFIG_FIELDS

    fields = tuple(field.strip() for field in raw.split(",") if field.strip())
    unknown = [field for field in fields if field not in CONFIG_FIELDS]
    if unknown:
        allowed = ", ".join(CONFIG_FIELDS)
        raise ValueError(f"Unknown field(s): {', '.join(unknown)}. Allowed fields: {allowed}")
    return fields


def format_key(parts: dict[str, str], fields: tuple[str, ...]) -> str:
    if not fields:
        return "all"
    return "+".join(parts[field] for field in fields)


def family_values(
    family_key: tuple[str, ...],
    family_fields: tuple[str, ...],
) -> dict[str, str]:
    values = {field: "pooled" for field in CONFIG_FIELDS}
    for field, value in zip(family_fields, family_key):
        values[field] = value
    if "defense" not in family_fields:
        values["defense"] = ""
    if "aggregator" not in family_fields:
        values["aggregator"] = ""
    return values


def parse_global_stats(path: Path, seed_folder: str) -> GlobalRun:
    metrics: dict[str, Optional[float]] = {}
    experiment = ""
    seed_id = ""
    in_final_accuracy = False

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("Experiment:"):
            experiment = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Seed:"):
            seed_id = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Final Global Accuracy:"):
            in_final_accuracy = True
        elif in_final_accuracy and stripped.startswith("Avg:"):
            metrics["final_accuracy"] = parse_optional_float(stripped.split(":", 1)[1])
        elif stripped.startswith("Bytes Sent") or stripped.startswith("Bytes Received"):
            in_final_accuracy = False
        elif stripped.startswith("Attack Success Rate:"):
            metrics["attack_success_rate"] = parse_optional_float(stripped.split(":", 1)[1])
        elif stripped.startswith("Accuracy On Non-Target Classes:"):
            metrics["accuracy_on_non_target_classes"] = parse_optional_float(
                stripped.split(":", 1)[1]
            )
        elif stripped.startswith("Rounds Until 50 Percent Nodes Integrated Malicious:"):
            # This network-timing metric is intentionally excluded from the
            # required-cycle calculation because it is highly layout-dependent.
            pass

    if not experiment:
        raise ValueError(f"Could not parse experiment name from {path}")
    if not seed_id:
        raise ValueError(f"Could not parse seed from {path}")

    run = GlobalRun(
        seed_folder=seed_folder,
        seed_id=seed_id,
        experiment=experiment,
        config_folder=path.parent,
        metrics=metrics,
    )

    return run


def node_files_for_seed(config_folder: Path, seed_id: str) -> list[Path]:
    seed_node_folder = config_folder / f"nodes_seed{seed_id}"
    if seed_node_folder.is_dir():
        return sorted(seed_node_folder.glob("node_*.txt"))
    return sorted(config_folder.rglob("node_*.txt"))


def parse_node_header_line(header: NodeHeader, line: str) -> None:
    stripped = line.strip()
    if stripped == "Kind: Benign":
        header.is_benign = True
    elif stripped.startswith("Included Direct Adversarial In Round:"):
        value = stripped.split(":", 1)[1].strip()
        header.included_direct_round = parse_optional_int(value)
    elif stripped.startswith("Rounds:"):
        header.rounds = int(stripped.split(":", 1)[1].strip())


def finalize_round(
    counts: ReputationCounts,
    header: NodeHeader,
    state: Optional[RoundRepState],
) -> None:
    if state is None or state.srba is None or state.frba is None or state.fpbr is None:
        return

    if not header.is_benign:
        return

    labeled_round = (
        header.included_direct_round is None
        or state.round_id <= header.included_direct_round
    )
    if not labeled_round:
        return

    malicious = state.adversarial_neighbors
    benign = state.benign_neighbors
    updates = malicious + benign
    if updates == 0:
        return

    if malicious > 0:
        counts.malicious_blocked += state.srba * malicious
        counts.malicious_not_blocked += state.frba * malicious
        counts.malicious_total += malicious

    if benign > 0:
        counts.regular_blocked += state.fpbr * benign
        counts.regular_total += benign

    counts.update_total += updates


def parse_node_reputation_counts(path: Path) -> tuple[ReputationCounts, Optional[int]]:
    header = NodeHeader()
    counts = ReputationCounts()
    current: Optional[RoundRepState] = None

    for line in path.read_text(encoding="utf-8").splitlines():
        round_match = ROUND_PATTERN.match(line.strip())
        if round_match:
            finalize_round(counts, header, current)
            current = RoundRepState(round_id=int(round_match.group(1)))
            continue

        if current is None:
            parse_node_header_line(header, line)
            continue

        blocking_match = BLOCKING_PATTERN.search(line)
        if blocking_match:
            current.srba = float(blocking_match.group("srba"))
            current.frba = float(blocking_match.group("frba"))
            current.fpbr = float(blocking_match.group("fpbr"))
            continue

        neighbor_match = NEIGHBOR_PATTERN.search(line)
        if neighbor_match:
            if neighbor_match.group(1) == "true":
                current.adversarial_neighbors += 1
            else:
                current.benign_neighbors += 1

    finalize_round(counts, header, current)
    return counts, header.rounds


def parse_reputation_metrics(
    config_folder: Path,
    seed_id: str,
    fpbr_denominator: str,
) -> dict[str, Optional[float]]:
    counts = ReputationCounts()
    for node_file in node_files_for_seed(config_folder, seed_id):
        node_counts, _ = parse_node_reputation_counts(node_file)
        counts.add(node_counts)
    return counts.rates(fpbr_denominator)


def collect_runs(
    results_folder: Path,
    fpbr_denominator: str,
) -> list[GlobalRun]:
    runs: list[GlobalRun] = []

    for seed_folder in sorted(results_folder.glob("seed_*")):
        if not seed_folder.is_dir():
            continue

        for global_file in sorted(seed_folder.rglob("global_stats_*.txt")):
            run = parse_global_stats(global_file, seed_folder.name)
            config_parts = parse_config_parts(run.experiment)

            if config_parts["defense"] == "Rep":
                rep_metrics = parse_reputation_metrics(
                    run.config_folder,
                    seed_id=run.seed_id,
                    fpbr_denominator=fpbr_denominator,
                )
                run.metrics.update(rep_metrics)

            runs.append(run)

    return runs


def summarize_metric(
    values: list[float],
    margin_error: float,
    z_critical: float,
) -> dict[str, Optional[float]]:
    cycle_count = len(values)
    avg = mean(values)

    if cycle_count == 1:
        return {
            "observed_cycles": float(cycle_count),
            "mean": avg,
            "sample_std": None,
            "required_cycles": None,
            "additional_cycles": None,
            "t_critical_observed": None,
            "ci_lower_observed": None,
            "ci_upper_observed": None,
            "ci_half_width_observed": None,
            "t_half_width_at_required_cycles": None,
        }

    sample_std = stdev(values)
    required_cycles = max(1, math.ceil(((z_critical * sample_std) / margin_error) ** 2))
    additional_cycles = max(0, required_cycles - cycle_count)

    t_observed = t_critical_0975(cycle_count - 1)
    observed_half_width = t_observed * sample_std / math.sqrt(cycle_count)

    if required_cycles <= 1:
        t_half_width_at_required_cycles = 0.0
    else:
        t_required = t_critical_0975(required_cycles - 1)
        t_half_width_at_required_cycles = (
            t_required * sample_std / math.sqrt(required_cycles)
        )

    return {
        "observed_cycles": float(cycle_count),
        "mean": avg,
        "sample_std": sample_std,
        "required_cycles": float(required_cycles),
        "additional_cycles": float(additional_cycles),
        "t_critical_observed": t_observed,
        "ci_lower_observed": avg - observed_half_width,
        "ci_upper_observed": avg + observed_half_width,
        "ci_half_width_observed": observed_half_width,
        "t_half_width_at_required_cycles": t_half_width_at_required_cycles,
    }


def build_rows(
    runs: list[GlobalRun],
    margin_error: float,
    z_critical: float,
    family_fields: tuple[str, ...],
    method_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    cells: dict[tuple[tuple[str, ...], str, str, str], list[float]] = {}
    skipped_none = 0

    for run in runs:
        parts = parse_config_parts(run.experiment)
        family_key = tuple(parts[field] for field in family_fields)
        method = format_key(parts, method_fields)

        for metric, value in run.metrics.items():
            if value is None:
                skipped_none += 1
                continue
            cell_key = (family_key, method, metric, run.cycle_id)
            cells.setdefault(cell_key, []).append(value)

    by_family_metric: dict[tuple[tuple[str, ...], str], dict[str, dict[str, float]]] = {}
    for (family_key, method, metric, cycle_id), values in cells.items():
        by_family_metric.setdefault((family_key, metric), {}).setdefault(method, {})[
            cycle_id
        ] = mean(values)

    rows: list[dict[str, object]] = []
    for (family_key, metric), method_values in sorted(by_family_metric.items()):
        family = "+".join(family_key) if family_key else "all"
        methods = sorted(method_values)

        if len(methods) < 2:
            method = methods[0]
            cycle_values = sorted(method_values[method].items())
            values = [value for _, value in cycle_values]
            summary = summarize_metric(values, margin_error, z_critical)
            summary["mean_value"] = summary.pop("mean")
            summary["sample_std_value"] = summary.pop("sample_std")

            row: dict[str, object] = {
                "analysis_type": "family_method_summary",
                "family": family,
                "family_fields": ",".join(family_fields) if family_fields else "none",
                "method_fields": ",".join(method_fields) if method_fields else "none",
                **family_values(family_key, family_fields),
                "baseline_method": "",
                "comparison_method": method,
                "metric": metric,
                "mean_baseline": "",
                "mean_comparison": summary["mean_value"],
                "mean_difference": "",
                "sample_std_difference": "",
                "cycle_ids": ";".join(cycle_id for cycle_id, _ in cycle_values),
                "margin_error_target": margin_error,
                "z_critical": z_critical,
            }
            row.update(summary)
            rows.append(row)
            continue

        for baseline_method, comparison_method in combinations(methods, 2):
            baseline_by_cycle = method_values[baseline_method]
            comparison_by_cycle = method_values[comparison_method]
            common_cycles = sorted(set(baseline_by_cycle) & set(comparison_by_cycle))
            if not common_cycles:
                continue

            baseline_values = [baseline_by_cycle[cycle_id] for cycle_id in common_cycles]
            comparison_values = [comparison_by_cycle[cycle_id] for cycle_id in common_cycles]
            differences = [
                comparison - baseline
                for baseline, comparison in zip(baseline_values, comparison_values)
            ]
            summary = summarize_metric(differences, margin_error, z_critical)
            summary["mean_difference"] = summary.pop("mean")
            summary["sample_std_difference"] = summary.pop("sample_std")

            row = {
                "analysis_type": "paired_family_comparison",
                "family": family,
                "family_fields": ",".join(family_fields) if family_fields else "none",
                "method_fields": ",".join(method_fields) if method_fields else "none",
                **family_values(family_key, family_fields),
                "baseline_method": baseline_method,
                "comparison_method": comparison_method,
                "metric": metric,
                "mean_baseline": mean(baseline_values),
                "mean_comparison": mean(comparison_values),
                "mean_value": "",
                "sample_std_value": "",
                "cycle_ids": ";".join(common_cycles),
                "margin_error_target": margin_error,
                "z_critical": z_critical,
            }
            row.update(summary)
            rows.append(row)

    if skipped_none:
        print(f"Skipped {skipped_none} unavailable metric values marked as None.")

    return rows


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "analysis_type",
        "family",
        "family_fields",
        "method_fields",
        "network_node_count",
        "topology",
        "attack",
        "byzantine_fraction",
        "defense",
        "aggregator",
        "baseline_method",
        "comparison_method",
        "metric",
        "observed_cycles",
        "required_cycles",
        "additional_cycles",
        "mean_baseline",
        "mean_comparison",
        "mean_difference",
        "sample_std_difference",
        "mean_value",
        "sample_std_value",
        "ci_lower_observed",
        "ci_upper_observed",
        "ci_half_width_observed",
        "t_critical_observed",
        "t_half_width_at_required_cycles",
        "margin_error_target",
        "z_critical",
        "cycle_ids",
    ]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def print_summary(rows: list[dict[str, object]], limit: int) -> None:
    rows_with_required = [
        row for row in rows if row.get("required_cycles") not in (None, "")
    ]
    rows_with_required.sort(
        key=lambda row: float(row["required_cycles"]), reverse=True
    )

    print(f"Computed {len(rows)} family/metric comparison summaries.")
    print(f"Top {min(limit, len(rows_with_required))} required cycle counts:")
    for row in rows_with_required[:limit]:
        std_value = row.get("sample_std_difference") or row.get("sample_std_value")
        comparison = (
            f"{row['comparison_method']} - {row['baseline_method']}"
            if row["analysis_type"] == "paired_family_comparison"
            else str(row["comparison_method"])
        )
        print(
            "  "
            f"{row['required_cycles']:>6.0f} cycles | "
            f"{row['metric']} | {row['family']} | {comparison} "
            f"(observed={row['observed_cycles']:.0f}, std={float(std_value):.6g})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate required simulation cycles from results/seed_* logs for a "
            "two-sided 95% confidence interval target margin of error using "
            "seed-paired family-level method comparisons."
        )
    )
    parser.add_argument(
        "--results-folder",
        type=Path,
        default=Path("results"),
        help="Folder containing one seed_* directory per simulation cycle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/required_cycles.csv"),
        help="CSV file to write.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional JSON file to write with the same rows as the CSV.",
    )
    parser.add_argument(
        "--margin-error",
        type=float,
        default=DEFAULT_MARGIN_ERROR,
        help="Target margin of error E for performance metrics.",
    )
    parser.add_argument(
        "--z-critical",
        type=float,
        default=DEFAULT_Z_0975,
        help="Normal critical value used in required-cycle formula.",
    )
    parser.add_argument(
        "--family-fields",
        default=",".join(DEFAULT_FAMILY_FIELDS),
        help=(
            "Comma-separated config fields that define a family before methods "
            "are compared. Default pools byzantine fractions within each "
            "network_node_count/topology/attack family."
        ),
    )
    parser.add_argument(
        "--method-fields",
        default=",".join(DEFAULT_METHOD_FIELDS),
        help=(
            "Comma-separated config fields that define the compared method. "
            "Default compares defense+aggregator."
        ),
    )
    parser.add_argument(
        "--fpbr-denominator",
        choices=["all_updates", "benign_updates"],
        default="all_updates",
        help=(
            "Use 'all_updates' for the FPBR definition in the thesis text; "
            "'benign_updates' matches the denominator used in the raw node log rate."
        ),
    )
    parser.add_argument(
        "--summary-limit",
        type=int,
        default=10,
        help="Number of highest required-cycle rows to print.",
    )

    args = parser.parse_args()

    runs = collect_runs(
        args.results_folder,
        fpbr_denominator=args.fpbr_denominator,
    )
    try:
        family_fields = parse_field_list(args.family_fields, DEFAULT_FAMILY_FIELDS)
        method_fields = parse_field_list(args.method_fields, DEFAULT_METHOD_FIELDS)
    except ValueError as exc:
        parser.error(str(exc))

    rows = build_rows(
        runs,
        args.margin_error,
        args.z_critical,
        family_fields=family_fields,
        method_fields=method_fields,
    )

    write_csv(rows, args.output)
    if args.json_output:
        write_json(rows, args.json_output)

    print_summary(rows, args.summary_limit)
    print(f"Wrote CSV: {args.output}")
    if args.json_output:
        print(f"Wrote JSON: {args.json_output}")


if __name__ == "__main__":
    main()
