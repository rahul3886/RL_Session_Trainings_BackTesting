"""
Generate post-retraining Lens 1 analysis and comparison artifacts.

This script compares the newly retrained Lens 1 checkpoints against the
pre-retrain baseline checkpoints on both validation and test splits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.pipeline import run_data_pipeline
from src.lens1.environment import Lens1LondonEnv
from src.lens1.encoder import (
    extract_all_encodings,
    analyse_encoding_correlations,
    analyse_sweep_accuracy,
    analyse_scenario_clusters,
)


def _evaluate_model(
    label: str,
    model_path: Path,
    split_name: str,
    split_df: pd.DataFrame,
    split_days: List[pd.Timestamp],
    settings,
    output_root: Path,
) -> Dict:
    model = PPO.load(str(model_path))
    env = Lens1LondonEnv(split_df, split_days, settings)

    split_output_dir = output_root / label / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    encodings_df = extract_all_encodings(model, env, split_days)
    corr = analyse_encoding_correlations(encodings_df, str(split_output_dir))
    sweep = analyse_sweep_accuracy(encodings_df, corr, str(split_output_dir))
    clusters = analyse_scenario_clusters(encodings_df, str(split_output_dir))

    dominant_counts = {}
    if len(clusters) > 0 and "dominant_scenario" in clusters.columns:
        dominant_counts = clusters["dominant_scenario"].value_counts().to_dict()

    summary = {
        "label": label,
        "split": split_name,
        "model_path": str(model_path),
        "rows": len(encodings_df),
        "combined_recall": sweep["combined_recall"],
        "sweep_high_recall": sweep["sweep_high_recall"],
        "sweep_low_recall": sweep["sweep_low_recall"],
        "total_sweeps": sweep["total_sweeps"],
        "cluster_count": len(clusters),
        "dominant_counts": dominant_counts,
    }
    return summary


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _write_markdown_report(
    summaries: List[Dict],
    report_path: Path,
) -> None:
    lines = [
        "# Lens 1 Retrain Comparison",
        "",
        "This report compares the retrained Lens 1 checkpoints against the",
        "pre-retrain baseline checkpoints on the validation and test splits.",
        "",
    ]

    for split_name in sorted({s["split"] for s in summaries}):
        split_rows = [s for s in summaries if s["split"] == split_name]
        split_rows.sort(key=lambda x: x["combined_recall"], reverse=True)

        lines.append(f"## {split_name.title()} Split")
        lines.append("")
        lines.append("| Model | Rows | Combined Sweep Recall | High Sweep Recall | Low Sweep Recall | Total Sweeps | Clusters |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for s in split_rows:
            lines.append(
                f"| {s['label']} | {s['rows']} | {_fmt_pct(s['combined_recall'])} | "
                f"{_fmt_pct(s['sweep_high_recall'])} | {_fmt_pct(s['sweep_low_recall'])} | "
                f"{s['total_sweeps']} | {s['cluster_count']} |"
            )
        lines.append("")

        best = split_rows[0]
        lines.append(f"Best on {split_name}: `{best['label']}` at {_fmt_pct(best['combined_recall'])}.")
        lines.append("")

        for s in split_rows:
            dominant = s["dominant_counts"]
            if dominant:
                lines.append(f"- `{s['label']}` dominant scenarios: {dominant}")
            else:
                lines.append(f"- `{s['label']}` dominant scenarios: unavailable")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-final", required=True)
    parser.add_argument("--baseline-best", required=True)
    parser.add_argument(
        "--output-dir",
        default="output/lens1_analysis/comparison",
        help="Directory for comparison artifacts.",
    )
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    settings = load_settings(str(project_root / "config" / "settings.yaml"))
    data = run_data_pipeline(settings, project_root)

    model_paths = {
        "baseline_final": Path(args.baseline_final),
        "baseline_best": Path(args.baseline_best),
        "retrained_final": project_root / "models" / "lens1" / "lens1_ppo_final.zip",
        "retrained_best": project_root / "models" / "lens1" / "best" / "best_model.zip",
    }

    output_root = project_root / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict] = []
    split_map = {
        "validate": (data["validate"], data["validate_days"]),
        "test": (data["test"], data["test_days"]),
    }

    for label, model_path in model_paths.items():
        if not model_path.exists():
            continue

        for split_name, (split_df, split_days) in split_map.items():
            summaries.append(
                _evaluate_model(
                    label=label,
                    model_path=model_path,
                    split_name=split_name,
                    split_df=split_df,
                    split_days=split_days,
                    settings=settings,
                    output_root=output_root,
                )
            )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_root / "lens1_retrain_comparison_summary.csv", index=False)
    _write_markdown_report(summaries, output_root / "lens1_retrain_comparison.md")


if __name__ == "__main__":
    main()
