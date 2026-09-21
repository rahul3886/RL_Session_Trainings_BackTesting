"""
Lens 1 Post-Training Encoder Analysis.

After training, this module extracts and analyses what Lens 1 learned:
1. Encoding Correlation Matrix vs NY outcomes
2. Scenario Clustering (k-means k=10)
3. Sweep Prediction Accuracy
4. Encoding Visualisation (PCA/UMAP to 2D)
"""

import os
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pathlib import Path

from src.lens1.opening_targets import infer_opening_outcome
from src.smc.liquidity import detect_sweep

logger = logging.getLogger(__name__)


def extract_all_encodings(
    model,
    env,
    trading_days: list,
) -> pd.DataFrame:
    """
    Run Lens 1 on all trading days and extract encodings + NY outcomes.

    Returns:
        DataFrame with columns: date, encoding[0..7], ny_direction,
        sweep_high, sweep_low, ny_close_vs_london_open, etc.
    """
    records = []
    sweep_threshold_lr = 0.12

    if getattr(env, "settings", None) is not None:
        sweep_threshold_lr = env.settings.smc.sweep_min_lr
        opening_window_candles = env.settings.lens1.opening_target_window_candles
    else:
        opening_window_candles = 4

    for i, date in enumerate(trading_days):
        try:
            # Reset env to specific day
            env.current_day_idx = i
            obs, info = env.reset()

            # Run through observation phase
            encoding = np.zeros(env.encoding_dim)
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if info.get("phase") == "evaluation" or done:
                    encoding = info.get("encoding", encoding)

            # Collect NY outcomes
            lp = env.london_profile
            ny = env.ny_candles

            if lp is None or ny is None or len(ny) == 0:
                continue

            ny_close = ny["close"].iloc[-1]
            ny_high = ny["high"].max()
            ny_low = ny["low"].min()
            london_open = lp.open_price
            london_mid = lp.mid
            sweep = detect_sweep(
                current_price=float(ny_close),
                session_high=lp.high,
                session_low=lp.low,
                candles_after=ny,
                threshold_lr=sweep_threshold_lr,
                require_reversal=True,
            )
            opening_outcome = infer_opening_outcome(
                lp,
                ny,
                sweep_threshold_lr=sweep_threshold_lr,
                window_candles=opening_window_candles,
            )

            record = {
                "date": str(date),
                "ny_direction": 1 if ny_close > london_mid else -1,
                "ny_close_vs_london_open": ny_close - london_open,
                "ny_range": ny_high - ny_low,
                "london_range": lp.range_points,
                "london_bias": lp.bias,
                "sweep_high": 1 if sweep["sweep_high"] else 0,
                "sweep_low": 1 if sweep["sweep_low"] else 0,
                "ny_respected_high": 1 if abs(ny_high - lp.high) < 5.0 else 0,
                "ny_respected_low": 1 if abs(ny_low - lp.low) < 5.0 else 0,
                "continuation": 1 if (lp.bias == 1 and ny_close > london_open) or
                                     (lp.bias == -1 and ny_close < london_open) else 0,
                "opening_trade_expected": int(opening_outcome.opening_trade_expected),
                "opening_side_bias": int(opening_outcome.opening_side_bias),
                "opening_style_bias": float(opening_outcome.opening_style_bias),
                "opening_urgency": float(opening_outcome.opening_urgency),
                "opening_confidence": float(opening_outcome.confidence),
                "opening_trigger_step": int(opening_outcome.trigger_step),
                "opening_setup_family": opening_outcome.opening_setup_family,
                "opening_scenario_label": opening_outcome.scenario_label,
            }

            # Add encoding dimensions
            for j in range(len(encoding)):
                record[f"encoding_{j}"] = encoding[j]

            records.append(record)

        except Exception as e:
            logger.warning(f"Failed to process day {date}: {e}")
            continue

    return pd.DataFrame(records)


def analyse_encoding_correlations(encodings_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Analysis 1: Encoding correlation matrix.
    Which encoding dimensions correlate with which NY outcomes?
    """
    os.makedirs(output_dir, exist_ok=True)

    encoding_cols = [c for c in encodings_df.columns if c.startswith("encoding_")]
    outcome_cols = [
        "ny_direction", "sweep_high", "sweep_low",
        "ny_respected_high", "ny_respected_low",
        "continuation", "ny_close_vs_london_open",
        "opening_trade_expected", "opening_side_bias",
        "opening_style_bias", "opening_urgency", "opening_confidence",
    ]

    # Compute correlations
    all_cols = encoding_cols + outcome_cols
    existing_cols = [c for c in all_cols if c in encodings_df.columns]
    corr = encodings_df[existing_cols].corr()

    # Extract just encoding vs outcome correlations
    enc_vs_outcome = corr.loc[encoding_cols, [c for c in outcome_cols if c in corr.columns]]

    # Save
    save_path = os.path.join(output_dir, "lens1_encoding_correlations.csv")
    enc_vs_outcome.to_csv(save_path)
    logger.info(f"Saved encoding correlations to {save_path}")

    # Log significant correlations
    for enc_col in encoding_cols:
        for out_col in enc_vs_outcome.columns:
            val = enc_vs_outcome.loc[enc_col, out_col]
            if abs(val) > 0.3:
                logger.info(f"  SIGNIFICANT: {enc_col} ↔ {out_col}: {val:.3f}")

    return enc_vs_outcome


def analyse_scenario_clusters(
    encodings_df: pd.DataFrame,
    output_dir: str,
    n_clusters: int = 10,
) -> pd.DataFrame:
    """
    Analysis 2: K-means clustering of encoding vectors.
    Labels each cluster with most common NY outcome.
    """
    from sklearn.cluster import KMeans

    os.makedirs(output_dir, exist_ok=True)

    encoding_cols = [c for c in encodings_df.columns if c.startswith("encoding_")]
    X = encodings_df[encoding_cols].values

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    encodings_df = encodings_df.copy()
    encodings_df["cluster"] = labels

    # Analyse each cluster
    cluster_summary = []
    for cluster_id in range(n_clusters):
        cluster_data = encodings_df[encodings_df["cluster"] == cluster_id]
        n = len(cluster_data)

        if n == 0:
            continue

        summary = {
            "cluster": cluster_id,
            "count": n,
            "pct_total": n / len(encodings_df) * 100,
            "avg_ny_direction": cluster_data["ny_direction"].mean(),
            "pct_sweep_high": cluster_data["sweep_high"].mean() * 100 if "sweep_high" in cluster_data.columns else 0,
            "pct_sweep_low": cluster_data["sweep_low"].mean() * 100 if "sweep_low" in cluster_data.columns else 0,
            "pct_continuation": cluster_data["continuation"].mean() * 100 if "continuation" in cluster_data.columns else 0,
            "avg_ny_range": cluster_data["ny_range"].mean() if "ny_range" in cluster_data.columns else 0,
            "opening_trade_expected_rate": cluster_data["opening_trade_expected"].mean() * 100 if "opening_trade_expected" in cluster_data.columns else 0,
            "avg_opening_side_bias": cluster_data["opening_side_bias"].mean() if "opening_side_bias" in cluster_data.columns else 0,
            "avg_opening_style_bias": cluster_data["opening_style_bias"].mean() if "opening_style_bias" in cluster_data.columns else 0,
            "avg_opening_urgency": cluster_data["opening_urgency"].mean() if "opening_urgency" in cluster_data.columns else 0,
            "pct_open_sweep_high": (cluster_data["opening_scenario_label"] == "S7_HighLiqGrab").mean() * 100 if "opening_scenario_label" in cluster_data.columns else 0,
            "pct_open_sweep_low": (cluster_data["opening_scenario_label"] == "S8_LowLiqGrab").mean() * 100 if "opening_scenario_label" in cluster_data.columns else 0,
            "pct_open_bull_continuation": (cluster_data["opening_scenario_label"] == "S1_BullContinuation").mean() * 100 if "opening_scenario_label" in cluster_data.columns else 0,
            "pct_open_bear_continuation": (cluster_data["opening_scenario_label"] == "S2_BearContinuation").mean() * 100 if "opening_scenario_label" in cluster_data.columns else 0,
            "pct_open_generic_continuation": (cluster_data["opening_scenario_label"] == "S3S4_Continuation").mean() * 100 if "opening_scenario_label" in cluster_data.columns else 0,
        }

        # Determine dominant scenario
        if summary["pct_open_sweep_high"] >= 20:
            summary["dominant_scenario"] = "S7_HighLiqGrab"
        elif summary["pct_open_sweep_low"] >= 20:
            summary["dominant_scenario"] = "S8_LowLiqGrab"
        elif summary["pct_open_bull_continuation"] >= 20:
            summary["dominant_scenario"] = "S1_BullContinuation"
        elif summary["pct_open_bear_continuation"] >= 20:
            summary["dominant_scenario"] = "S2_BearContinuation"
        elif summary["opening_trade_expected_rate"] < 25:
            summary["dominant_scenario"] = "Mixed/Ranging"
        elif summary["pct_open_generic_continuation"] >= 20 or abs(summary["avg_opening_side_bias"]) >= 0.25:
            summary["dominant_scenario"] = "S3S4_Continuation"
        else:
            summary["dominant_scenario"] = "Mixed/Ranging"

        # Add encoding centroids
        for j, col in enumerate(encoding_cols):
            summary[f"centroid_{j}"] = cluster_data[col].mean()

        cluster_summary.append(summary)

    result = pd.DataFrame(cluster_summary)
    save_path = os.path.join(output_dir, "lens1_scenario_clusters.csv")
    result.to_csv(save_path, index=False)
    logger.info(f"Saved scenario clusters to {save_path}")

    # Also save full labelled data
    full_path = os.path.join(output_dir, "lens1_encodings_labelled.csv")
    encodings_df.to_csv(full_path, index=False)

    return result


def analyse_sweep_accuracy(
    encodings_df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    output_dir: str,
) -> Dict:
    """
    Analysis 3: Sweep prediction accuracy.
    What % of actual sweeps did Lens 1 predict?
    Finds the best sweep-predicting dimension dynamically.
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    encoding_cols = [c for c in encodings_df.columns if c.startswith("encoding_")]

    # Find best dimension for sweep_high
    if "sweep_high" in corr_matrix.columns:
        sweep_high_corrs = corr_matrix.loc[encoding_cols, "sweep_high"].abs()
        best_sweep_high_col = sweep_high_corrs.idxmax()
    else:
        best_sweep_high_col = "encoding_3"

    # Find best dimension for sweep_low
    if "sweep_low" in corr_matrix.columns:
        sweep_low_corrs = corr_matrix.loc[encoding_cols, "sweep_low"].abs()
        best_sweep_low_col = sweep_low_corrs.idxmax()
    else:
        best_sweep_low_col = "encoding_4"

    logger.info(f"Best sweep_high predictor: {best_sweep_high_col}")
    logger.info(f"Best sweep_low predictor:  {best_sweep_low_col}")

    sweep_high_mean = encodings_df[best_sweep_high_col].mean()
    sweep_low_mean = encodings_df[best_sweep_low_col].mean()

    # High sweeps
    actual_sweep_high = encodings_df[encodings_df["sweep_high"] == 1]
    if len(actual_sweep_high) > 0:
        predicted = actual_sweep_high[actual_sweep_high[best_sweep_high_col] > sweep_high_mean]
        results["sweep_high_total"] = len(actual_sweep_high)
        results["sweep_high_predicted"] = len(predicted)
        results["sweep_high_recall"] = len(predicted) / len(actual_sweep_high)
    else:
        results["sweep_high_total"] = 0
        results["sweep_high_predicted"] = 0
        results["sweep_high_recall"] = 0.0

    # Low sweeps
    actual_sweep_low = encodings_df[encodings_df["sweep_low"] == 1]
    if len(actual_sweep_low) > 0:
        predicted = actual_sweep_low[actual_sweep_low[best_sweep_low_col] > sweep_low_mean]
        results["sweep_low_total"] = len(actual_sweep_low)
        results["sweep_low_predicted"] = len(predicted)
        results["sweep_low_recall"] = len(predicted) / len(actual_sweep_low)
    else:
        results["sweep_low_total"] = 0
        results["sweep_low_predicted"] = 0
        results["sweep_low_recall"] = 0.0

    # Combined
    total_sweeps = results["sweep_high_total"] + results["sweep_low_total"]
    total_predicted = results["sweep_high_predicted"] + results["sweep_low_predicted"]
    results["combined_recall"] = total_predicted / max(total_sweeps, 1)
    results["total_sweeps"] = total_sweeps

    # Save
    save_path = os.path.join(output_dir, "lens1_sweep_accuracy.txt")
    with open(save_path, "w") as f:
        f.write("Lens 1 Sweep Prediction Accuracy\n")
        f.write("=" * 50 + "\n\n")
        for k, v in results.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n  Target: > 60% recall on sweeps\n")
        f.write(f"  Status: {'PASS' if results['combined_recall'] > 0.6 else 'NEEDS IMPROVEMENT'}\n")

    logger.info(f"Sweep accuracy: {results['combined_recall']:.1%} "
                f"({total_predicted}/{total_sweeps} sweeps predicted)")

    return results


def visualise_encodings(encodings_df: pd.DataFrame, output_dir: str):
    """
    Analysis 4: PCA/UMAP visualisation of encoding vectors.
    Colour each point by NY outcome.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    os.makedirs(output_dir, exist_ok=True)

    encoding_cols = [c for c in encodings_df.columns if c.startswith("encoding_")]
    X = encodings_df[encoding_cols].values

    if len(X) < 3:
        logger.warning("Not enough data points for visualisation")
        return

    # PCA to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Colour by NY direction
    colors = encodings_df["ny_direction"].values
    sc1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap="RdYlGn", alpha=0.6, s=20)
    axes[0].set_title("Lens 1 Encoding — NY Direction")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.colorbar(sc1, ax=axes[0], label="NY Direction")

    # Plot 2: Colour by sweep events
    sweep_colors = encodings_df["sweep_high"].values + encodings_df["sweep_low"].values * 2
    sc2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=sweep_colors, cmap="Set1", alpha=0.6, s=20)
    axes[1].set_title("Lens 1 Encoding — Sweep Events")
    axes[1].set_xlabel(f"PC1")
    axes[1].set_ylabel(f"PC2")

    # Plot 3: Colour by cluster (if available)
    if "cluster" in encodings_df.columns:
        cluster_colors = encodings_df["cluster"].values
        sc3 = axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_colors, cmap="tab10", alpha=0.6, s=20)
        axes[2].set_title("Lens 1 Encoding — Clusters")
        plt.colorbar(sc3, ax=axes[2], label="Cluster")
    else:
        # Colour by continuation
        cont_colors = encodings_df.get("continuation", pd.Series(np.zeros(len(encodings_df)))).values
        sc3 = axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=cont_colors, cmap="coolwarm", alpha=0.6, s=20)
        axes[2].set_title("Lens 1 Encoding — Continuation")

    axes[2].set_xlabel(f"PC1")
    axes[2].set_ylabel(f"PC2")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "lens1_encoding_map.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved encoding visualisation to {save_path}")

    # Try UMAP if available
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = encodings_df["ny_direction"].values
        sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, cmap="RdYlGn", alpha=0.6, s=20)
        ax.set_title("Lens 1 Encoding — UMAP (NY Direction)")
        plt.colorbar(sc, ax=ax, label="NY Direction")
        umap_path = os.path.join(output_dir, "lens1_encoding_umap.png")
        plt.savefig(umap_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved UMAP visualisation to {umap_path}")
    except ImportError:
        logger.info("UMAP not available, skipping UMAP visualisation")


def run_full_analysis(model, env, trading_days: list, output_dir: str):
    """Run all 4 analyses and save outputs."""
    logger.info("=" * 60)
    logger.info("LENS 1 POST-TRAINING ANALYSIS")
    logger.info("=" * 60)

    # Extract all encodings
    logger.info("\nExtracting encodings for all trading days...")
    encodings_df = extract_all_encodings(model, env, trading_days)
    logger.info(f"  Extracted {len(encodings_df)} day encodings")

    if len(encodings_df) < 10:
        logger.warning("Too few encodings for meaningful analysis")
        return encodings_df

    # Analysis 1: Correlations
    logger.info("\n--- Analysis 1: Encoding Correlations ---")
    corr_matrix = analyse_encoding_correlations(encodings_df, output_dir)

    # Analysis 2: Scenario Clustering
    logger.info("\n--- Analysis 2: Scenario Clustering ---")
    cluster_df = analyse_scenario_clusters(encodings_df, output_dir)

    # Analysis 3: Sweep Accuracy
    logger.info("\n--- Analysis 3: Sweep Prediction Accuracy ---")
    analyse_sweep_accuracy(encodings_df, corr_matrix, output_dir)

    # Analysis 4: Visualisation
    logger.info("\n--- Analysis 4: Encoding Visualisation ---")
    visualise_encodings(encodings_df, output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("LENS 1 ANALYSIS COMPLETE")
    logger.info(f"All outputs saved to {output_dir}")
    logger.info("=" * 60)

    return encodings_df
