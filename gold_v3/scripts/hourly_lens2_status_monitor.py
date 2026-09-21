"""
Hourly status monitor for the long Lens 2 retraining run.

Writes a latest-status markdown file plus an append-only history log until the
configured Lens 2 retrain completes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings


TOTAL_TIMESTEPS_RE = re.compile(r"total_timesteps\s*\|?\s*([0-9]+)")
RESUME_STEP_RE = re.compile(r"lens2_(\d+)_steps\.zip$", re.IGNORECASE)


@dataclass
class Lens2Status:
    checked_at_utc: str
    run_state: str
    absolute_target_timesteps: int
    start_timesteps: int
    latest_total_timesteps: int
    progress_pct_of_resume_run: float
    latest_checkpoint_name: str
    latest_checkpoint_time_utc: str
    best_checkpoint_name: str
    best_checkpoint_time_utc: str
    log_name: str
    log_last_update_utc: str
    notes: str


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _parse_resume_step(resume_checkpoint: str) -> int:
    match = RESUME_STEP_RE.search(resume_checkpoint.replace("/", "\\"))
    if not match:
        return 0
    return int(match.group(1))


def _find_latest_log(log_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _extract_latest_timesteps(log_path: Path) -> int:
    latest = 0
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return 0
    for match in TOTAL_TIMESTEPS_RE.finditer(text):
        latest = max(latest, int(match.group(1)))
    return latest


def _latest_checkpoint(model_dir: Path) -> Optional[Path]:
    checkpoints = sorted(model_dir.glob("lens2_*_steps.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def _best_checkpoint(model_dir: Path) -> Optional[Path]:
    best = model_dir / "best" / "best_model.zip"
    return best if best.exists() else None


def _determine_state(
    latest_log: Optional[Path],
    latest_timesteps: int,
    absolute_target: int,
    stale_minutes: int,
) -> tuple[str, str]:
    if latest_log is None:
        return "not_started", "No matching Lens2 resume log was found yet."

    try:
        log_text = latest_log.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return "missing_log", "The Lens2 resume log disappeared while monitoring."

    if "Lens 2 training complete." in log_text:
        return "completed", "Lens 2 retraining completed and wrote its final summary."

    age_seconds = time.time() - latest_log.stat().st_mtime
    if latest_timesteps >= absolute_target:
        return "completed", "Latest timesteps reached or exceeded the configured target."
    if age_seconds > stale_minutes * 60:
        return "stopped", f"Log has not updated for more than {stale_minutes} minutes."
    return "running", "Lens 2 retraining is still actively updating."


def _build_status(config_path: Path, model_dir: Path, log_dir: Path, log_pattern: str, stale_minutes: int) -> Lens2Status:
    settings = load_settings(str(config_path))
    start_timesteps = _parse_resume_step(settings.lens2.resume_checkpoint)
    absolute_target = start_timesteps + int(settings.lens2.total_timesteps)

    latest_log = _find_latest_log(log_dir, log_pattern)
    latest_timesteps = _extract_latest_timesteps(latest_log) if latest_log else 0
    latest_cp = _latest_checkpoint(model_dir)
    best_cp = _best_checkpoint(model_dir)
    run_state, notes = _determine_state(latest_log, latest_timesteps, absolute_target, stale_minutes)

    if settings.lens2.total_timesteps > 0:
        progress_pct = ((latest_timesteps - start_timesteps) / settings.lens2.total_timesteps) * 100.0
    else:
        progress_pct = 0.0
    progress_pct = max(0.0, min(progress_pct, 100.0))

    return Lens2Status(
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
        run_state=run_state,
        absolute_target_timesteps=absolute_target,
        start_timesteps=start_timesteps,
        latest_total_timesteps=latest_timesteps,
        progress_pct_of_resume_run=progress_pct,
        latest_checkpoint_name=latest_cp.name if latest_cp else "none",
        latest_checkpoint_time_utc=_utc_iso(latest_cp.stat().st_mtime) if latest_cp else "n/a",
        best_checkpoint_name=best_cp.name if best_cp else "none",
        best_checkpoint_time_utc=_utc_iso(best_cp.stat().st_mtime) if best_cp else "n/a",
        log_name=latest_log.name if latest_log else "none",
        log_last_update_utc=_utc_iso(latest_log.stat().st_mtime) if latest_log else "n/a",
        notes=notes,
    )


def _write_latest(status: Lens2Status, output_md: Path, output_json: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    md = "\n".join(
        [
            "# Lens2 Hourly Status",
            "",
            f"- Checked at UTC: `{status.checked_at_utc}`",
            f"- State: `{status.run_state}`",
            f"- Progress: `{status.latest_total_timesteps:,}` / `{status.absolute_target_timesteps:,}` "
            f"(`{status.progress_pct_of_resume_run:.2f}%` of target run)",
            f"- Latest checkpoint: `{status.latest_checkpoint_name}` at `{status.latest_checkpoint_time_utc}`",
            f"- Best checkpoint: `{status.best_checkpoint_name}` at `{status.best_checkpoint_time_utc}`",
            f"- Source log: `{status.log_name}` updated `{status.log_last_update_utc}`",
            f"- Notes: {status.notes}",
            "",
        ]
    )
    output_md.write_text(md, encoding="utf-8")
    output_json.write_text(json.dumps(asdict(status), indent=2), encoding="utf-8")


def _append_history(status: Lens2Status, history_md: Path) -> None:
    history_md.parent.mkdir(parents=True, exist_ok=True)
    entry = "\n".join(
        [
            f"## {status.checked_at_utc}",
            f"- State: `{status.run_state}`",
            f"- Progress: `{status.latest_total_timesteps:,}` / `{status.absolute_target_timesteps:,}` "
            f"(`{status.progress_pct_of_resume_run:.2f}%` of target run)",
            f"- Latest checkpoint: `{status.latest_checkpoint_name}`",
            f"- Best checkpoint: `{status.best_checkpoint_name}`",
            f"- Log: `{status.log_name}`",
            f"- Notes: {status.notes}",
            "",
        ]
    )
    with history_md.open("a", encoding="utf-8") as handle:
        handle.write(entry)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write hourly Lens2 retrain status snapshots.")
    parser.add_argument("--config", default="config/resume_lens2_retrain_500k_after_lens1.yaml")
    parser.add_argument("--model-dir", default="models/lens2_retrain_500k_after_lens1")
    parser.add_argument("--log-dir", default="output/training_logs")
    parser.add_argument("--log-pattern", default="resume_lens2_retrain*.out.log")
    parser.add_argument("--output-md", default="output/training_logs/lens2_hourly_status.md")
    parser.add_argument("--output-json", default="output/training_logs/lens2_hourly_status.json")
    parser.add_argument("--history-md", default="output/training_logs/lens2_hourly_status_history.md")
    parser.add_argument("--interval-minutes", type=int, default=60)
    parser.add_argument("--stale-minutes", type=int, default=75)
    parser.add_argument("--once", action="store_true", help="Write one snapshot and exit.")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    config_path = (project_root / args.config).resolve()
    model_dir = (project_root / args.model_dir).resolve()
    log_dir = (project_root / args.log_dir).resolve()
    output_md = (project_root / args.output_md).resolve()
    output_json = (project_root / args.output_json).resolve()
    history_md = (project_root / args.history_md).resolve()

    while True:
        status = _build_status(config_path, model_dir, log_dir, args.log_pattern, args.stale_minutes)
        _write_latest(status, output_md, output_json)
        _append_history(status, history_md)

        if args.once:
            break
        if status.run_state == "completed":
            break

        time.sleep(max(args.interval_minutes, 1) * 60)


if __name__ == "__main__":
    main()
