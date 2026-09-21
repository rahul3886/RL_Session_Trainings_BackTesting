"""
Model checkpoint resolution helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _resolve_model_root(project_root: Path | str, model_dir: str) -> Path:
    root = Path(project_root)
    model_root = Path(model_dir)
    if not model_root.is_absolute():
        model_root = root / model_root
    return model_root.resolve()


def resolve_model_checkpoint(
    project_root: Path | str,
    model_dir: str,
    model_prefix: str,
    selector: Optional[str] = None,
) -> Path:
    """
    Resolve a model checkpoint selector into a concrete ``.zip`` path.
    """
    selector_norm = (selector or "final").strip()
    selector_key = selector_norm.lower()
    model_root = _resolve_model_root(project_root, model_dir)

    if selector_key in {"", "final"}:
        path = model_root / f"{model_prefix}_ppo_final.zip"
    elif selector_key in {"best", "best_eval"}:
        path = model_root / "best" / "best_model.zip"
    elif selector_key == "structural_best":
        path = model_root / "structural_best" / "structural_best_model.zip"
    elif selector_key.startswith("step:"):
        step_value = selector_key.split(":", 1)[1].strip()
        if not step_value.isdigit():
            raise ValueError(f"Invalid checkpoint selector: {selector_norm}")
        path = model_root / f"{model_prefix}_{step_value}_steps.zip"
    else:
        path = Path(selector_norm)
        if not path.is_absolute():
            path = Path(project_root) / path

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    return path


def maybe_resolve_model_checkpoint(
    project_root: Path | str,
    model_dir: str,
    model_prefix: str,
    selector: Optional[str] = None,
) -> Optional[Path]:
    selector_norm = (selector or "").strip().lower()
    if selector_norm in {"", "none", "new"}:
        return None
    return resolve_model_checkpoint(project_root, model_dir, model_prefix, selector)
