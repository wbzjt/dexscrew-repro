#!/usr/bin/env python3
"""Summarize IsaacGym CoDrive sim2sim reference JSON snapshots."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


SNAPSHOT_RE = re.compile(r"isaacgym_ref_env(?P<env>\d+)_step(?P<step>\d+)_(?P<phase>[^.]+)\.json$")
ACTIVE_BODY_TOKENS = ("right_index", "right_thumb")
OBJECT_BODY_TOKENS = ("codrive", "lightbulb", "object", "nut")


def as_scalar(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        if not value:
            return default
        return as_scalar(value[0], default=default)
    return default


def vector(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float64)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64).reshape(-1)
    return np.asarray([float(value)], dtype=np.float64)


def discover_snapshots(reference_dir: Path, phase: str, env_id: int | None) -> list[tuple[int, Path]]:
    snapshots: list[tuple[int, Path]] = []
    for path in reference_dir.glob("isaacgym_ref_env*_step*_*.json"):
        match = SNAPSHOT_RE.match(path.name)
        if match is None:
            continue
        if match.group("phase") != phase:
            continue
        current_env = int(match.group("env"))
        if env_id is not None and current_env != env_id:
            continue
        snapshots.append((int(match.group("step")), path))
    snapshots.sort(key=lambda item: item[0])
    return snapshots


def infer_nut_partner(payload: dict[str, Any], tolerance: float) -> tuple[str, float, str]:
    nut_force = as_scalar(payload.get("nut_contact_force_norm"), default=0.0)
    if not np.isfinite(nut_force) or nut_force <= tolerance:
        return "none", nut_force, "no_nut_force"

    candidates: list[tuple[float, str, float]] = []
    for body in payload.get("top_contact_bodies", []) or []:
        name = str(body.get("name", ""))
        force = as_scalar(body.get("force_norm"), default=0.0)
        if any(token in name for token in OBJECT_BODY_TOKENS):
            continue
        if not any(token in name for token in ACTIVE_BODY_TOKENS):
            continue
        diff = abs(force - nut_force)
        rel = diff / max(abs(nut_force), 1.0)
        if diff <= tolerance or rel <= tolerance:
            candidates.append((diff, name, force))

    if not candidates:
        return "none", nut_force, "unmatched_force"
    candidates.sort(key=lambda item: item[0])
    best_diff, best_name, best_force = candidates[0]
    if len(candidates) > 1 and abs(candidates[1][0] - best_diff) <= tolerance:
        return "ambiguous", nut_force, "multiple_matches"
    return best_name, best_force, "matched"


def body_force(payload: dict[str, Any], name: str) -> float:
    for body in payload.get("top_contact_bodies", []) or []:
        if body.get("name") == name:
            return as_scalar(body.get("force_norm"), default=0.0)
    return 0.0


def summarize_snapshot(step: int, path: Path, tolerance: float) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    partner, partner_force, partner_status = infer_nut_partner(payload, tolerance)
    action = vector(payload.get("policy_action"))
    mask = vector(payload.get("env_last_action_mask"))
    active_action = action
    if mask.size == action.size and action.size:
        active_action = action[np.abs(mask) > 0.5]

    return {
        "step": int(step),
        "path": str(path),
        "phase": str(payload.get("phase", "")),
        "dt": as_scalar(payload.get("dt")),
        "control_freq_inv": int(as_scalar(payload.get("control_freq_inv"), default=0)),
        "object_scale": as_scalar(payload.get("object_scale")),
        "nut_dof_pos": as_scalar(payload.get("nut_dof_pos")),
        "nut_dof_vel": as_scalar(payload.get("nut_dof_vel")),
        "nut_contact_force_norm": as_scalar(payload.get("nut_contact_force_norm"), default=0.0),
        "nut_partner": partner,
        "nut_partner_force_norm": float(partner_force),
        "nut_partner_status": partner_status,
        "index_tactile_link_2_force_norm": body_force(payload, "right_index_tactile_link_2"),
        "thumb_tactile_link_1_force_norm": body_force(payload, "right_thumb_tactile_link_1"),
        "index_tip_force_norm": body_force(payload, "right_index_tip"),
        "thumb_tip_force_norm": body_force(payload, "right_thumb_tip"),
        "max_abs_policy_action": float(np.max(np.abs(action))) if action.size else 0.0,
        "active_action_saturation_fraction": (
            float(np.mean(np.abs(active_action) >= 0.99)) if active_action.size else 0.0
        ),
    }


def summarize_reference(reference_dir: Path, phase: str, env_id: int | None, tolerance: float) -> tuple[dict, list[dict]]:
    snapshots = discover_snapshots(reference_dir, phase=phase, env_id=env_id)
    if not snapshots:
        raise FileNotFoundError(f"No {phase} snapshots found in {reference_dir}")

    rows = [summarize_snapshot(step, path, tolerance=tolerance) for step, path in snapshots]
    partners = [str(row["nut_partner"]) for row in rows]
    partner_counts = Counter(partners)
    active_partners = [partner for partner in partners if partner not in ("none", "ambiguous")]
    active_counts = Counter(active_partners)
    dominant_partner = ""
    dominant_count = 0
    if active_counts:
        dominant_partner, dominant_count = active_counts.most_common(1)[0]

    nut_pos = np.asarray([row["nut_dof_pos"] for row in rows], dtype=np.float64)
    nut_vel = np.asarray([row["nut_dof_vel"] for row in rows], dtype=np.float64)
    nut_force = np.asarray([row["nut_contact_force_norm"] for row in rows], dtype=np.float64)
    action_sat = np.asarray([row["active_action_saturation_fraction"] for row in rows], dtype=np.float64)
    object_scales = [row["object_scale"] for row in rows if np.isfinite(row["object_scale"])]

    summary = {
        "reference_dir": str(reference_dir),
        "phase": phase,
        "env_id": env_id,
        "steps": len(rows),
        "first_step": int(rows[0]["step"]),
        "last_step": int(rows[-1]["step"]),
        "dt": rows[0]["dt"],
        "control_freq_inv": rows[0]["control_freq_inv"],
        "policy_rate_hz": (
            float(1.0 / (rows[0]["dt"] * rows[0]["control_freq_inv"]))
            if rows[0]["dt"] and rows[0]["control_freq_inv"]
            else math.nan
        ),
        "object_scale": float(object_scales[0]) if object_scales else math.nan,
        "nut_axis_delta": float(nut_pos[-1] - nut_pos[0]) if len(nut_pos) else 0.0,
        "nut_mean_axis_vel": float(np.nanmean(nut_vel)) if len(nut_vel) else 0.0,
        "nut_max_abs_axis_vel": float(np.nanmax(np.abs(nut_vel))) if len(nut_vel) else 0.0,
        "mean_nut_contact_force_norm": float(np.nanmean(nut_force)) if len(nut_force) else 0.0,
        "max_nut_contact_force_norm": float(np.nanmax(nut_force)) if len(nut_force) else 0.0,
        "partner_counts": dict(partner_counts),
        "dominant_nut_partner": dominant_partner,
        "dominant_nut_partner_fraction": float(dominant_count / max(1, len(active_partners))),
        "index_tactile_partner_fraction": float(
            sum(partner == "right_index_tactile_link_2" for partner in partners) / max(1, len(rows))
        ),
        "thumb_tactile_partner_fraction": float(
            sum(partner == "right_thumb_tactile_link_1" for partner in partners) / max(1, len(rows))
        ),
        "no_matched_partner_fraction": float(
            sum(partner == "none" for partner in partners) / max(1, len(rows))
        ),
        "ambiguous_partner_fraction": float(
            sum(partner == "ambiguous" for partner in partners) / max(1, len(rows))
        ),
        "mean_active_action_saturation_fraction": float(np.nanmean(action_sat)) if len(action_sat) else 0.0,
        "max_active_action_saturation_fraction": float(np.nanmax(action_sat)) if len(action_sat) else 0.0,
        "partner_sequence": partners,
    }
    return summary, rows


def write_tsv(rows: list[dict], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=Path, help="Directory containing isaacgym_ref_env*_step*_pre_step.json")
    parser.add_argument("--phase", default="pre_step", help="Snapshot phase to analyze")
    parser.add_argument("--env-id", type=int, default=None, help="Optional env id filter")
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=1e-4,
        help="Absolute or relative force tolerance for inferring nut partner bodies",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-tsv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_dir = args.reference_dir.expanduser().resolve()
    summary, rows = summarize_reference(
        reference_dir=reference_dir,
        phase=args.phase,
        env_id=args.env_id,
        tolerance=args.match_tolerance,
    )
    output_json = args.output_json.expanduser().resolve() if args.output_json else reference_dir / "isaacgym_reference_summary.json"
    output_tsv = args.output_tsv.expanduser().resolve() if args.output_tsv else reference_dir / "isaacgym_reference_steps.tsv"
    output_json.write_text(json.dumps({"summary": summary, "steps": rows}, indent=2), encoding="utf-8")
    write_tsv(rows, output_tsv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
