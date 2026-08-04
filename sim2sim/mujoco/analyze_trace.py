#!/usr/bin/env python3
"""Summarize MuJoCo CoDrive rollout CSV traces."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np


def numeric_column(rows: list[dict], key: str) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw in ("", "None", None):
            values.append(np.nan)
        else:
            values.append(float(raw))
    return np.asarray(values, dtype=np.float64)


def finite_or_zero(values: np.ndarray) -> np.ndarray:
    if np.isnan(values).all():
        return np.zeros_like(values, dtype=np.float64)
    return np.nan_to_num(values, nan=0.0)


def columns_with_prefix(rows: list[dict], prefix: str) -> list[str]:
    if not rows:
        return []
    return [key for key in rows[0].keys() if key.startswith(prefix)]


def contact_pair_statistics(rows: list[dict]) -> dict:
    step_counter: Counter[str] = Counter()
    event_counter: Counter[str] = Counter()
    active_steps = 0
    event_total = 0
    for row in rows:
        pairs = [pair for pair in str(row.get("object_contact_pairs", "") or "").split(";") if pair]
        if pairs:
            active_steps += 1
        for pair in set(pairs):
            step_counter[pair] += 1
        for pair in pairs:
            event_counter[pair] += 1
            event_total += 1

    dominant_pair = ""
    dominant_steps = 0
    dominant_events = 0
    if step_counter:
        dominant_pair, dominant_steps = step_counter.most_common(1)[0]
        dominant_events = event_counter.get(dominant_pair, 0)

    denom_steps = max(1, active_steps)
    denom_events = max(1, event_total)
    return {
        "dominant_object_contact_pair": dominant_pair,
        "dominant_object_contact_pair_steps": int(dominant_steps),
        "dominant_object_contact_pair_step_fraction": float(dominant_steps / denom_steps),
        "dominant_object_contact_pair_events": int(dominant_events),
        "dominant_object_contact_pair_event_fraction": float(dominant_events / denom_events),
        "object_contact_pair_event_count": int(event_total),
        "object_contact_pair_unique_count": int(len(event_counter)),
    }


def summarize_trace(csv_path: Path) -> dict:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    time_s = numeric_column(rows, "time_s")
    object_x = numeric_column(rows, "object_x")
    object_y = numeric_column(rows, "object_y")
    object_z = numeric_column(rows, "object_z")
    object_geom_x = numeric_column(rows, "object_geom_x")
    object_geom_y = numeric_column(rows, "object_geom_y")
    object_geom_z = numeric_column(rows, "object_geom_z")
    object_yaw = numeric_column(rows, "object_yaw")
    object_axis_pos = numeric_column(rows, "object_axis_pos")
    object_axis_vel = numeric_column(rows, "object_axis_vel")
    object_angvel_z = numeric_column(rows, "object_angvel_z")
    if np.isnan(object_axis_pos).all():
        object_axis_pos = object_yaw.copy()
    if np.isnan(object_axis_vel).all():
        object_axis_vel = object_angvel_z.copy()
    object_pos = np.stack([object_x, object_y, object_z], axis=1)
    if np.isnan(object_geom_x).all():
        object_geom_pos = object_pos.copy()
        object_geom_z = object_z.copy()
    else:
        object_geom_pos = np.stack([object_geom_x, object_geom_y, object_geom_z], axis=1)
    object_drift = np.linalg.norm(object_pos - object_pos[0], axis=1)
    object_geom_drift = np.linalg.norm(object_geom_pos - object_geom_pos[0], axis=1)

    action_cols = columns_with_prefix(rows, "action_")
    target_cols = columns_with_prefix(rows, "target_")
    q_cols = columns_with_prefix(rows, "q_")
    tau_cols = columns_with_prefix(rows, "tau_")
    active_tokens = ("right_index", "right_thumb")
    active_action_cols = [c for c in action_cols if any(tok in c for tok in active_tokens)]
    active_target_cols = [c for c in target_cols if any(tok in c for tok in active_tokens)]
    active_q_cols = [c.replace("target_", "q_") for c in active_target_cols]

    actions = np.stack([numeric_column(rows, c) for c in action_cols], axis=1) if action_cols else np.zeros((len(rows), 0))
    active_actions = (
        np.stack([numeric_column(rows, c) for c in active_action_cols], axis=1)
        if active_action_cols
        else np.zeros((len(rows), 0))
    )
    tau = np.stack([numeric_column(rows, c) for c in tau_cols], axis=1) if tau_cols else np.zeros((len(rows), 0))

    tracking_error = np.zeros((len(rows), 0))
    if active_target_cols and active_q_cols:
        targets = np.stack([numeric_column(rows, c) for c in active_target_cols], axis=1)
        q = np.stack([numeric_column(rows, c) for c in active_q_cols], axis=1)
        tracking_error = targets - q

    ncon = numeric_column(rows, "ncon")
    active_contact = numeric_column(rows, "active_contact_count")
    finger_names = ("index", "middle", "ring", "thumb")
    active_contact = finite_or_zero(active_contact)
    finger_contacts = {finger: finite_or_zero(numeric_column(rows, f"{finger}_object_contact_count")) for finger in finger_names}
    finger_forces = {finger: finite_or_zero(numeric_column(rows, f"{finger}_object_contact_force")) for finger in finger_names}
    finger_tip_contacts = {finger: finite_or_zero(numeric_column(rows, f"{finger}_tip_contact_count")) for finger in finger_names}
    finger_tip_forces = {finger: finite_or_zero(numeric_column(rows, f"{finger}_tip_contact_force")) for finger in finger_names}
    active_force = finite_or_zero(numeric_column(rows, "active_contact_force"))
    active_tip_contact = finite_or_zero(numeric_column(rows, "active_tip_contact_count"))
    active_tip_force = finite_or_zero(numeric_column(rows, "active_tip_contact_force"))
    index_thumb_overlap = np.logical_and(finger_contacts["index"] > 0, finger_contacts["thumb"] > 0)
    index_only_contact = np.logical_and(finger_contacts["index"] > 0, finger_contacts["thumb"] <= 0)
    thumb_only_contact = np.logical_and(finger_contacts["thumb"] > 0, finger_contacts["index"] <= 0)
    no_active_contact = np.logical_and(finger_contacts["index"] <= 0, finger_contacts["thumb"] <= 0)
    index_thumb_tip_overlap = np.logical_and(finger_tip_contacts["index"] > 0, finger_tip_contacts["thumb"] > 0)

    yaw_delta = float(object_yaw[-1] - object_yaw[0])
    # unwrap for a less jumpy integrated delta if long traces cross +/- pi.
    yaw_unwrapped = np.unwrap(object_yaw)
    yaw_unwrapped_delta = float(yaw_unwrapped[-1] - yaw_unwrapped[0])

    summary = {
        "csv": str(csv_path),
        "steps": int(len(rows)),
        "duration_s": float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0,
        "max_abs_action": float(np.nanmax(np.abs(actions))) if actions.size else 0.0,
        "active_action_saturation_fraction": (
            float(np.mean(np.abs(active_actions) >= 0.99)) if active_actions.size else 0.0
        ),
        "all_action_saturation_fraction": (
            float(np.mean(np.abs(actions) >= 0.99)) if actions.size else 0.0
        ),
        "max_abs_tau": float(np.nanmax(np.abs(tau))) if tau.size else 0.0,
        "active_tracking_rmse": (
            float(math.sqrt(np.nanmean(tracking_error * tracking_error))) if tracking_error.size else 0.0
        ),
        "active_tracking_max_abs": (
            float(np.nanmax(np.abs(tracking_error))) if tracking_error.size else 0.0
        ),
        "max_contract_violation": float(np.nanmax(numeric_column(rows, "max_contract_violation"))),
        "max_ncon": int(np.nanmax(ncon)),
        "mean_ncon": float(np.nanmean(ncon)),
        "active_contact_fraction": float(np.mean(active_contact > 0)),
        "mean_active_contact_count": float(np.nanmean(active_contact)),
        "max_active_contact_count": int(np.nanmax(active_contact)),
        "mean_active_contact_force": float(np.nanmean(active_force)),
        "max_active_contact_force": float(np.nanmax(active_force)),
        "active_tip_contact_fraction": float(np.mean(active_tip_contact > 0)),
        "mean_active_tip_contact_count": float(np.nanmean(active_tip_contact)),
        "max_active_tip_contact_count": int(np.nanmax(active_tip_contact)),
        "mean_active_tip_contact_force": float(np.nanmean(active_tip_force)),
        "max_active_tip_contact_force": float(np.nanmax(active_tip_force)),
        "index_thumb_overlap_fraction": float(np.mean(index_thumb_overlap)),
        "index_only_contact_fraction": float(np.mean(index_only_contact)),
        "thumb_only_contact_fraction": float(np.mean(thumb_only_contact)),
        "no_active_contact_fraction": float(np.mean(no_active_contact)),
        "mean_index_thumb_min_contact_count": float(
            np.mean(np.minimum(finger_contacts["index"], finger_contacts["thumb"]))
        ),
        "index_thumb_tip_overlap_fraction": float(np.mean(index_thumb_tip_overlap)),
        "mean_index_thumb_tip_min_contact_count": float(
            np.mean(np.minimum(finger_tip_contacts["index"], finger_tip_contacts["thumb"]))
        ),
        "object_initial_pos": [float(v) for v in object_pos[0]],
        "object_final_pos": [float(v) for v in object_pos[-1]],
        "object_initial_geom_pos": [float(v) for v in object_geom_pos[0]],
        "object_final_geom_pos": [float(v) for v in object_geom_pos[-1]],
        "object_final_z": float(object_z[-1]),
        "object_final_geom_z": float(object_geom_z[-1]),
        "object_max_drift": float(np.nanmax(object_drift)),
        "object_final_drift": float(object_drift[-1]),
        "object_max_geom_drift": float(np.nanmax(object_geom_drift)),
        "object_final_geom_drift": float(object_geom_drift[-1]),
        "object_yaw_delta": yaw_delta,
        "object_yaw_unwrapped_delta": yaw_unwrapped_delta,
        "object_axis_delta": float(object_axis_pos[-1] - object_axis_pos[0]),
        "object_mean_axis_vel": float(np.nanmean(object_axis_vel)),
        "object_max_abs_axis_vel": float(np.nanmax(np.abs(object_axis_vel))),
        "object_positive_axis_vel_fraction": float(np.mean(object_axis_vel > 0.0)),
        "object_mean_angvel_z": float(np.nanmean(object_angvel_z)),
        "object_max_abs_angvel_z": float(np.nanmax(np.abs(object_angvel_z))),
    }
    summary.update(contact_pair_statistics(rows))
    for finger in finger_names:
        summary[f"{finger}_contact_fraction"] = float(np.mean(finger_contacts[finger] > 0))
        summary[f"mean_{finger}_contact_count"] = float(np.nanmean(finger_contacts[finger]))
        summary[f"max_{finger}_contact_count"] = int(np.nanmax(finger_contacts[finger]))
        summary[f"mean_{finger}_contact_force"] = float(np.nanmean(finger_forces[finger]))
        summary[f"max_{finger}_contact_force"] = float(np.nanmax(finger_forces[finger]))
        summary[f"{finger}_tip_contact_fraction"] = float(np.mean(finger_tip_contacts[finger] > 0))
        summary[f"mean_{finger}_tip_contact_count"] = float(np.nanmean(finger_tip_contacts[finger]))
        summary[f"max_{finger}_tip_contact_count"] = int(np.nanmax(finger_tip_contacts[finger]))
        summary[f"mean_{finger}_tip_contact_force"] = float(np.nanmean(finger_tip_forces[finger]))
        summary[f"max_{finger}_tip_contact_force"] = float(np.nanmax(finger_tip_forces[finger]))
    return summary


def write_tsv(summary: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()), delimiter="\t")
        writer.writeheader()
        writer.writerow(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Trace CSV from run_codrive_sim2sim.py")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-tsv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.expanduser().resolve()
    summary = summarize_trace(csv_path)
    output_json = args.output_json.expanduser().resolve() if args.output_json else csv_path.with_name("analysis_summary.json")
    output_tsv = args.output_tsv.expanduser().resolve() if args.output_tsv else csv_path.with_name("analysis_summary.tsv")
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_tsv(summary, output_tsv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
