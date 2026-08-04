#!/usr/bin/env python3
"""Sweep MuJoCo reset poses and report immediate object/finger contacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

try:
    import mujoco
except ModuleNotFoundError as exc:
    raise SystemExit("MuJoCo Python bindings are required: pip install mujoco") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim2sim.mujoco.run_codrive_sim2sim import (  # noqa: E402
    active_tip_geometry,
    apply_finger_contact_mode,
    apply_init_pose_file,
    apply_object_contact_mode,
    apply_reset,
    apply_reset_source,
    clamp_init_q_to_limits,
    configure_joint_limits,
    contact_summary,
    joint_mapping,
    load_task_contract,
)


def parse_float_list(text: str | None) -> list[float] | None:
    if text is None:
        return None
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one float")
    return values


def default_or_values(values: list[float] | None, default: float) -> list[float]:
    return [float(default)] if values is None else values


def classify_reset(summary: dict) -> str:
    index = int(summary.get("index_object_contact_count", 0))
    thumb = int(summary.get("thumb_object_contact_count", 0))
    active = int(summary.get("active_contact_count", 0))
    if active == 0:
        return "no_active_contact"
    if index > 0 and thumb > 0:
        return "index_thumb_contact"
    if index > 0:
        return "index_only_contact"
    if thumb > 0:
        return "thumb_only_contact"
    return "non_active_contact"


def reset_score(summary: dict) -> tuple[int, int, int, int]:
    index = int(summary.get("index_object_contact_count", 0))
    thumb = int(summary.get("thumb_object_contact_count", 0))
    active = int(summary.get("active_contact_count", 0))
    object_contacts = int(summary.get("object_contact_count", 0))
    return (active, abs(index - thumb), object_contacts, index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=REPO_ROOT / "sim2sim/mujoco/scene_dexh13_lightbulb_hinge_fingertips.xml")
    parser.add_argument("--task-config", type=Path, default=REPO_ROOT / "sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml")
    parser.add_argument("--init-pose-file", type=Path)
    parser.add_argument(
        "--reset-source",
        choices=["yaml", "isaacgym_screwdriver"],
        default="yaml",
        help="Reset source passed through to run_codrive_sim2sim.py.",
    )
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--joint-limit-mode", choices=["task", "model"], default="task")
    parser.add_argument("--finger-contact-mode", choices=["full", "active_pad_proxy", "tip_proxy"], default="tip_proxy")
    parser.add_argument(
        "--object-contact-mode",
        choices=["default", "low_friction_debug", "high_hinge_friction_debug", "low_friction_high_hinge_debug"],
        default="default",
    )
    parser.add_argument("--object-x-values", default="0.010,0.012,0.014,0.016,0.018")
    parser.add_argument("--object-y-values", default="-0.030,-0.026,-0.023,-0.020,-0.017")
    parser.add_argument("--object-z-values", default="0.018,0.020,0.022,0.024,0.026")
    parser.add_argument("--hand-x-values")
    parser.add_argument("--hand-y-values")
    parser.add_argument("--hand-z-values", default="0.245,0.249,0.253,0.257")
    parser.add_argument("--hand-roll-values")
    parser.add_argument("--hand-pitch-values", default="0.330,0.350,0.370,0.390,0.410")
    parser.add_argument("--hand-yaw-values")
    parser.add_argument("--settle-steps", type=int, default=0, help="Optional passive MuJoCo steps after reset before measuring contacts.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/sim2sim_mujoco_alignment/reset_contact_sweep")
    args = parser.parse_args()
    args.scene = args.scene.expanduser().resolve()
    args.task_config = args.task_config.expanduser().resolve()
    if args.init_pose_file is not None:
        args.init_pose_file = args.init_pose_file.expanduser().resolve()

    contract = apply_reset_source(apply_init_pose_file(load_task_contract(args.task_config), args.init_pose_file), args.reset_source)
    contract.dt = float(args.dt)
    contract.finger_contact_mode = args.finger_contact_mode
    contract.object_contact_mode = args.object_contact_mode
    if not args.output_dir.is_absolute():
        args.output_dir = REPO_ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    model.opt.timestep = contract.dt
    contract.disabled_finger_mesh_contact_geoms = apply_finger_contact_mode(model, contract.finger_contact_mode)
    contract.modified_object_contact_geoms, contract.object_hinge_frictionloss = apply_object_contact_mode(
        model, contract.object_contact_mode
    )
    configure_joint_limits(model, contract, args.joint_limit_mode)
    clamp_init_q_to_limits(contract)
    data = mujoco.MjData(model)
    qpos_adrs, dof_adrs = joint_mapping(model)

    oxs = parse_float_list(args.object_x_values)
    oys = parse_float_list(args.object_y_values)
    ozs = parse_float_list(args.object_z_values)
    hxs = default_or_values(parse_float_list(args.hand_x_values), float(contract.hand_root_pos[0]))
    hys = default_or_values(parse_float_list(args.hand_y_values), float(contract.hand_root_pos[1]))
    hzs = default_or_values(parse_float_list(args.hand_z_values), float(contract.hand_root_pos[2]))
    hrs = default_or_values(parse_float_list(args.hand_roll_values), float(contract.hand_root_rpy[0]))
    hps = default_or_values(parse_float_list(args.hand_pitch_values), float(contract.hand_root_rpy[1]))
    hysaw = default_or_values(parse_float_list(args.hand_yaw_values), float(contract.hand_root_rpy[2]))

    rows = []
    for object_pos, hand_pos, hand_rpy in product(
        product(oxs, oys, ozs),
        product(hxs, hys, hzs),
        product(hrs, hps, hysaw),
    ):
        contract.object_init_pos = np.asarray(object_pos, dtype=np.float32)
        contract.hand_root_pos = np.asarray(hand_pos, dtype=np.float32)
        contract.hand_root_rpy = np.asarray(hand_rpy, dtype=np.float32)
        apply_reset(model, data, contract, qpos_adrs, dof_adrs)
        for _ in range(max(0, int(args.settle_steps))):
            mujoco.mj_step(model, data)
        summary = contact_summary(model, data)
        summary.update(active_tip_geometry(model, data))
        label = classify_reset(summary)
        score = reset_score(summary)
        rows.append(
            {
                "object_x": object_pos[0],
                "object_y": object_pos[1],
                "object_z": object_pos[2],
                "hand_x": hand_pos[0],
                "hand_y": hand_pos[1],
                "hand_z": hand_pos[2],
                "hand_roll": hand_rpy[0],
                "hand_pitch": hand_rpy[1],
                "hand_yaw": hand_rpy[2],
                "reset_source": contract.reset_source,
                "reset_class": label,
                "reset_score": ":".join(str(v) for v in score),
                **summary,
            }
        )

    rows.sort(key=lambda row: tuple(int(v) for v in row["reset_score"].split(":")))
    out_path = args.output_dir / "reset_contact_summary.tsv"
    fieldnames = list(rows[0].keys()) if rows else ["reset_class"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    class_counts = {}
    for row in rows:
        class_counts[row["reset_class"]] = class_counts.get(row["reset_class"], 0) + 1
    best = rows[: min(10, len(rows))]
    print(
        json.dumps(
            {
                "rows": len(rows),
                "class_counts": class_counts,
                "finger_contact_mode": contract.finger_contact_mode,
                "object_contact_mode": contract.object_contact_mode,
                "reset_source": contract.reset_source,
                "disabled_finger_mesh_contact_geoms": int(contract.disabled_finger_mesh_contact_geoms),
                "modified_object_contact_geoms": int(contract.modified_object_contact_geoms),
                "object_hinge_frictionloss": float(contract.object_hinge_frictionloss),
                "tsv": str(out_path),
                "best": best,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
