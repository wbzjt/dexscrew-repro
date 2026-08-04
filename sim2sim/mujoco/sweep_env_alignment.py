#!/usr/bin/env python3
"""Sweep MuJoCo pose/rate/joint-limit alignment settings for CoDrive sim2sim."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from itertools import product
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "sim2sim/mujoco/run_codrive_sim2sim.py"
ROLLOUT_SUMMARY_FIELDS = [
    "object_final_z",
    "object_final_geom_z",
    "object_final_drift",
    "object_max_drift",
    "object_final_geom_drift",
    "object_max_geom_drift",
    "object_yaw_delta",
    "object_axis_delta",
    "object_mean_axis_vel",
    "object_max_abs_axis_vel",
    "object_positive_axis_vel_fraction",
    "object_mean_angvel_z",
    "object_max_abs_angvel_z",
    "mean_index_tip_to_object_geom_dist",
    "min_index_tip_to_object_geom_dist",
    "mean_thumb_tip_to_object_geom_dist",
    "min_thumb_tip_to_object_geom_dist",
    "mean_index_thumb_tip_dist",
    "min_index_thumb_tip_dist",
    "max_ncon",
    "mean_ncon",
    "active_contact_fraction",
    "mean_active_contact_count",
    "max_active_contact_count",
    "mean_active_contact_force",
    "max_active_contact_force",
    "active_tip_contact_fraction",
    "mean_active_tip_contact_count",
    "max_active_tip_contact_count",
    "mean_active_tip_contact_force",
    "max_active_tip_contact_force",
    "finger_contact_mode",
    "object_contact_mode",
    "reset_source",
    "disabled_finger_mesh_contact_geoms",
    "modified_object_contact_geoms",
    "object_hinge_frictionloss",
    "index_thumb_overlap_fraction",
    "index_only_contact_fraction",
    "thumb_only_contact_fraction",
    "no_active_contact_fraction",
    "mean_index_thumb_min_contact_count",
    "index_thumb_tip_overlap_fraction",
    "mean_index_thumb_tip_min_contact_count",
    "dominant_object_contact_pair",
    "dominant_object_contact_pair_step_fraction",
    "dominant_object_contact_pair_event_fraction",
    "object_contact_pair_event_count",
    "object_contact_pair_unique_count",
]
for _finger in ("index", "middle", "ring", "thumb"):
    ROLLOUT_SUMMARY_FIELDS.extend(
        [
            f"{_finger}_contact_fraction",
            f"mean_{_finger}_contact_count",
            f"max_{_finger}_contact_count",
            f"mean_{_finger}_contact_force",
            f"max_{_finger}_contact_force",
            f"{_finger}_tip_contact_fraction",
            f"mean_{_finger}_tip_contact_count",
            f"max_{_finger}_tip_contact_count",
            f"mean_{_finger}_tip_contact_force",
            f"max_{_finger}_tip_contact_force",
        ]
    )


def parse_vec3_list(text: str) -> list[tuple[float, float, float]]:
    out = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        values = [float(v) for v in item.split(",")]
        if len(values) != 3:
            raise ValueError(f"Expected X,Y,Z triple, got: {item}")
        out.append((values[0], values[1], values[2]))
    if not out:
        raise ValueError("No vector triples parsed")
    return out


def parse_dt_decimations(text: str) -> list[tuple[float, int]]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        dt_s, dec_s = item.split(":")
        out.append((float(dt_s), int(dec_s)))
    if not out:
        raise ValueError("No dt:decimation pairs parsed")
    return out


def parse_csv_list(text: str) -> list[str]:
    return [v.strip() for v in text.split(",") if v.strip()]


def parse_float_list(text: str | None) -> list[float] | None:
    if text is None:
        return None
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one float")
    return values


def run_one(
    args,
    mode: str,
    dt: float,
    decimation: int,
    object_pos,
    hand_root_pos,
    hand_root_rpy,
    limit_mode: str,
    hard_clamp: bool,
    run_dir: Path,
) -> dict:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--mode",
        mode,
        "--scene",
        str(args.scene),
        "--task-config",
        str(args.task_config),
        "--dt",
        str(dt),
        "--control-decimation",
        str(decimation),
        "--policy-hz",
        str(args.policy_hz),
        "--object-pos",
        str(object_pos[0]),
        str(object_pos[1]),
        str(object_pos[2]),
        "--joint-limit-mode",
        limit_mode,
        "--reset-source",
        args.reset_source,
        "--finger-contact-mode",
        args.finger_contact_mode,
        "--object-contact-mode",
        args.object_contact_mode,
        "--policy-steps",
        str(args.policy_steps),
        "--output-dir",
        str(run_dir),
    ]
    if args.init_pose_file is not None:
        cmd.extend(["--init-pose-file", str(args.init_pose_file)])
    if hand_root_pos is not None:
        cmd.extend(
            [
                "--hand-root-pos",
                str(hand_root_pos[0]),
                str(hand_root_pos[1]),
                str(hand_root_pos[2]),
            ]
        )
    if hand_root_rpy is not None:
        cmd.extend(
            [
                "--hand-root-rpy",
                str(hand_root_rpy[0]),
                str(hand_root_rpy[1]),
                str(hand_root_rpy[2]),
            ]
        )
    if hard_clamp:
        cmd.append("--hard-clamp-joints")
    if mode == "poke":
        cmd.extend(["--poke-joint", str(args.poke_joint), "--poke-action", str(args.poke_action)])

    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    record = {
        "mode": mode,
        "scene": str(args.scene),
        "task_config": str(args.task_config),
        "init_pose_file": None if args.init_pose_file is None else str(args.init_pose_file),
        "dt": dt,
        "decimation": decimation,
        "policy_hz": args.policy_hz,
        "object_x": object_pos[0],
        "object_y": object_pos[1],
        "object_z": object_pos[2],
        "hand_root_x": None if hand_root_pos is None else hand_root_pos[0],
        "hand_root_y": None if hand_root_pos is None else hand_root_pos[1],
        "hand_root_z": None if hand_root_pos is None else hand_root_pos[2],
        "hand_root_roll": None if hand_root_rpy is None else hand_root_rpy[0],
        "hand_root_pitch": None if hand_root_rpy is None else hand_root_rpy[1],
        "hand_root_yaw": None if hand_root_rpy is None else hand_root_rpy[2],
        "joint_limit_mode": limit_mode,
        "reset_source": args.reset_source,
        "finger_contact_mode": args.finger_contact_mode,
        "object_contact_mode": args.object_contact_mode,
        "hard_clamp": hard_clamp,
        "returncode": proc.returncode,
        "run_dir": str(run_dir),
    }
    if proc.returncode != 0:
        record.update(
            {
                "status": "error",
                "error_tail": (proc.stderr or proc.stdout).strip()[-500:],
            }
        )
        return record

    try:
        summary = json.loads(proc.stdout)
    except json.JSONDecodeError:
        record.update({"status": "bad_json", "error_tail": proc.stdout.strip()[-500:]})
        return record

    record.update(
        {
            "status": "ok",
            "physics_steps": summary.get("physics_steps"),
            "max_abs_q": summary.get("max_abs_q"),
            "max_abs_tau": summary.get("max_abs_tau"),
            "max_abs_action": summary.get("max_abs_action"),
            "max_contract_violation": summary.get("max_contract_violation"),
            "hard_clamp_events": summary.get("hard_clamp_events"),
            "csv": summary.get("csv"),
        }
    )
    for field in ROLLOUT_SUMMARY_FIELDS:
        record[field] = summary.get(field)
    record["coarse_two_finger_contact_fraction"] = min(
        float(record.get("index_contact_fraction") or 0.0),
        float(record.get("thumb_contact_fraction") or 0.0),
    )
    record["two_finger_contact_fraction"] = float(record.get("index_thumb_overlap_fraction") or 0.0)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-steps", type=int, default=10)
    parser.add_argument("--policy-hz", type=float, default=20.0)
    parser.add_argument(
        "--scene",
        type=Path,
        default=RUNNER.parent / "scene_dexh13_lightbulb.xml",
        help="MuJoCo XML scene passed through to run_codrive_sim2sim.py.",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=REPO_ROOT / "sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml",
        help="Task YAML passed through to run_codrive_sim2sim.py.",
    )
    parser.add_argument("--init-pose-file", type=Path, help="Optional init-pose YAML passed through to runner.")
    parser.add_argument("--modes", default="zero", help="Comma list from zero,poke,policy.")
    parser.add_argument(
        "--dt-decimations",
        default="0.001:50,0.002:25,0.005:10",
        help="Comma list of dt:decimation pairs. Keep dt*decimation near 1/policy_hz.",
    )
    parser.add_argument(
        "--object-positions",
        default="0,0,1.0;0.012,-0.018,0.05;0.012,-0.018,0.08;0.012,-0.018,0.12",
        help="Semicolon list of X,Y,Z triples.",
    )
    parser.add_argument("--object-x-values", help="Comma list of object x values. Requires y/z grid values too.")
    parser.add_argument("--object-y-values", help="Comma list of object y values. Requires x/z grid values too.")
    parser.add_argument("--object-z-values", help="Comma list of object z values. Requires x/y grid values too.")
    parser.add_argument("--hand-root-positions", help="Optional semicolon list of hand root X,Y,Z triples.")
    parser.add_argument("--hand-root-rpys", help="Optional semicolon list of hand root roll,pitch,yaw triples.")
    parser.add_argument("--joint-limit-modes", default="task", help="Comma list from task,model.")
    parser.add_argument(
        "--reset-source",
        choices=["yaml", "isaacgym_screwdriver"],
        default="yaml",
        help="Reset source passed through to run_codrive_sim2sim.py.",
    )
    parser.add_argument(
        "--finger-contact-mode",
        choices=["full", "active_pad_proxy", "tip_proxy"],
        default="full",
        help="Pass through to run_codrive_sim2sim.py.",
    )
    parser.add_argument(
        "--object-contact-mode",
        choices=["default", "low_friction_debug", "high_hinge_friction_debug", "low_friction_high_hinge_debug"],
        default="default",
        help="Pass through to run_codrive_sim2sim.py.",
    )
    parser.add_argument("--include-hard-clamp", action="store_true")
    parser.add_argument("--poke-joint", type=int, default=0)
    parser.add_argument("--poke-action", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/sim2sim_mujoco_alignment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.scene = args.scene.expanduser().resolve()
    args.task_config = args.task_config.expanduser().resolve()
    if args.init_pose_file is not None:
        args.init_pose_file = args.init_pose_file.expanduser().resolve()
    modes = parse_csv_list(args.modes)
    dt_decimations = parse_dt_decimations(args.dt_decimations)
    grid_values = (
        parse_float_list(args.object_x_values),
        parse_float_list(args.object_y_values),
        parse_float_list(args.object_z_values),
    )
    if any(values is not None for values in grid_values):
        if not all(values is not None for values in grid_values):
            raise ValueError("--object-x-values, --object-y-values, and --object-z-values must be provided together")
        object_positions = list(product(grid_values[0], grid_values[1], grid_values[2]))
    else:
        object_positions = parse_vec3_list(args.object_positions)
    hand_root_positions = [None] if not args.hand_root_positions else parse_vec3_list(args.hand_root_positions)
    hand_root_rpys = [None] if not args.hand_root_rpys else parse_vec3_list(args.hand_root_rpys)
    limit_modes = parse_csv_list(args.joint_limit_modes)
    hard_clamp_options = [False, True] if args.include_hard_clamp else [False]

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_runs = (
        len(modes)
        * len(dt_decimations)
        * len(object_positions)
        * len(hand_root_positions)
        * len(hand_root_rpys)
        * len(limit_modes)
        * len(hard_clamp_options)
    )
    for idx, (mode, (dt, decimation), object_pos, hand_root_pos, hand_root_rpy, limit_mode, hard_clamp) in enumerate(
        product(modes, dt_decimations, object_positions, hand_root_positions, hand_root_rpys, limit_modes, hard_clamp_options)
    ):
        run_dir = args.output_dir / (
            f"run_{idx:04d}_{mode}_dt{dt:g}_dec{decimation}_"
            f"obj{object_pos[0]:+.3f}_{object_pos[1]:+.3f}_{object_pos[2]:+.3f}_"
            f"hand{0.0 if hand_root_pos is None else hand_root_pos[0]:+.3f}_"
            f"{0.0 if hand_root_pos is None else hand_root_pos[1]:+.3f}_"
            f"{0.0 if hand_root_pos is None else hand_root_pos[2]:+.3f}_"
            f"lim{limit_mode}_hard{int(hard_clamp)}"
        )
        record = run_one(args, mode, dt, decimation, object_pos, hand_root_pos, hand_root_rpy, limit_mode, hard_clamp, run_dir)
        records.append(record)
        print(
            f"[{idx + 1:03d}/{total_runs:03d}] "
            f"{record['status']} mode={mode} dt={dt:g} dec={decimation} obj={object_pos} "
            f"hand={hand_root_pos} rpy={hand_root_rpy} "
            f"limit={limit_mode} hard={hard_clamp} violation={record.get('max_contract_violation')}"
        )

    summary_path = args.output_dir / "alignment_summary.tsv"
    fieldnames = sorted({key for record in records for key in record.keys()})
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)

    ok_records = [r for r in records if r.get("status") == "ok"]
    ok_records.sort(
        key=lambda r: (
            float(r.get("max_contract_violation") or 1e9),
            float(r.get("max_abs_tau") or 1e9),
            -float(r.get("object_z") or 0.0),
        )
    )
    print(f"\nsummary: {summary_path}")
    print("top ok records:")
    for record in ok_records[:8]:
        print(
            f"  mode={record['mode']} dt={record['dt']} dec={record['decimation']} "
            f"obj=({record['object_x']},{record['object_y']},{record['object_z']}) "
            f"limit={record['joint_limit_mode']} hard={record['hard_clamp']} "
            f"viol={record['max_contract_violation']} tau={record['max_abs_tau']}"
        )

    contact_records = [r for r in ok_records if r.get("mode") == "policy"]
    contact_records.sort(
        key=lambda r: (
            -float(r.get("two_finger_contact_fraction") or 0.0),
            -float(r.get("coarse_two_finger_contact_fraction") or 0.0),
            -abs(float(r.get("object_yaw_delta") or 0.0)),
            float(r.get("object_final_drift") or 1e9),
            float(r.get("max_contract_violation") or 1e9),
        )
    )
    if contact_records:
        print("\ntop contact records:")
        for record in contact_records[:8]:
            print(
                f"  obj=({record['object_x']},{record['object_y']},{record['object_z']}) "
                f"dt={record['dt']} dec={record['decimation']} "
                f"overlap={record.get('two_finger_contact_fraction')} "
                f"coarse_two={record.get('coarse_two_finger_contact_fraction')} "
                f"idx={record.get('index_contact_fraction')} thumb={record.get('thumb_contact_fraction')} "
                f"yaw={record.get('object_yaw_delta')} drift={record.get('object_final_drift')} "
                f"viol={record.get('max_contract_violation')}"
            )
    tip_distance_records = [r for r in contact_records if r.get("mean_index_tip_to_object_geom_dist") is not None]
    tip_distance_records.sort(
        key=lambda r: (
            float(r.get("mean_index_tip_to_object_geom_dist") or 1e9)
            + float(r.get("mean_thumb_tip_to_object_geom_dist") or 1e9),
            abs(
                float(r.get("mean_index_tip_to_object_geom_dist") or 1e9)
                - float(r.get("mean_thumb_tip_to_object_geom_dist") or 1e9)
            ),
            -float(r.get("object_axis_delta") or 0.0),
        )
    )
    if tip_distance_records:
        print("\ntop tip-distance records:")
        for record in tip_distance_records[:8]:
            print(
                f"  obj=({record['object_x']},{record['object_y']},{record['object_z']}) "
                f"hand=({record['hand_root_x']},{record['hand_root_y']},{record['hand_root_z']}) "
                f"rpy=({record['hand_root_roll']},{record['hand_root_pitch']},{record['hand_root_yaw']}) "
                f"idx_d={record.get('mean_index_tip_to_object_geom_dist')} "
                f"thumb_d={record.get('mean_thumb_tip_to_object_geom_dist')} "
                f"idx={record.get('index_tip_contact_fraction')} "
                f"thumb={record.get('thumb_tip_contact_fraction')} "
                f"tip_overlap={record.get('index_thumb_tip_overlap_fraction')} "
                f"axis={record.get('object_axis_delta')}"
            )


if __name__ == "__main__":
    main()
