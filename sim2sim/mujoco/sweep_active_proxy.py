#!/usr/bin/env python3
"""Sweep generated active distal proxy geometry from a fixed IsaacGym state."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "sim2sim/mujoco/run_codrive_sim2sim.py"
DEFAULT_REFERENCE_STATE = (
    REPO_ROOT
    / "outputs/sim2sim_isaacgym_reference/codrive_ref_cpu_20_bodycontacts/isaacgym_ref_env0_step0000_pre_step.json"
)
DEFAULT_REFERENCE_SUMMARY = (
    REPO_ROOT / "outputs/sim2sim_isaacgym_reference/codrive_ref_cpu_20_bodycontacts/isaacgym_reference_summary.json"
)


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError(f"No floats parsed from {text!r}")
    return values


def parse_vec_list(text: str, width: int) -> list[tuple[float, ...]]:
    out: list[tuple[float, ...]] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        values = tuple(float(v.strip()) for v in item.replace(":", ",").split(",") if v.strip())
        if len(values) != width:
            raise ValueError(f"Expected {width} values in {item!r}, got {len(values)}")
        out.append(values)
    if not out:
        raise ValueError(f"No vectors parsed from {text!r}")
    return out


def slug_float(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def slug_vec(prefix: str, values: tuple[float, ...]) -> str:
    return prefix + "x".join(slug_float(v) for v in values)


def simplify_partner(text: str) -> str:
    if not text or text == "none":
        return "none"
    if "right_index_tactile_link_2" in text or "right_index_distal_proxy" in text:
        return "index_tactile"
    if "right_thumb_tactile_link_1" in text or "right_thumb_distal_proxy" in text:
        return "thumb_tactile"
    if "right_index_tip" in text:
        return "index_tip"
    if "right_thumb_tip" in text:
        return "thumb_tip"
    if "right_index" in text:
        return "index_other"
    if "right_thumb" in text:
        return "thumb_other"
    return "other"


def simplify_contact_pairs(raw: str) -> str:
    pairs = [pair for pair in str(raw or "").split(";") if pair]
    tokens = {simplify_partner(pair) for pair in pairs}
    tokens.discard("none")
    if not tokens:
        return "none"
    active = {token for token in tokens if token.startswith("index") or token.startswith("thumb")}
    if not active:
        return "other"
    if len(active) == 1:
        return next(iter(active))
    if "index_tactile" in active and "thumb_tactile" in active:
        return "index_thumb_tactile"
    if any(token.startswith("index") for token in active) and any(token.startswith("thumb") for token in active):
        return "index_thumb_mixed"
    return "+".join(sorted(active))


def trace_sequence(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [simplify_contact_pairs(row.get("object_contact_pairs", "")) for row in rows]


def load_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", payload)
    return {
        "axis_delta": float(summary.get("nut_axis_delta", math.nan)),
        "dominant_fraction": float(summary.get("dominant_nut_partner_fraction", 0.60)),
        "sequence": [simplify_partner(item) for item in summary.get("partner_sequence", [])],
    }


def sequence_match_fraction(candidate: list[str], reference: list[str]) -> float:
    if not candidate or not reference:
        return math.nan
    n = min(len(candidate), len(reference))
    matches = 0
    for cand, ref in zip(candidate[:n], reference[:n]):
        if cand == ref:
            matches += 1
        elif ref == "index_tactile" and cand in {"index_tactile", "index_thumb_tactile"}:
            matches += 1
        elif ref == "thumb_tactile" and cand in {"thumb_tactile", "index_thumb_tactile"}:
            matches += 1
        elif ref == "none" and cand == "none":
            matches += 1
    return float(matches / max(1, n))


def score_candidate(policy: dict[str, Any], zero: dict[str, Any], reference: dict[str, Any], sequence_match: float) -> float:
    axis = float(policy.get("object_axis_delta") or 0.0)
    ref_axis = float(reference.get("axis_delta") or math.nan)
    axis_term = abs(axis - ref_axis) / abs(ref_axis) if math.isfinite(ref_axis) and abs(ref_axis) > 1e-9 else 0.0

    zero_axis = abs(float(zero.get("object_axis_delta") or 0.0))
    zero_term = max(0.0, zero_axis - 0.03) * 5.0

    dominant_fraction = float(policy.get("dominant_object_contact_pair_step_fraction") or 0.0)
    ref_dominant = float(reference.get("dominant_fraction") or 0.60)
    dominant_term = max(0.0, dominant_fraction - max(ref_dominant, 0.60))

    index_contact = float(policy.get("index_contact_fraction") or 0.0)
    thumb_contact = float(policy.get("thumb_contact_fraction") or 0.0)
    balance_term = abs(index_contact - thumb_contact) * 0.25

    shortcut_term = 0.0
    dominant_pair = str(policy.get("dominant_object_contact_pair") or "")
    if "right_thumb_tip" in dominant_pair or "right_index_tip" in dominant_pair:
        shortcut_term += 0.50
    if dominant_fraction >= 0.80:
        shortcut_term += 0.20

    sequence_term = 0.5 * (1.0 - sequence_match) if math.isfinite(sequence_match) else 0.25
    contract_violation = float(policy.get("max_contract_violation") or 0.0)
    contract_term = max(0.0, contract_violation - 0.03) * 2.0
    return float(axis_term + zero_term + dominant_term + balance_term + shortcut_term + sequence_term + contract_term)


def run_one(args: argparse.Namespace, run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    common = [
        sys.executable,
        str(RUNNER),
        "--scene",
        str(args.scene),
        "--task-config",
        str(args.task_config),
        "--reference-state-json",
        str(args.reference_state_json),
        "--joint-limit-mode",
        args.joint_limit_mode,
        "--policy-steps",
        str(args.policy_steps),
        "--finger-contact-mode",
        args.finger_contact_mode,
        "--active-proxy-profile",
        args.active_proxy_profile,
        "--index-proxy-pos",
        *[str(v) for v in config["index_proxy_pos"]],
        "--thumb-proxy-pos",
        *[str(v) for v in config["thumb_proxy_pos"]],
        "--active-proxy-size",
        str(config["active_proxy_size"]),
        "--active-proxy-margin",
        str(config["active_proxy_margin"]),
        "--index-proxy-size",
        str(config["index_proxy_size"]),
        "--thumb-proxy-size",
        str(config["thumb_proxy_size"]),
        "--index-proxy-margin",
        str(config["index_proxy_margin"]),
        "--thumb-proxy-margin",
        str(config["thumb_proxy_margin"]),
        "--index-proxy-half-length",
        str(config["index_proxy_half_length"]),
        "--thumb-proxy-half-length",
        str(config["thumb_proxy_half_length"]),
        "--object-friction",
        *[str(v) for v in config["object_friction"]],
        "--object-solref",
        *[str(v) for v in config["object_solref"]],
        "--object-solimp",
        *[str(v) for v in config["object_solimp"]],
        "--object-margin",
        str(config["object_margin"]),
        "--active-tip-margin",
        str(config["active_tip_margin"]),
    ]
    if args.no_clamp_init_q:
        common.append("--no-clamp-init-q")

    policy_dir = run_dir / "policy"
    zero_dir = run_dir / "zero"
    policy_cmd = common + ["--mode", "policy", "--output-dir", str(policy_dir)]
    zero_cmd = common + ["--mode", "zero", "--output-dir", str(zero_dir)]
    policy_proc = subprocess.run(policy_cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    zero_proc = subprocess.run(zero_cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    record: dict[str, Any] = {
        **config,
        "run_dir": str(run_dir),
        "policy_returncode": policy_proc.returncode,
        "zero_returncode": zero_proc.returncode,
    }
    if policy_proc.returncode != 0 or zero_proc.returncode != 0:
        record["status"] = "error"
        record["stderr"] = (policy_proc.stderr + "\n" + zero_proc.stderr)[-2000:]
        return record

    policy_summary = json.loads((policy_dir / "policy_summary.json").read_text(encoding="utf-8"))
    zero_summary = json.loads((zero_dir / "zero_summary.json").read_text(encoding="utf-8"))
    sequence = trace_sequence(policy_dir / "policy_trace.csv")
    sequence_match = sequence_match_fraction(sequence, args.reference["sequence"])
    record.update(
        {
            "status": "ok",
            "policy_csv": str(policy_dir / "policy_trace.csv"),
            "zero_csv": str(zero_dir / "zero_trace.csv"),
            "sequence": sequence,
            "sequence_match_fraction": sequence_match,
            "object_axis_delta": float(policy_summary.get("object_axis_delta") or 0.0),
            "zero_object_axis_delta": float(zero_summary.get("object_axis_delta") or 0.0),
            "policy_minus_zero_axis_delta": float(policy_summary.get("object_axis_delta") or 0.0)
            - float(zero_summary.get("object_axis_delta") or 0.0),
            "dominant_object_contact_pair": str(policy_summary.get("dominant_object_contact_pair") or ""),
            "dominant_object_contact_pair_step_fraction": float(
                policy_summary.get("dominant_object_contact_pair_step_fraction") or 0.0
            ),
            "index_contact_fraction": float(policy_summary.get("index_contact_fraction") or 0.0),
            "thumb_contact_fraction": float(policy_summary.get("thumb_contact_fraction") or 0.0),
            "index_thumb_overlap_fraction": float(policy_summary.get("index_thumb_overlap_fraction") or 0.0),
            "active_contact_fraction": float(policy_summary.get("active_contact_fraction") or 0.0),
            "mean_active_contact_force": float(policy_summary.get("mean_active_contact_force") or 0.0),
            "max_active_contact_force": float(policy_summary.get("max_active_contact_force") or 0.0),
            "max_contract_violation": float(policy_summary.get("max_contract_violation") or 0.0),
        }
    )
    record["score"] = score_candidate(policy_summary, zero_summary, args.reference, sequence_match)
    return record


def flatten(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    for key in ("index_proxy_pos", "thumb_proxy_pos", "object_friction", "object_solref", "object_solimp", "sequence"):
        value = out.get(key)
        if isinstance(value, (list, tuple)):
            out[key] = ";".join(str(v) for v in value)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=REPO_ROOT / "sim2sim/mujoco/scene_dexh13_lightbulb_hinge_isaacgym_parity.xml")
    parser.add_argument("--task-config", type=Path, default=REPO_ROOT / "sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml")
    parser.add_argument("--reference-state-json", type=Path, default=DEFAULT_REFERENCE_STATE)
    parser.add_argument("--reference-summary-json", type=Path, default=DEFAULT_REFERENCE_SUMMARY)
    parser.add_argument("--policy-steps", type=int, default=20)
    parser.add_argument("--joint-limit-mode", choices=["task", "model"], default="task")
    parser.add_argument("--no-clamp-init-q", action="store_true", default=True)
    parser.add_argument("--finger-contact-mode", choices=["active_distal_proxy", "active_proxy_only"], default="active_distal_proxy")
    parser.add_argument("--active-proxy-profile", choices=["distal_spheres", "tactile_capsules"], default="distal_spheres")
    parser.add_argument("--index-proxy-positions", default="0,0.006,0.0035;0,0.008,0.0035;0,0.010,0.0035")
    parser.add_argument("--thumb-proxy-positions", default="0,0.007,0.0035;0,0.009,0.0035;0,0.011,0.0035")
    parser.add_argument("--active-proxy-sizes", default="0.0035,0.0045,0.0055")
    parser.add_argument("--active-proxy-margins", default="0.0005,0.001")
    parser.add_argument("--index-proxy-sizes", help="Comma-separated index proxy radii. Defaults to --active-proxy-sizes.")
    parser.add_argument("--thumb-proxy-sizes", help="Comma-separated thumb proxy radii. Defaults to --active-proxy-sizes.")
    parser.add_argument("--index-proxy-margins", help="Comma-separated index proxy margins. Defaults to --active-proxy-margins.")
    parser.add_argument("--thumb-proxy-margins", help="Comma-separated thumb proxy margins. Defaults to --active-proxy-margins.")
    parser.add_argument("--index-proxy-half-lengths", default="0.0115")
    parser.add_argument("--thumb-proxy-half-lengths", default="0.0148")
    parser.add_argument("--object-friction", default="0.8,0.005,0.0001")
    parser.add_argument("--object-solref", default="0.015,1")
    parser.add_argument("--object-solimp", default="0.8,0.98,0.001")
    parser.add_argument("--object-margins", default="0.001,0.002")
    parser.add_argument("--active-tip-margins", default="0.0005,0.001")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/sim2sim_mujoco_contact_sweeps/active_proxy_geometry")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.scene = args.scene.expanduser().resolve()
    args.task_config = args.task_config.expanduser().resolve()
    args.reference_state_json = args.reference_state_json.expanduser().resolve()
    args.reference_summary_json = args.reference_summary_json.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reference = load_reference(args.reference_summary_json)

    configs = []
    index_proxy_sizes = parse_float_list(args.index_proxy_sizes or args.active_proxy_sizes)
    thumb_proxy_sizes = parse_float_list(args.thumb_proxy_sizes or args.active_proxy_sizes)
    index_proxy_margins = parse_float_list(args.index_proxy_margins or args.active_proxy_margins)
    thumb_proxy_margins = parse_float_list(args.thumb_proxy_margins or args.active_proxy_margins)
    for (
        index_pos,
        thumb_pos,
        index_proxy_size,
        thumb_proxy_size,
        index_proxy_margin,
        thumb_proxy_margin,
        index_proxy_half_length,
        thumb_proxy_half_length,
        object_margin,
        active_tip_margin,
    ) in product(
        parse_vec_list(args.index_proxy_positions, 3),
        parse_vec_list(args.thumb_proxy_positions, 3),
        index_proxy_sizes,
        thumb_proxy_sizes,
        index_proxy_margins,
        thumb_proxy_margins,
        parse_float_list(args.index_proxy_half_lengths),
        parse_float_list(args.thumb_proxy_half_lengths),
        parse_float_list(args.object_margins),
        parse_float_list(args.active_tip_margins),
    ):
        active_proxy_size = max(float(index_proxy_size), float(thumb_proxy_size))
        active_proxy_margin = max(float(index_proxy_margin), float(thumb_proxy_margin))
        configs.append(
            {
                "index_proxy_pos": tuple(index_pos),
                "thumb_proxy_pos": tuple(thumb_pos),
                "active_proxy_size": active_proxy_size,
                "active_proxy_margin": active_proxy_margin,
                "index_proxy_size": float(index_proxy_size),
                "thumb_proxy_size": float(thumb_proxy_size),
                "index_proxy_margin": float(index_proxy_margin),
                "thumb_proxy_margin": float(thumb_proxy_margin),
                "index_proxy_half_length": float(index_proxy_half_length),
                "thumb_proxy_half_length": float(thumb_proxy_half_length),
                "object_margin": float(object_margin),
                "active_tip_margin": float(active_tip_margin),
                "object_friction": tuple(parse_vec_list(args.object_friction, 3)[0]),
                "object_solref": tuple(parse_vec_list(args.object_solref, 2)[0]),
                "object_solimp": tuple(parse_vec_list(args.object_solimp, 3)[0]),
            }
        )
    if args.max_runs is not None:
        configs = configs[: max(0, int(args.max_runs))]

    rows = []
    for idx, config in enumerate(configs):
        run_name = "_".join(
            [
                f"{idx:04d}",
                slug_vec("idx", config["index_proxy_pos"]),
                slug_vec("th", config["thumb_proxy_pos"]),
                f"size{slug_float(config['active_proxy_size'])}",
                f"pm{slug_float(config['active_proxy_margin'])}",
                f"is{slug_float(config['index_proxy_size'])}",
                f"ts{slug_float(config['thumb_proxy_size'])}",
                f"im{slug_float(config['index_proxy_margin'])}",
                f"tmarg{slug_float(config['thumb_proxy_margin'])}",
                f"il{slug_float(config['index_proxy_half_length'])}",
                f"tl{slug_float(config['thumb_proxy_half_length'])}",
                f"om{slug_float(config['object_margin'])}",
                f"tm{slug_float(config['active_tip_margin'])}",
            ]
        )
        rows.append(run_one(args, args.output_dir / run_name, config))

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ok_rows.sort(key=lambda row: float(row.get("score", math.inf)))
    rows_sorted = ok_rows + [row for row in rows if row.get("status") != "ok"]
    tsv_path = args.output_dir / "active_proxy_sweep.tsv"
    json_path = args.output_dir / "active_proxy_sweep.json"
    fieldnames: list[str] = []
    for row in rows_sorted:
        for key in flatten(row).keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(flatten(row))
    json_path.write_text(
        json.dumps({"reference": args.reference, "rows": rows_sorted}, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "runs": len(rows),
                "ok": len(ok_rows),
                "reference": args.reference,
                "tsv": str(tsv_path),
                "json": str(json_path),
                "best": rows_sorted[:5],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
