#!/usr/bin/env python3
"""Sweep MuJoCo object contact parameters from a fixed IsaacGym reference state."""

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
        raise ValueError(f"No float values parsed from {text!r}")
    return values


def parse_vec_list(text: str, width: int) -> list[tuple[float, ...] | None]:
    if text.strip().lower() in {"none", "null", ""}:
        return [None]
    out: list[tuple[float, ...] | None] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"none", "null"}:
            out.append(None)
            continue
        normalized = item.replace(":", ",")
        values = tuple(float(v.strip()) for v in normalized.split(",") if v.strip())
        if len(values) != width:
            raise ValueError(f"Expected {width} values in {item!r}, got {len(values)}")
        out.append(values)
    if not out:
        raise ValueError(f"No vectors parsed from {text!r}")
    return out


def parse_int_list(text: str) -> list[int | None]:
    if text.strip().lower() in {"none", "null", ""}:
        return [None]
    out: list[int | None] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"none", "null"}:
            out.append(None)
        else:
            out.append(int(item))
    if not out:
        raise ValueError(f"No int values parsed from {text!r}")
    return out


def parse_path_list(text: str) -> list[str | None]:
    if text.strip().lower() in {"none", "null", ""}:
        return [None]
    out: list[str | None] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"none", "null"}:
            out.append(None)
        else:
            out.append(str((REPO_ROOT / item).resolve() if not Path(item).expanduser().is_absolute() else Path(item).expanduser().resolve()))
    if not out:
        raise ValueError(f"No path values parsed from {text!r}")
    return out


def slug_float(value: float) -> str:
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return text


def slug_vec(prefix: str, values: tuple[float, ...] | None) -> str:
    if values is None:
        return f"{prefix}none"
    return f"{prefix}" + "x".join(slug_float(v) for v in values)


def slug_path(prefix: str, value: str | None) -> str:
    if not value:
        return f"{prefix}default"
    return f"{prefix}{Path(value).stem}"


def load_reference(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "axis_delta": math.nan,
            "dominant_fraction": 0.60,
            "sequence": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", payload)
    sequence = [simplify_partner(item) for item in summary.get("partner_sequence", [])]
    return {
        "axis_delta": float(summary.get("nut_axis_delta", math.nan)),
        "dominant_fraction": float(summary.get("dominant_nut_partner_fraction", 0.60)),
        "sequence": sequence,
    }


def simplify_partner(text: str) -> str:
    if not text or text == "none":
        return "none"
    if "right_index_tactile_link_2" in text:
        return "index_tactile"
    if "right_thumb_tactile_link_1" in text:
        return "thumb_tactile"
    if "right_index_tip" in text:
        return "index_tip"
    if "right_thumb_tip" in text:
        return "thumb_tip"
    if "right_index" in text:
        return "index_other"
    if "right_thumb" in text:
        return "thumb_other"
    if text == "ambiguous":
        return "ambiguous"
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


def trace_sequence(csv_path: Path | None) -> list[str]:
    if csv_path is None or not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [simplify_contact_pairs(row.get("object_contact_pairs", "")) for row in rows]


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


def score_candidate(summary: dict[str, Any], reference: dict[str, Any], sequence_match: float) -> float:
    axis = float(summary.get("object_axis_delta") or 0.0)
    ref_axis = float(reference.get("axis_delta") or math.nan)
    if math.isfinite(ref_axis) and abs(ref_axis) > 1e-9:
        axis_term = abs(axis - ref_axis) / abs(ref_axis)
    else:
        axis_term = 0.0

    dominant_fraction = float(summary.get("dominant_object_contact_pair_step_fraction") or 0.0)
    ref_dominant = float(reference.get("dominant_fraction") or 0.60)
    dominant_term = max(0.0, dominant_fraction - max(ref_dominant, 0.60))

    index_contact = float(summary.get("index_contact_fraction") or 0.0)
    thumb_contact = float(summary.get("thumb_contact_fraction") or 0.0)
    overlap = float(summary.get("index_thumb_overlap_fraction") or 0.0)
    active_balance_term = abs(index_contact - thumb_contact) * 0.25
    low_overlap_term = max(0.0, 0.20 - overlap) * 0.5

    dominant_pair = str(summary.get("dominant_object_contact_pair") or "")
    shortcut_term = 0.0
    if "right_thumb_tip" in dominant_pair or "right_index_tip" in dominant_pair:
        shortcut_term += 0.75
    if dominant_fraction >= 0.80:
        shortcut_term += 0.25

    sequence_term = 0.0
    if math.isfinite(sequence_match):
        sequence_term = 0.5 * (1.0 - sequence_match)

    contract_violation = float(summary.get("max_contract_violation") or 0.0)
    contract_term = max(0.0, contract_violation - 0.02) * 2.0

    return float(axis_term + dominant_term + active_balance_term + low_overlap_term + shortcut_term + sequence_term + contract_term)


def run_candidate(args: argparse.Namespace, run_dir: Path, config: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--mode",
        "policy",
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
        config["finger_contact_mode"],
        "--object-contact-mode",
        args.object_contact_mode,
        "--output-dir",
        str(run_dir),
    ]
    if args.no_clamp_init_q:
        cmd.append("--no-clamp-init-q")
    if config["object_contact0_mesh"]:
        cmd.extend(["--object-contact0-mesh", str(config["object_contact0_mesh"])])
    if config["object_contact1_mesh"]:
        cmd.extend(["--object-contact1-mesh", str(config["object_contact1_mesh"])])
    if config["object_friction"] is not None:
        cmd.extend(["--object-friction", *[str(v) for v in config["object_friction"]]])
    if config["object_solref"] is not None:
        cmd.extend(["--object-solref", *[str(v) for v in config["object_solref"]]])
    if config["object_solimp"] is not None:
        cmd.extend(["--object-solimp", *[str(v) for v in config["object_solimp"]]])
    if config["object_condim"] is not None:
        cmd.extend(["--object-condim", str(config["object_condim"])])
    if any(abs(float(v)) > 0.0 for v in config["object_contact_pos_offset"]):
        cmd.extend(["--object-contact-pos-offset", *[str(v) for v in config["object_contact_pos_offset"]]])
    if abs(float(config["object_contact_z_offset"])) > 0.0:
        cmd.extend(["--object-contact-z-offset", str(config["object_contact_z_offset"])])
    if config["hinge_frictionloss"] is not None:
        cmd.extend(["--hinge-frictionloss", str(config["hinge_frictionloss"])])

    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    record = {
        **config,
        "run_dir": str(run_dir),
        "returncode": proc.returncode,
    }
    if proc.returncode != 0:
        record.update({"status": "error", "error_tail": (proc.stderr or proc.stdout).strip()[-800:]})
        return record
    try:
        summary = json.loads(proc.stdout)
    except json.JSONDecodeError:
        record.update({"status": "bad_json", "error_tail": proc.stdout.strip()[-800:]})
        return record

    csv_path = Path(summary.get("csv", "")) if summary.get("csv") else None
    sequence = trace_sequence(csv_path)
    sequence_match = sequence_match_fraction(sequence, reference["sequence"])
    score = score_candidate(summary, reference, sequence_match)
    record.update(
        {
            "status": "ok",
            "score": score,
            "sequence_match_fraction": sequence_match,
            "sequence": sequence,
            "csv": str(csv_path) if csv_path is not None else "",
        }
    )
    for key in (
        "object_axis_delta",
        "object_contact0_mesh",
        "object_contact1_mesh",
        "dominant_object_contact_pair",
        "dominant_object_contact_pair_step_fraction",
        "dominant_object_contact_pair_event_fraction",
        "index_contact_fraction",
        "thumb_contact_fraction",
        "index_thumb_overlap_fraction",
        "index_only_contact_fraction",
        "thumb_only_contact_fraction",
        "active_contact_fraction",
        "mean_active_contact_force",
        "max_active_contact_force",
        "min_object_contact_dist",
        "min_active_contact_dist",
        "object_contact_pos_offset",
        "object_contact_z_offset",
        "max_contract_violation",
        "object_hinge_frictionloss",
    ):
        record[key] = summary.get(key)
    return record


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    for key in ("object_friction", "object_solref", "object_solimp", "object_contact_pos_offset", "sequence"):
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
    parser.add_argument("--finger-contact-modes", default="full")
    parser.add_argument("--contact0-meshes", default="none")
    parser.add_argument("--contact1-meshes", default="none")
    parser.add_argument(
        "--object-contact-mode",
        choices=["default", "low_friction_debug", "high_hinge_friction_debug", "low_friction_high_hinge_debug"],
        default="default",
    )
    parser.add_argument("--frictions", default="none;1.0,0.005,0.0001;1.5,0.005,0.0001;2.0,0.005,0.0001")
    parser.add_argument("--solrefs", default="none;0.005,1;0.01,1;0.02,1")
    parser.add_argument("--solimps", default="none")
    parser.add_argument("--condims", default="none")
    parser.add_argument("--contact-pos-offsets", default="0,0,0")
    parser.add_argument("--contact-z-offsets", default="0.0")
    parser.add_argument("--hinge-frictionlosses", default="none")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/sim2sim_mujoco_contact_sweeps/object_contact_reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.scene = args.scene.expanduser().resolve()
    args.task_config = args.task_config.expanduser().resolve()
    args.reference_state_json = args.reference_state_json.expanduser().resolve()
    args.reference_summary_json = args.reference_summary_json.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_reference(args.reference_summary_json)
    finger_modes = [item.strip() for item in args.finger_contact_modes.split(",") if item.strip()]
    contact0_meshes = parse_path_list(args.contact0_meshes)
    contact1_meshes = parse_path_list(args.contact1_meshes)
    frictions = parse_vec_list(args.frictions, 3)
    solrefs = parse_vec_list(args.solrefs, 2)
    solimps = parse_vec_list(args.solimps, 3)
    condims = parse_int_list(args.condims)
    contact_pos_offsets = parse_vec_list(args.contact_pos_offsets, 3)
    contact_z_offsets = parse_float_list(args.contact_z_offsets)
    hinges = parse_vec_list(args.hinge_frictionlosses, 1)

    configs = []
    for (
        finger_mode,
        contact0_mesh,
        contact1_mesh,
        friction,
        solref,
        solimp,
        condim,
        contact_pos_offset,
        contact_z_offset,
        hinge,
    ) in product(
        finger_modes,
        contact0_meshes,
        contact1_meshes,
        frictions,
        solrefs,
        solimps,
        condims,
        contact_pos_offsets,
        contact_z_offsets,
        hinges,
    ):
        configs.append(
            {
                "finger_contact_mode": finger_mode,
                "object_contact0_mesh": contact0_mesh or "",
                "object_contact1_mesh": contact1_mesh or "",
                "object_friction": friction,
                "object_solref": solref,
                "object_solimp": solimp,
                "object_condim": condim,
                "object_contact_pos_offset": tuple(float(v) for v in contact_pos_offset),
                "object_contact_z_offset": float(contact_z_offset),
                "hinge_frictionloss": None if hinge is None else float(hinge[0]),
            }
        )
    if args.max_runs is not None:
        configs = configs[: max(0, int(args.max_runs))]

    rows = []
    for index, config in enumerate(configs):
        run_name = "_".join(
            [
                f"{index:04d}",
                config["finger_contact_mode"],
                slug_path("c0", config["object_contact0_mesh"]),
                slug_path("c1", config["object_contact1_mesh"]),
                slug_vec("fric", config["object_friction"]),
                slug_vec("solref", config["object_solref"]),
                slug_vec("solimp", config["object_solimp"]),
                f"condim{config['object_condim'] if config['object_condim'] is not None else 'none'}",
                slug_vec("pos", config["object_contact_pos_offset"]),
                f"z{slug_float(config['object_contact_z_offset'])}",
                f"hinge{slug_float(config['hinge_frictionloss']) if config['hinge_frictionloss'] is not None else 'none'}",
            ]
        )
        rows.append(run_candidate(args, args.output_dir / run_name, config, reference))

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ok_rows.sort(key=lambda row: float(row.get("score", math.inf)))
    rows_sorted = ok_rows + [row for row in rows if row.get("status") != "ok"]

    tsv_path = args.output_dir / "object_contact_sweep.tsv"
    json_path = args.output_dir / "object_contact_sweep.json"
    fieldnames: list[str] = []
    for row in rows_sorted:
        for key in flatten_record(row).keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(flatten_record(row))
    json_path.write_text(
        json.dumps(
            {
                "reference": reference,
                "rows": rows_sorted,
                "best": ok_rows[:10],
                "tsv": str(tsv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "runs": len(rows),
                "ok": len(ok_rows),
                "reference": reference,
                "tsv": str(tsv_path),
                "json": str(json_path),
                "best": ok_rows[:5],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
