#!/usr/bin/env python3
"""Run the standard CoDrive MuJoCo sim2sim validation pack."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "sim2sim/mujoco/run_codrive_sim2sim.py"


def run_rollout(args, mode: str, output_dir: Path) -> dict:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--mode",
        mode,
        "--scene",
        str(args.scene),
        "--task-config",
        str(args.task_config),
        "--checkpoint",
        str(args.checkpoint),
        "--dt",
        str(args.dt),
        "--policy-hz",
        str(args.policy_hz),
        "--joint-limit-mode",
        args.joint_limit_mode,
        "--finger-contact-mode",
        args.finger_contact_mode,
        "--object-contact-mode",
        args.object_contact_mode,
        "--policy-steps",
        str(args.policy_steps),
        "--output-dir",
        str(output_dir),
    ]
    if args.object_pos is not None:
        cmd.extend(
            [
                "--object-pos",
                str(args.object_pos[0]),
                str(args.object_pos[1]),
                str(args.object_pos[2]),
            ]
        )
    if args.hand_root_pos is not None:
        cmd.extend(
            [
                "--hand-root-pos",
                str(args.hand_root_pos[0]),
                str(args.hand_root_pos[1]),
                str(args.hand_root_pos[2]),
            ]
        )
    if args.hand_root_rpy is not None:
        cmd.extend(
            [
                "--hand-root-rpy",
                str(args.hand_root_rpy[0]),
                str(args.hand_root_rpy[1]),
                str(args.hand_root_rpy[2]),
            ]
        )
    if args.init_pose_file is not None:
        cmd.extend(["--init-pose-file", str(args.init_pose_file)])
    if args.render_video:
        cmd.extend(
            [
                "--render-video",
                "--render-width",
                str(args.render_width),
                "--render-height",
                str(args.render_height),
                "--render-every",
                str(args.render_every),
            ]
        )
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{mode} rollout failed:\n{proc.stderr or proc.stdout}")
    try:
        summary = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{mode} rollout did not return JSON:\n{proc.stdout[-2000:]}") from exc
    (output_dir / f"{mode}_stdout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def f(summary: dict, key: str, default: float = 0.0) -> float:
    value = summary.get(key, default)
    return float(default if value is None else value)


def build_report(args, policy: dict, zero: dict) -> dict:
    policy_axis = f(policy, "object_axis_delta")
    zero_axis = f(zero, "object_axis_delta")
    axis_delta_gain = policy_axis - zero_axis
    checks = {
        "policy_returned_json": True,
        "zero_returned_json": True,
        "policy_rate_aligned": abs(f(policy, "policy_rate_hz") - float(args.policy_hz)) < 1e-6,
        "zero_rate_aligned": abs(f(zero, "policy_rate_hz") - float(args.policy_hz)) < 1e-6,
        "policy_limit_ok": f(policy, "max_contract_violation") <= args.max_contract_violation,
        "zero_limit_ok": f(zero, "max_contract_violation") <= args.max_contract_violation,
        "policy_drives_axis_vs_zero": axis_delta_gain >= args.min_axis_gain,
        "policy_has_active_contact": f(policy, "active_contact_fraction") >= args.min_active_contact_fraction,
        "zero_has_low_active_contact": f(zero, "active_contact_fraction") <= args.max_zero_active_contact_fraction,
        "two_finger_contact_ok": (
            f(policy, "index_contact_fraction") >= args.min_two_finger_contact_fraction
            and f(policy, "thumb_contact_fraction") >= args.min_two_finger_contact_fraction
            and f(policy, "index_thumb_overlap_fraction") >= args.min_index_thumb_overlap_fraction
        ),
    }
    if args.anti_shortcut_gate:
        checks.update(
            {
                "dominant_pair_not_overconcentrated": (
                    f(policy, "dominant_object_contact_pair_step_fraction")
                    <= args.max_dominant_contact_pair_fraction
                ),
                "index_only_not_dominant": f(policy, "index_only_contact_fraction") <= args.max_index_only_fraction,
                "thumb_participates": f(policy, "thumb_contact_fraction") >= args.min_thumb_contact_fraction,
            }
        )
        if args.finger_contact_mode == "tip_proxy" or args.require_tip_overlap:
            checks["tip_two_finger_overlap_ok"] = (
                f(policy, "index_thumb_tip_overlap_fraction") >= args.min_index_thumb_tip_overlap_fraction
            )
    hard_pass = all(
        checks[name]
        for name in (
            "policy_returned_json",
            "zero_returned_json",
            "policy_rate_aligned",
            "zero_rate_aligned",
            "policy_limit_ok",
            "zero_limit_ok",
            "policy_drives_axis_vs_zero",
            "policy_has_active_contact",
            "zero_has_low_active_contact",
        )
    )
    anti_shortcut_pass = True
    if args.anti_shortcut_gate:
        anti_shortcut_names = (
            "dominant_pair_not_overconcentrated",
            "index_only_not_dominant",
            "thumb_participates",
            "tip_two_finger_overlap_ok",
        )
        anti_shortcut_pass = all(checks[name] for name in anti_shortcut_names if name in checks)
    verdict = "policy_runs_and_drives_hinge"
    if not hard_pass:
        verdict = "validation_failed"
    elif args.anti_shortcut_gate and not anti_shortcut_pass:
        verdict = "validation_failed_single_finger_shortcut"
    elif checks["two_finger_contact_ok"]:
        verdict = "policy_runs_with_two_finger_contact"
    return {
        "verdict": verdict,
        "checks": checks,
        "thresholds": {
            "max_contract_violation": args.max_contract_violation,
            "min_axis_gain": args.min_axis_gain,
            "min_active_contact_fraction": args.min_active_contact_fraction,
            "max_zero_active_contact_fraction": args.max_zero_active_contact_fraction,
            "min_two_finger_contact_fraction": args.min_two_finger_contact_fraction,
            "min_index_thumb_overlap_fraction": args.min_index_thumb_overlap_fraction,
            "anti_shortcut_gate": bool(args.anti_shortcut_gate),
            "max_dominant_contact_pair_fraction": args.max_dominant_contact_pair_fraction,
            "max_index_only_fraction": args.max_index_only_fraction,
            "min_thumb_contact_fraction": args.min_thumb_contact_fraction,
            "min_index_thumb_tip_overlap_fraction": args.min_index_thumb_tip_overlap_fraction,
        },
        "policy_minus_zero": {
            "object_axis_delta": axis_delta_gain,
            "object_yaw_delta": f(policy, "object_yaw_delta") - f(zero, "object_yaw_delta"),
        },
        "policy": policy,
        "zero": zero,
    }


def write_markdown(report: dict, path: Path) -> None:
    policy = report["policy"]
    zero = report["zero"]
    index_fraction = f(policy, "index_contact_fraction")
    thumb_fraction = f(policy, "thumb_contact_fraction")
    overlap_fraction = f(policy, "index_thumb_overlap_fraction")
    if report["verdict"] == "validation_failed_single_finger_shortcut":
        contact_note = (
            "The policy drives the hinge, but the anti-shortcut gate fails; "
            "the motion is likely dominated by single-finger friction rather than faithful two-finger behavior."
        )
    elif report["checks"].get("two_finger_contact_ok"):
        contact_note = "The current pose satisfies the configured simultaneous index+thumb contact threshold."
    elif index_fraction >= 0.5 and thumb_fraction < 0.5:
        contact_note = "The current validated hinge pose is index-dominant; thumb contact is still not matched to the CoDrive two-finger objective."
    elif thumb_fraction >= 0.5 and index_fraction < 0.5:
        contact_note = "The current validated hinge pose is thumb-dominant; index contact is still not matched to the CoDrive two-finger objective."
    elif overlap_fraction > 0.0:
        contact_note = "The current hinge pose has brief simultaneous index+thumb contact, but not enough to satisfy the configured overlap threshold."
    elif index_fraction > 0.0 and thumb_fraction > 0.0:
        contact_note = "The current hinge pose has separate index and thumb contact at different steps, but no simultaneous two-finger overlap."
    else:
        contact_note = "The current hinge pose drives the object without reproducing the CoDrive two-finger contact objective."
    lines = [
        "# CoDrive MuJoCo Sim2Sim Validation Report",
        "",
        f"Verdict: `{report['verdict']}`",
        "",
        "## Key Metrics",
        "",
        "| Metric | Policy | Zero | Delta |",
        "|---|---:|---:|---:|",
        (
            f"| object_axis_delta | {f(policy, 'object_axis_delta'):.6f} | "
            f"{f(zero, 'object_axis_delta'):.6f} | "
            f"{report['policy_minus_zero']['object_axis_delta']:.6f} |"
        ),
        (
            f"| object_yaw_delta | {f(policy, 'object_yaw_delta'):.6f} | "
            f"{f(zero, 'object_yaw_delta'):.6f} | "
            f"{report['policy_minus_zero']['object_yaw_delta']:.6f} |"
        ),
        f"| max_contract_violation | {f(policy, 'max_contract_violation'):.6f} | {f(zero, 'max_contract_violation'):.6f} | |",
        f"| active_contact_fraction | {f(policy, 'active_contact_fraction'):.6f} | {f(zero, 'active_contact_fraction'):.6f} | |",
        f"| index_contact_fraction | {f(policy, 'index_contact_fraction'):.6f} | {f(zero, 'index_contact_fraction'):.6f} | |",
        f"| thumb_contact_fraction | {f(policy, 'thumb_contact_fraction'):.6f} | {f(zero, 'thumb_contact_fraction'):.6f} | |",
        f"| index_thumb_overlap_fraction | {f(policy, 'index_thumb_overlap_fraction'):.6f} | {f(zero, 'index_thumb_overlap_fraction'):.6f} | |",
        f"| active_tip_contact_fraction | {f(policy, 'active_tip_contact_fraction'):.6f} | {f(zero, 'active_tip_contact_fraction'):.6f} | |",
        f"| index_tip_contact_fraction | {f(policy, 'index_tip_contact_fraction'):.6f} | {f(zero, 'index_tip_contact_fraction'):.6f} | |",
        f"| thumb_tip_contact_fraction | {f(policy, 'thumb_tip_contact_fraction'):.6f} | {f(zero, 'thumb_tip_contact_fraction'):.6f} | |",
        f"| index_thumb_tip_overlap_fraction | {f(policy, 'index_thumb_tip_overlap_fraction'):.6f} | {f(zero, 'index_thumb_tip_overlap_fraction'):.6f} | |",
        f"| index_only_contact_fraction | {f(policy, 'index_only_contact_fraction'):.6f} | {f(zero, 'index_only_contact_fraction'):.6f} | |",
        f"| thumb_only_contact_fraction | {f(policy, 'thumb_only_contact_fraction'):.6f} | {f(zero, 'thumb_only_contact_fraction'):.6f} | |",
        (
            f"| dominant_contact_pair_step_fraction | "
            f"{f(policy, 'dominant_object_contact_pair_step_fraction'):.6f} | "
            f"{f(zero, 'dominant_object_contact_pair_step_fraction'):.6f} | |"
        ),
        f"| active_action_saturation_proxy | {f(policy, 'max_abs_action'):.6f} | {f(zero, 'max_abs_action'):.6f} | |",
        "",
        "## Dominant Contact",
        "",
        f"- policy: `{policy.get('dominant_object_contact_pair', '')}`",
        f"- zero: `{zero.get('dominant_object_contact_pair', '')}`",
        "",
        "## Checks",
        "",
    ]
    for name, value in report["checks"].items():
        lines.append(f"- `{name}`: {'PASS' if value else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- policy csv: `{policy.get('csv')}`",
            f"- zero csv: `{zero.get('csv')}`",
            f"- policy video: `{policy.get('video')}`",
            f"- zero video: `{zero.get('video')}`",
            "",
            "## Interpretation",
            "",
            "- The frozen policy loads and runs in MuJoCo without NaN or interface errors.",
            "- The hinge scene records the screw-like one-DOF axis directly as `object_axis_delta`.",
            f"- {contact_note}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        default=REPO_ROOT / "sim2sim/mujoco/scene_dexh13_lightbulb_hinge_fingertips.xml",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=REPO_ROOT / "sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "sim2real/codrive/model_best_codrive.ckpt")
    parser.add_argument("--init-pose-file", type=Path)
    parser.add_argument(
        "--object-pos",
        type=float,
        nargs=3,
        help="Optional object root pose override. If omitted, the task YAML init pose is used.",
    )
    parser.add_argument("--hand-root-pos", type=float, nargs=3, help="Optional mocap hand root position override.")
    parser.add_argument("--hand-root-rpy", type=float, nargs=3, help="Optional mocap hand root RPY override.")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--policy-hz", type=float, default=20.0)
    parser.add_argument("--policy-steps", type=int, default=200)
    parser.add_argument("--joint-limit-mode", choices=["task", "model"], default="task")
    parser.add_argument("--finger-contact-mode", choices=["full", "active_pad_proxy", "tip_proxy"], default="full")
    parser.add_argument(
        "--object-contact-mode",
        choices=["default", "low_friction_debug", "high_hinge_friction_debug", "low_friction_high_hinge_debug"],
        default="default",
    )
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--max-contract-violation", type=float, default=0.002)
    parser.add_argument("--min-axis-gain", type=float, default=0.2)
    parser.add_argument("--min-active-contact-fraction", type=float, default=0.5)
    parser.add_argument("--max-zero-active-contact-fraction", type=float, default=0.05)
    parser.add_argument("--min-two-finger-contact-fraction", type=float, default=0.1)
    parser.add_argument("--min-index-thumb-overlap-fraction", type=float, default=0.1)
    parser.add_argument("--anti-shortcut-gate", action="store_true", help="Fail runs dominated by a single index/contact-pair rubbing shortcut.")
    parser.add_argument("--max-dominant-contact-pair-fraction", type=float, default=0.6)
    parser.add_argument("--max-index-only-fraction", type=float, default=0.5)
    parser.add_argument("--min-thumb-contact-fraction", type=float, default=0.25)
    parser.add_argument("--min-index-thumb-tip-overlap-fraction", type=float, default=0.1)
    parser.add_argument("--require-tip-overlap", action="store_true", help="Require index+thumb fingertip/proxy simultaneous overlap even outside tip_proxy mode.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/sim2sim_mujoco_validation/codrive_hinge")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.scene = args.scene.expanduser().resolve()
    args.task_config = args.task_config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.init_pose_file is not None:
        args.init_pose_file = args.init_pose_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    policy = run_rollout(args, "policy", args.output_dir / "policy")
    zero = run_rollout(args, "zero", args.output_dir / "zero")
    report = build_report(args, policy, zero)
    report_path = args.output_dir / "validation_report.json"
    report_md = args.output_dir / "validation_report.md"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, report_md)
    print(json.dumps({"report": str(report_path), "markdown": str(report_md), "verdict": report["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
