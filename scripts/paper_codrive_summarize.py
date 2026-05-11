#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


RAW_FIELDS = [
    "method",
    "algo",
    "train_seed",
    "eval_seed",
    "seed",
    "task",
    "num_envs",
    "mode",
    "checkpoint",
    "status",
    "fixed_steps",
    "fixed_step_reward",
    "done_rate",
    "episode_n",
    "episode_return_mean",
    "episode_return_std",
    "episode_len_mean",
    "episode_len_std",
    "screw_progress_rad_mean",
    "screw_progress_rad_p25",
    "screw_progress_rad_p50",
    "screw_progress_rad_p75",
    "success_2pi_rate",
    "policy_ms_mean",
    "policy_ms_p50",
    "policy_ms_p95",
    "cuda_peak_mem_mib",
    "json_path",
]


AGG_METRICS = [
    "fixed_step_reward",
    "done_rate",
    "episode_return_mean",
    "episode_len_mean",
    "screw_progress_rad_mean",
    "success_2pi_rate",
    "policy_ms_mean",
    "policy_ms_p50",
    "policy_ms_p95",
]


def is_number(value):
    if value in ("", None):
        return False
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value)


def stat(values):
    vals = [float(v) for v in values if is_number(v)]
    if not vals:
        return {"n": 0, "mean": "", "std": "", "min": "", "max": ""}
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    return {
        "n": len(vals),
        "mean": mean,
        "std": std,
        "min": min(vals),
        "max": max(vals),
    }


def flatten(payload, json_path):
    row = {field: "" for field in RAW_FIELDS}
    for field in RAW_FIELDS:
        if field in payload:
            row[field] = payload[field]
    row["json_path"] = str(json_path)
    return row


def write_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-glob", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--agg-out", required=True)
    parser.add_argument("--group-by", nargs="+", default=["method", "mode"])
    args = parser.parse_args()

    rows = []
    for path in sorted(Path().glob(args.json_glob)):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.append(flatten(payload, path))
    write_csv(args.raw_out, rows, RAW_FIELDS)

    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k, "") for k in args.group_by)
        grouped[key].append(row)

    agg_fields = list(args.group_by) + ["n", "all_status_ok"]
    for metric in AGG_METRICS:
        agg_fields.extend(
            [f"{metric}_mean", f"{metric}_std", f"{metric}_min", f"{metric}_max"]
        )

    agg_rows = []
    for key, group_rows in sorted(grouped.items()):
        out = {field: value for field, value in zip(args.group_by, key)}
        out["n"] = len(group_rows)
        out["all_status_ok"] = all(r.get("status") == "ok" for r in group_rows)
        for metric in AGG_METRICS:
            s = stat([r.get(metric, "") for r in group_rows])
            out[f"{metric}_mean"] = s["mean"]
            out[f"{metric}_std"] = s["std"]
            out[f"{metric}_min"] = s["min"]
            out[f"{metric}_max"] = s["max"]
        agg_rows.append(out)
    write_csv(args.agg_out, agg_rows, agg_fields)

    print(f"rows={len(rows)} groups={len(agg_rows)} raw={args.raw_out} agg={args.agg_out}")


if __name__ == "__main__":
    main()
