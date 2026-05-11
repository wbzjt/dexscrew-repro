#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def write_latex_table(path, rows, columns, caption, label):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(name for name, _ in columns) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            cells = []
            for _, key in columns:
                val = row.get(key, "")
                if key.endswith("_mean") or key.endswith("_std") or key.endswith("_rate"):
                    val = fnum(val)
                cells.append(str(val).replace("_", "\\_"))
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def maybe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def plot_main(run_dir, plt):
    rows = read_csv(run_dir / "aggregate_csv/main_aggregate.csv")
    rows = [r for r in rows if r.get("mode") == "eval" and r.get("fixed_step_reward_mean")]
    rows.sort(key=lambda r: float(r["fixed_step_reward_mean"]), reverse=True)
    if not rows:
        return
    methods = [r["method"] for r in rows]
    rewards = [float(r["fixed_step_reward_mean"]) for r in rows]
    errs = [float(r.get("fixed_step_reward_std") or 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(methods, rewards, yerr=errs, color="#4C78A8")
    ax.set_ylabel("Fixed-Step Reward")
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(run_dir / "figures/main_fixed_step_reward.png", dpi=200)
    plt.close(fig)


def plot_latency(run_dir, plt):
    points = []
    for path in sorted((run_dir / "aggregate_csv").glob("latency_nfe*_aggregate.csv")):
        nfe = path.stem.split("_aggregate")[0].replace("latency_nfe", "")
        for row in read_csv(path):
            if row.get("mode") != "latency":
                continue
            if not row.get("policy_ms_mean_mean"):
                continue
            points.append((row["method"], int(nfe), float(row["policy_ms_mean_mean"])))
    rewards = {}
    for path in sorted((run_dir / "aggregate_csv").glob("nfe*_aggregate.csv")):
        nfe = path.stem.split("_aggregate")[0].replace("nfe", "")
        for row in read_csv(path):
            if row.get("mode") == "eval" and row.get("fixed_step_reward_mean"):
                rewards[(row["method"], int(nfe))] = float(row["fixed_step_reward_mean"])
    data = [(m, n, ms, rewards.get((m, n))) for m, n, ms in points if rewards.get((m, n)) is not None]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in sorted(set(m for m, _, _, _ in data)):
        xs = [ms for m, _, ms, _ in data if m == method]
        ys = [rw for m, _, _, rw in data if m == method]
        labels = [n for m, n, _, _ in data if m == method]
        ax.plot(xs, ys, marker="o", label=method)
        for x, y, n in zip(xs, ys, labels):
            ax.annotate(str(n), (x, y), fontsize=8)
    ax.set_xlabel("Policy Latency Mean (ms)")
    ax.set_ylabel("Fixed-Step Reward")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "figures/reward_latency_pareto.png", dpi=200)
    plt.close(fig)


def plot_robustness(run_dir, plt):
    nominal = {}
    nom_path = run_dir / "aggregate_csv/robust_nominal_aggregate.csv"
    for row in read_csv(nom_path):
        if row.get("mode") == "eval" and row.get("fixed_step_reward_mean"):
            nominal[row["method"]] = float(row["fixed_step_reward_mean"])
    if not nominal:
        return
    rows = []
    for path in sorted((run_dir / "aggregate_csv").glob("robust_*_aggregate.csv")):
        stress = path.stem.replace("robust_", "").replace("_aggregate", "")
        if stress == "nominal":
            continue
        for row in read_csv(path):
            if row.get("mode") == "eval" and row.get("fixed_step_reward_mean") and row["method"] in nominal:
                rows.append((stress, row["method"], float(row["fixed_step_reward_mean"]) - nominal[row["method"]]))
    if not rows:
        return
    stresses = sorted(set(s for s, _, _ in rows))
    methods = sorted(set(m for _, m, _ in rows))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / max(len(methods), 1)
    xbase = list(range(len(stresses)))
    for i, method in enumerate(methods):
        vals = []
        for stress in stresses:
            found = [v for s, m, v in rows if s == stress and m == method]
            vals.append(found[0] if found else 0.0)
        xs = [x + (i - len(methods) / 2) * width for x in xbase]
        ax.bar(xs, vals, width=width, label=method)
    ax.set_xticks(xbase)
    ax.set_xticklabels(stresses, rotation=25)
    ax.set_ylabel("Reward Delta vs Nominal")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(run_dir / "figures/robustness_degradation.png", dpi=200)
    plt.close(fig)


def build_tables(run_dir):
    main = read_csv(run_dir / "aggregate_csv/main_aggregate.csv")
    main = [r for r in main if r.get("mode") == "eval"]
    main.sort(key=lambda r: float(r.get("fixed_step_reward_mean") or -1e9), reverse=True)
    write_latex_table(
        run_dir / "tables/main_table.tex",
        main,
        [
            ("Method", "method"),
            ("Reward", "fixed_step_reward_mean"),
            ("Done", "done_rate_mean"),
            ("Return", "episode_return_mean_mean"),
            ("2pi", "success_2pi_rate_mean"),
        ],
        "Unified CoDriveThesis evaluation.",
        "tab:codrive-main",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    build_tables(run_dir)
    plt = maybe_import_matplotlib()
    if plt is not None:
        plot_main(run_dir, plt)
        plot_latency(run_dir, plt)
        plot_robustness(run_dir, plt)
    with open(run_dir / "artifact_summary.txt", "w", encoding="utf-8") as f:
        f.write("artifact generation complete\n")
        f.write(f"matplotlib_available={plt is not None}\n")


if __name__ == "__main__":
    main()
