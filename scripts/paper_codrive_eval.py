#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import isaacgym  # noqa: F401
import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from dexscrew.algo.ppo.ppo import PPO
from dexscrew.algo.student import (
    ProprioAdapt,
    PureBC,
    DiffusionLatentStudent,
    ConsistencyLatentStudent,
    FlowMatchingLatentStudent,
    BCStudent,
    DAggerStudent,
    DOTPG,
    DOTPGStudent,
)
from dexscrew.algo.ppo.diffusion_action_chunk_student import DiffusionActionChunkStudent
from dexscrew.tasks import isaacgym_task_map
from dexscrew.utils.misc import set_np_formatting, set_seed
from dexscrew.utils.reformat import omegaconf_to_dict


ALGO_MAP = {
    "PPO": PPO,
    "ProprioAdapt": ProprioAdapt,
    "PureBC": PureBC,
    "DiffusionLatentStudent": DiffusionLatentStudent,
    "ConsistencyLatentStudent": ConsistencyLatentStudent,
    "FlowMatchingLatentStudent": FlowMatchingLatentStudent,
    "DiffusionActionChunkStudent": DiffusionActionChunkStudent,
    "BCStudent": BCStudent,
    "DAggerStudent": DAggerStudent,
    "DOTPG": DOTPG,
    "DOTPGStudent": DOTPGStudent,
}


def register_resolvers():
    resolvers = {
        "eq": lambda x, y: x.lower() == y.lower(),
        "contains": lambda x, y: x.lower() in y.lower(),
        "if": lambda pred, a, b: a if pred else b,
        "resolve_default": lambda default, arg: default if arg == "" else arg,
    }
    for name, fn in resolvers.items():
        try:
            OmegaConf.register_new_resolver(name, fn)
        except ValueError:
            pass


def scalarize(value):
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def flatten_info(info, prefix=""):
    out = {}
    if not isinstance(info, dict):
        return out
    for key, value in info.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_info(value, name))
            continue
        scalar = scalarize(value)
        if scalar is not None and math.isfinite(scalar):
            out[name] = scalar
    return out


def add_info(acc, info):
    for key, value in flatten_info(info).items():
        total, count = acc.get(key, (0.0, 0))
        acc[key] = (total + float(value), count + 1)


def mean_info(acc):
    return {
        key: total / max(count, 1)
        for key, (total, count) in sorted(acc.items())
        if count > 0
    }


def stats(values):
    values = [float(v) for v in values]
    if not values:
        return {
            "n": 0,
            "mean": "",
            "std": "",
            "min": "",
            "max": "",
            "p25": "",
            "p50": "",
            "p75": "",
        }
    values_sorted = sorted(values)
    tensor = torch.tensor(values_sorted, dtype=torch.float32)
    return {
        "n": len(values_sorted),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=True).item()) if len(values_sorted) > 1 else 0.0,
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "p25": float(torch.quantile(tensor, 0.25).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p75": float(torch.quantile(tensor, 0.75).item()),
    }


def resolve_checkpoint(path):
    if not path:
        return ""
    matches = glob.glob(path) if "*" in path else [path]
    if len(matches) != 1:
        raise FileNotFoundError(f"checkpoint glob resolved to {len(matches)} files: {path}")
    return os.path.abspath(matches[0])


def compose_config(args):
    register_resolvers()
    config_dir = os.path.abspath(args.config_dir)
    overrides = list(args.overrides)
    if args.checkpoint:
        overrides.append(f"checkpoint={args.checkpoint}")
    if args.algo:
        overrides.append(f"train.algo={args.algo}")
    if args.output_name:
        overrides.append(f"train.ppo.output_name={args.output_name}")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    cfg.test = True
    if cfg.checkpoint:
        checkpoint = resolve_checkpoint(to_absolute_path(str(cfg.checkpoint)))
        cfg.checkpoint = checkpoint
        OmegaConf.update(cfg, "train.load_path", checkpoint, force_add=True)
    return cfg


def build_agent(cfg):
    set_np_formatting()
    cfg.seed = set_seed(cfg.seed)
    cfg_dict = omegaconf_to_dict(cfg)
    env = isaacgym_task_map[cfg.task_name](
        config=cfg_dict["task"],
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
    )
    output_dir = os.path.join("outputs", cfg.train.ppo.output_name)
    os.makedirs(output_dir, exist_ok=True)
    algo_name = str(cfg.train.algo)
    if algo_name not in ALGO_MAP:
        raise KeyError(f"unsupported algo: {algo_name}")
    agent = ALGO_MAP[algo_name](env, output_dir, full_config=cfg)
    if cfg.train.load_path:
        agent.restore_test(cfg.train.load_path)
    if hasattr(agent, "set_eval"):
        agent.set_eval()
    return env, agent


def ppo_action(agent, obs_dict):
    if agent.normalize_point_cloud:
        point_cloud = agent.point_cloud_mean_std(
            obs_dict["point_cloud_info"].reshape(-1, 3)
        ).reshape((obs_dict["obs"].shape[0], -1, 3))
    else:
        point_cloud = obs_dict["point_cloud_info"]
    input_dict = {
        "obs": agent.running_mean_std(obs_dict["obs"]),
        "priv_info": agent.priv_mean_std(obs_dict["priv_info"])
        if agent.normalize_priv
        else obs_dict["priv_info"],
        "proprio_hist": obs_dict["proprio_hist"],
        "point_cloud_info": point_cloud,
    }
    mu, extrin, _ = agent.model.act_inference(input_dict)
    return mu, {"extrin_record": extrin}


def dotpg_action(agent, obs_dict):
    state, _, _ = agent.get_state_from_obs(obs_dict)
    return agent.policy(state), {}


def action_for(agent, obs_dict, algo_name):
    if hasattr(agent, "_eval_select_action"):
        result = agent._eval_select_action(obs_dict)
        if isinstance(result, tuple):
            return result
        return result, {}
    if algo_name == "PPO":
        return ppo_action(agent, obs_dict)
    if algo_name in ("DOTPG", "DOTPGStudent"):
        return dotpg_action(agent, obs_dict)
    raise KeyError(f"no action adapter for algo={algo_name}")


def sanitize_action(action):
    action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
    return torch.clamp(action, -1.0, 1.0).contiguous()


def screw_pos(env):
    value = getattr(env, "nut_dof_pos", None)
    if value is None:
        return None
    return value.detach().float().view(-1).clone()


def run_fixed(env, agent, algo_name, steps):
    obs_dict = env.reset()
    reward_sum = 0.0
    done_sum = 0.0
    info_acc = {}
    t0 = time.time()
    with torch.no_grad():
        for _ in range(int(steps)):
            action, step_kwargs = action_for(agent, obs_dict, algo_name)
            obs_dict, reward, done, info = env.step(sanitize_action(action), **step_kwargs)
            reward_sum += float(reward.float().mean().detach().cpu())
            done_sum += float(done.float().mean().detach().cpu())
            add_info(info_acc, info)
    return {
        "fixed_steps": int(steps),
        "fixed_step_reward": reward_sum / float(max(int(steps), 1)),
        "done_rate": done_sum / float(max(int(steps), 1)),
        "fixed_wall_time_sec": time.time() - t0,
        "fixed_info_means": mean_info(info_acc),
    }


def run_episode(env, agent, algo_name, target_episodes, max_steps):
    obs_dict = env.reset()
    nenv = int(env.num_envs)
    ep_return = torch.zeros(nenv, device=env.rl_device)
    ep_len = torch.zeros(nenv, device=env.rl_device)
    start_screw = screw_pos(env)
    if start_screw is None:
        start_screw = torch.zeros(nenv, device=env.rl_device)
    else:
        start_screw = start_screw.to(env.rl_device)
    returns = []
    lengths = []
    progresses = []
    info_acc = {}
    steps = 0
    t0 = time.time()
    with torch.no_grad():
        while len(returns) < int(target_episodes) and steps < int(max_steps):
            action, step_kwargs = action_for(agent, obs_dict, algo_name)
            obs_dict, reward, done, info = env.step(sanitize_action(action), **step_kwargs)
            add_info(info_acc, info)
            ep_return += reward.detach()
            ep_len += 1.0
            current_screw = screw_pos(env)
            if current_screw is None:
                current_screw = start_screw
            else:
                current_screw = current_screw.to(env.rl_device)
            done_mask = done.detach().bool().view(-1)
            if bool(done_mask.any().item()):
                idx = done_mask.nonzero(as_tuple=False).view(-1)
                progress = current_screw[idx] - start_screw[idx]
                returns.extend(ep_return[idx].detach().cpu().view(-1).tolist())
                lengths.extend(ep_len[idx].detach().cpu().view(-1).tolist())
                progresses.extend(progress.detach().cpu().view(-1).tolist())
                ep_return[idx] = 0.0
                ep_len[idx] = 0.0
                start_screw[idx] = 0.0
            steps += 1
    progress_stats = stats(progresses)
    return_stats = stats(returns)
    len_stats = stats(lengths)
    success_threshold = 2.0 * math.pi
    success_count = sum(1 for p in progresses if p >= success_threshold)
    return {
        "episode_target": int(target_episodes),
        "episode_max_steps": int(max_steps),
        "episode_steps": int(steps),
        "episode_n": int(len(returns)),
        "episode_return_mean": return_stats["mean"],
        "episode_return_std": return_stats["std"],
        "episode_len_mean": len_stats["mean"],
        "episode_len_std": len_stats["std"],
        "screw_progress_rad_mean": progress_stats["mean"],
        "screw_progress_rad_p25": progress_stats["p25"],
        "screw_progress_rad_p50": progress_stats["p50"],
        "screw_progress_rad_p75": progress_stats["p75"],
        "success_2pi_rate": success_count / float(max(len(progresses), 1)),
        "episode_wall_time_sec": time.time() - t0,
        "episode_info_means": mean_info(info_acc),
    }


def run_latency(env, agent, algo_name, warmup, measure):
    obs_dict = env.reset()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for _ in range(int(warmup)):
        with torch.no_grad():
            action, _ = action_for(agent, obs_dict, algo_name)
            del action
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples = []
    with torch.no_grad():
        for _ in range(int(measure)):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            action, _ = action_for(agent, obs_dict, algo_name)
            del action
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
    stat = stats(samples)
    peak_mem = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0
    return {
        "latency_batch_size": int(env.num_envs),
        "latency_warmup": int(warmup),
        "latency_measure": int(measure),
        "policy_ms_mean": stat["mean"],
        "policy_ms_p50": stat["p50"],
        "policy_ms_p95": float(torch.quantile(torch.tensor(samples), 0.95).item()) if samples else "",
        "policy_ms_min": stat["min"],
        "policy_ms_max": stat["max"],
        "cuda_peak_mem_mib": peak_mem,
    }


def write_outputs(payload, json_path, csv_path=None):
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        flat = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        continue
                    flat[f"{key}.{subkey}"] = subvalue
            else:
                flat[key] = value
        exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(flat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval", "latency"], default="eval")
    parser.add_argument("--method", required=True)
    parser.add_argument("--train-seed", default="")
    parser.add_argument("--algo", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output-name", default="")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--fixed-steps", type=int, default=2048)
    parser.add_argument("--episode-target", type=int, default=256)
    parser.add_argument("--episode-max-steps", type=int, default=8192)
    parser.add_argument("--latency-warmup", type=int, default=200)
    parser.add_argument("--latency-measure", type=int, default=1000)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = compose_config(args)
    env, agent = build_agent(cfg)
    algo_name = str(cfg.train.algo)
    payload = {
        "method": args.method,
        "algo": algo_name,
        "train_seed": args.train_seed,
        "eval_seed": int(cfg.seed),
        "checkpoint": str(cfg.train.load_path),
        "seed": int(cfg.seed),
        "task": str(cfg.task_name),
        "num_envs": int(env.num_envs),
        "mode": args.mode,
        "status": "ok",
    }
    if args.mode == "latency":
        payload.update(run_latency(env, agent, algo_name, args.latency_warmup, args.latency_measure))
        print(
            "PaperLatencySummary "
            f"method={args.method} seed={cfg.seed} batch={env.num_envs} "
            f"policy_ms_mean={payload['policy_ms_mean']:.6f} "
            f"policy_ms_p50={payload['policy_ms_p50']:.6f} "
            f"policy_ms_p95={payload['policy_ms_p95']:.6f}"
        )
    else:
        payload.update(run_fixed(env, agent, algo_name, args.fixed_steps))
        payload.update(run_episode(env, agent, algo_name, args.episode_target, args.episode_max_steps))
        print(
            "PaperEvalSummary "
            f"method={args.method} seed={cfg.seed} steps={payload['fixed_steps']} "
            f"fixed_step_reward={payload['fixed_step_reward']:.6f} "
            f"done_rate={payload['done_rate']:.6f} "
            f"episode_n={payload['episode_n']} "
            f"episode_return_mean={payload['episode_return_mean']}"
        )
    write_outputs(payload, args.json_out, args.csv_out)


if __name__ == "__main__":
    main()
