# --------------------------------------------------------
# Latent Diffusion student distillation (separate from ProprioAdapt)
# --------------------------------------------------------

import os
import time
import torch
import torch.nn as nn
from termcolor import cprint
from tensorboardX import SummaryWriter

from dexscrew.algo.ppo.padapt import ProprioAdapt
from dexscrew.utils.misc import tprint


class LatentDiffusionHead(nn.Module):
    def __init__(self, proprio_hist_dim, proprio_dim, latent_dim, num_steps, hidden_dim=256, t_dim=64):
        super().__init__()
        self.hist_encoder = nn.Sequential(
            nn.Linear(proprio_hist_dim * proprio_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.t_embed = nn.Embedding(num_steps, t_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + t_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, proprio_hist, x_t, t):
        hist_feat = self.hist_encoder(proprio_hist.reshape(proprio_hist.shape[0], -1))
        t_feat = self.t_embed(t)
        denoise_in = torch.cat([hist_feat, x_t, t_feat], dim=-1)
        return self.denoiser(denoise_in)


class DiffusionLatentStudent(ProprioAdapt):
    """Diffusion student that replaces deterministic latent prediction with latent diffusion."""

    def __init__(self, env, output_dir, full_config, student_dim=24):
        super().__init__(env, output_dir, full_config, student_dim=student_dim)

        # Keep diffusion artifacts isolated from ProprioAdapt artifacts.
        self.writer.close()
        self.nn_dir = os.path.join(self.output_dir, "stage2_diffusion_nn")
        self.tb_dir = os.path.join(self.output_dir, "stage2_diffusion_tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)

        # Freeze teacher/student backbone; train diffusion head only.
        for p in self.model.parameters():
            p.requires_grad = False
        self.diffusion_student_trainable_param_patterns = (
            self._resolve_diffusion_trainable_param_patterns()
        )
        extra_trainable_params = []
        if self.diffusion_student_trainable_param_patterns:
            for name, p in self.model.named_parameters():
                if self._is_diffusion_trainable_param(name):
                    p.requires_grad = True
                    extra_trainable_params.append(p)

        self.latent_dim = self.model.adapt_tconv.low_dim_proj.out_features
        self.diffusion_steps = int(self.ppo_config.get("diffusion_steps", 10))
        self.diffusion_steps_infer = int(
            self.ppo_config.get("diffusion_steps_infer", self.diffusion_steps)
        )
        # Keep inference schedule valid under current training schedule.
        self.diffusion_steps_infer = min(self.diffusion_steps_infer, self.diffusion_steps)
        self.stochastic_infer = bool(self.ppo_config.get("diffusion_stochastic_infer", False))
        # Optional residual mode: diffuse only the correction around the frozen adapt_tconv latent.
        self.residual_base = bool(self.ppo_config.get("diffusion_residual_base", False))
        # Optional obs-noise curriculum: linearly ramp env observation noise during training.
        self.obs_noise_curriculum = bool(
            self.ppo_config.get("diffusion_obs_noise_curriculum", False)
        )
        self.obs_noise_curriculum_start = int(
            self.ppo_config.get("diffusion_obs_noise_curriculum_start", 0)
        )
        self.obs_noise_curriculum_steps = max(
            1, int(self.ppo_config.get("diffusion_obs_noise_curriculum_steps", 1))
        )
        self.obs_noise_curriculum_mode = str(
            self.ppo_config.get("diffusion_obs_noise_curriculum_mode", "linear")
        )
        self.obs_noise_curriculum_t_phase_ratio = float(
            self.ppo_config.get("diffusion_obs_noise_curriculum_t_phase_ratio", 0.6)
        )
        self.obs_noise_curriculum_t_phase_ratio = min(
            max(self.obs_noise_curriculum_t_phase_ratio, 0.0), 1.0
        )
        self.obs_noise_curriculum_hold_ratio = float(
            self.ppo_config.get("diffusion_obs_noise_curriculum_hold_ratio", 0.2)
        )
        self.obs_noise_curriculum_hold_ratio = min(
            max(self.obs_noise_curriculum_hold_ratio, 0.0), 1.0
        )
        self.obs_noise_e_base = float(getattr(self.env, "random_obs_noise_e_scale", 0.0))
        self.obs_noise_t_base = float(getattr(self.env, "random_obs_noise_t_scale", 0.0))
        self.obs_noise_e_target = float(
            self.ppo_config.get("diffusion_obs_noise_e_target", self.obs_noise_e_base)
        )
        self.obs_noise_t_target = float(
            self.ppo_config.get("diffusion_obs_noise_t_target", self.obs_noise_t_base)
        )

        beta_start = float(self.ppo_config.get("diffusion_beta_start", 1e-4))
        beta_end = float(self.ppo_config.get("diffusion_beta_end", 2e-2))
        self.betas = torch.linspace(beta_start, beta_end, self.diffusion_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        self.diffusion_model = LatentDiffusionHead(
            proprio_hist_dim=self.proprio_hist_dim,
            proprio_dim=self.proprio_dim,
            latent_dim=self.latent_dim,
            num_steps=self.diffusion_steps,
            hidden_dim=int(self.ppo_config.get("diffusion_hidden_dim", 256)),
            t_dim=int(self.ppo_config.get("diffusion_t_dim", 64)),
        ).to(self.device)

        self.diffusion_loss_coef = float(self.ppo_config.get("diffusion_loss_coef", 1.0))
        self.bc_loss_coef = float(self.ppo_config.get("bc_loss_coef", 1.0))
        self.latent_recon_coef = float(
            self.ppo_config.get("diffusion_latent_recon_coef", 0.0)
        )
        self.base_action_anchor_coef = float(
            self.ppo_config.get("diffusion_base_action_anchor_coef", 0.0)
        )
        self.optim = torch.optim.Adam(
            list(self.diffusion_model.parameters()) + extra_trainable_params,
            lr=float(self.ppo_config.get("diffusion_lr", 3e-4)),
        )
        self.diffusion_extra_trainable_param_count = int(
            sum(p.numel() for p in extra_trainable_params)
        )
        if self.diffusion_student_trainable_param_patterns:
            tprint(
                "DiffusionLatent extra trainable patterns: "
                f"{list(self.diffusion_student_trainable_param_patterns)} | "
                f"extra params: {self.diffusion_extra_trainable_param_count}"
            )
        else:
            tprint("DiffusionLatent extra trainable patterns: [] | extra params: 0")

    def _resolve_diffusion_trainable_param_patterns(self):
        patterns = self.ppo_config.get("diffusion_student_trainable_param_patterns", [])
        if patterns is None:
            return tuple()
        if isinstance(patterns, (str, bytes)):
            patterns = [patterns]
        elif not isinstance(patterns, (list, tuple)):
            try:
                patterns = list(patterns)
            except TypeError as exc:
                raise TypeError(
                    "train.ppo.diffusion_student_trainable_param_patterns must be list/tuple/str, "
                    f"got {type(patterns)}"
                ) from exc
        cleaned = []
        for p in patterns:
            token = str(p).strip()
            if token:
                cleaned.append(token)
        return tuple(cleaned)

    def _is_diffusion_trainable_param(self, param_name):
        return any(
            pattern in param_name for pattern in self.diffusion_student_trainable_param_patterns
        )

    def _gather_vec(self, vec, t):
        return vec[t].unsqueeze(-1)

    def _q_sample(self, x0, t, noise):
        s1 = self._gather_vec(self.sqrt_alpha_bars, t)
        s2 = self._gather_vec(self.sqrt_one_minus_alpha_bars, t)
        return s1 * x0 + s2 * noise

    def _predict_x0(self, x_t, t, eps_pred):
        s1 = self._gather_vec(self.sqrt_alpha_bars, t)
        s2 = self._gather_vec(self.sqrt_one_minus_alpha_bars, t)
        return (x_t - s2 * eps_pred) / (s1 + 1e-8)

    @torch.no_grad()
    def _get_base_latent(self, proprio_hist):
        base_latent = self.model.adapt_tconv(proprio_hist)
        return torch.tanh(base_latent)

    @torch.no_grad()
    def sample_latent(self, proprio_hist):
        batch_size = proprio_hist.shape[0]
        base_latent = self._get_base_latent(proprio_hist) if self.residual_base else None
        # Keep deterministic eval truly deterministic across seeds.
        if self.stochastic_infer:
            x = torch.randn(batch_size, self.latent_dim, device=self.device)
        else:
            x = torch.zeros(batch_size, self.latent_dim, device=self.device)
        for t_idx in reversed(range(self.diffusion_steps_infer)):
            t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            eps_pred = self.diffusion_model(proprio_hist, x, t)

            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alpha_bars[t_idx]
            beta_t = self.betas[t_idx]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t + 1e-8)) * eps_pred
            )
            if t_idx > 0 and self.stochastic_infer:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean
        if base_latent is not None:
            x = x + base_latent
        return torch.tanh(x)

    def _apply_obs_noise_curriculum(self):
        if not self.obs_noise_curriculum:
            return
        progress = (self.agent_steps - self.obs_noise_curriculum_start) / float(
            self.obs_noise_curriculum_steps
        )
        progress = min(max(progress, 0.0), 1.0)

        if self.obs_noise_curriculum_mode == "staged_te":
            split = self.obs_noise_curriculum_t_phase_ratio
            if split <= 0.0:
                cur_t = self.obs_noise_t_target
                e_progress = progress
                cur_e = self.obs_noise_e_base + (
                    self.obs_noise_e_target - self.obs_noise_e_base
                ) * e_progress
            elif split >= 1.0:
                t_progress = progress
                cur_t = self.obs_noise_t_base + (
                    self.obs_noise_t_target - self.obs_noise_t_base
                ) * t_progress
                cur_e = self.obs_noise_e_base
            elif progress <= split:
                t_progress = progress / split
                cur_t = self.obs_noise_t_base + (
                    self.obs_noise_t_target - self.obs_noise_t_base
                ) * t_progress
                cur_e = self.obs_noise_e_base
            else:
                cur_t = self.obs_noise_t_target
                e_progress = (progress - split) / (1.0 - split)
                cur_e = self.obs_noise_e_base + (
                    self.obs_noise_e_target - self.obs_noise_e_base
                ) * e_progress
        elif self.obs_noise_curriculum_mode == "staged_hold_te":
            hold = self.obs_noise_curriculum_hold_ratio
            split = max(self.obs_noise_curriculum_t_phase_ratio, hold)
            if progress <= hold:
                cur_t = self.obs_noise_t_base
                cur_e = self.obs_noise_e_base
            elif split >= 1.0:
                t_progress = (progress - hold) / max(1e-8, 1.0 - hold)
                cur_t = self.obs_noise_t_base + (
                    self.obs_noise_t_target - self.obs_noise_t_base
                ) * t_progress
                cur_e = self.obs_noise_e_base
            elif progress <= split:
                t_progress = (progress - hold) / max(1e-8, split - hold)
                cur_t = self.obs_noise_t_base + (
                    self.obs_noise_t_target - self.obs_noise_t_base
                ) * t_progress
                cur_e = self.obs_noise_e_base
            else:
                cur_t = self.obs_noise_t_target
                e_progress = (progress - split) / max(1e-8, 1.0 - split)
                cur_e = self.obs_noise_e_base + (
                    self.obs_noise_e_target - self.obs_noise_e_base
                ) * e_progress
        else:
            cur_e = self.obs_noise_e_base + (
                self.obs_noise_e_target - self.obs_noise_e_base
            ) * progress
            cur_t = self.obs_noise_t_base + (
                self.obs_noise_t_target - self.obs_noise_t_base
            ) * progress
        self.env.random_obs_noise_e_scale = cur_e
        self.env.random_obs_noise_t_scale = cur_t
        self.direct_info["obs_noise_e_scale_cur"] = cur_e
        self.direct_info["obs_noise_t_scale_cur"] = cur_t
        self.direct_info["obs_noise_curriculum_progress"] = progress
        self.direct_info["obs_noise_curriculum_mode"] = (
            2.0
            if self.obs_noise_curriculum_mode == "staged_hold_te"
            else (1.0 if self.obs_noise_curriculum_mode == "staged_te" else 0.0)
        )

    def set_eval(self):
        super().set_eval()
        self.diffusion_model.eval()

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        c = 0
        eval_reward_sum = 0.0
        eval_done_sum = 0.0
        while True:
            if self.normalize_point_cloud:
                point_cloud_info = self.point_cloud_mean_std(
                    obs_dict["point_cloud_info"].reshape(-1, 3)
                ).reshape((obs_dict["obs"].shape[0], -1, 3))
            else:
                point_cloud_info = obs_dict["point_cloud_info"]

            proprio_hist = self.sa_mean_std(obs_dict["proprio_hist"].detach())
            obs = self.running_mean_std(obs_dict["obs"])
            latent = self.sample_latent(proprio_hist)
            student_obs_input = torch.cat([obs, latent], dim=-1)
            student_x = self.model.actor_mlp(student_obs_input)
            mu = self.model.mu(student_x)

            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, _ = self.env.step(mu)
            c += 1
            print(f"Step {c}")
            if self.test_num_steps > 0:
                eval_reward_sum += float(r.float().mean().detach().cpu())
                eval_done_sum += float(done.float().mean().detach().cpu())
                if c >= self.test_num_steps:
                    avg_reward = eval_reward_sum / float(c)
                    avg_done_rate = eval_done_sum / float(c)
                    print(
                        "EvalSummary "
                        f"steps={c} avg_reward={avg_reward:.6f} avg_done_rate={avg_done_rate:.6f}"
                    )
                    break

    def train(self):
        _t = time.time()
        _last_t = time.time()

        obs_dict = self.env.reset()
        self.agent_steps += self.batch_size
        while self.agent_steps <= 1e9:
            self._apply_obs_noise_curriculum()
            if self.normalize_point_cloud:
                point_cloud_info = self.point_cloud_mean_std(
                    obs_dict["point_cloud_info"].reshape(-1, 3)
                ).reshape((obs_dict["obs"].shape[0], -1, 3))
            else:
                point_cloud_info = obs_dict["point_cloud_info"]

            input_dict = {
                "obs": self.running_mean_std(obs_dict["obs"]).detach(),
                "priv_info": self.priv_mean_std(obs_dict["priv_info"])
                if self.normalize_priv
                else obs_dict["priv_info"],
                "proprio_hist": self.sa_mean_std(obs_dict["proprio_hist"].detach()),
                "point_cloud_info": point_cloud_info,
            }

            with torch.no_grad():
                _, _, _, _, e_gt = self.model._actor_critic(input_dict)
                teacher_obs_input = torch.cat([input_dict["obs"], e_gt.detach()], dim=-1)
                teacher_x = self.model.actor_mlp(teacher_obs_input)
                teacher_mu = self.model.mu(teacher_x)
                residual_base_latent = (
                    self._get_base_latent(input_dict["proprio_hist"])
                    if self.residual_base
                    else None
                )
                base_student_mu = None
                if self.base_action_anchor_coef > 0.0:
                    base_latent_anchor = self._get_base_latent(input_dict["proprio_hist"])
                    base_obs_input = torch.cat([input_dict["obs"], base_latent_anchor], dim=-1)
                    base_x = self.model.actor_mlp(base_obs_input)
                    base_student_mu = torch.clamp(self.model.mu(base_x), -1.0, 1.0)

            batch_size = e_gt.shape[0]
            t = torch.randint(0, self.diffusion_steps, (batch_size,), device=self.device)
            target_latent = e_gt.detach()
            target_x0 = (
                target_latent - residual_base_latent
                if residual_base_latent is not None
                else target_latent
            )
            noise = torch.randn_like(target_x0)
            x_t = self._q_sample(target_x0, t, noise)
            eps_pred = self.diffusion_model(input_dict["proprio_hist"], x_t, t)
            diffusion_loss = ((eps_pred - noise) ** 2).mean()

            x0_pred = self._predict_x0(x_t, t, eps_pred)
            if residual_base_latent is not None:
                x0_pred = x0_pred + residual_base_latent
            pred_latent = torch.tanh(x0_pred)
            latent_recon_loss = ((pred_latent - target_latent) ** 2).mean()
            student_obs_input = torch.cat([input_dict["obs"], pred_latent], dim=-1)
            student_x = self.model.actor_mlp(student_obs_input)
            student_mu = self.model.mu(student_x)
            student_mu_clamped = torch.clamp(student_mu, -1, 1)
            bc_loss = torch.sum(
                self.recon_criterion(student_mu_clamped, torch.clamp(teacher_mu, -1, 1)),
                dim=-1,
            ).mean()
            if base_student_mu is not None:
                base_action_anchor_loss = torch.sum(
                    self.recon_criterion(student_mu_clamped, base_student_mu),
                    dim=-1,
                ).mean()
            else:
                base_action_anchor_loss = torch.zeros(
                    (), device=self.device, dtype=bc_loss.dtype
                )

            loss = (
                self.diffusion_loss_coef * diffusion_loss
                + self.bc_loss_coef * bc_loss
                + self.latent_recon_coef * latent_recon_loss
                + self.base_action_anchor_coef * base_action_anchor_loss
            )
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            mu_env = torch.clamp(student_mu.detach(), -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu_env)
            self.agent_steps += self.batch_size

            self.step_reward += r
            self.step_length += 1
            done_indices = done.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])

            not_dones = 1.0 - done.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            self.direct_info["diffusion_loss"] = float(diffusion_loss.detach().cpu())
            self.direct_info["bc_loss"] = float(bc_loss.detach().cpu())
            self.direct_info["latent_recon_loss"] = float(latent_recon_loss.detach().cpu())
            self.direct_info["base_action_anchor_loss"] = float(
                base_action_anchor_loss.detach().cpu()
            )
            self.direct_info["total_loss"] = float(loss.detach().cpu())
            self.direct_info["done_rate"] = float(done.float().mean().detach().cpu())
            self._update_env_info(info)
            self.log_tensorboard()

            if self.agent_steps % 1e8 == 0:
                self.save(os.path.join(self.nn_dir, f"{self.agent_steps // 1e8}00m"))
                self.save(os.path.join(self.nn_dir, "model_last"))

            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_rewards:
                self.save(os.path.join(self.nn_dir, "model_best"))
                self.best_rewards = mean_rewards

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = (
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                f"Last FPS: {last_fps:.1f} | Current Best: {self.best_rewards:.2f}"
            )
            tprint(info_string)

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        cprint("careful, using non-strict matching", "red", attrs=["bold"])
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"], strict=False)
        if "running_mean_std" in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if "priv_mean_std" in checkpoint and self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint["priv_mean_std"])
        if "point_cloud_mean_std" in checkpoint and self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint["point_cloud_mean_std"])

        # Resume diffusion student state when available; fallback to teacher-only init otherwise.
        if "diffusion_model" in checkpoint:
            self.diffusion_model.load_state_dict(checkpoint["diffusion_model"])
            cprint("Loaded diffusion_model for train resume", "green")
        if "diffusion_optim" in checkpoint:
            self.optim.load_state_dict(checkpoint["diffusion_optim"])
            cprint("Loaded diffusion optimizer state for train resume", "green")

    def restore_test(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"], strict=False)
        if "diffusion_model" in checkpoint:
            self.diffusion_model.load_state_dict(checkpoint["diffusion_model"])
        if "running_mean_std" in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if "sa_mean_std" in checkpoint:
            self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])
        if "priv_mean_std" in checkpoint and self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint["priv_mean_std"])
        if "point_cloud_mean_std" in checkpoint and self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint["point_cloud_mean_std"])

    def save(self, name):
        weights = {
            "model": self.model.state_dict(),
            "diffusion_model": self.diffusion_model.state_dict(),
        }
        if self.running_mean_std:
            weights["running_mean_std"] = self.running_mean_std.state_dict()
        if self.sa_mean_std:
            weights["sa_mean_std"] = self.sa_mean_std.state_dict()
        if self.priv_mean_std:
            weights["priv_mean_std"] = self.priv_mean_std.state_dict()
        if self.normalize_point_cloud:
            weights["point_cloud_mean_std"] = self.point_cloud_mean_std.state_dict()
        weights["diffusion_optim"] = self.optim.state_dict()
        torch.save(weights, f"{name}.ckpt")
