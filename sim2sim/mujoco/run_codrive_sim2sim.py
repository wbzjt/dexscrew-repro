#!/usr/bin/env python3
"""Minimal MuJoCo sim2sim scaffold for the frozen CoDrive PAdapt policy."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn

try:
    import mujoco
except ModuleNotFoundError as exc:
    raise SystemExit("MuJoCo Python bindings are required: pip install mujoco") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexscrew.algo.models.block import TemporalConv  # noqa: E402
from dexscrew.algo.models.models import MLP  # noqa: E402


POLICY_JOINT_NAMES = [
    "right_index_joint_0",
    "right_index_joint_1",
    "right_index_joint_2",
    "right_index_joint_3",
    "right_middle_joint_0",
    "right_middle_joint_1",
    "right_middle_joint_2",
    "right_middle_joint_3",
    "right_ring_joint_0",
    "right_ring_joint_1",
    "right_ring_joint_2",
    "right_ring_joint_3",
    "right_thumb_joint_0",
    "right_thumb_joint_1",
    "right_thumb_joint_2",
    "right_thumb_joint_3",
]

FINGER_PREFIXES = {
    "index": "right_index_",
    "middle": "right_middle_",
    "ring": "right_ring_",
    "thumb": "right_thumb_",
}
ACTIVE_TACTILE_GEOMS = {
    "right_index_tactile_link_2",
    "right_thumb_tactile_link_1",
    "right_index_tip",
    "right_thumb_tip",
}
ACTIVE_PROXY_GEOMS = {
    "right_index_distal_proxy",
    "right_thumb_distal_proxy",
    "right_index_tactile_capsule_0",
    "right_index_tactile_capsule_1",
    "right_index_tactile_capsule_2",
    "right_thumb_tactile_capsule_0",
    "right_thumb_tactile_capsule_1",
    "right_thumb_tactile_capsule_2",
}
ACTIVE_PROXY_PROFILES = {"none", "distal_spheres", "tactile_capsules"}
ACTIVE_CONTACT0_PAIR_PROFILES = {
    "none": (),
    "balanced_soft": (
        ("right_index_tactile_link_2", "codrive_lightbulb_contact0", 4, (0.8, 0.005, 0.0001), (0.015, 1.0), (0.8, 0.98, 0.001)),
        ("right_thumb_tactile_link_1", "codrive_lightbulb_contact0", 4, (0.8, 0.005, 0.0001), (0.015, 1.0), (0.8, 0.98, 0.001)),
    ),
    "index_release": (
        ("right_index_tactile_link_2", "codrive_lightbulb_contact0", 4, (0.35, 0.001, 0.0001), (0.02, 1.0), (0.8, 0.98, 0.001)),
        ("right_thumb_tactile_link_1", "codrive_lightbulb_contact0", 4, (0.8, 0.005, 0.0001), (0.015, 1.0), (0.8, 0.98, 0.001)),
    ),
    "index_only_release": (
        ("right_index_tactile_link_2", "codrive_lightbulb_contact0", 4, (0.35, 0.001, 0.0001), (0.02, 1.0), (0.8, 0.98, 0.001)),
    ),
    "index_only_soft": (
        ("right_index_tactile_link_2", "codrive_lightbulb_contact0", 4, (0.6, 0.003, 0.0001), (0.018, 1.0), (0.8, 0.98, 0.001)),
    ),
    "thumb_guard": (
        ("right_index_tactile_link_2", "codrive_lightbulb_contact0", 4, (0.8, 0.005, 0.0001), (0.015, 1.0), (0.8, 0.98, 0.001)),
        ("right_thumb_tactile_link_1", "codrive_lightbulb_contact0", 4, (0.45, 0.001, 0.0001), (0.02, 1.0), (0.8, 0.98, 0.001)),
    ),
    "thumb_only_guard": (
        ("right_thumb_tactile_link_1", "codrive_lightbulb_contact0", 4, (0.45, 0.001, 0.0001), (0.02, 1.0), (0.8, 0.98, 0.001)),
    ),
    "low_pair": (
        ("right_index_tactile_link_2", "codrive_lightbulb_contact0", 4, (0.45, 0.001, 0.0001), (0.02, 1.0), (0.8, 0.98, 0.001)),
        ("right_thumb_tactile_link_1", "codrive_lightbulb_contact0", 4, (0.45, 0.001, 0.0001), (0.02, 1.0), (0.8, 0.98, 0.001)),
    ),
}
OBJECT_BODY_NAME = "codrive_lightbulb"
OBJECT_NUT_BODY_NAME = "codrive_lightbulb_nut"
OBJECT_FREE_JOINT_NAME = "codrive_lightbulb_freejoint"
OBJECT_HINGE_JOINT_NAME = "codrive_lightbulb_hinge"
SCALABLE_OBJECT_MESHES = {"codrive_lightbulb_contact0", "codrive_lightbulb_contact1"}
SCALABLE_OBJECT_GEOM_BASE_RADIUS = 0.03
SCALABLE_OBJECT_GEOM_BASE_HALFHEIGHT = 0.005
SCALABLE_OBJECT_BOLT_Z = 0.005
SCALABLE_OBJECT_CONTACT_Z = 0.06

DEFAULT_INIT_Q = np.array(
    [
        0.3426670736,
        0.925279522,
        0.2828971177,
        1.1292990017,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.3599407768,
        1.5656383038,
        0.220286703,
        0.411966312,
    ],
    dtype=np.float32,
)
ISAACGYM_SCREWDRIVER_INIT_Q = np.array(
    [
        0.34266707360744476,
        1.2325279521942139,
        0.28289711773395538,
        1.1292990016937256,
        0.00888176321983337,
        0.7502785110473633,
        0.8105450248718262,
        0.825642421245575,
        -0.3587684601545334,
        1.3127488565444946,
        0.0030379545688629,
        1.5381801319122314,
        -0.35994077682495117,
        1.5656383037567139,
        0.5520286703109741,
        0.531196631193161,
    ],
    dtype=np.float32,
)
ISAACGYM_SCREWDRIVER_HAND_ROOT_POS = np.array([0.14, 0.072, 0.177], dtype=np.float32)
ISAACGYM_SCREWDRIVER_HAND_ROOT_RPY = np.array([3.0543261909900767, 0.0, math.pi], dtype=np.float32)
DEFAULT_LOWER = np.array(
    [
        -0.35,
        0.0,
        0.0,
        0.0,
        -0.001,
        0.0,
        0.0,
        0.0,
        -0.001,
        0.0,
        0.0,
        0.0,
        -0.35,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)
DEFAULT_UPPER = np.array(
    [
        0.35,
        1.57,
        1.57,
        1.57,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.35,
        1.57,
        1.57,
        1.57,
    ],
    dtype=np.float32,
)


@dataclass
class TaskContract:
    dt: float = 0.005
    control_decimation: int = 10
    target_policy_hz: float = 20.0
    action_scale: float = 0.05
    pgain: float = 3.0
    dgain: float = 0.01
    torque_limit: float = 300.0
    action_mask_indices: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 11)
    init_q: np.ndarray = field(default_factory=lambda: DEFAULT_INIT_Q.copy())
    lower: np.ndarray = field(default_factory=lambda: DEFAULT_LOWER.copy())
    upper: np.ndarray = field(default_factory=lambda: DEFAULT_UPPER.copy())
    hand_root_pos: np.ndarray = field(default_factory=lambda: np.array([0.11, 0.020, 0.217], dtype=np.float32))
    hand_root_rpy: np.ndarray = field(default_factory=lambda: np.array([3.1415, 0.3, 3.1415], dtype=np.float32))
    hand_root_quat_wxyz: np.ndarray | None = None
    init_qvel: np.ndarray | None = None
    base_obj_scale: float = 1.0
    object_scale: float | None = None
    hand_root_pos_z_scale_comp: float = 0.0
    object_init_pos: np.ndarray = field(default_factory=lambda: np.array([0.012, -0.018, 0.05], dtype=np.float32))
    object_root_quat_wxyz: np.ndarray | None = None
    object_root_linvel: np.ndarray | None = None
    object_root_angvel: np.ndarray | None = None
    object_axis_pos: float = 0.0
    object_axis_vel: float = 0.0
    object_contact0_mesh: str = ""
    object_contact1_mesh: str = ""
    active_pair_profile: str = "none"
    active_proxy_profile: str = "none"
    index_proxy_pos: tuple[float, float, float] = (0.0, 0.008, 0.0035)
    thumb_proxy_pos: tuple[float, float, float] = (0.0, 0.009, 0.0035)
    active_proxy_size: float = 0.0045
    active_proxy_margin: float = 0.001
    index_proxy_size: float | None = None
    thumb_proxy_size: float | None = None
    index_proxy_margin: float | None = None
    thumb_proxy_margin: float | None = None
    index_proxy_half_length: float | None = None
    thumb_proxy_half_length: float | None = None
    active_tactile_friction_override: tuple[float, float, float] | None = None
    active_tactile_solref_override: tuple[float, float] | None = None
    active_tactile_solimp_override: tuple[float, float, float] | None = None
    active_tactile_margin_override: float | None = None
    active_tip_margin_override: float | None = None
    init_target: np.ndarray | None = None
    reference_action: np.ndarray | None = None
    reference_state_json: str = ""
    reference_phase: str = ""
    reference_hand_dof_names: tuple[str, ...] = ()
    init_q_clipped_count: int = 0
    finger_contact_mode: str = "full"
    object_contact_mode: str = "default"
    object_friction_override: tuple[float, float, float] | None = None
    object_solref_override: tuple[float, float] | None = None
    object_solimp_override: tuple[float, float, float] | None = None
    object_condim_override: int | None = None
    object_margin_override: float | None = None
    object_gap_override: float | None = None
    object_contact_pos_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    object_contact_z_offset: float = 0.0
    hinge_frictionloss_override: float | None = None
    reset_source: str = "yaml"
    disabled_finger_mesh_contact_geoms: int = 0
    modified_object_contact_geoms: int = 0
    object_hinge_frictionloss: float = 0.0


class ProprioHistory:
    def __init__(self, length: int = 30, dof: int = 16):
        self.length = int(length)
        self.dof = int(dof)
        self.frame_dim = self.dof * 2
        self.buffer = np.zeros((self.length, self.frame_dim), dtype=np.float32)

    def reset(self, q: np.ndarray, target: np.ndarray) -> None:
        frame = np.concatenate([q, target]).astype(np.float32)
        self.buffer[:] = frame[None, :]

    def append(self, q: np.ndarray, target: np.ndarray) -> None:
        frame = np.concatenate([q, target]).astype(np.float32)
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = frame

    def obs(self) -> np.ndarray:
        return self.buffer[-3:].reshape(1, self.frame_dim * 3).copy()

    def hist(self) -> np.ndarray:
        return self.buffer.reshape(1, self.length, self.frame_dim).copy()


class ProprioAdaptPolicy:
    def __init__(self, actor_mlp: nn.Module, mu: nn.Module, adapt_tconv: nn.Module, stats: dict):
        self.actor_mlp = actor_mlp.cpu().eval()
        self.mu = mu.cpu().eval()
        self.adapt_tconv = adapt_tconv.cpu().eval()
        self.running_mean = stats["running_mean"].float().cpu()
        self.running_var = stats["running_var"].float().cpu()
        self.sa_mean = stats["sa_mean"].float().cpu()
        self.sa_var = stats["sa_var"].float().cpu()
        self.obs_dim = int(self.running_mean.numel())
        self.proprio_len, self.proprio_dim = [int(v) for v in self.sa_mean.shape]
        self.action_dim = int(self.mu.weight.shape[0])
        self.extrin_dim = int(self.mu.weight.shape[1])

    @staticmethod
    def _strip_prefix(state_dict: dict, prefix: str) -> dict:
        return {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }

    @staticmethod
    def _load_checkpoint(path: Path) -> dict:
        try:
            return torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(str(path), map_location="cpu")
        except pickle.UnpicklingError:
            return torch.load(str(path), map_location="cpu")

    @classmethod
    def from_checkpoint(cls, path: Path) -> "ProprioAdaptPolicy":
        ckpt = cls._load_checkpoint(path)
        required = {"model", "running_mean_std", "sa_mean_std"}
        missing = sorted(required - set(ckpt.keys()))
        if missing:
            raise KeyError(f"Missing ProprioAdapt checkpoint keys: {missing}")

        model_sd = ckpt["model"]
        for key in (
            "actor_mlp.mlp.0.weight",
            "actor_mlp.mlp.2.weight",
            "actor_mlp.mlp.4.weight",
            "mu.weight",
            "adapt_tconv.low_dim_proj.weight",
        ):
            if key not in model_sd:
                raise KeyError(f"Checkpoint does not look like a PAdapt stage-2 model: missing {key}")

        state_dim = int(model_sd["actor_mlp.mlp.0.weight"].shape[1])
        hidden_units = [
            int(model_sd["actor_mlp.mlp.0.weight"].shape[0]),
            int(model_sd["actor_mlp.mlp.2.weight"].shape[0]),
            int(model_sd["actor_mlp.mlp.4.weight"].shape[0]),
        ]
        action_dim = int(model_sd["mu.weight"].shape[0])
        obs_dim = int(ckpt["running_mean_std"]["running_mean"].numel())
        proprio_len, proprio_dim = ckpt["sa_mean_std"]["running_mean"].shape
        extrin_dim = state_dim - obs_dim
        if extrin_dim <= 0:
            raise ValueError(f"Invalid policy dimensions: state_dim={state_dim}, obs_dim={obs_dim}")

        actor_mlp = MLP(units=hidden_units, input_size=state_dim)
        actor_mlp.load_state_dict(cls._strip_prefix(model_sd, "actor_mlp."))

        mu = nn.Linear(hidden_units[-1], action_dim)
        mu.load_state_dict(cls._strip_prefix(model_sd, "mu."))

        adapt_hidden_dim = int(model_sd["adapt_tconv.channel_transform.0.weight"].shape[0])
        adapt_tconv = TemporalConv(int(proprio_dim), extrin_dim, hidden_dim=adapt_hidden_dim)
        adapt_tconv.load_state_dict(cls._strip_prefix(model_sd, "adapt_tconv."))

        stats = {
            "running_mean": ckpt["running_mean_std"]["running_mean"],
            "running_var": ckpt["running_mean_std"]["running_var"],
            "sa_mean": ckpt["sa_mean_std"]["running_mean"],
            "sa_var": ckpt["sa_mean_std"]["running_var"],
        }
        policy = cls(actor_mlp, mu, adapt_tconv, stats)
        if policy.obs_dim != 96 or policy.proprio_len != 30 or policy.proprio_dim != 32:
            raise ValueError(
                "Unexpected policy interface: "
                f"obs={policy.obs_dim}, hist=({policy.proprio_len},{policy.proprio_dim})"
            )
        return policy

    def act(self, obs: np.ndarray, proprio_hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obs_t = torch.from_numpy(obs.astype(np.float32))
        hist_t = torch.from_numpy(proprio_hist.astype(np.float32))
        with torch.no_grad():
            obs_t = torch.clamp(obs_t, -5.0, 5.0)
            norm_obs = (obs_t - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
            norm_obs = torch.clamp(norm_obs, -5.0, 5.0)
            norm_hist = (hist_t - self.sa_mean) / torch.sqrt(self.sa_var + 1e-5)
            norm_hist = torch.clamp(norm_hist, -5.0, 5.0)
            extrin = torch.tanh(self.adapt_tconv(norm_hist))
            action = self.mu(self.actor_mlp(torch.cat([norm_obs, extrin], dim=-1)))
            action = torch.clamp(action, -1.0, 1.0)
        return action.cpu().numpy().reshape(-1), extrin.cpu().numpy().reshape(-1)


def load_task_contract(path: Path) -> TaskContract:
    contract = TaskContract()
    if not path.exists():
        return contract
    try:
        import yaml
    except ModuleNotFoundError:
        return contract

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env = cfg.get("env", {})
    sim = cfg.get("sim", {})
    ctrl = env.get("controller", {})
    asset = env.get("asset", {})
    obj = env.get("object", {})

    contract.base_obj_scale = float(env.get("baseObjScale", contract.base_obj_scale))
    contract.object_scale = contract.base_obj_scale
    contract.dt = float(sim.get("dt", contract.dt))
    contract.control_decimation = int(ctrl.get("controlFrequencyInv", contract.control_decimation))
    contract.action_scale = float(ctrl.get("action_scale", contract.action_scale))
    contract.pgain = float(ctrl.get("pgain", contract.pgain))
    contract.dgain = float(ctrl.get("dgain", contract.dgain))
    contract.torque_limit = float(ctrl.get("torque_limit", contract.torque_limit))
    contract.action_mask_indices = tuple(int(i) for i in env.get("action_mask_indices", contract.action_mask_indices))
    if "dofLowerLimits" in asset:
        contract.lower = np.asarray(asset["dofLowerLimits"], dtype=np.float32)
    if "dofUpperLimits" in asset:
        contract.upper = np.asarray(asset["dofUpperLimits"], dtype=np.float32)
    if "handInitPose" in asset:
        pose = asset["handInitPose"]
        contract.init_q = np.asarray([pose[name] for name in POLICY_JOINT_NAMES], dtype=np.float32)
    if "handRootPos" in asset:
        contract.hand_root_pos = np.asarray(asset["handRootPos"], dtype=np.float32)
    if "handRootRPY" in asset:
        contract.hand_root_rpy = np.asarray(asset["handRootRPY"], dtype=np.float32)
    contract.hand_root_pos_z_scale_comp = float(
        asset.get("handRootPosZScaleComp", contract.hand_root_pos_z_scale_comp)
    )
    if contract.hand_root_pos_z_scale_comp != 0.0 and contract.base_obj_scale != 1.0:
        contract.hand_root_pos = contract.hand_root_pos.copy()
        contract.hand_root_pos[2] += contract.hand_root_pos_z_scale_comp * (contract.base_obj_scale - 1.0)
    if "init_pos" in obj:
        obj_pos = np.asarray(obj["init_pos"], dtype=np.float32)
        if obj_pos.shape == (3,):
            contract.object_init_pos = obj_pos.copy()
            if abs(float(contract.object_init_pos[2])) < 1e-6:
                contract.object_init_pos[2] = 0.05
    return contract


def apply_init_pose_file(contract: TaskContract, path: Path | None) -> TaskContract:
    if path is None:
        return contract
    try:
        import yaml
    except ModuleNotFoundError:
        raise RuntimeError("--init-pose-file requires PyYAML")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    asset = cfg.get("env", {}).get("asset", cfg)
    if "handRootPos" in asset:
        contract.hand_root_pos = np.asarray(asset["handRootPos"], dtype=np.float32)
    if "handRootRPY" in asset:
        contract.hand_root_rpy = np.asarray(asset["handRootRPY"], dtype=np.float32)
    if "handInitPose" in asset:
        pose = asset["handInitPose"]
        contract.init_q = np.asarray([pose[name] for name in POLICY_JOINT_NAMES], dtype=np.float32)
    return contract


def apply_reset_source(contract: TaskContract, reset_source: str) -> TaskContract:
    contract.reset_source = reset_source
    if reset_source == "yaml":
        return contract
    if reset_source != "isaacgym_screwdriver":
        raise ValueError(f"Unknown reset source: {reset_source}")

    contract.init_q = ISAACGYM_SCREWDRIVER_INIT_Q.copy()
    contract.hand_root_pos = ISAACGYM_SCREWDRIVER_HAND_ROOT_POS.copy()
    contract.hand_root_rpy = ISAACGYM_SCREWDRIVER_HAND_ROOT_RPY.copy()
    contract.object_init_pos = contract.object_init_pos.copy()
    contract.object_init_pos[2] = 0.0
    return contract


def vector_from_reference(payload: dict, key: str, length: int | None = None) -> np.ndarray | None:
    value = payload.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if length is not None:
        if array.size < length:
            return None
        array = array[:length]
    return array


def apply_reference_state_json(contract: TaskContract, path: Path | None) -> TaskContract:
    if path is None:
        return contract
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    contract.reference_state_json = str(path)
    contract.reference_phase = str(payload.get("phase", ""))
    if contract.reference_phase and contract.reference_phase != "pre_step":
        raise ValueError(
            "--reference-state-json replay expects a pre_step snapshot. "
            f"Got phase={contract.reference_phase!r}; use isaacgym_ref_*_pre_step.json."
        )

    hand_dof_names = payload.get("hand_dof_names")
    if hand_dof_names is not None:
        names = tuple(str(v) for v in hand_dof_names[: len(POLICY_JOINT_NAMES)])
        contract.reference_hand_dof_names = names
        expected = tuple(POLICY_JOINT_NAMES)
        if names != expected:
            raise ValueError(
                "IsaacGym hand_dof_names do not match MuJoCo POLICY_JOINT_NAMES. "
                f"expected={list(expected)}, got={list(names)}"
            )

    q = vector_from_reference(payload, "hand_dof_pos", 16)
    if q is not None:
        contract.init_q = q.astype(np.float32)
    object_scale = vector_from_reference(payload, "object_scale", 1)
    if object_scale is not None:
        contract.object_scale = float(object_scale[0])
    qvel = vector_from_reference(payload, "hand_dof_vel", 16)
    if qvel is not None:
        contract.init_qvel = qvel.astype(np.float32)

    for key in ("cur_targets", "prev_targets", "init_pose_buf"):
        target = vector_from_reference(payload, key, 16)
        if target is not None:
            contract.init_target = target.astype(np.float32)
            break

    hand_root = vector_from_reference(payload, "hand_root_state_xyzw", 7)
    if hand_root is not None:
        contract.hand_root_pos = hand_root[:3].astype(np.float32)
        qx, qy, qz, qw = [float(v) for v in hand_root[3:7]]
        contract.hand_root_quat_wxyz = np.asarray([qw, qx, qy, qz], dtype=np.float64)

    object_root = vector_from_reference(payload, "object_root_state_xyzw", 7)
    if object_root is not None:
        contract.object_init_pos = object_root[:3].astype(np.float32)
        qx, qy, qz, qw = [float(v) for v in object_root[3:7]]
        contract.object_root_quat_wxyz = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    object_root_full = vector_from_reference(payload, "object_root_state_xyzw", 13)
    if object_root_full is not None:
        contract.object_root_linvel = object_root_full[7:10].astype(np.float32)
        contract.object_root_angvel = object_root_full[10:13].astype(np.float32)

    nut_dof = vector_from_reference(payload, "nut_dof_pos", 1)
    if nut_dof is not None:
        contract.object_axis_pos = float(nut_dof[0])
    nut_dof_vel = vector_from_reference(payload, "nut_dof_vel", 1)
    if nut_dof_vel is not None:
        contract.object_axis_vel = float(nut_dof_vel[0])

    action = vector_from_reference(payload, "policy_action", 16)
    if action is not None:
        contract.reference_action = action.astype(np.float32)
    return contract


def _resolve_xml_file_attrs(root: ET.Element, base_dir: Path) -> None:
    for elem in root.iter():
        file_attr = elem.get("file")
        if not file_attr:
            continue
        path = Path(file_attr)
        if not path.is_absolute():
            elem.set("file", str((base_dir / path).resolve()))


def _set_proxy_contact_attrs(geom: ET.Element, margin: float) -> None:
    geom.set("rgba", "0.1 0.9 0.3 0.35")
    geom.set("condim", "4")
    geom.set("friction", "0.8 0.003 0.0001")
    geom.set("solref", "0.015 1")
    geom.set("solimp", "0.8 0.98 0.001")
    geom.set("margin", f"{float(margin):.9g}")


def _add_distal_sphere_proxy_geoms(root: ET.Element, contract: TaskContract) -> int:
    proxy_specs = (
        (
            "right_index_tactile_link_2",
            "right_index_distal_proxy",
            contract.index_proxy_pos,
            contract.index_proxy_size,
            contract.index_proxy_margin,
        ),
        (
            "right_thumb_tactile_link_1",
            "right_thumb_distal_proxy",
            contract.thumb_proxy_pos,
            contract.thumb_proxy_size,
            contract.thumb_proxy_margin,
        ),
    )
    added = 0
    for body_name, geom_name, pos, size_override, margin_override in proxy_specs:
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            continue
        if body.find(f"./geom[@name='{geom_name}']") is not None:
            continue
        size = contract.active_proxy_size if size_override is None else size_override
        margin = contract.active_proxy_margin if margin_override is None else margin_override
        geom = ET.SubElement(body, "geom")
        geom.set("name", geom_name)
        geom.set("type", "sphere")
        geom.set("pos", " ".join(f"{float(v):.9g}" for v in pos))
        geom.set("size", f"{float(size):.9g}")
        _set_proxy_contact_attrs(geom, margin)
        added += 1
    return added


def _add_tactile_capsule_proxy_geoms(root: ET.Element, contract: TaskContract) -> int:
    index_radius = contract.index_proxy_size if contract.index_proxy_size is not None else 0.0035
    thumb_radius = contract.thumb_proxy_size if contract.thumb_proxy_size is not None else 0.0035
    index_margin = contract.index_proxy_margin if contract.index_proxy_margin is not None else contract.active_proxy_margin
    thumb_margin = contract.thumb_proxy_margin if contract.thumb_proxy_margin is not None else contract.active_proxy_margin
    index_half_length = contract.index_proxy_half_length if contract.index_proxy_half_length is not None else 0.0115
    thumb_half_length = contract.thumb_proxy_half_length if contract.thumb_proxy_half_length is not None else 0.0148
    proxy_specs = (
        (
            "right_index_tactile_link_2",
            "right_index_tactile_capsule",
            contract.index_proxy_pos,
            index_radius,
            index_margin,
            index_half_length,
            (-0.007, 0.0, 0.007),
        ),
        (
            "right_thumb_tactile_link_1",
            "right_thumb_tactile_capsule",
            contract.thumb_proxy_pos,
            thumb_radius,
            thumb_margin,
            thumb_half_length,
            (-0.009, 0.0, 0.009),
        ),
    )
    added = 0
    for body_name, geom_prefix, center, radius, margin, half_length_y, x_offsets in proxy_specs:
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            continue
        cx, cy, cz = [float(v) for v in center]
        for idx, x_offset in enumerate(x_offsets):
            geom_name = f"{geom_prefix}_{idx}"
            if body.find(f"./geom[@name='{geom_name}']") is not None:
                continue
            x = cx + float(x_offset)
            geom = ET.SubElement(body, "geom")
            geom.set("name", geom_name)
            geom.set("type", "capsule")
            geom.set(
                "fromto",
                " ".join(
                    f"{float(v):.9g}"
                    for v in (x, cy - half_length_y, cz, x, cy + half_length_y, cz)
                ),
            )
            geom.set("size", f"{float(radius):.9g}")
            _set_proxy_contact_attrs(geom, margin)
        added += 1
    return added


def _add_active_proxy_geoms(root: ET.Element, contract: TaskContract) -> int:
    if contract.active_proxy_profile == "distal_spheres":
        return _add_distal_sphere_proxy_geoms(root, contract)
    if contract.active_proxy_profile == "tactile_capsules":
        return _add_tactile_capsule_proxy_geoms(root, contract)
    return 0


def prepare_scene_xml(scene: Path, contract: TaskContract, out_dir: Path) -> Path:
    needs_xml = False
    if contract.object_contact0_mesh or contract.object_contact1_mesh:
        needs_xml = True
    if contract.active_pair_profile != "none":
        needs_xml = True
    if contract.active_proxy_profile != "none":
        needs_xml = True
    if contract.object_scale is not None and abs(float(contract.object_scale) - float(contract.base_obj_scale)) >= 1e-6:
        needs_xml = True
    if not needs_xml:
        return scene

    tree = ET.parse(scene)
    root = tree.getroot()
    _resolve_xml_file_attrs(root, scene.resolve().parent)

    mesh_overrides = {
        "codrive_lightbulb_contact0": contract.object_contact0_mesh,
        "codrive_lightbulb_contact1": contract.object_contact1_mesh,
    }
    for mesh in root.findall(".//mesh"):
        override = mesh_overrides.get(mesh.get("name") or "")
        if override:
            mesh.set("file", str(Path(override).expanduser().resolve()))

    scale = float(contract.object_scale if contract.object_scale is not None else contract.base_obj_scale)
    for mesh in root.findall(".//mesh"):
        if mesh.get("name") in SCALABLE_OBJECT_MESHES:
            mesh.set("scale", f"{scale:.9g} {scale:.9g} {scale:.9g}")

    for body in root.findall(".//body"):
        if body.get("name") == "codrive_lightbulb_bolt":
            body.set("pos", f"0 0 {SCALABLE_OBJECT_BOLT_Z * scale:.9g}")

    for geom in root.findall(".//geom"):
        name = geom.get("name")
        if name == "codrive_lightbulb_base":
            geom.set(
                "size",
                f"{SCALABLE_OBJECT_GEOM_BASE_RADIUS * scale:.9g} "
                f"{SCALABLE_OBJECT_GEOM_BASE_HALFHEIGHT * scale:.9g}",
            )
        elif name in SCALABLE_OBJECT_MESHES:
            geom.set("pos", f"0 0 {SCALABLE_OBJECT_CONTACT_Z * scale:.9g}")

    pair_profile = ACTIVE_CONTACT0_PAIR_PROFILES.get(contract.active_pair_profile)
    if pair_profile is None:
        raise ValueError(f"Unknown active pair profile: {contract.active_pair_profile}")
    if pair_profile:
        contact_elem = root.find("contact")
        if contact_elem is None:
            contact_elem = ET.SubElement(root, "contact")
        for geom1, geom2, condim, friction, solref, solimp in pair_profile:
            pair = ET.SubElement(contact_elem, "pair")
            pair.set("geom1", geom1)
            pair.set("geom2", geom2)
            pair.set("condim", str(condim))
            pair.set("friction", " ".join(f"{float(v):.9g}" for v in friction))
            pair.set("solref", " ".join(f"{float(v):.9g}" for v in solref))
            pair.set("solimp", " ".join(f"{float(v):.9g}" for v in solimp))

    if contract.active_proxy_profile not in ACTIVE_PROXY_PROFILES:
        raise ValueError(f"Unknown active proxy profile: {contract.active_proxy_profile}")
    if contract.active_proxy_profile != "none":
        added = _add_active_proxy_geoms(root, contract)
        if added < 2:
            for include in root.findall(".//include"):
                include_file = include.get("file")
                if not include_file:
                    continue
                include_path = Path(include_file)
                include_tree = ET.parse(include_path)
                include_root = include_tree.getroot()
                _resolve_xml_file_attrs(include_root, include_path.resolve().parent)
                include_added = _add_active_proxy_geoms(include_root, contract)
                if include_added:
                    include_out = out_dir / f"{include_path.stem}_proxy_{contract.active_proxy_profile}.xml"
                    include_tree.write(include_out, encoding="unicode")
                    include.set("file", str(include_out.resolve()))
                    added += include_added
            if added < 2:
                raise ValueError("Could not inject both active distal proxy geoms")

    suffix = f"object_scale_{scale:.6g}"
    if contract.active_pair_profile != "none":
        suffix += f"_pair_{contract.active_pair_profile}"
    if contract.active_proxy_profile != "none":
        suffix += f"_proxy_{contract.active_proxy_profile}"
    out_path = out_dir / f"{scene.stem}_{suffix}.xml"
    tree.write(out_path, encoding="unicode")
    return out_path


def quat_wxyz_from_rpy_xyz(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in quat]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def mj_name(model, obj_type, idx: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, idx)
    return "" if name is None else name


def joint_mapping(model) -> tuple[np.ndarray, np.ndarray]:
    qpos_adrs = []
    dof_adrs = []
    missing = []
    for name in POLICY_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            missing.append(name)
            continue
        qpos_adrs.append(int(model.jnt_qposadr[jid]))
        dof_adrs.append(int(model.jnt_dofadr[jid]))
    if missing:
        raise KeyError(f"MuJoCo model is missing policy joints: {missing}")
    return np.asarray(qpos_adrs, dtype=np.int32), np.asarray(dof_adrs, dtype=np.int32)


def configure_joint_limits(model, contract: TaskContract, mode: str) -> None:
    for idx, name in enumerate(POLICY_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        if mode == "task":
            model.jnt_range[jid, 0] = float(contract.lower[idx])
            model.jnt_range[jid, 1] = float(contract.upper[idx])
            model.jnt_limited[jid] = 1
        elif mode == "model":
            contract.lower[idx] = float(model.jnt_range[jid, 0])
            contract.upper[idx] = float(model.jnt_range[jid, 1])
        else:
            raise ValueError(f"Unknown joint limit mode: {mode}")


def clamp_init_q_to_limits(contract: TaskContract) -> None:
    clipped = np.clip(contract.init_q, contract.lower, contract.upper).astype(np.float32)
    contract.init_q_clipped_count = int(np.count_nonzero(np.abs(clipped - contract.init_q) > 1e-6))
    contract.init_q = clipped


def apply_reset(model, data, contract: TaskContract, qpos_adrs: np.ndarray, dof_adrs: np.ndarray) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[qpos_adrs] = contract.init_q
    if contract.init_qvel is not None:
        data.qvel[dof_adrs] = contract.init_qvel
    else:
        data.qvel[dof_adrs] = 0.0

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_palm")
    if body_id >= 0:
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id >= 0:
            data.mocap_pos[mocap_id] = contract.hand_root_pos
            if contract.hand_root_quat_wxyz is not None:
                data.mocap_quat[mocap_id] = contract.hand_root_quat_wxyz
            else:
                data.mocap_quat[mocap_id] = quat_wxyz_from_rpy_xyz(contract.hand_root_rpy)

    joint_type, object_joint_id = object_joint_info(model)
    if joint_type == "free":
        adr = int(model.jnt_qposadr[object_joint_id])
        dadr = int(model.jnt_dofadr[object_joint_id])
        data.qpos[adr : adr + 3] = contract.object_init_pos
        if contract.object_root_quat_wxyz is not None:
            data.qpos[adr + 3 : adr + 7] = contract.object_root_quat_wxyz
        else:
            data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if contract.object_root_linvel is not None:
            data.qvel[dadr : dadr + 3] = contract.object_root_linvel
        else:
            data.qvel[dadr : dadr + 3] = 0.0
        if contract.object_root_angvel is not None:
            data.qvel[dadr + 3 : dadr + 6] = contract.object_root_angvel
        else:
            data.qvel[dadr + 3 : dadr + 6] = 0.0
    elif joint_type == "hinge":
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME)
        if body_id >= 0:
            model.body_pos[body_id] = contract.object_init_pos
        adr = int(model.jnt_qposadr[object_joint_id])
        dadr = int(model.jnt_dofadr[object_joint_id])
        data.qpos[adr] = float(contract.object_axis_pos)
        data.qvel[dadr] = float(contract.object_axis_vel)
    mujoco.mj_forward(model, data)


def read_q(data, qpos_adrs: np.ndarray) -> np.ndarray:
    return data.qpos[qpos_adrs].astype(np.float32).copy()


def read_qvel(data, dof_adrs: np.ndarray) -> np.ndarray:
    return data.qvel[dof_adrs].astype(np.float32).copy()


def clamp_locked(vec: np.ndarray, mask_indices: Iterable[int]) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float32).copy()
    for idx in mask_indices:
        out[int(idx)] = 0.0
    return out


def masked_action(action: np.ndarray, mask_indices: Iterable[int]) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    for idx in mask_indices:
        out[int(idx)] = 0.0
    return out


def pd_torque(q: np.ndarray, qvel: np.ndarray, target: np.ndarray, contract: TaskContract) -> np.ndarray:
    tau = contract.pgain * (target - q) - contract.dgain * qvel
    return np.clip(tau, -contract.torque_limit, contract.torque_limit).astype(np.float32)


def apply_joint_torque(data, dof_adrs: np.ndarray, tau: np.ndarray) -> None:
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[dof_adrs] = tau


def body_name_for_geom(model, geom_id: int) -> str:
    if geom_id < 0:
        return ""
    body_id = int(model.geom_bodyid[geom_id])
    return mj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def geom_name_for_id(model, geom_id: int) -> str:
    if geom_id < 0:
        return ""
    name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name:
        return name
    body_name = body_name_for_geom(model, geom_id)
    return f"geom_{geom_id}@{body_name}"


def mesh_name_for_geom(model, geom_id: int) -> str:
    if geom_id < 0:
        return ""
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        return ""
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return ""
    return mj_name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)


def finger_for_body(body_name: str) -> str | None:
    for finger, prefix in FINGER_PREFIXES.items():
        if body_name.startswith(prefix):
            return finger
    return None


def object_contact_body(body_name: str) -> bool:
    return body_name == OBJECT_BODY_NAME or body_name.startswith(f"{OBJECT_BODY_NAME}_")


def apply_finger_contact_mode(model, mode: str) -> int:
    if mode == "full":
        return 0
    if mode not in {
        "active_pad_proxy",
        "tip_proxy",
        "no_index_tactile2",
        "index_tip_thumb_pad_proxy",
        "active_distal_proxy",
        "active_proxy_only",
    }:
        raise ValueError(f"Unknown finger contact mode: {mode}")

    if mode == "no_index_tactile2":
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_index_tactile_link_2")
        if geom_id < 0:
            raise ValueError("right_index_tactile_link_2 geom not found")
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
        return 1

    active_tip_geoms = {"right_index_tip", "right_thumb_tip"}
    active_pad_meshes = {
        "right_index_tactile_link_0",
        "right_index_tactile_link_1",
        "right_index_tactile_link_2",
        "right_thumb_tactile_link_0",
        "right_thumb_tactile_link_1",
    }
    index_tip_thumb_pad_geoms = {"right_index_tip", "right_thumb_tip"}
    thumb_pad_meshes = {
        "right_thumb_tactile_link_0",
        "right_thumb_tactile_link_1",
    }
    disabled = 0
    for geom_id in range(model.ngeom):
        body_name = body_name_for_geom(model, geom_id)
        geom_name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        mesh_name = mesh_name_for_geom(model, geom_id)
        finger = finger_for_body(body_name)
        if finger not in {"index", "thumb"}:
            continue
        keep_contact = geom_name in active_tip_geoms
        if mode == "active_pad_proxy":
            keep_contact = keep_contact or mesh_name in active_pad_meshes
        elif mode == "index_tip_thumb_pad_proxy":
            keep_contact = geom_name in index_tip_thumb_pad_geoms or (
                finger == "thumb" and mesh_name in thumb_pad_meshes
            )
        elif mode == "active_distal_proxy":
            keep_contact = geom_name in active_tip_geoms or geom_name in ACTIVE_PROXY_GEOMS
        elif mode == "active_proxy_only":
            keep_contact = geom_name in ACTIVE_PROXY_GEOMS
        if not keep_contact:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
            disabled += 1
    return disabled


def hinge_frictionloss(model) -> float:
    hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_HINGE_JOINT_NAME)
    if hinge_id < 0:
        return 0.0
    dof_id = int(model.jnt_dofadr[hinge_id])
    return float(model.dof_frictionloss[dof_id])


def apply_object_contact_mode(model, mode: str) -> tuple[int, float]:
    if mode not in {"default", "low_friction_debug", "high_hinge_friction_debug", "low_friction_high_hinge_debug"}:
        raise ValueError(f"Unknown object contact mode: {mode}")

    modified_geoms = 0
    if mode in {"low_friction_debug", "low_friction_high_hinge_debug"}:
        for geom_id in range(model.ngeom):
            geom_name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name.startswith(f"{OBJECT_BODY_NAME}_contact"):
                model.geom_friction[geom_id, :] = np.array([0.5, 0.001, 0.0001], dtype=np.float64)
                modified_geoms += 1

    if mode in {"high_hinge_friction_debug", "low_friction_high_hinge_debug"}:
        hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_HINGE_JOINT_NAME)
        if hinge_id >= 0:
            dof_id = int(model.jnt_dofadr[hinge_id])
            model.dof_frictionloss[dof_id] = 1.0

    return modified_geoms, hinge_frictionloss(model)


def apply_contact_overrides(model, contract: TaskContract) -> None:
    if contract.active_tactile_friction_override is not None:
        for geom_id in range(model.ngeom):
            geom_name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name in ACTIVE_TACTILE_GEOMS:
                model.geom_friction[geom_id, :] = np.asarray(contract.active_tactile_friction_override, dtype=np.float64)

    if contract.active_tactile_solref_override is not None:
        for geom_id in range(model.ngeom):
            geom_name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name in ACTIVE_TACTILE_GEOMS:
                model.geom_solref[geom_id, :] = np.asarray(contract.active_tactile_solref_override, dtype=np.float64)

    if contract.active_tactile_solimp_override is not None:
        for geom_id in range(model.ngeom):
            geom_name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name in ACTIVE_TACTILE_GEOMS:
                model.geom_solimp[geom_id, :3] = np.asarray(contract.active_tactile_solimp_override, dtype=np.float64)

    if contract.active_tactile_margin_override is not None:
        for geom_id in range(model.ngeom):
            geom_name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name in ACTIVE_TACTILE_GEOMS:
                model.geom_margin[geom_id] = float(contract.active_tactile_margin_override)

    if contract.active_tip_margin_override is not None:
        for geom_id in range(model.ngeom):
            geom_name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name in {"right_index_tip", "right_thumb_tip"}:
                model.geom_margin[geom_id] = float(contract.active_tip_margin_override)

    if contract.object_friction_override is not None:
        for geom_id in range(model.ngeom):
            geom_name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name.startswith(f"{OBJECT_BODY_NAME}_contact"):
                model.geom_friction[geom_id, :] = np.asarray(contract.object_friction_override, dtype=np.float64)
                contract.modified_object_contact_geoms += 1

    if contract.object_solref_override is not None:
        for geom_id in range(model.ngeom):
            geom_name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name.startswith(f"{OBJECT_BODY_NAME}_contact"):
                model.geom_solref[geom_id, :] = np.asarray(contract.object_solref_override, dtype=np.float64)
                contract.modified_object_contact_geoms += 1

    if contract.object_solimp_override is not None:
        for geom_id in range(model.ngeom):
            geom_name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name.startswith(f"{OBJECT_BODY_NAME}_contact"):
                model.geom_solimp[geom_id, :3] = np.asarray(contract.object_solimp_override, dtype=np.float64)
                contract.modified_object_contact_geoms += 1

    if contract.object_condim_override is not None:
        for geom_id in range(model.ngeom):
            geom_name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name.startswith(f"{OBJECT_BODY_NAME}_contact"):
                model.geom_condim[geom_id] = int(contract.object_condim_override)
                contract.modified_object_contact_geoms += 1

    if contract.object_margin_override is not None:
        for geom_id in range(model.ngeom):
            geom_name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name.startswith(f"{OBJECT_BODY_NAME}_contact"):
                model.geom_margin[geom_id] = float(contract.object_margin_override)
                contract.modified_object_contact_geoms += 1

    if contract.object_gap_override is not None:
        for geom_id in range(model.ngeom):
            geom_name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name.startswith(f"{OBJECT_BODY_NAME}_contact"):
                model.geom_gap[geom_id] = float(contract.object_gap_override)
                contract.modified_object_contact_geoms += 1

    contact_pos_offset = np.asarray(contract.object_contact_pos_offset, dtype=np.float64).copy()
    contact_pos_offset[2] += float(contract.object_contact_z_offset)
    if np.any(np.abs(contact_pos_offset) > 0.0):
        for geom_id in range(model.ngeom):
            geom_name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name.startswith(f"{OBJECT_BODY_NAME}_contact"):
                model.geom_pos[geom_id, :] += contact_pos_offset
                contract.modified_object_contact_geoms += 1

    if contract.hinge_frictionloss_override is not None:
        hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_HINGE_JOINT_NAME)
        if hinge_id >= 0:
            dof_id = int(model.jnt_dofadr[hinge_id])
            model.dof_frictionloss[dof_id] = float(contract.hinge_frictionloss_override)

    contract.object_hinge_frictionloss = hinge_frictionloss(model)


def object_pose_body_id(model) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_NUT_BODY_NAME)
    if body_id >= 0:
        return int(body_id)
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME))


def object_joint_info(model) -> tuple[str, int]:
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_FREE_JOINT_NAME)
    if free_id >= 0:
        return "free", int(free_id)
    hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_HINGE_JOINT_NAME)
    if hinge_id >= 0:
        return "hinge", int(hinge_id)
    return "none", -1


def mean_object_geom_pos(model, data) -> np.ndarray | None:
    positions = []
    for geom_id in range(model.ngeom):
        name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name.startswith(f"{OBJECT_BODY_NAME}_contact"):
            positions.append(data.geom_xpos[geom_id].copy())
    if not positions:
        return None
    return np.mean(np.asarray(positions, dtype=np.float64), axis=0)


def object_state(model, data) -> dict:
    joint_type, joint_id = object_joint_info(model)
    if joint_id < 0:
        return {}
    qadr = int(model.jnt_qposadr[joint_id])
    dadr = int(model.jnt_dofadr[joint_id])
    axis_pos = 0.0
    axis_vel = 0.0
    linvel = np.zeros(3, dtype=np.float64)
    angvel = np.zeros(3, dtype=np.float64)
    if joint_type == "free":
        pos = data.qpos[qadr : qadr + 3]
        quat = data.qpos[qadr + 3 : qadr + 7]
        linvel = data.qvel[dadr : dadr + 3]
        angvel = data.qvel[dadr + 3 : dadr + 6]
        axis_pos = yaw_from_quat_wxyz(quat)
        axis_vel = float(angvel[2])
    elif joint_type == "hinge":
        body_id = object_pose_body_id(model)
        if body_id < 0:
            return {}
        pos = data.xpos[body_id]
        quat = data.xquat[body_id]
        cvel = data.cvel[body_id]
        angvel = cvel[:3]
        linvel = cvel[3:]
        axis_pos = float(data.qpos[qadr])
        axis_vel = float(data.qvel[dadr])
    else:
        return {}
    geom_pos = mean_object_geom_pos(model, data)
    if geom_pos is None:
        geom_pos = np.asarray(pos, dtype=np.float64)
    return {
        "object_joint_type": joint_type,
        "object_x": float(pos[0]),
        "object_y": float(pos[1]),
        "object_z": float(pos[2]),
        "object_geom_x": float(geom_pos[0]),
        "object_geom_y": float(geom_pos[1]),
        "object_geom_z": float(geom_pos[2]),
        "object_qw": float(quat[0]),
        "object_qx": float(quat[1]),
        "object_qy": float(quat[2]),
        "object_qz": float(quat[3]),
        "object_yaw": float(yaw_from_quat_wxyz(quat)),
        "object_axis_pos": float(axis_pos),
        "object_axis_vel": float(axis_vel),
        "object_linvel_x": float(linvel[0]),
        "object_linvel_y": float(linvel[1]),
        "object_linvel_z": float(linvel[2]),
        "object_angvel_x": float(angvel[0]),
        "object_angvel_y": float(angvel[1]),
        "object_angvel_z": float(angvel[2]),
    }


def active_tip_geometry(model, data) -> dict:
    out = {}
    object_geom_pos = mean_object_geom_pos(model, data)
    for finger in ("index", "thumb"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"right_{finger}_tip")
        if geom_id < 0:
            continue
        pos = data.geom_xpos[geom_id].copy()
        out[f"{finger}_tip_x"] = float(pos[0])
        out[f"{finger}_tip_y"] = float(pos[1])
        out[f"{finger}_tip_z"] = float(pos[2])
        if object_geom_pos is not None:
            out[f"{finger}_tip_to_object_geom_dist"] = float(np.linalg.norm(pos - object_geom_pos))
    if "index_tip_x" in out and "thumb_tip_x" in out:
        index_pos = np.array([out["index_tip_x"], out["index_tip_y"], out["index_tip_z"]], dtype=np.float64)
        thumb_pos = np.array([out["thumb_tip_x"], out["thumb_tip_y"], out["thumb_tip_z"]], dtype=np.float64)
        out["index_thumb_tip_dist"] = float(np.linalg.norm(index_pos - thumb_pos))
    return out


def contact_summary(model, data) -> dict:
    counts = {finger: 0 for finger in FINGER_PREFIXES}
    force_sums = {finger: 0.0 for finger in FINGER_PREFIXES}
    tip_counts = {finger: 0 for finger in FINGER_PREFIXES}
    tip_force_sums = {finger: 0.0 for finger in FINGER_PREFIXES}
    active_count = 0
    active_force = 0.0
    active_tip_count = 0
    active_tip_force = 0.0
    object_count = 0
    min_object_contact_dist = math.inf
    min_active_contact_dist = math.inf
    force = np.zeros(6, dtype=np.float64)
    object_pairs = []
    object_pair_details = []
    finger_geoms = {finger: set() for finger in FINGER_PREFIXES}
    finger_tip_geoms = {finger: set() for finger in FINGER_PREFIXES}

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = body_name_for_geom(model, geom1)
        body2 = body_name_for_geom(model, geom2)
        geom_name1 = geom_name_for_id(model, geom1)
        geom_name2 = geom_name_for_id(model, geom2)
        if object_contact_body(body1):
            finger = finger_for_body(body2)
            finger_geom_name = geom_name2
            object_geom_name = geom_name1
        elif object_contact_body(body2):
            finger = finger_for_body(body1)
            finger_geom_name = geom_name1
            object_geom_name = geom_name2
        else:
            continue
        object_pairs.append(f"{body1}/{geom_name1}<->{body2}/{geom_name2}")
        object_count += 1
        min_object_contact_dist = min(min_object_contact_dist, float(contact.dist))
        mujoco.mj_contactForce(model, data, i, force)
        normal_force = float(force[0])
        tangent_force = float(np.linalg.norm(force[1:3]))
        object_pair_details.append(
            f"{body1}/{geom_name1}<->{body2}/{geom_name2}"
            f"|dist={float(contact.dist):.9g}|fn={normal_force:.9g}|ft={tangent_force:.9g}"
        )
        if finger is None:
            continue
        force_norm = float(np.linalg.norm(force[:3]))
        counts[finger] += 1
        force_sums[finger] += force_norm
        finger_geoms[finger].add(f"{finger_geom_name}->{object_geom_name}")
        if finger_geom_name == f"right_{finger}_tip":
            tip_counts[finger] += 1
            tip_force_sums[finger] += force_norm
            finger_tip_geoms[finger].add(f"{finger_geom_name}->{object_geom_name}")
        if finger in ("index", "thumb"):
            active_count += 1
            active_force += force_norm
            min_active_contact_dist = min(min_active_contact_dist, float(contact.dist))
            if finger_geom_name == f"right_{finger}_tip":
                active_tip_count += 1
                active_tip_force += force_norm

    out = {
        "object_contact_count": int(object_count),
        "active_contact_count": int(active_count),
        "active_contact_force": float(active_force),
        "active_tip_contact_count": int(active_tip_count),
        "active_tip_contact_force": float(active_tip_force),
        "object_contact_pairs": ";".join(sorted(object_pairs)),
        "object_contact_details": ";".join(sorted(object_pair_details)),
        "min_object_contact_dist": 0.0 if math.isinf(min_object_contact_dist) else float(min_object_contact_dist),
        "min_active_contact_dist": 0.0 if math.isinf(min_active_contact_dist) else float(min_active_contact_dist),
    }
    for finger in FINGER_PREFIXES:
        out[f"{finger}_object_contact_count"] = int(counts[finger])
        out[f"{finger}_object_contact_force"] = float(force_sums[finger])
        out[f"{finger}_tip_contact_count"] = int(tip_counts[finger])
        out[f"{finger}_tip_contact_force"] = float(tip_force_sums[finger])
        out[f"{finger}_contact_geoms"] = ";".join(sorted(finger_geoms[finger]))
        out[f"{finger}_tip_contact_geoms"] = ";".join(sorted(finger_tip_geoms[finger]))
    return out


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


def contact_geom_table(model) -> list[dict]:
    rows = []
    for geom_id in range(model.ngeom):
        body_name = body_name_for_geom(model, geom_id)
        geom_name = geom_name_for_id(model, geom_id)
        finger = finger_for_body(body_name)
        is_object = object_contact_body(body_name) or geom_name.startswith(f"{OBJECT_BODY_NAME}_contact")
        if finger is None and not is_object:
            continue
        rows.append(
            {
                "geom_id": int(geom_id),
                "geom_name": geom_name,
                "body_name": body_name,
                "finger": "" if finger is None else finger,
                "mesh_name": mesh_name_for_geom(model, geom_id),
                "geom_type": int(model.geom_type[geom_id]),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "friction": [float(v) for v in model.geom_friction[geom_id]],
                "margin": float(model.geom_margin[geom_id]),
                "gap": float(model.geom_gap[geom_id]),
            }
        )
    return rows


def prefixed_contact_summary(model, data, prefix: str = "reset") -> dict:
    summary = contact_summary(model, data)
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def summarize_model(model, qpos_adrs: np.ndarray, dof_adrs: np.ndarray, contract: TaskContract) -> dict:
    joints = []
    for i, name in enumerate(POLICY_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joints.append(
            {
                "policy_index": i,
                "name": name,
                "joint_id": int(jid),
                "qposadr": int(qpos_adrs[i]),
                "dofadr": int(dof_adrs[i]),
                "range": [float(v) for v in model.jnt_range[jid]],
                "dof_damping": float(model.dof_damping[dof_adrs[i]]),
                "dof_armature": float(model.dof_armature[dof_adrs[i]]),
                "dof_frictionloss": float(model.dof_frictionloss[dof_adrs[i]]),
                "joint_stiffness": float(model.jnt_stiffness[jid]),
                "contract_lower": float(contract.lower[i]),
                "contract_upper": float(contract.upper[i]),
            }
        )
    actuators = []
    for aid in range(model.nu):
        actuators.append(
            {
                "id": aid,
                "name": mj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid),
                "trntype": int(model.actuator_trntype[aid]),
                "trnid": [int(v) for v in model.actuator_trnid[aid]],
                "ctrlrange": [float(v) for v in model.actuator_ctrlrange[aid]],
            }
        )
    policy_interval = float(contract.dt * contract.control_decimation)
    policy_rate = float(1.0 / policy_interval)
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "timestep": float(model.opt.timestep),
        "policy_interval_s": policy_interval,
        "policy_rate_hz": policy_rate,
        "target_policy_hz": float(contract.target_policy_hz),
        "policy_rate_error_hz": float(policy_rate - contract.target_policy_hz),
        "control": {
            "dt": contract.dt,
            "decimation": contract.control_decimation,
            "action_scale": contract.action_scale,
            "pgain": contract.pgain,
            "dgain": contract.dgain,
            "torque_limit": contract.torque_limit,
            "action_mask_indices": list(contract.action_mask_indices),
        },
        "reset_pose": {
            "hand_root_pos": [float(v) for v in contract.hand_root_pos],
            "hand_root_rpy": [float(v) for v in contract.hand_root_rpy],
            "hand_root_quat_wxyz": None
            if contract.hand_root_quat_wxyz is None
            else [float(v) for v in contract.hand_root_quat_wxyz],
            "base_obj_scale": float(contract.base_obj_scale),
            "object_scale": None if contract.object_scale is None else float(contract.object_scale),
            "hand_root_pos_z_scale_comp": float(contract.hand_root_pos_z_scale_comp),
            "object_init_pos": [float(v) for v in contract.object_init_pos],
            "object_root_quat_wxyz": None
            if contract.object_root_quat_wxyz is None
            else [float(v) for v in contract.object_root_quat_wxyz],
            "object_root_linvel": None
            if contract.object_root_linvel is None
            else [float(v) for v in contract.object_root_linvel],
            "object_root_angvel": None
            if contract.object_root_angvel is None
            else [float(v) for v in contract.object_root_angvel],
            "object_axis_pos": float(contract.object_axis_pos),
            "object_axis_vel": float(contract.object_axis_vel),
            "object_contact0_mesh": contract.object_contact0_mesh,
            "object_contact1_mesh": contract.object_contact1_mesh,
            "active_pair_profile": contract.active_pair_profile,
            "active_proxy_profile": contract.active_proxy_profile,
            "index_proxy_pos": [float(v) for v in contract.index_proxy_pos],
            "thumb_proxy_pos": [float(v) for v in contract.thumb_proxy_pos],
            "active_proxy_size": float(contract.active_proxy_size),
            "active_proxy_margin": float(contract.active_proxy_margin),
            "index_proxy_size": None if contract.index_proxy_size is None else float(contract.index_proxy_size),
            "thumb_proxy_size": None if contract.thumb_proxy_size is None else float(contract.thumb_proxy_size),
            "index_proxy_margin": None if contract.index_proxy_margin is None else float(contract.index_proxy_margin),
            "thumb_proxy_margin": None if contract.thumb_proxy_margin is None else float(contract.thumb_proxy_margin),
            "index_proxy_half_length": None
            if contract.index_proxy_half_length is None
            else float(contract.index_proxy_half_length),
            "thumb_proxy_half_length": None
            if contract.thumb_proxy_half_length is None
            else float(contract.thumb_proxy_half_length),
            "active_tactile_friction_override": None
            if contract.active_tactile_friction_override is None
            else [float(v) for v in contract.active_tactile_friction_override],
            "active_tactile_solref_override": None
            if contract.active_tactile_solref_override is None
            else [float(v) for v in contract.active_tactile_solref_override],
            "active_tactile_solimp_override": None
            if contract.active_tactile_solimp_override is None
            else [float(v) for v in contract.active_tactile_solimp_override],
            "reference_state_json": contract.reference_state_json,
            "reference_phase": contract.reference_phase,
            "reference_hand_dof_names": list(contract.reference_hand_dof_names),
            "init_qvel": None
            if contract.init_qvel is None
            else [float(v) for v in contract.init_qvel],
            "init_target": None
            if contract.init_target is None
            else [float(v) for v in contract.init_target],
            "reference_action": None
            if contract.reference_action is None
            else [float(v) for v in contract.reference_action],
            "init_q_clipped_count": int(contract.init_q_clipped_count),
            "finger_contact_mode": contract.finger_contact_mode,
            "object_contact_mode": contract.object_contact_mode,
            "object_friction_override": None
            if contract.object_friction_override is None
            else [float(v) for v in contract.object_friction_override],
            "object_solref_override": None
            if contract.object_solref_override is None
            else [float(v) for v in contract.object_solref_override],
            "object_solimp_override": None
            if contract.object_solimp_override is None
            else [float(v) for v in contract.object_solimp_override],
            "object_condim_override": None
            if contract.object_condim_override is None
            else int(contract.object_condim_override),
            "object_contact_pos_offset": [float(v) for v in contract.object_contact_pos_offset],
            "object_contact_z_offset": float(contract.object_contact_z_offset),
            "hinge_frictionloss_override": None
            if contract.hinge_frictionloss_override is None
            else float(contract.hinge_frictionloss_override),
            "reset_source": contract.reset_source,
            "disabled_finger_mesh_contact_geoms": int(contract.disabled_finger_mesh_contact_geoms),
            "modified_object_contact_geoms": int(contract.modified_object_contact_geoms),
            "object_hinge_frictionloss": float(contract.object_hinge_frictionloss),
        },
        "contact_geoms": contact_geom_table(model),
        "joints": joints,
        "actuators": actuators,
    }


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def apply_cli_overrides(contract: TaskContract, args: argparse.Namespace) -> TaskContract:
    if args.dt is not None:
        contract.dt = float(args.dt)
    if args.policy_hz is not None:
        contract.target_policy_hz = float(args.policy_hz)
    if args.control_decimation is None and args.policy_hz is not None:
        target_interval = 1.0 / float(args.policy_hz)
        contract.control_decimation = max(1, int(round(target_interval / contract.dt)))
    elif args.control_decimation is not None:
        contract.control_decimation = int(args.control_decimation)
    if args.object_pos is not None:
        contract.object_init_pos = np.asarray(args.object_pos, dtype=np.float32)
    if args.object_scale is not None:
        contract.object_scale = float(args.object_scale)
    if args.object_contact0_mesh is not None:
        contract.object_contact0_mesh = str(args.object_contact0_mesh.expanduser().resolve())
    if args.object_contact1_mesh is not None:
        contract.object_contact1_mesh = str(args.object_contact1_mesh.expanduser().resolve())
    contract.active_pair_profile = args.active_pair_profile
    contract.active_proxy_profile = args.active_proxy_profile
    if args.index_proxy_pos is not None:
        contract.index_proxy_pos = tuple(float(v) for v in args.index_proxy_pos)
    if args.thumb_proxy_pos is not None:
        contract.thumb_proxy_pos = tuple(float(v) for v in args.thumb_proxy_pos)
    if args.active_proxy_size is not None:
        contract.active_proxy_size = float(args.active_proxy_size)
    if args.active_proxy_margin is not None:
        contract.active_proxy_margin = float(args.active_proxy_margin)
    if args.index_proxy_size is not None:
        contract.index_proxy_size = float(args.index_proxy_size)
    if args.thumb_proxy_size is not None:
        contract.thumb_proxy_size = float(args.thumb_proxy_size)
    if args.index_proxy_margin is not None:
        contract.index_proxy_margin = float(args.index_proxy_margin)
    if args.thumb_proxy_margin is not None:
        contract.thumb_proxy_margin = float(args.thumb_proxy_margin)
    if args.index_proxy_half_length is not None:
        contract.index_proxy_half_length = float(args.index_proxy_half_length)
    if args.thumb_proxy_half_length is not None:
        contract.thumb_proxy_half_length = float(args.thumb_proxy_half_length)
    if args.active_tactile_friction is not None:
        contract.active_tactile_friction_override = tuple(float(v) for v in args.active_tactile_friction)
    if args.active_tactile_solref is not None:
        contract.active_tactile_solref_override = tuple(float(v) for v in args.active_tactile_solref)
    if args.active_tactile_solimp is not None:
        contract.active_tactile_solimp_override = tuple(float(v) for v in args.active_tactile_solimp)
    if args.active_tactile_margin is not None:
        contract.active_tactile_margin_override = float(args.active_tactile_margin)
    if args.active_tip_margin is not None:
        contract.active_tip_margin_override = float(args.active_tip_margin)
    if args.hand_root_pos is not None:
        contract.hand_root_pos = np.asarray(args.hand_root_pos, dtype=np.float32)
    if args.hand_root_rpy is not None:
        contract.hand_root_rpy = np.asarray(args.hand_root_rpy, dtype=np.float32)
        contract.hand_root_quat_wxyz = None
    contract.finger_contact_mode = args.finger_contact_mode
    contract.object_contact_mode = args.object_contact_mode
    if args.object_friction is not None:
        contract.object_friction_override = tuple(float(v) for v in args.object_friction)
    if args.object_solref is not None:
        contract.object_solref_override = tuple(float(v) for v in args.object_solref)
    if args.object_solimp is not None:
        contract.object_solimp_override = tuple(float(v) for v in args.object_solimp)
    if args.object_condim is not None:
        contract.object_condim_override = int(args.object_condim)
    if args.object_margin is not None:
        contract.object_margin_override = float(args.object_margin)
    if args.object_gap is not None:
        contract.object_gap_override = float(args.object_gap)
    if args.object_contact_pos_offset is not None:
        contract.object_contact_pos_offset = np.asarray(args.object_contact_pos_offset, dtype=np.float64)
    if args.object_contact_z_offset is not None:
        contract.object_contact_z_offset = float(args.object_contact_z_offset)
    if args.hinge_frictionloss is not None:
        contract.hinge_frictionloss_override = float(args.hinge_frictionloss)
    return contract


def add_vector_columns(row: dict, prefix: str, values: np.ndarray) -> None:
    for idx, name in enumerate(POLICY_JOINT_NAMES):
        row[f"{prefix}_{idx:02d}_{name}"] = float(values[idx])


def hard_clamp_joint_state(data, qpos_adrs: np.ndarray, dof_adrs: np.ndarray, contract: TaskContract) -> int:
    q = data.qpos[qpos_adrs].copy()
    clipped = np.clip(q, contract.lower, contract.upper)
    changed = np.abs(clipped - q) > 1e-8
    if np.any(changed):
        data.qpos[qpos_adrs] = clipped
        data.qvel[dof_adrs[changed]] = 0.0
    return int(np.count_nonzero(changed))


class FfmpegVideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float):
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            f"{self.fps:g}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(self.path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        stderr = self.proc.stderr.read().decode("utf-8", errors="replace") if self.proc.stderr else ""
        ret = self.proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {ret}: {stderr[-1000:]}")


def make_render_camera(contract: TaskContract):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 135.0
    cam.elevation = -25.0
    cam.distance = 0.32
    object_contact_z_offset = 0.06 * float(contract.base_obj_scale)
    cam.lookat[:] = [
        float(contract.object_init_pos[0]),
        float(contract.object_init_pos[1]),
        float(contract.object_init_pos[2] + object_contact_z_offset),
    ]
    return cam


def rollout(args, contract: TaskContract) -> dict:
    out_dir = ensure_output_dir(args.output_dir)
    scene_path = prepare_scene_xml(args.scene, contract, out_dir)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = contract.dt
    contract.disabled_finger_mesh_contact_geoms = apply_finger_contact_mode(model, contract.finger_contact_mode)
    contract.modified_object_contact_geoms, contract.object_hinge_frictionloss = apply_object_contact_mode(
        model, contract.object_contact_mode
    )
    apply_contact_overrides(model, contract)
    configure_joint_limits(model, contract, args.joint_limit_mode)
    if not args.no_clamp_init_q:
        clamp_init_q_to_limits(contract)
    data = mujoco.MjData(model)
    qpos_adrs, dof_adrs = joint_mapping(model)
    apply_reset(model, data, contract, qpos_adrs, dof_adrs)

    summary = summarize_model(model, qpos_adrs, dof_adrs, contract)
    summary["scene_path"] = str(scene_path)
    summary["reset_contact_summary"] = contact_summary(model, data)
    (out_dir / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    policy = None
    if args.mode == "policy":
        policy = ProprioAdaptPolicy.from_checkpoint(args.checkpoint)

    target = contract.init_target.copy() if contract.init_target is not None else contract.init_q.copy()
    history = ProprioHistory(length=30, dof=16)
    history.reset(clamp_locked(read_q(data, qpos_adrs), contract.action_mask_indices), target)
    rows = []
    max_abs_q = 0.0
    max_abs_tau = 0.0
    max_abs_action = 0.0
    max_contract_violation = 0.0
    hard_clamp_events = 0
    start = time.time()
    renderer = None
    video_writer = None
    video_path = None
    render_camera = None
    viewer = None
    viewer_wall_start = None
    viewer_sim_start = None
    viewer_step_count = 0
    viewer_closed = False
    if args.render_video:
        video_path = out_dir / f"{args.mode}_rollout.mp4"
        renderer = mujoco.Renderer(model, height=args.render_height, width=args.render_width)
        render_camera = make_render_camera(contract)
        video_writer = FfmpegVideoWriter(
            video_path,
            width=args.render_width,
            height=args.render_height,
            fps=args.render_fps,
        )
    if args.viewer:
        try:
            import mujoco.viewer as mujoco_viewer
        except Exception as exc:
            raise RuntimeError("MuJoCo viewer is unavailable; check your display/OpenGL setup") from exc
        viewer = mujoco_viewer.launch_passive(model, data)
        viewer_sync_every = max(1, int(args.viewer_sync_every))
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0
        viewer.cam.distance = 0.32
        object_contact_z_offset = 0.06 * float(contract.base_obj_scale)
        viewer.cam.lookat[:] = [
            float(contract.object_init_pos[0]),
            float(contract.object_init_pos[1]),
            float(contract.object_init_pos[2] + object_contact_z_offset),
        ]
        viewer.sync()
        viewer_wall_start = time.time()
        viewer_sim_start = float(data.time)

    policy_step = 0
    try:
        while args.policy_steps <= 0 or policy_step < args.policy_steps:
            if viewer is not None and not viewer.is_running():
                break
            q = read_q(data, qpos_adrs)
            q_policy = clamp_locked(q, contract.action_mask_indices)
            history.append(q_policy, target)

            if args.mode == "zero":
                action = np.zeros(16, dtype=np.float32)
            elif args.mode == "poke":
                action = np.zeros(16, dtype=np.float32)
                if policy_step >= args.poke_start:
                    action[args.poke_joint] = float(args.poke_action)
            elif args.mode == "policy":
                assert policy is not None
                action, _extrin = policy.act(history.obs(), history.hist())
            elif args.mode == "reference_action":
                if contract.reference_action is None:
                    raise ValueError("--mode reference_action requires --reference-state-json with policy_action")
                action = contract.reference_action.copy()
            else:
                raise ValueError(f"Rollout mode expected zero/poke/policy/reference_action, got {args.mode}")

            action = masked_action(action, contract.action_mask_indices)
            target = target + contract.action_scale * action
            target = np.clip(target, contract.lower, contract.upper).astype(np.float32)
            target = clamp_locked(target, contract.action_mask_indices)

            last_tau = np.zeros(16, dtype=np.float32)
            for _ in range(contract.control_decimation):
                q = read_q(data, qpos_adrs)
                qvel = read_qvel(data, dof_adrs)
                last_tau = pd_torque(q, qvel, target, contract)
                apply_joint_torque(data, dof_adrs, last_tau)
                mujoco.mj_step(model, data)
                if args.hard_clamp_joints:
                    hard_clamp_events += hard_clamp_joint_state(data, qpos_adrs, dof_adrs, contract)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError(f"Non-finite MuJoCo state at policy_step={policy_step}")
                if viewer is not None:
                    viewer_step_count += 1
                    if viewer_step_count % viewer_sync_every == 0:
                        if not viewer.is_running():
                            viewer_closed = True
                            break
                        viewer.sync()
                        if args.viewer_realtime_factor > 0.0:
                            assert viewer_wall_start is not None and viewer_sim_start is not None
                            sim_elapsed = float(data.time) - viewer_sim_start
                            target_wall_elapsed = sim_elapsed / float(args.viewer_realtime_factor)
                            sleep_s = viewer_wall_start + target_wall_elapsed - time.time()
                            if sleep_s > 0.0:
                                time.sleep(min(sleep_s, 0.05))
                if viewer_closed:
                    break
            if viewer_closed:
                break

            q = read_q(data, qpos_adrs)
            qvel = read_qvel(data, dof_adrs)
            max_abs_q = max(max_abs_q, float(np.max(np.abs(q))))
            max_abs_tau = max(max_abs_tau, float(np.max(np.abs(last_tau))))
            max_abs_action = max(max_abs_action, float(np.max(np.abs(action))))
            violation = np.maximum(q - contract.upper, contract.lower - q)
            violation = float(max(0.0, np.max(violation)))
            max_contract_violation = max(max_contract_violation, violation)
            row = {
                "policy_step": policy_step,
                "time_s": policy_step * contract.dt * contract.control_decimation,
                "max_abs_action": float(np.max(np.abs(action))),
                "max_abs_tau": float(np.max(np.abs(last_tau))),
                "max_abs_q": float(np.max(np.abs(q))),
                "max_abs_qvel": float(np.max(np.abs(qvel))),
                "max_contract_violation": violation,
                "ncon": int(data.ncon),
                "hard_clamp_events": int(hard_clamp_events),
            }
            row.update(object_state(model, data))
            row.update(active_tip_geometry(model, data))
            row.update(contact_summary(model, data))
            add_vector_columns(row, "action", action)
            add_vector_columns(row, "target", target)
            add_vector_columns(row, "q", q)
            add_vector_columns(row, "qvel", qvel)
            add_vector_columns(row, "tau", last_tau)
            rows.append(row)

            if video_writer is not None and renderer is not None and policy_step % args.render_every == 0:
                renderer.update_scene(data, camera=render_camera)
                video_writer.write(renderer.render())
            policy_step += 1
    except KeyboardInterrupt:
        pass
    finally:
        if video_writer is not None:
            video_writer.close()
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()

    csv_path = out_dir / f"{args.mode}_trace.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["policy_step"])
        writer.writeheader()
        writer.writerows(rows)
    if not rows:
        rollout_summary = {
            "mode": args.mode,
            "policy_steps": 0,
            "requested_policy_steps": int(args.policy_steps),
            "physics_steps": 0,
            "elapsed_wall_s": time.time() - start,
            "dt": float(contract.dt),
            "control_decimation": int(contract.control_decimation),
            "policy_rate_hz": float(1.0 / (contract.dt * contract.control_decimation)),
            "target_policy_hz": float(contract.target_policy_hz),
            "joint_limit_mode": args.joint_limit_mode,
            "hard_clamp_joints": bool(args.hard_clamp_joints),
            "finger_contact_mode": contract.finger_contact_mode,
            "object_contact_mode": contract.object_contact_mode,
            "object_friction_override": None
            if contract.object_friction_override is None
            else [float(v) for v in contract.object_friction_override],
            "object_solref_override": None
            if contract.object_solref_override is None
            else [float(v) for v in contract.object_solref_override],
            "object_solimp_override": None
            if contract.object_solimp_override is None
            else [float(v) for v in contract.object_solimp_override],
            "object_condim_override": None
            if contract.object_condim_override is None
            else int(contract.object_condim_override),
            "object_contact_pos_offset": [float(v) for v in contract.object_contact_pos_offset],
            "object_contact_z_offset": float(contract.object_contact_z_offset),
            "hinge_frictionloss_override": None
            if contract.hinge_frictionloss_override is None
            else float(contract.hinge_frictionloss_override),
            "reset_source": contract.reset_source,
            "disabled_finger_mesh_contact_geoms": int(contract.disabled_finger_mesh_contact_geoms),
            "modified_object_contact_geoms": int(contract.modified_object_contact_geoms),
            "object_hinge_frictionloss": float(contract.object_hinge_frictionloss),
            "object_init_pos": [float(v) for v in contract.object_init_pos],
            "object_scale": None if contract.object_scale is None else float(contract.object_scale),
            "object_axis_pos": float(contract.object_axis_pos),
            "object_axis_vel": float(contract.object_axis_vel),
            "object_contact0_mesh": contract.object_contact0_mesh,
            "object_contact1_mesh": contract.object_contact1_mesh,
            "active_pair_profile": contract.active_pair_profile,
            "active_proxy_profile": contract.active_proxy_profile,
            "index_proxy_pos": [float(v) for v in contract.index_proxy_pos],
            "thumb_proxy_pos": [float(v) for v in contract.thumb_proxy_pos],
            "active_proxy_size": float(contract.active_proxy_size),
            "active_proxy_margin": float(contract.active_proxy_margin),
            "index_proxy_size": None if contract.index_proxy_size is None else float(contract.index_proxy_size),
            "thumb_proxy_size": None if contract.thumb_proxy_size is None else float(contract.thumb_proxy_size),
            "index_proxy_margin": None if contract.index_proxy_margin is None else float(contract.index_proxy_margin),
            "thumb_proxy_margin": None if contract.thumb_proxy_margin is None else float(contract.thumb_proxy_margin),
            "index_proxy_half_length": None
            if contract.index_proxy_half_length is None
            else float(contract.index_proxy_half_length),
            "thumb_proxy_half_length": None
            if contract.thumb_proxy_half_length is None
            else float(contract.thumb_proxy_half_length),
            "active_tactile_friction_override": None
            if contract.active_tactile_friction_override is None
            else [float(v) for v in contract.active_tactile_friction_override],
            "active_tactile_solref_override": None
            if contract.active_tactile_solref_override is None
            else [float(v) for v in contract.active_tactile_solref_override],
            "active_tactile_solimp_override": None
            if contract.active_tactile_solimp_override is None
            else [float(v) for v in contract.active_tactile_solimp_override],
            "hand_root_pos": [float(v) for v in contract.hand_root_pos],
            "hand_root_rpy": [float(v) for v in contract.hand_root_rpy],
            "hand_root_quat_wxyz": None
            if contract.hand_root_quat_wxyz is None
            else [float(v) for v in contract.hand_root_quat_wxyz],
            "reference_state_json": contract.reference_state_json,
            "reference_phase": contract.reference_phase,
            "init_q_clipped_count": int(contract.init_q_clipped_count),
            "csv": str(csv_path),
        }
        rollout_summary.update(prefixed_contact_summary(model, data))
        (out_dir / f"{args.mode}_summary.json").write_text(json.dumps(rollout_summary, indent=2), encoding="utf-8")
        return rollout_summary

    ncon_values = np.asarray([float(row.get("ncon", 0.0)) for row in rows], dtype=np.float64)
    active_counts = np.asarray([float(row.get("active_contact_count", 0.0)) for row in rows], dtype=np.float64)
    finger_counts = {
        finger: np.asarray([float(row.get(f"{finger}_object_contact_count", 0.0)) for row in rows], dtype=np.float64)
        for finger in FINGER_PREFIXES
    }
    finger_forces = {
        finger: np.asarray([float(row.get(f"{finger}_object_contact_force", 0.0)) for row in rows], dtype=np.float64)
        for finger in FINGER_PREFIXES
    }
    finger_tip_counts = {
        finger: np.asarray([float(row.get(f"{finger}_tip_contact_count", 0.0)) for row in rows], dtype=np.float64)
        for finger in FINGER_PREFIXES
    }
    finger_tip_forces = {
        finger: np.asarray([float(row.get(f"{finger}_tip_contact_force", 0.0)) for row in rows], dtype=np.float64)
        for finger in FINGER_PREFIXES
    }
    active_forces = np.asarray([float(row.get("active_contact_force", 0.0)) for row in rows], dtype=np.float64)
    active_tip_counts = np.asarray([float(row.get("active_tip_contact_count", 0.0)) for row in rows], dtype=np.float64)
    active_tip_forces = np.asarray([float(row.get("active_tip_contact_force", 0.0)) for row in rows], dtype=np.float64)
    object_contact_dists = np.asarray(
        [float(row.get("min_object_contact_dist", 0.0)) for row in rows if float(row.get("object_contact_count", 0.0)) > 0],
        dtype=np.float64,
    )
    active_contact_dists = np.asarray(
        [float(row.get("min_active_contact_dist", 0.0)) for row in rows if float(row.get("active_contact_count", 0.0)) > 0],
        dtype=np.float64,
    )
    index_contact_active = finger_counts["index"] > 0
    thumb_contact_active = finger_counts["thumb"] > 0
    index_thumb_overlap = np.logical_and(index_contact_active, thumb_contact_active)
    index_only_contact = np.logical_and(index_contact_active, np.logical_not(thumb_contact_active))
    thumb_only_contact = np.logical_and(thumb_contact_active, np.logical_not(index_contact_active))
    no_active_contact = np.logical_and(np.logical_not(index_contact_active), np.logical_not(thumb_contact_active))
    index_tip_contact_active = finger_tip_counts["index"] > 0
    thumb_tip_contact_active = finger_tip_counts["thumb"] > 0
    index_thumb_tip_overlap = np.logical_and(index_tip_contact_active, thumb_tip_contact_active)
    object_pos = np.asarray(
        [[float(row.get("object_x", 0.0)), float(row.get("object_y", 0.0)), float(row.get("object_z", 0.0))] for row in rows],
        dtype=np.float64,
    )
    object_geom_pos = np.asarray(
        [
            [
                float(row.get("object_geom_x", 0.0)),
                float(row.get("object_geom_y", 0.0)),
                float(row.get("object_geom_z", 0.0)),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    object_yaw = np.asarray([float(row.get("object_yaw", 0.0)) for row in rows], dtype=np.float64)
    object_axis_pos = np.asarray([float(row.get("object_axis_pos", 0.0)) for row in rows], dtype=np.float64)
    object_axis_vel = np.asarray([float(row.get("object_axis_vel", 0.0)) for row in rows], dtype=np.float64)
    object_angvel_z = np.asarray([float(row.get("object_angvel_z", 0.0)) for row in rows], dtype=np.float64)
    index_tip_dist = np.asarray(
        [float(row.get("index_tip_to_object_geom_dist", np.nan)) for row in rows], dtype=np.float64
    )
    thumb_tip_dist = np.asarray(
        [float(row.get("thumb_tip_to_object_geom_dist", np.nan)) for row in rows], dtype=np.float64
    )
    index_thumb_tip_dist = np.asarray([float(row.get("index_thumb_tip_dist", np.nan)) for row in rows], dtype=np.float64)
    object_drift = np.linalg.norm(object_pos - object_pos[0], axis=1) if len(object_pos) else np.zeros(0)
    object_geom_drift = (
        np.linalg.norm(object_geom_pos - object_geom_pos[0], axis=1) if len(object_geom_pos) else np.zeros(0)
    )

    rollout_summary = {
        "mode": args.mode,
        "policy_steps": len(rows),
        "requested_policy_steps": int(args.policy_steps),
        "physics_steps": len(rows) * contract.control_decimation,
        "elapsed_wall_s": time.time() - start,
        "max_abs_q": max_abs_q,
        "max_abs_tau": max_abs_tau,
        "max_abs_action": max_abs_action,
        "max_contract_violation": max_contract_violation,
        "hard_clamp_events": int(hard_clamp_events),
        "dt": float(contract.dt),
        "control_decimation": int(contract.control_decimation),
        "policy_rate_hz": float(1.0 / (contract.dt * contract.control_decimation)),
        "target_policy_hz": float(contract.target_policy_hz),
        "joint_limit_mode": args.joint_limit_mode,
        "hard_clamp_joints": bool(args.hard_clamp_joints),
        "finger_contact_mode": contract.finger_contact_mode,
        "object_contact_mode": contract.object_contact_mode,
        "object_friction_override": None
        if contract.object_friction_override is None
        else [float(v) for v in contract.object_friction_override],
        "object_solref_override": None
        if contract.object_solref_override is None
        else [float(v) for v in contract.object_solref_override],
        "object_solimp_override": None
        if contract.object_solimp_override is None
        else [float(v) for v in contract.object_solimp_override],
        "object_condim_override": None
        if contract.object_condim_override is None
        else int(contract.object_condim_override),
        "object_contact_pos_offset": [float(v) for v in contract.object_contact_pos_offset],
        "object_contact_z_offset": float(contract.object_contact_z_offset),
        "hinge_frictionloss_override": None
        if contract.hinge_frictionloss_override is None
        else float(contract.hinge_frictionloss_override),
        "reset_source": contract.reset_source,
        "disabled_finger_mesh_contact_geoms": int(contract.disabled_finger_mesh_contact_geoms),
        "modified_object_contact_geoms": int(contract.modified_object_contact_geoms),
        "object_hinge_frictionloss": float(contract.object_hinge_frictionloss),
        "object_init_pos": [float(v) for v in contract.object_init_pos],
        "object_scale": None if contract.object_scale is None else float(contract.object_scale),
        "object_axis_pos": float(contract.object_axis_pos),
        "object_axis_vel": float(contract.object_axis_vel),
        "object_contact0_mesh": contract.object_contact0_mesh,
        "object_contact1_mesh": contract.object_contact1_mesh,
        "active_pair_profile": contract.active_pair_profile,
        "active_proxy_profile": contract.active_proxy_profile,
        "index_proxy_pos": [float(v) for v in contract.index_proxy_pos],
        "thumb_proxy_pos": [float(v) for v in contract.thumb_proxy_pos],
        "active_proxy_size": float(contract.active_proxy_size),
        "active_proxy_margin": float(contract.active_proxy_margin),
        "index_proxy_size": None if contract.index_proxy_size is None else float(contract.index_proxy_size),
        "thumb_proxy_size": None if contract.thumb_proxy_size is None else float(contract.thumb_proxy_size),
        "index_proxy_margin": None if contract.index_proxy_margin is None else float(contract.index_proxy_margin),
        "thumb_proxy_margin": None if contract.thumb_proxy_margin is None else float(contract.thumb_proxy_margin),
        "index_proxy_half_length": None
        if contract.index_proxy_half_length is None
        else float(contract.index_proxy_half_length),
        "thumb_proxy_half_length": None
        if contract.thumb_proxy_half_length is None
        else float(contract.thumb_proxy_half_length),
        "active_tactile_friction_override": None
        if contract.active_tactile_friction_override is None
        else [float(v) for v in contract.active_tactile_friction_override],
        "active_tactile_solref_override": None
        if contract.active_tactile_solref_override is None
        else [float(v) for v in contract.active_tactile_solref_override],
        "active_tactile_solimp_override": None
        if contract.active_tactile_solimp_override is None
        else [float(v) for v in contract.active_tactile_solimp_override],
        "reference_state_json": contract.reference_state_json,
        "reference_phase": contract.reference_phase,
        "reset_object_contact_count": summary["reset_contact_summary"]["object_contact_count"],
        "reset_active_contact_count": summary["reset_contact_summary"]["active_contact_count"],
        "reset_index_contact_count": summary["reset_contact_summary"]["index_object_contact_count"],
        "reset_thumb_contact_count": summary["reset_contact_summary"]["thumb_object_contact_count"],
        "reset_object_contact_pairs": summary["reset_contact_summary"]["object_contact_pairs"],
        "object_final_pos": [float(v) for v in object_pos[-1]],
        "object_final_geom_pos": [float(v) for v in object_geom_pos[-1]],
        "object_final_z": float(object_pos[-1, 2]),
        "object_final_geom_z": float(object_geom_pos[-1, 2]),
        "object_final_drift": float(object_drift[-1]),
        "object_max_drift": float(np.max(object_drift)),
        "object_final_geom_drift": float(object_geom_drift[-1]),
        "object_max_geom_drift": float(np.max(object_geom_drift)),
        "object_yaw_delta": float(np.unwrap(object_yaw)[-1] - np.unwrap(object_yaw)[0]),
        "object_axis_delta": float(object_axis_pos[-1] - object_axis_pos[0]),
        "object_mean_axis_vel": float(np.mean(object_axis_vel)),
        "object_max_abs_axis_vel": float(np.max(np.abs(object_axis_vel))),
        "object_positive_axis_vel_fraction": float(np.mean(object_axis_vel > 0.0)),
        "object_mean_angvel_z": float(np.mean(object_angvel_z)),
        "object_max_abs_angvel_z": float(np.max(np.abs(object_angvel_z))),
        "mean_index_tip_to_object_geom_dist": float(np.nanmean(index_tip_dist)),
        "min_index_tip_to_object_geom_dist": float(np.nanmin(index_tip_dist)),
        "mean_thumb_tip_to_object_geom_dist": float(np.nanmean(thumb_tip_dist)),
        "min_thumb_tip_to_object_geom_dist": float(np.nanmin(thumb_tip_dist)),
        "mean_index_thumb_tip_dist": float(np.nanmean(index_thumb_tip_dist)),
        "min_index_thumb_tip_dist": float(np.nanmin(index_thumb_tip_dist)),
        "max_ncon": int(np.max(ncon_values)),
        "mean_ncon": float(np.mean(ncon_values)),
        "active_contact_fraction": float(np.mean(active_counts > 0)),
        "mean_active_contact_count": float(np.mean(active_counts)),
        "max_active_contact_count": int(np.max(active_counts)),
        "mean_active_contact_force": float(np.mean(active_forces)),
        "max_active_contact_force": float(np.max(active_forces)),
        "min_object_contact_dist": float(np.min(object_contact_dists)) if object_contact_dists.size else 0.0,
        "min_active_contact_dist": float(np.min(active_contact_dists)) if active_contact_dists.size else 0.0,
        "active_tip_contact_fraction": float(np.mean(active_tip_counts > 0)),
        "mean_active_tip_contact_count": float(np.mean(active_tip_counts)),
        "max_active_tip_contact_count": int(np.max(active_tip_counts)),
        "mean_active_tip_contact_force": float(np.mean(active_tip_forces)),
        "max_active_tip_contact_force": float(np.max(active_tip_forces)),
        "index_thumb_overlap_fraction": float(np.mean(index_thumb_overlap)),
        "index_only_contact_fraction": float(np.mean(index_only_contact)),
        "thumb_only_contact_fraction": float(np.mean(thumb_only_contact)),
        "no_active_contact_fraction": float(np.mean(no_active_contact)),
        "mean_index_thumb_min_contact_count": float(np.mean(np.minimum(finger_counts["index"], finger_counts["thumb"]))),
        "index_thumb_tip_overlap_fraction": float(np.mean(index_thumb_tip_overlap)),
        "mean_index_thumb_tip_min_contact_count": float(
            np.mean(np.minimum(finger_tip_counts["index"], finger_tip_counts["thumb"]))
        ),
        "hand_root_pos": [float(v) for v in contract.hand_root_pos],
        "hand_root_rpy": [float(v) for v in contract.hand_root_rpy],
        "hand_root_quat_wxyz": None
        if contract.hand_root_quat_wxyz is None
        else [float(v) for v in contract.hand_root_quat_wxyz],
        "base_obj_scale": float(contract.base_obj_scale),
        "object_scale": None if contract.object_scale is None else float(contract.object_scale),
        "hand_root_pos_z_scale_comp": float(contract.hand_root_pos_z_scale_comp),
        "init_q_clipped_count": int(contract.init_q_clipped_count),
        "finger_contact_mode": contract.finger_contact_mode,
        "reset_source": contract.reset_source,
        "disabled_finger_mesh_contact_geoms": int(contract.disabled_finger_mesh_contact_geoms),
        "csv": str(csv_path),
    }
    rollout_summary.update(contact_pair_statistics(rows))
    if video_path is not None:
        rollout_summary["video"] = str(video_path)
    for finger in FINGER_PREFIXES:
        rollout_summary[f"{finger}_contact_fraction"] = float(np.mean(finger_counts[finger] > 0))
        rollout_summary[f"mean_{finger}_contact_count"] = float(np.mean(finger_counts[finger]))
        rollout_summary[f"max_{finger}_contact_count"] = int(np.max(finger_counts[finger]))
        rollout_summary[f"mean_{finger}_contact_force"] = float(np.mean(finger_forces[finger]))
        rollout_summary[f"max_{finger}_contact_force"] = float(np.max(finger_forces[finger]))
        rollout_summary[f"{finger}_tip_contact_fraction"] = float(np.mean(finger_tip_counts[finger] > 0))
        rollout_summary[f"mean_{finger}_tip_contact_count"] = float(np.mean(finger_tip_counts[finger]))
        rollout_summary[f"max_{finger}_tip_contact_count"] = int(np.max(finger_tip_counts[finger]))
        rollout_summary[f"mean_{finger}_tip_contact_force"] = float(np.mean(finger_tip_forces[finger]))
        rollout_summary[f"max_{finger}_tip_contact_force"] = float(np.max(finger_tip_forces[finger]))
    (out_dir / f"{args.mode}_summary.json").write_text(json.dumps(rollout_summary, indent=2), encoding="utf-8")
    return rollout_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["audit", "zero", "poke", "policy", "reference_action"],
        default="audit",
        help="audit prints mappings; zero/poke/policy/reference_action run a short rollout.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=REPO_ROOT / "sim2sim/mujoco/scene_dexh13_lightbulb.xml",
        help="MuJoCo XML scene.",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=REPO_ROOT / "sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml",
        help="Frozen CoDrive task YAML for control/reset contract.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "sim2real/codrive/model_best_codrive.ckpt",
        help="Frozen PAdapt checkpoint for --mode policy.",
    )
    parser.add_argument(
        "--init-pose-file",
        type=Path,
        help="Optional YAML with handRootPos, handRootRPY, and handInitPose overrides.",
    )
    parser.add_argument(
        "--reference-state-json",
        type=Path,
        help="Optional IsaacGym sim2sim reference JSON. Overrides q/root/target/action when present.",
    )
    parser.add_argument("--policy-steps", type=int, default=20, help="Number of policy steps. Use 0 with --viewer to run until the viewer closes or Ctrl+C.")
    parser.add_argument("--dt", type=float, help="Override MuJoCo timestep.")
    parser.add_argument("--control-decimation", type=int, help="Override low-level steps per policy action.")
    parser.add_argument(
        "--policy-hz",
        type=float,
        help="Target policy inference rate. If --control-decimation is omitted, it is derived from --dt.",
    )
    parser.add_argument("--object-pos", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Override object freejoint position.")
    parser.add_argument("--object-scale", type=float, help="Override CoDrive object mesh/contact scale for generated scene replay.")
    parser.add_argument("--object-contact0-mesh", type=Path, help="Override codrive_lightbulb_contact0 mesh file in generated scene.")
    parser.add_argument("--object-contact1-mesh", type=Path, help="Override codrive_lightbulb_contact1 mesh file in generated scene.")
    parser.add_argument(
        "--active-pair-profile",
        choices=sorted(ACTIVE_CONTACT0_PAIR_PROFILES),
        default="none",
        help="Generate explicit MuJoCo contact pairs for active tactile/contact0 diagnostics.",
    )
    parser.add_argument(
        "--active-proxy-profile",
        choices=sorted(ACTIVE_PROXY_PROFILES),
        default="none",
        help="Generate local active distal contact proxy geoms in a temporary scene.",
    )
    parser.add_argument(
        "--index-proxy-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Local position for the generated index distal proxy geom.",
    )
    parser.add_argument(
        "--thumb-proxy-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Local position for the generated thumb distal proxy geom.",
    )
    parser.add_argument(
        "--active-proxy-size",
        type=float,
        help="Sphere radius for generated active distal proxy geoms.",
    )
    parser.add_argument(
        "--active-proxy-margin",
        type=float,
        help="MuJoCo margin for generated active distal proxy geoms.",
    )
    parser.add_argument(
        "--index-proxy-size",
        type=float,
        help="Override radius for the generated index distal proxy geom.",
    )
    parser.add_argument(
        "--thumb-proxy-size",
        type=float,
        help="Override radius for the generated thumb distal proxy geom.",
    )
    parser.add_argument(
        "--index-proxy-margin",
        type=float,
        help="Override MuJoCo margin for the generated index distal proxy geom.",
    )
    parser.add_argument(
        "--thumb-proxy-margin",
        type=float,
        help="Override MuJoCo margin for the generated thumb distal proxy geom.",
    )
    parser.add_argument(
        "--index-proxy-half-length",
        type=float,
        help="Override half length for generated index tactile capsule proxy geoms.",
    )
    parser.add_argument(
        "--thumb-proxy-half-length",
        type=float,
        help="Override half length for generated thumb tactile capsule proxy geoms.",
    )
    parser.add_argument(
        "--active-tactile-friction",
        type=float,
        nargs=3,
        metavar=("SLIDE", "TORSION", "ROLL"),
        help="Override friction for active index/thumb tactile and tip geoms.",
    )
    parser.add_argument(
        "--active-tactile-solref",
        type=float,
        nargs=2,
        metavar=("TIMECONST", "DAMPING"),
        help="Override solref for active index/thumb tactile and tip geoms.",
    )
    parser.add_argument(
        "--active-tactile-solimp",
        type=float,
        nargs=3,
        metavar=("MIN", "MAX", "WIDTH"),
        help="Override first three solimp values for active index/thumb tactile and tip geoms.",
    )
    parser.add_argument(
        "--active-tactile-margin",
        type=float,
        help="Override MuJoCo geom margin for active index/thumb tactile and tip geoms.",
    )
    parser.add_argument(
        "--active-tip-margin",
        type=float,
        help="Override MuJoCo geom margin for active fingertip sphere geoms only.",
    )
    parser.add_argument("--hand-root-pos", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Override mocap hand root position.")
    parser.add_argument("--hand-root-rpy", type=float, nargs=3, metavar=("R", "P", "Y"), help="Override mocap hand root RPY.")
    parser.add_argument(
        "--reset-source",
        choices=["yaml", "isaacgym_screwdriver"],
        default="yaml",
        help="Reset source for init q/root pose before explicit CLI overrides. yaml preserves task YAML values; isaacgym_screwdriver mirrors xhand_pasini.py screwdriver_inclined reset constants.",
    )
    parser.add_argument(
        "--joint-limit-mode",
        choices=["task", "model"],
        default="task",
        help="task overwrites MuJoCo joint ranges from frozen YAML; model keeps MJCF ranges and clips targets to them.",
    )
    parser.add_argument(
        "--finger-contact-mode",
        choices=[
            "full",
            "active_pad_proxy",
            "tip_proxy",
            "no_index_tactile2",
            "index_tip_thumb_pad_proxy",
            "active_distal_proxy",
            "active_proxy_only",
        ],
        default="full",
        help="full keeps mesh contacts; active_pad_proxy keeps active tactile pads/tips; tip_proxy keeps active fingertip spheres only; no_index_tactile2 disables the reset-biased index tactile mesh; index_tip_thumb_pad_proxy keeps index tip plus thumb pad/tip contacts; active_distal_proxy keeps active tips plus generated distal proxy geoms; active_proxy_only keeps only generated active distal proxy geoms.",
    )
    parser.add_argument(
        "--object-contact-mode",
        choices=["default", "low_friction_debug", "high_hinge_friction_debug", "low_friction_high_hinge_debug"],
        default="default",
        help="Diagnostic contact changes for checking single-finger friction shortcuts.",
    )
    parser.add_argument(
        "--object-friction",
        type=float,
        nargs=3,
        metavar=("SLIDE", "TORSION", "ROLL"),
        help="Override friction for codrive_lightbulb_contact* geoms after object contact mode is applied.",
    )
    parser.add_argument(
        "--object-solref",
        type=float,
        nargs=2,
        metavar=("TIMECONST", "DAMPING"),
        help="Override solref for codrive_lightbulb_contact* geoms after object contact mode is applied.",
    )
    parser.add_argument(
        "--object-solimp",
        type=float,
        nargs=3,
        metavar=("MIN", "MAX", "WIDTH"),
        help="Override the first three solimp values for codrive_lightbulb_contact* geoms.",
    )
    parser.add_argument(
        "--object-condim",
        type=int,
        choices=[1, 3, 4, 6],
        help="Override condim for codrive_lightbulb_contact* geoms.",
    )
    parser.add_argument(
        "--object-margin",
        type=float,
        help="Override MuJoCo geom margin for codrive_lightbulb_contact* geoms.",
    )
    parser.add_argument(
        "--object-gap",
        type=float,
        help="Override MuJoCo geom gap for codrive_lightbulb_contact* geoms.",
    )
    parser.add_argument(
        "--object-contact-pos-offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Additive local position offset for codrive_lightbulb_contact* geoms after scene load.",
    )
    parser.add_argument(
        "--object-contact-z-offset",
        type=float,
        help="Additive local z offset for codrive_lightbulb_contact* geoms after scene load.",
    )
    parser.add_argument(
        "--hinge-frictionloss",
        type=float,
        help="Override codrive_lightbulb_hinge frictionloss after object contact mode is applied.",
    )
    parser.add_argument("--no-clamp-init-q", action="store_true", help="Do not clip reset qpos to selected joint limits.")
    parser.add_argument("--hard-clamp-joints", action="store_true", help="Diagnostic only: clamp qpos/qvel to joint limits after each step.")
    parser.add_argument("--poke-joint", type=int, default=0)
    parser.add_argument("--poke-action", type=float, default=0.5)
    parser.add_argument("--poke-start", type=int, default=2)
    parser.add_argument("--render-video", action="store_true", help="Write an MP4 rollout video through MuJoCo offscreen rendering.")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--render-fps", type=float, default=20.0)
    parser.add_argument("--render-every", type=int, default=1, help="Render every N policy steps.")
    parser.add_argument("--viewer", action="store_true", help="Open a live MuJoCo viewer while the rollout runs.")
    parser.add_argument("--viewer-sync-every", type=int, default=10, help="Sync the live viewer every N physics steps.")
    parser.add_argument(
        "--viewer-realtime-factor",
        type=float,
        default=1.0,
        help="Viewer playback speed. 1.0 is realtime; 0 disables sleeping and runs as fast as possible.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/sim2sim_mujoco_mvp",
        help="Directory for JSON/CSV traces.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.scene = args.scene.expanduser().resolve()
    args.task_config = args.task_config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.init_pose_file is not None:
        args.init_pose_file = args.init_pose_file.expanduser().resolve()
    if args.reference_state_json is not None:
        args.reference_state_json = args.reference_state_json.expanduser().resolve()
    if args.object_contact0_mesh is not None:
        args.object_contact0_mesh = args.object_contact0_mesh.expanduser().resolve()
    if args.object_contact1_mesh is not None:
        args.object_contact1_mesh = args.object_contact1_mesh.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    contract = apply_cli_overrides(
        apply_reference_state_json(
            apply_reset_source(apply_init_pose_file(load_task_contract(args.task_config), args.init_pose_file), args.reset_source),
            args.reference_state_json,
        ),
        args,
    )

    if args.mode == "audit":
        out_dir = ensure_output_dir(args.output_dir)
        scene_path = prepare_scene_xml(args.scene, contract, out_dir)
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        model.opt.timestep = contract.dt
        contract.disabled_finger_mesh_contact_geoms = apply_finger_contact_mode(model, contract.finger_contact_mode)
        contract.modified_object_contact_geoms, contract.object_hinge_frictionloss = apply_object_contact_mode(
            model, contract.object_contact_mode
        )
        apply_contact_overrides(model, contract)
        configure_joint_limits(model, contract, args.joint_limit_mode)
        if not args.no_clamp_init_q:
            clamp_init_q_to_limits(contract)
        qpos_adrs, dof_adrs = joint_mapping(model)
        data = mujoco.MjData(model)
        apply_reset(model, data, contract, qpos_adrs, dof_adrs)
        summary = summarize_model(model, qpos_adrs, dof_adrs, contract)
        summary["reset_contact_summary"] = contact_summary(model, data)
        (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    summary = rollout(args, contract)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
