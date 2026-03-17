#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_pure_physics_trig_fullgrid.py
==================================

High-capacity pure-physics PI-DeepONet training for FULL-GRID sensors.
Optimized for strong CUDA GPUs such as H200 SXM.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")

from pure_physics_trig_fullgrid_model import PIDeepONetTrigFullGrid


def get_best_device(explicit_device: str | None = None) -> str:
    if explicit_device:
        return explicit_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_acceleration(args, device: str) -> tuple[str, object, bool]:
    """
    Returns
    -------
    amp_dtype_name: str
    scaler: GradScaler-like or None
    compile_enabled: bool
    """
    amp_dtype_name = args.amp_dtype
    if amp_dtype_name == "auto":
        amp_dtype_name = "bf16" if device == "cuda" else "none"

    if device == "cuda":
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        if args.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

    scaler = None
    if device == "cuda" and amp_dtype_name == "fp16":
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    compile_enabled = False
    return amp_dtype_name, scaler, compile_enabled


def maybe_compile_model(model: torch.nn.Module, args, device: str) -> torch.nn.Module:
    if not args.compile:
        return model
    if device != "cuda":
        print("[compile] skipped: only enabled for CUDA in this script")
        return model
    if not hasattr(torch, "compile"):
        print("[compile] skipped: torch.compile not available in this PyTorch build")
        return model
    try:
        model = torch.compile(
            model,
            mode=args.compile_mode,
            fullgraph=False,
            dynamic=False,
        )
        print(f"[compile] enabled: mode={args.compile_mode}")
    except Exception as e:
        print(f"[compile] failed, falling back to eager mode: {e}")
    return model


def get_autocast_context(device: str, amp_dtype_name: str):
    if device != "cuda" or amp_dtype_name == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


class WaveDatasetFullGrid:
    def __init__(self, data_path: str, device: str = "cpu") -> None:
        self.device = device
        data = np.load(data_path)

        self.x1_coords = data["x1_coords"].astype(np.float32)
        self.x2_coords = data["x2_coords"].astype(np.float32)
        self.Nx = int(data["Nx"])
        self.Ny = int(data["Ny"])
        self.N_dof = self.Nx * self.Ny

        self.u0_all = torch.tensor(data["u0"], dtype=torch.float32, device=device)
        self.v0_all = torch.tensor(data["v0"], dtype=torch.float32, device=device)
        self.params_all = torch.tensor(data["params"], dtype=torch.float32, device=device)

        self.n_samples = self.u0_all.shape[0]
        self.sensor_indices = np.arange(self.N_dof, dtype=np.int64)
        self.sensor_x1 = self.x1_coords.copy()
        self.sensor_x2 = self.x2_coords.copy()
        self.n_sensors = self.N_dof

        self.sensor_x1_t = torch.tensor(self.sensor_x1, dtype=torch.float32, device=device)
        self.sensor_x2_t = torch.tensor(self.sensor_x2, dtype=torch.float32, device=device)

        print(f"\n{'='*72}")
        print("FULL-GRID TRIG PURE-PHYSICS DATASET")
        print(f"{'='*72}")
        print(f"  Samples: {self.n_samples}")
        print(f"  Grid: {self.Nx} x {self.Ny} = {self.N_dof} DOFs")
        print(f"  Sensors: FULL GRID = {self.n_sensors}")
        print(f"{'='*72}\n")

    def get_branch_input(self, sample_idx: int) -> torch.Tensor:
        u0 = self.u0_all[sample_idx]
        v0 = self.v0_all[sample_idx]
        params = self.params_all[sample_idx]
        return torch.cat([u0, v0, params], dim=0)

    def get_branch_batch_from_indices(self, sample_idx: np.ndarray) -> torch.Tensor:
        idx_t = torch.as_tensor(sample_idx, dtype=torch.long, device=self.device)
        u0 = self.u0_all[idx_t]
        v0 = self.v0_all[idx_t]
        params = self.params_all[idx_t]
        return torch.cat([u0, v0, params], dim=1)

    def sample_ic_batch(self, batch_size: int, rng=None) -> Tuple[torch.Tensor, ...]:
        if rng is None:
            rng = np.random.default_rng()

        sample_idx = rng.integers(0, self.n_samples, size=batch_size)
        spatial_idx = rng.integers(0, self.N_dof, size=batch_size)

        idx_s = torch.as_tensor(sample_idx, dtype=torch.long, device=self.device)
        idx_x = torch.as_tensor(spatial_idx, dtype=torch.long, device=self.device)

        branch = self.get_branch_batch_from_indices(sample_idx)
        trunk = torch.stack([
            torch.zeros(batch_size, device=self.device, dtype=torch.float32),
            torch.tensor(self.x1_coords[spatial_idx], device=self.device, dtype=torch.float32),
            torch.tensor(self.x2_coords[spatial_idx], device=self.device, dtype=torch.float32),
        ], dim=1)
        u0 = self.u0_all[idx_s, idx_x]
        v0 = self.v0_all[idx_s, idx_x]
        return branch, trunk, u0, v0

    def sample_pde_batch(
        self,
        batch_size: int,
        T: float,
        *,
        late_fraction: float = 0.7,
        rng=None,
    ) -> Tuple[torch.Tensor, ...]:
        if rng is None:
            rng = np.random.default_rng()

        sample_idx = rng.integers(0, self.n_samples, size=batch_size)
        params_idx = torch.as_tensor(sample_idx, dtype=torch.long, device=self.device)
        branch = self.get_branch_batch_from_indices(sample_idx)
        params_batch = self.params_all[params_idx]

        n_late = int(batch_size * late_fraction)
        t = np.empty(batch_size, dtype=np.float32)
        if n_late > 0:
            t[:n_late] = rng.uniform(0.5 * T, T, size=n_late).astype(np.float32)
        if batch_size - n_late > 0:
            t[n_late:] = rng.uniform(0.0, 0.5 * T, size=batch_size - n_late).astype(np.float32)
        rng.shuffle(t)

        x1 = rng.uniform(0.0, 1.0, size=batch_size).astype(np.float32)
        x2 = rng.uniform(0.0, 1.0, size=batch_size).astype(np.float32)

        trunk = torch.stack([
            torch.tensor(t, device=self.device),
            torch.tensor(x1, device=self.device),
            torch.tensor(x2, device=self.device),
        ], dim=1)
        return branch, trunk, params_batch


def build_branch_from_sensor_state(u_sensors: torch.Tensor, v_sensors: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    return torch.cat([u_sensors, v_sensors, params], dim=1)


def predict_state_at_sensor_time(
    model: torch.nn.Module,
    branch_input: torch.Tensor,
    sensor_x1: torch.Tensor,
    sensor_x2: torch.Tensor,
    t_vals: torch.Tensor,
    *,
    sensor_chunk_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized sensor prediction with chunking over sensor dimension.

    Returns
    -------
    u_sensors: [B, Ns]
    v_sensors: [B, Ns]
    """
    B = branch_input.shape[0]
    Ns = sensor_x1.shape[0]

    u_chunks = []
    v_chunks = []

    for start in range(0, Ns, sensor_chunk_size):
        end = min(start + sensor_chunk_size, Ns)
        chunk_n = end - start

        x1 = sensor_x1[start:end]
        x2 = sensor_x2[start:end]

        t_grid = t_vals[:, None].expand(B, chunk_n)
        x1_grid = x1[None, :].expand(B, chunk_n)
        x2_grid = x2[None, :].expand(B, chunk_n)

        trunk = torch.stack([t_grid, x1_grid, x2_grid], dim=-1).reshape(B * chunk_n, 3)
        trunk = trunk.requires_grad_(True)

        br = branch_input[:, None, :].expand(B, chunk_n, branch_input.shape[1]).reshape(B * chunk_n, -1)
        derivs = model.forward_with_grad(br, trunk)

        u = derivs["u"].reshape(B, chunk_n)
        v = derivs["u_t"].reshape(B, chunk_n)

        u_chunks.append(u)
        v_chunks.append(v)

    return torch.cat(u_chunks, dim=1), torch.cat(v_chunks, dim=1)


def compute_loss_semigroup(
    model,
    dataset,
    batch_size,
    T,
    *,
    dt_min=0.2,
    dt_max=2.0,
    sensor_chunk_size=1024,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    sample_idx = rng.integers(0, dataset.n_samples, size=batch_size)
    branch0 = dataset.get_branch_batch_from_indices(sample_idx)
    params_batch = dataset.params_all[torch.as_tensor(sample_idx, dtype=torch.long, device=dataset.device)]

    dt_vals = torch.tensor(
        rng.uniform(dt_min, dt_max, size=batch_size).astype(np.float32),
        device=dataset.device,
    )
    t1_np = np.array([rng.uniform(0.0, max(T - float(dt), 1e-6)) for dt in dt_vals.detach().cpu().numpy()], dtype=np.float32)
    t1_vals = torch.tensor(t1_np, device=dataset.device)
    t2_vals = t1_vals + dt_vals

    u_t1, v_t1 = predict_state_at_sensor_time(
        model,
        branch0,
        dataset.sensor_x1_t,
        dataset.sensor_x2_t,
        t1_vals,
        sensor_chunk_size=sensor_chunk_size,
    )
    branch_step = build_branch_from_sensor_state(u_t1, v_t1, params_batch)

    u_direct, v_direct = predict_state_at_sensor_time(
        model,
        branch0,
        dataset.sensor_x1_t,
        dataset.sensor_x2_t,
        t2_vals,
        sensor_chunk_size=sensor_chunk_size,
    )
    u_step, v_step = predict_state_at_sensor_time(
        model,
        branch_step,
        dataset.sensor_x1_t,
        dataset.sensor_x2_t,
        dt_vals,
        sensor_chunk_size=sensor_chunk_size,
    )

    loss_u = F.mse_loss(u_step, u_direct)
    loss_v = F.mse_loss(v_step, v_direct)
    return loss_u + loss_v, loss_u, loss_v


def compute_loss_amplitude(
    model,
    dataset,
    batch_size,
    T,
    *,
    late_start_frac=0.70,
    l1_weight_power=2.0,
    tau_u=1e-4,
    tau_v=1e-4,
    tau_weight=5.0,
    velocity_weight=0.25,
    dt_min=0.2,
    dt_max=2.0,
    sensor_chunk_size=1024,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    sample_idx = rng.integers(0, dataset.n_samples, size=batch_size)
    branch_batch = dataset.get_branch_batch_from_indices(sample_idx)

    late_t0 = max(late_start_frac * T, 0.0)
    if late_t0 >= T:
        late_t0 = max(0.0, T - dt_max - 1e-6)

    dt_high = min(dt_max, max(T - late_t0, dt_min + 1e-6))
    dt_vals = torch.tensor(
        rng.uniform(dt_min, dt_high, size=batch_size).astype(np.float32),
        device=dataset.device,
    )

    t_now_np = np.empty(batch_size, dtype=np.float32)
    dt_cpu = dt_vals.detach().cpu().numpy()
    for i, dt in enumerate(dt_cpu):
        upper = max(T - float(dt), late_t0 + 1e-6)
        lower = min(late_t0, upper - 1e-6)
        t_now_np[i] = rng.uniform(lower, upper)

    t_now_vals = torch.tensor(t_now_np, device=dataset.device)
    t_next_vals = t_now_vals + dt_vals

    u_now, v_now = predict_state_at_sensor_time(
        model,
        branch_batch,
        dataset.sensor_x1_t,
        dataset.sensor_x2_t,
        t_now_vals,
        sensor_chunk_size=sensor_chunk_size,
    )
    u_next, v_next = predict_state_at_sensor_time(
        model,
        branch_batch,
        dataset.sensor_x1_t,
        dataset.sensor_x2_t,
        t_next_vals,
        sensor_chunk_size=sensor_chunk_size,
    )

    w_next = (t_next_vals / max(T, 1e-6)).pow(l1_weight_power).unsqueeze(1)

    late_l1_u = (w_next * torch.abs(u_next)).mean()
    late_l1_v = (w_next * torch.abs(v_next)).mean()

    tau_pen_u = (w_next * F.relu(torch.abs(u_next) - tau_u)).mean()
    tau_pen_v = (w_next * F.relu(torch.abs(v_next) - tau_v)).mean()

    rms_now = torch.sqrt(torch.mean(u_now ** 2, dim=1) + velocity_weight * torch.mean(v_now ** 2, dim=1) + 1e-12)
    rms_next = torch.sqrt(torch.mean(u_next ** 2, dim=1) + velocity_weight * torch.mean(v_next ** 2, dim=1) + 1e-12)
    mono_pen = F.relu(rms_next - rms_now).mean()

    loss_amp = (
        late_l1_u
        + velocity_weight * late_l1_v
        + tau_weight * (tau_pen_u + velocity_weight * tau_pen_v)
        + mono_pen
    )

    stats = {
        "rms_now": rms_now.mean().detach(),
        "rms_next": rms_next.mean().detach(),
        "late_l1_u": late_l1_u.detach(),
        "late_l1_v": late_l1_v.detach(),
        "tau_pen_u": tau_pen_u.detach(),
        "tau_pen_v": tau_pen_v.detach(),
        "mono_pen": mono_pen.detach(),
        "t_now_mean": t_now_vals.mean().detach(),
        "t_next_mean": t_next_vals.mean().detach(),
    }
    return loss_amp, stats


def compute_loss_pde(model, branch_input, trunk_input, params_batch):
    derivs = model.forward_with_grad(branch_input, trunk_input)

    u = derivs["u"]
    u_t = derivs["u_t"]
    u_tt = derivs["u_tt"]
    lap_u = derivs["laplacian_u"]
    lap_ut = derivs["laplacian_ut"]
    t_vals = trunk_input[:, 0]

    gamma0 = params_batch[:, 0]
    a_gamma = params_batch[:, 1]
    beta0 = params_batch[:, 2]
    a_beta = params_batch[:, 3]
    alpha = params_batch[:, 4]

    gamma_t = gamma0 * (1.0 + a_gamma * torch.sin(t_vals))
    beta_t = beta0 * (1.0 + a_beta * torch.sin(t_vals))

    residual = u_tt - lap_u - gamma_t * lap_ut + beta_t * u_t - alpha * u + u ** 3
    return torch.mean(residual ** 2), residual


def compute_loss_ic(model, branch_input, trunk_input, u0_target, v0_target):
    derivs = model.forward_with_grad(branch_input, trunk_input)
    u_pred = derivs["u"]
    v_pred = derivs["u_t"]
    loss_u0 = F.mse_loss(u_pred, u0_target)
    loss_v0 = F.mse_loss(v_pred, v0_target)
    return loss_u0 + loss_v0, loss_u0, loss_v0


def build_optimizer(model: torch.nn.Module, config: dict, device: str):
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.0)
    fused = bool(config.get("fused_optimizer", False) and device == "cuda")

    if config.get("optimizer", "adamw").lower() == "adam":
        if fused:
            try:
                return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
            except TypeError:
                pass
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if fused:
        try:
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
        except TypeError:
            pass
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train(model, dataset, config):
    device = dataset.device
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    save_best = out_dir / config.get("save_name", "best_pure_physics_trig_fullgrid.pt")
    save_last = out_dir / "last_pure_physics_trig_fullgrid.pt"

    optimizer = build_optimizer(model, config, device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["n_epochs"],
        eta_min=config.get("lr_min", 1e-6),
    )

    amp_dtype_name = config["amp_dtype"]
    scaler = None
    if device == "cuda" and amp_dtype_name == "fp16":
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    rng = np.random.default_rng(config.get("seed", 42))
    history = {
        "epoch": [],
        "loss_total": [],
        "loss_pde": [],
        "loss_ic": [],
        "loss_u0": [],
        "loss_v0": [],
        "loss_sg": [],
        "loss_sg_u": [],
        "loss_sg_v": [],
        "loss_amp": [],
        "loss_amp_l1_u": [],
        "loss_amp_l1_v": [],
        "loss_amp_tau_u": [],
        "loss_amp_tau_v": [],
        "loss_amp_mono": [],
        "lr": [],
        "T_curr": [],
    }

    best_loss = float("inf")
    start_epoch = 1

    n_epochs = config["n_epochs"]
    stage = config.get("stage", "phase1")
    if stage == "phase1":
        if config["w_sg"] == 0.0:
            config["w_sg"] = 0.0
        if config["w_amp"] == 0.0:
            config["w_amp"] = 50.0
    elif stage == "phase2":
        if config["w_sg"] == 0.0:
            config["w_sg"] = 0.01
        if config["w_amp"] == 0.0:
            config["w_amp"] = 30.0
    elif stage == "phase3":
        if config["w_sg"] == 0.0:
            config["w_sg"] = 0.02
        if config["w_amp"] == 0.0:
            config["w_amp"] = 20.0

    w_pde = config["w_pde"]
    w_ic = config["w_ic"]
    w_sg = config["w_sg"]
    w_amp = config["w_amp"]
    T_final = config["T"]

    if config.get("resume"):
        resume_path = Path(config["resume"])
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
                if "optimizer_state" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                best_loss = ckpt.get("best_loss", best_loss)
                start_epoch = ckpt.get("epoch", 0) + 1
            else:
                model.load_state_dict(ckpt)
            print(f"Resumed from epoch {start_epoch}")

    if config.get("curriculum", False):
        T_start = config.get("resume_T_start")
        if T_start is None:
            T_start = config.get("T_start", 5.0)
        T_schedule = np.linspace(T_start, T_final, n_epochs)
    else:
        T_schedule = np.full(n_epochs, T_final)

    print(f"\n{'='*88}")
    print("PURE PHYSICS TRAINING WITH TRIG MODEL — FULL GRID")
    print(f"{'='*88}")
    print(f"  Epochs: {n_epochs}")
    print(f"  T_final: {T_final}")
    print(f"  Weights: PDE={w_pde}, IC={w_ic}, SG={w_sg}, AMP={w_amp}")
    print(f"  Curriculum: {config.get('curriculum', False)}")
    print(f"  Sensor chunk size: {config['sensor_chunk_size']}")
    print(f"  Optimizer: {config.get('optimizer', 'adamw')} | fused={config.get('fused_optimizer', False)}")
    print(f"  AMP: {amp_dtype_name}")
    print(f"  TF32: {config.get('allow_tf32', False)}")
    print(f"  Compile: {config.get('compile', False)}")
    if config.get("curriculum", False):
        print(f"  T_start: {T_schedule[0]:.2f} -> T_end: {T_schedule[-1]:.2f}")
    print(f"{'='*88}\n")

    t0 = time.time()

    for local_epoch in range(1, n_epochs + 1):
        epoch = start_epoch + local_epoch - 1
        model.train()
        optimizer.zero_grad(set_to_none=True)

        T_curr = float(T_schedule[local_epoch - 1])

        with get_autocast_context(device, amp_dtype_name):
            br_ic, tr_ic, u0_target, v0_target = dataset.sample_ic_batch(config["batch_ic"], rng=rng)
            loss_ic, loss_u0, loss_v0 = compute_loss_ic(model, br_ic, tr_ic, u0_target, v0_target)

            br_pde, tr_pde, params_pde = dataset.sample_pde_batch(
                config["batch_pde"],
                T=T_curr,
                late_fraction=config.get("late_fraction", 0.7),
                rng=rng,
            )
            loss_pde, _ = compute_loss_pde(model, br_pde, tr_pde, params_pde)

            if w_sg > 0.0:
                loss_sg, loss_sg_u, loss_sg_v = compute_loss_semigroup(
                    model=model,
                    dataset=dataset,
                    batch_size=config["batch_sg"],
                    T=T_curr,
                    dt_min=config["sg_dt_min"],
                    dt_max=config["sg_dt_max"],
                    sensor_chunk_size=config["sensor_chunk_size"],
                    rng=rng,
                )
            else:
                loss_sg = torch.tensor(0.0, device=device)
                loss_sg_u = torch.tensor(0.0, device=device)
                loss_sg_v = torch.tensor(0.0, device=device)

            if w_amp > 0.0:
                loss_amp, amp_stats = compute_loss_amplitude(
                    model=model,
                    dataset=dataset,
                    batch_size=config["batch_amp"],
                    T=T_curr,
                    late_start_frac=config["amp_late_start_frac"],
                    l1_weight_power=config["amp_l1_weight_power"],
                    tau_u=config["amp_tau_u"],
                    tau_v=config["amp_tau_v"],
                    tau_weight=config["amp_tau_weight"],
                    velocity_weight=config["amp_velocity_weight"],
                    dt_min=config["amp_dt_min"],
                    dt_max=config["amp_dt_max"],
                    sensor_chunk_size=config["sensor_chunk_size"],
                    rng=rng,
                )
            else:
                loss_amp = torch.tensor(0.0, device=device)
                amp_stats = {
                    "rms_now": torch.tensor(0.0, device=device),
                    "rms_next": torch.tensor(0.0, device=device),
                    "late_l1_u": torch.tensor(0.0, device=device),
                    "late_l1_v": torch.tensor(0.0, device=device),
                    "tau_pen_u": torch.tensor(0.0, device=device),
                    "tau_pen_v": torch.tensor(0.0, device=device),
                    "mono_pen": torch.tensor(0.0, device=device),
                    "t_now_mean": torch.tensor(0.0, device=device),
                    "t_next_mean": torch.tensor(0.0, device=device),
                }

            total = w_pde * loss_pde + w_ic * loss_ic + w_sg * loss_sg + w_amp * loss_amp

        if scaler is not None:
            scaler.scale(total).backward()
            if config.get("clip_norm"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            if config.get("clip_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_norm"])
            optimizer.step()

        scheduler.step()

        if epoch % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

        if local_epoch % config["log_every"] == 0 or local_epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            history["epoch"].append(epoch)
            history["loss_total"].append(float(total.item()))
            history["loss_pde"].append(float(loss_pde.item()))
            history["loss_ic"].append(float(loss_ic.item()))
            history["loss_u0"].append(float(loss_u0.item()))
            history["loss_v0"].append(float(loss_v0.item()))
            history["loss_sg"].append(float(loss_sg.item()))
            history["loss_sg_u"].append(float(loss_sg_u.item()))
            history["loss_sg_v"].append(float(loss_sg_v.item()))
            history["loss_amp"].append(float(loss_amp.item()))
            history["loss_amp_l1_u"].append(float(amp_stats["late_l1_u"].item()))
            history["loss_amp_l1_v"].append(float(amp_stats["late_l1_v"].item()))
            history["loss_amp_tau_u"].append(float(amp_stats["tau_pen_u"].item()))
            history["loss_amp_tau_v"].append(float(amp_stats["tau_pen_v"].item()))
            history["loss_amp_mono"].append(float(amp_stats["mono_pen"].item()))
            history["lr"].append(float(lr))
            history["T_curr"].append(float(T_curr))

            elapsed = (time.time() - t0) / 60.0
            print(
                f"Ep {epoch:5d} | "
                f"PDE {loss_pde.item():.3e} | "
                f"IC {loss_ic.item():.3e} (u0:{loss_u0.item():.3e}, v0:{loss_v0.item():.3e}) | "
                f"SG {loss_sg.item():.3e} (u:{loss_sg_u.item():.3e}, v:{loss_sg_v.item():.3e}) | "
                f"AMP {loss_amp.item():.3e} | "
                f"Total {total.item():.3e} | "
                f"T_curr={T_curr:.1f} | "
                f"LR {lr:.2e} | "
                f"{elapsed:.1f}min"
            )
            if w_amp > 0.0:
                print(
                    "         "
                    f"rms_now={amp_stats['rms_now'].item():.3e}, "
                    f"rms_next={amp_stats['rms_next'].item():.3e}, "
                    f"L1u={amp_stats['late_l1_u'].item():.3e}, "
                    f"TauU={amp_stats['tau_pen_u'].item():.3e}, "
                    f"Mono={amp_stats['mono_pen'].item():.3e}, "
                    f"t=({amp_stats['t_now_mean'].item():.2f}->{amp_stats['t_next_mean'].item():.2f})"
                )

        if local_epoch % config.get("save_last_every", 200) == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "history": history,
                },
                save_last,
            )

        if local_epoch % config["eval_every"] == 0:
            curr_loss = float(total.item())
            if curr_loss < best_loss:
                best_loss = curr_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "best_loss": best_loss,
                    },
                    save_best,
                )
                print(f"  *** Saved best: {best_loss:.4e}")

    print(f"\nDone! Best checkpoint: {save_best}")
    return history, save_best


def main():
    parser = argparse.ArgumentParser(description="Train full-grid pure-physics trig PI-DeepONet")
    parser.add_argument("--data", type=str, default="wave_data_v2.npz")
    parser.add_argument("--output_dir", type=str, default="results_pure_physics_trig_fullgrid")
    parser.add_argument("--save_name", type=str, default="best_pure_physics_trig_fullgrid.pt")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"])

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--p", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--first_omega_0", type=float, default=20.0)
    parser.add_argument("--hidden_omega_0", type=float, default=20.0)

    parser.add_argument("--n_epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--fused_optimizer", action="store_true")
    parser.add_argument("--batch_pde", type=int, default=4096)
    parser.add_argument("--batch_ic", type=int, default=1024)

    parser.add_argument("--stage", type=str, default="phase1", choices=["phase1", "phase2", "phase3"])
    parser.add_argument("--w_sg", type=float, default=0.0)
    parser.add_argument("--batch_sg", type=int, default=2)
    parser.add_argument("--sg_dt_min", type=float, default=0.2)
    parser.add_argument("--sg_dt_max", type=float, default=2.0)

    parser.add_argument("--w_amp", type=float, default=0.0)
    parser.add_argument("--batch_amp", type=int, default=2)
    parser.add_argument("--amp_dt_min", type=float, default=0.2)
    parser.add_argument("--amp_dt_max", type=float, default=2.0)
    parser.add_argument("--amp_late_start_frac", type=float, default=0.70)
    parser.add_argument("--amp_l1_weight_power", type=float, default=2.0)
    parser.add_argument("--amp_tau_u", type=float, default=1e-4)
    parser.add_argument("--amp_tau_v", type=float, default=1e-4)
    parser.add_argument("--amp_tau_weight", type=float, default=5.0)
    parser.add_argument("--amp_velocity_weight", type=float, default=0.25)

    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_ic", type=float, default=1000.0)

    parser.add_argument("--T", type=float, default=30.0)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--T_start", type=float, default=5.0)
    parser.add_argument("--resume_T_start", type=float, default=None)
    parser.add_argument("--late_fraction", type=float, default=0.7)

    parser.add_argument("--sensor_chunk_size", type=int, default=1024)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_last_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "none", "bf16", "fp16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--cudnn_benchmark", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])

    args = parser.parse_args()
    device = get_best_device(args.device)
    amp_dtype_name, _, _ = setup_acceleration(args, device)
    args.amp_dtype = amp_dtype_name

    dataset = WaveDatasetFullGrid(data_path=args.data, device=device)
    model = PIDeepONetTrigFullGrid(
        n_sensors=dataset.n_sensors,
        n_params=5,
        hidden_dim=args.hidden_dim,
        p=args.p,
        n_layers=args.n_layers,
        first_omega_0=args.first_omega_0,
        hidden_omega_0=args.hidden_omega_0,
    ).to(device)
    model = maybe_compile_model(model, args, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, Params: {n_params:,}")

    config = vars(args)
    history, _ = train(model, dataset, config)

    history_path = Path(args.output_dir) / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
