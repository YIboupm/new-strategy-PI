#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np


def sample_smooth_field(X, Y, rng, n_modes=6, amp_scale=1.0):
    """
    生成满足零边界趋势的平滑初值：
        envelope * sum_{k,l} a_{kl} sin(k pi x) sin(l pi y)
    """
    field = np.zeros_like(X, dtype=np.float32)

    for _ in range(n_modes):
        kx = int(rng.integers(1, 6))
        ky = int(rng.integers(1, 6))
        a = rng.normal(0.0, 1.0) / (kx**2 + ky**2)
        phase_x = rng.uniform(-0.2, 0.2)
        phase_y = rng.uniform(-0.2, 0.2)

        sx = np.sin(np.pi * kx * np.clip(X + phase_x, 0.0, 1.0))
        sy = np.sin(np.pi * ky * np.clip(Y + phase_y, 0.0, 1.0))
        field += a * sx * sy

    # 轻微边界包络，进一步压边界
    envelope = X * (1.0 - X) * Y * (1.0 - Y)
    field = field * envelope

    # 归一化到可控幅值
    max_abs = np.max(np.abs(field)) + 1e-8
    field = field / max_abs
    field = field * amp_scale

    return field.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="wave_data_v2.npz")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    # 初值振幅范围
    parser.add_argument("--u0_min", type=float, default=0.2)
    parser.add_argument("--u0_max", type=float, default=1.0)
    parser.add_argument("--v0_min", type=float, default=0.0)
    parser.add_argument("--v0_max", type=float, default=0.3)

    # PDE 参数范围: [gamma0, a_gamma, beta0, a_beta, alpha]
    parser.add_argument("--gamma0_min", type=float, default=0.8)
    parser.add_argument("--gamma0_max", type=float, default=1.5)
    parser.add_argument("--a_gamma_min", type=float, default=0.05)
    parser.add_argument("--a_gamma_max", type=float, default=0.4)

    parser.add_argument("--beta0_min", type=float, default=1.0)
    parser.add_argument("--beta0_max", type=float, default=2.0)
    parser.add_argument("--a_beta_min", type=float, default=0.05)
    parser.add_argument("--a_beta_max", type=float, default=0.35)

    parser.add_argument("--alpha_min", type=float, default=1.0)
    parser.add_argument("--alpha_max", type=float, default=2.0)

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    x = np.linspace(0.0, 1.0, args.Nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, args.Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")

    x1_coords = X.reshape(-1).astype(np.float32)
    x2_coords = Y.reshape(-1).astype(np.float32)
    n_dof = args.Nx * args.Ny

    u0 = np.zeros((args.n_samples, n_dof), dtype=np.float32)
    v0 = np.zeros((args.n_samples, n_dof), dtype=np.float32)
    params = np.zeros((args.n_samples, 5), dtype=np.float32)

    for i in range(args.n_samples):
        amp_u = rng.uniform(args.u0_min, args.u0_max)
        amp_v = rng.uniform(args.v0_min, args.v0_max)

        u0_field = sample_smooth_field(X, Y, rng, n_modes=6, amp_scale=amp_u)
        v0_field = sample_smooth_field(X, Y, rng, n_modes=6, amp_scale=amp_v)

        u0[i] = u0_field.reshape(-1)
        v0[i] = v0_field.reshape(-1)

        gamma0 = rng.uniform(args.gamma0_min, args.gamma0_max)
        a_gamma = rng.uniform(args.a_gamma_min, args.a_gamma_max)
        beta0 = rng.uniform(args.beta0_min, args.beta0_max)
        a_beta = rng.uniform(args.a_beta_min, args.a_beta_max)
        alpha = rng.uniform(args.alpha_min, args.alpha_max)

        params[i] = np.array([gamma0, a_gamma, beta0, a_beta, alpha], dtype=np.float32)

    np.savez_compressed(
        args.output,
        x1_coords=x1_coords,
        x2_coords=x2_coords,
        Nx=np.int32(args.Nx),
        Ny=np.int32(args.Ny),
        u0=u0,
        v0=v0,
        params=params,
    )

    print("=" * 60)
    print("Dataset generated successfully")
    print(f"output   : {args.output}")
    print(f"samples  : {args.n_samples}")
    print(f"grid     : {args.Nx} x {args.Ny}")
    print(f"u0 shape : {u0.shape}")
    print(f"v0 shape : {v0.shape}")
    print(f"p shape  : {params.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()