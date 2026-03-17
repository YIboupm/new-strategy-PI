#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pure_physics_trig_fullgrid_model.py
===================================

High-capacity PI-DeepONet with pure sine (SIREN-style) activations.
Designed for FULL-GRID branch sensors (64x64 -> 4096 sensors).

Main differences vs previous trig model
---------------------------------------
1. Full-grid branch input is supported natively.
2. Trunk uses PURE sine layers on raw (t, x1, x2), no Fourier features.
3. Higher default capacity is expected in training script.
4. Same public interface:
      - forward(branch_input, trunk_input)
      - forward_with_grad(branch_input, trunk_input)

PDE target
----------
u_tt - Δu - gamma(t) Δu_t + beta(t) u_t - alpha u + u^3 = 0
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class TrigMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        n_layers: int = 8,
        first_omega_0: float = 20.0,
        hidden_omega_0: float = 20.0,
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2")

        layers = [
            SineLayer(
                input_dim,
                hidden_dim,
                is_first=True,
                omega_0=first_omega_0,
            )
        ]

        for _ in range(n_layers - 2):
            layers.append(
                SineLayer(
                    hidden_dim,
                    hidden_dim,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        final_linear = nn.Linear(hidden_dim, output_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / hidden_omega_0
            final_linear.weight.uniform_(-bound, bound)
            final_linear.bias.uniform_(-bound, bound)

        layers.append(final_linear)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BranchNetTrig(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 512,
        output_dim: int = 512,
        n_layers: int = 8,
        first_omega_0: float = 20.0,
        hidden_omega_0: float = 20.0,
    ) -> None:
        super().__init__()
        self.net = TrigMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrunkNetTrig(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        *,
        hidden_dim: int = 512,
        output_dim: int = 512,
        n_layers: int = 8,
        first_omega_0: float = 20.0,
        hidden_omega_0: float = 20.0,
    ) -> None:
        super().__init__()
        self.net = TrigMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PIDeepONetTrigFullGrid(nn.Module):
    """
    Pure-sine PI-DeepONet.

    Hard Dirichlet BC on [0,1]^2:
        D(x) = x1 (1-x1) x2 (1-x2)
    so the predicted field is u = D(x) * u_hat.
    """

    def __init__(
        self,
        *,
        n_sensors: int,
        n_params: int = 5,
        hidden_dim: int = 512,
        p: int = 512,
        n_layers: int = 8,
        first_omega_0: float = 20.0,
        hidden_omega_0: float = 20.0,
    ) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.n_params = n_params
        branch_input_dim = 2 * n_sensors + n_params

        self.branch = BranchNetTrig(
            input_dim=branch_input_dim,
            hidden_dim=hidden_dim,
            output_dim=p,
            n_layers=n_layers,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )
        self.trunk = TrunkNetTrig(
            input_dim=3,
            hidden_dim=hidden_dim,
            output_dim=p,
            n_layers=n_layers,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _bc_factor(trunk_input: torch.Tensor) -> torch.Tensor:
        x1 = trunk_input[:, 1]
        x2 = trunk_input[:, 2]
        return x1 * (1.0 - x1) * x2 * (1.0 - x2)

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        b = self.branch(branch_input)
        t = self.trunk(trunk_input)
        u_hat = torch.sum(b * t, dim=-1) + self.bias
        return self._bc_factor(trunk_input) * u_hat

    def forward_with_grad(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        trunk_input = trunk_input.requires_grad_(True)
        u = self.forward(branch_input, trunk_input)

        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=trunk_input,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        u_t = grad_u[:, 0]
        u_x1 = grad_u[:, 1]
        u_x2 = grad_u[:, 2]

        grad_ut = torch.autograd.grad(
            outputs=u_t,
            inputs=trunk_input,
            grad_outputs=torch.ones_like(u_t),
            create_graph=True,
            retain_graph=True,
        )[0]
        u_tt = grad_ut[:, 0]
        u_tx1 = grad_ut[:, 1]
        u_tx2 = grad_ut[:, 2]

        grad_ux1 = torch.autograd.grad(
            outputs=u_x1,
            inputs=trunk_input,
            grad_outputs=torch.ones_like(u_x1),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_ux2 = torch.autograd.grad(
            outputs=u_x2,
            inputs=trunk_input,
            grad_outputs=torch.ones_like(u_x2),
            create_graph=True,
            retain_graph=True,
        )[0]

        u_x1x1 = grad_ux1[:, 1]
        u_x2x2 = grad_ux2[:, 2]
        lap_u = u_x1x1 + u_x2x2

        grad_utx1 = torch.autograd.grad(
            outputs=u_tx1,
            inputs=trunk_input,
            grad_outputs=torch.ones_like(u_tx1),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_utx2 = torch.autograd.grad(
            outputs=u_tx2,
            inputs=trunk_input,
            grad_outputs=torch.ones_like(u_tx2),
            create_graph=True,
            retain_graph=True,
        )[0]

        ut_x1x1 = grad_utx1[:, 1]
        ut_x2x2 = grad_utx2[:, 2]
        lap_ut = ut_x1x1 + ut_x2x2

        return {
            "u": u,
            "u_t": u_t,
            "u_tt": u_tt,
            "u_x1": u_x1,
            "u_x2": u_x2,
            "laplacian_u": lap_u,
            "laplacian_ut": lap_ut,
        }
