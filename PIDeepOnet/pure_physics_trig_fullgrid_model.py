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
3. Initial conditions are HARD-EMBEDDED:
      u(t,x) = u0(x) + t v0(x) + t^2 D(x) * correction_theta(...)
   so that automatically:
      u(0,x)   = u0(x)
      u_t(0,x)= v0(x)
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
    Pure-sine PI-DeepONet with HARD-EMBEDDED IC.

    Output structure:
        u(t,x) = u0_interp(x) + t * v0_interp(x) + t^2 * D(x) * correction(t,x)

    where
        D(x) = x1 (1-x1) x2 (1-x2)

    Therefore:
        u(0,x)    = u0(x)
        u_t(0,x)  = v0(x)
    """

    def __init__(
        self,
        *,
        n_sensors: int,
        Nx: int,
        Ny: int,
        n_params: int = 5,
        hidden_dim: int = 512,
        p: int = 512,
        n_layers: int = 8,
        first_omega_0: float = 20.0,
        hidden_omega_0: float = 20.0,
    ) -> None:
        super().__init__()

        if n_sensors != Nx * Ny:
            raise ValueError(
                f"Expected n_sensors = Nx * Ny for full-grid mode, "
                f"but got n_sensors={n_sensors}, Nx*Ny={Nx*Ny}"
            )

        self.n_sensors = n_sensors
        self.Nx = Nx
        self.Ny = Ny
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

    def _split_branch_input(
        self,
        branch_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        branch_input shape: [B, 2*n_sensors + n_params]
        returns:
            u0_grid:  [B, n_sensors]
            v0_grid:  [B, n_sensors]
            params:   [B, n_params]
        """
        ns = self.n_sensors
        u0_grid = branch_input[:, :ns]
        v0_grid = branch_input[:, ns:2 * ns]
        params = branch_input[:, 2 * ns:]
        return u0_grid, v0_grid, params

    def _bilinear_interp_from_fullgrid(
        self,
        field_flat: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bilinear interpolation on a regular interior grid:
            x_i = i/(Nx+1), i=1,...,Nx
            y_j = j/(Ny+1), j=1,...,Ny

        Parameters
        ----------
        field_flat : [B, Nx*Ny]
        x1, x2     : [B] in [0,1]

        Returns
        -------
        values : [B]
        """
        B = field_flat.shape[0]
        field = field_flat.view(B, self.Nx, self.Ny)

        # Continuous index coordinates relative to interior grid
        gx = x1 * (self.Nx + 1) - 1.0
        gy = x2 * (self.Ny + 1) - 1.0

        # Clamp to valid interpolation range
        eps = 1e-6
        gx = torch.clamp(gx, 0.0, self.Nx - 1.0 - eps)
        gy = torch.clamp(gy, 0.0, self.Ny - 1.0 - eps)

        i0 = torch.floor(gx).long()
        j0 = torch.floor(gy).long()
        i1 = i0 + 1
        j1 = j0 + 1

        i1 = torch.clamp(i1, max=self.Nx - 1)
        j1 = torch.clamp(j1, max=self.Ny - 1)

        wx = gx - i0.float()
        wy = gy - j0.float()

        batch_idx = torch.arange(B, device=field.device)

        f00 = field[batch_idx, i0, j0]
        f10 = field[batch_idx, i1, j0]
        f01 = field[batch_idx, i0, j1]
        f11 = field[batch_idx, i1, j1]

        values = (
            (1.0 - wx) * (1.0 - wy) * f00
            + wx * (1.0 - wy) * f10
            + (1.0 - wx) * wy * f01
            + wx * wy * f11
        )
        return values

    def _ic_embedded_terms(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns interpolated u0(x), v0(x) at the trunk spatial points.
        """
        u0_grid, v0_grid, _ = self._split_branch_input(branch_input)
        x1 = trunk_input[:, 1]
        x2 = trunk_input[:, 2]

        u0_interp = self._bilinear_interp_from_fullgrid(u0_grid, x1, x2)
        v0_interp = self._bilinear_interp_from_fullgrid(v0_grid, x1, x2)
        return u0_interp, v0_interp

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        # Hard-embedded initial conditions
        u0_interp, v0_interp = self._ic_embedded_terms(branch_input, trunk_input)

        tvals = trunk_input[:, 0]
        bc = self._bc_factor(trunk_input)

        # Learn only the correction term
        b = self.branch(branch_input)
        tr = self.trunk(trunk_input)
        correction_hat = torch.sum(b * tr, dim=-1) + self.bias

        correction = (tvals ** 2) * bc * correction_hat
        u = u0_interp + tvals * v0_interp + correction
        return u

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
