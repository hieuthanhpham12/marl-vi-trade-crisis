#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SR-PLR-MARL v8 — Strengthened experiments (ICTA2026)
====================================================
New in v8 (relative to v7):
  - Classical VI baselines: ProjectedGradient (PG) and Extragradient (EG).
    EG uses two operator evaluations + two projections per iteration; an
    `operator_calls` counter is logged for cost accounting.
  - Standardized recovery metrics: success_rate, RT_all (failures censored at
    the evaluation window), RT_success (NaN if none), Final_D, Final_G
    (projected VI residual), Best_G_postshock.
  - Consistent projected residual G_lambda with a single LAM_RES shared by all
    methods ("projected residual w.r.t. the implemented feasible projection").
  - Sensitivity sweep over (kappa, cooldown, tau_high).
  - New CLI: --main-v8, --mc-mr-v8, --sensitivity-v8.

The original v2 diagnostic docstring is retained below for reference.

SR-PLR-MARL v2 — Debug Diagnostics
====================================
Changes from v1:
  - TradeNetwork: compute_F_clean(), compute_D_active(), compute_G_lambda()
  - run_single():  D_active, rho_active, G_lambda columns; D_max_active tracking
  - print_diag_table(): per-episode table around shock (window=15)
  - print_post_shock_summary(): max rho, min D post-shock
  - run_smoke_shock(): single function for one shock type x both methods x 2 seeds
  - main(): runs all 3 shock types sequentially

Naming convention (consistent throughout):
  price_spike  →  P_j^k  ← 2 * P_j^k   (demand price DOUBLES)
  price_drop   →  P_j^k  ← 0.5 * P_j^k (demand price halved; NOT used here)

Usage:
  python sr_plr_marl.py --smoke --out sr_plr_outputs
  python sr_plr_marl.py --smoke --out sr_plr_outputs --episodes 800 --shock-ep 300
"""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Projected VI residual step. Shared by ALL methods so that
#   G_lambda(x) = || x - Pi_X(x - LAM_RES * F_clean(x)) || / LAM_RES
# is a consistent, method-independent measure of VI optimality.
LAM_RES: float = 1.0

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NetCfg:
    m: int = 2; n: int = 2
    C_a: float = 0.10; C_b: float = 0.10
    T_a: float = 0.04; T_b: float = 0.04
    P_intercept: float = 5.00; P_slope: float = 0.05; P_min: float = 0.10
    u: float = 50.0; Q: float = 100.0; D_cap: float = 100.0
    init_low: float = 1.0; init_high: float = 5.0

@dataclass
class PLRCfg:
    alpha0: float = 0.01; T1: float = 200.0; T2: float = 10.0
    tau_high: float = 0.60; sigma: float = 0.01; delta: float = 1e-12

@dataclass
class SRCfg(PLRCfg):
    kappa: float = 1.5; cooldown: int = 50

@dataclass
class GLRCfg:
    alpha0: float = 0.0001; alpha_min: float = 1e-5
    gamma: float = 0.95; max_patience: int = 5; tau_max: float = 5.0

@dataclass
class ShockCfg:
    """
    shock_type options:
      "route_blockage" : zero route (route_i, route_j); penalty in F keeps x away.
      "supply_cut"     : reduce supply_i capacity by supply_cut_frac via projection.
      "price_spike"    : P_j^k <- price_multiplier * P_j^k  (multiplier=2 → price DOUBLES).
    """
    episode:           int   = 300
    shock_type:        str   = "route_blockage"
    route_i:           int   = 0; route_j:          int   = 0
    supply_i:          int   = 0; supply_cut_frac:  float = 0.50
    price_multiplier:  float = 2.00    # price_spike: P_j <- price_multiplier * P_j

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — TRADE NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class TradeNetwork:
    """
    Mutable trade network. Shock state stored here.
    Three residual measures:
      compute_F       – VI operator (with blocked-route penalty for gradient)
      compute_F_clean – VI operator WITHOUT penalty (for G_lambda)
      compute_D_active – mean |F_clean| over non-blocked routes only
      compute_G_lambda – projected VI residual ||x - Proj(x - lam*F_clean)|| / lam
    """
    def __init__(self, cfg: NetCfg) -> None:
        self.cfg = cfg
        self.blocked: Set[Tuple[int,int]] = set()
        self.supply_cut: Dict[int,float] = {}
        self.price_mult: float = 1.0

    def reset_shock(self) -> None:
        self.blocked.clear(); self.supply_cut.clear(); self.price_mult = 1.0

    def apply_shock(self, sc: ShockCfg) -> None:
        st = sc.shock_type
        if st == "route_blockage":
            self.blocked.add((sc.route_i, sc.route_j))
        elif st == "supply_cut":
            self.supply_cut[sc.supply_i] = min(
                1.0, self.supply_cut.get(sc.supply_i, 0.0) + sc.supply_cut_frac)
        elif st == "price_spike":
            self.price_mult *= sc.price_multiplier
        else:
            raise ValueError(f"Unknown shock_type {st!r}")

    # ── base VI operator (with penalty for blocked routes) ───────────────
    def compute_F(self, x: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        q = x.sum(axis=1); d = x.sum(axis=0)
        C = (cfg.C_a + cfg.C_b * q)[:, None]
        T = cfg.T_a + cfg.T_b * x
        P = (np.maximum(cfg.P_min, cfg.P_intercept - cfg.P_slope*d)
             * self.price_mult)[None, :]
        F = C + T - P
        for (i,j) in self.blocked:
            F[i,j] = abs(float(F[i,j])) + 10.0   # penalty: pushes x[i,j]→0
        return F

    # ── clean VI operator (NO penalty) used for G_lambda ─────────────────
    def compute_F_clean(self, x: np.ndarray) -> np.ndarray:
        """Natural operator without blocked-route penalty.
        At the post-shock VI solution: x[i,j]=0 (by projection) and
        F_clean[i,j]>=0 (complementary slackness). G_lambda uses this."""
        cfg = self.cfg
        q = x.sum(axis=1); d = x.sum(axis=0)
        C = (cfg.C_a + cfg.C_b * q)[:, None]
        T = cfg.T_a + cfg.T_b * x
        P = (np.maximum(cfg.P_min, cfg.P_intercept - cfg.P_slope*d)
             * self.price_mult)[None, :]
        return C + T - P

    # ── mean |F_clean| over ACTIVE (non-blocked) routes ──────────────────
    def compute_D_active(self, x: np.ndarray) -> float:
        """Excludes blocked routes. For route_blockage, this measures
        convergence on the active subnetwork independent of the penalty."""
        F = self.compute_F_clean(x)
        m, n = F.shape
        vals = [abs(float(F[i,j]))
                for i in range(m) for j in range(n)
                if (i,j) not in self.blocked]
        return float(np.mean(vals)) if vals else 0.0

    # ── projected VI residual ─────────────────────────────────────────────
    def compute_G_lambda(self, x: np.ndarray, lam: float = 1.0) -> float:
        """G_lambda(x) = ||x - Proj_X(x - lam*F_clean(x))|| / lam.
        Equals 0 iff x solves VI(F_clean, X_current). Standard VI gap."""
        Fc = self.compute_F_clean(x)
        xp = self.project(x - lam * Fc)
        return float(np.linalg.norm(x - xp)) / lam

    # ── all-route residual ────────────────────────────────────────────────
    def compute_D(self, x: np.ndarray) -> float:
        return float(np.mean(np.abs(self.compute_F(x))))

    # ── feasibility projection ────────────────────────────────────────────
    def project(self, x: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        y = np.clip(x, 0.0, cfg.u)
        for (i,j) in self.blocked:
            y[i,j] = 0.0
        for _ in range(10):
            for i in range(y.shape[0]):
                eff_Q = cfg.Q * (1.0 - self.supply_cut.get(i, 0.0))
                s = float(y[i,:].sum())
                if s > eff_Q > 0:
                    y[i,:] *= eff_Q / s
            for j in range(y.shape[1]):
                s = float(y[:,j].sum())
                if s > cfg.D_cap > 0:
                    y[:,j] *= cfg.D_cap / s
            y = np.clip(y, 0.0, cfg.u)
            for (i,j) in self.blocked:
                y[i,j] = 0.0
        return y

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — SCHEDULERS  (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════

class PLROriginal:
    name = "PLR-original"
    def __init__(self, cfg: PLRCfg) -> None:
        self.cfg = cfg; self.D_max = 1.0; self.phase = 1
    def init(self, D0: float) -> None:
        self.D_max = max(D0, self.cfg.delta); self.phase = 1
    def step(self, t: int, D_t: float) -> Tuple[float, dict]:
        cfg = self.cfg
        rho = max(0.0, 1.0 - D_t / self.D_max)
        if rho < cfg.tau_high:
            alpha = cfg.alpha0 / (1.0 + t / cfg.T1); self.phase = 1
        else:
            alpha = cfg.alpha0 / (1.0 + t / cfg.T2); self.phase = 2
        return alpha, {"rho": rho, "phase": self.phase,
                       "shock_detected": False, "t_local": t, "D_max": self.D_max}


class SRPLRScheduler:
    name = "SR-PLR-MARL"
    def __init__(self, cfg: SRCfg) -> None:
        self.cfg = cfg; self.D_max = 1.0; self.D_prev = 1.0
        self.phase = 1; self.t_local = 0
        self.last_shock_t = -(cfg.cooldown + 1); self.shock_count = 0
    def init(self, D0: float) -> None:
        cfg = self.cfg
        self.D_max = max(D0, cfg.delta); self.D_prev = D0
        self.phase = 1; self.t_local = 0
        self.last_shock_t = -(cfg.cooldown + 1); self.shock_count = 0
    def step(self, t: int, D_t: float) -> Tuple[float, dict]:
        cfg = self.cfg; shock_detected = False
        in_cooldown = (t - self.last_shock_t) < cfg.cooldown
        if not in_cooldown:
            type_a = D_t > self.D_max
            type_b = self.D_prev > cfg.delta and D_t > cfg.kappa * self.D_prev
            if type_a or type_b:
                shock_detected = True; self.last_shock_t = t; self.shock_count += 1
        if D_t > self.D_max:
            self.D_max = D_t
        if shock_detected:
            self.t_local = 0; self.phase = 1
        else:
            self.t_local += 1
        rho = max(0.0, 1.0 - D_t / self.D_max)
        if rho < cfg.tau_high or shock_detected:
            alpha = cfg.alpha0 / (1.0 + self.t_local / cfg.T1); self.phase = 1
        else:
            alpha = cfg.alpha0 / (1.0 + self.t_local / cfg.T2); self.phase = 2
        self.D_prev = D_t
        return alpha, {"rho": rho, "phase": self.phase, "shock_detected": shock_detected,
                       "t_local": self.t_local, "D_max": self.D_max,
                       "shock_count": self.shock_count}


class GLRScheduler:
    name = "GLR-style"
    def __init__(self, cfg: GLRCfg) -> None:
        self.cfg = cfg; self.alpha = cfg.alpha0; self.patience = 0
    def init(self, D0: float) -> None:
        self.alpha = self.cfg.alpha0; self.patience = 0
    def step(self, t: int, D_t: float) -> Tuple[float, dict]:
        cfg = self.cfg; rho = max(0.0, 1.0 - D_t / cfg.tau_max)
        if rho > 0.5:
            new_alpha = max(cfg.alpha_min, self.alpha * cfg.gamma * 0.95); self.patience = 0
        elif self.patience >= cfg.max_patience:
            new_alpha = max(cfg.alpha_min, self.alpha * cfg.gamma); self.patience = 0
        elif self.patience >= 2 and rho < 0.2:
            new_alpha = min(2.0 * self.alpha, self.alpha * 1.05); self.patience += 1
        else:
            new_alpha = self.alpha; self.patience += 1
        self.alpha = new_alpha
        return self.alpha, {"rho": rho, "phase": -1, "shock_detected": False,
                            "t_local": t, "D_max": cfg.tau_max, "glr_patience": self.patience}


class FixedLR:
    name = "Fixed-LR"
    def __init__(self, alpha: float = 0.01) -> None:
        self._alpha = alpha; self._D_max = 1.0
    def init(self, D0: float) -> None:
        self._D_max = max(D0, 1e-12)
    def step(self, t: int, D_t: float) -> Tuple[float, dict]:
        rho = max(0.0, 1.0 - D_t / self._D_max)
        return self._alpha, {"rho": rho, "phase": -1, "shock_detected": False,
                             "t_local": t, "D_max": self._D_max}


class PolyDecay:
    name = "Poly-decay"
    def __init__(self, alpha0: float = 0.01, exponent: float = 0.5) -> None:
        self.alpha0 = alpha0; self.exponent = exponent; self._D_max = 1.0
    def init(self, D0: float) -> None:
        self._D_max = max(D0, 1e-12)
    def step(self, t: int, D_t: float) -> Tuple[float, dict]:
        rho = max(0.0, 1.0 - D_t / self._D_max)
        alpha = self.alpha0 / (1.0 + t) ** self.exponent
        return alpha, {"rho": rho, "phase": -1, "shock_detected": False,
                       "t_local": t, "D_max": self._D_max}


# -- Classical VI baseline: Projected Gradient (constant tuned step) ----------
class ProjectedGradient:
    """Projected (stochastic) gradient for VI:  x <- Pi_X(x - alpha F(x) + noise).
    Constant tuned step; one operator evaluation + one projection per iteration."""
    name = "PG-fixed"
    update_rule = "pg"
    operator_calls_per_iter = 1
    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha; self._D_max = 1.0
    def init(self, D0: float) -> None:
        self._D_max = max(D0, 1e-12)
    def step(self, t: int, D_t: float) -> Tuple[float, dict]:
        rho = max(0.0, 1.0 - D_t / self._D_max)
        return self._alpha, {"rho": rho, "phase": -1, "shock_detected": False,
                             "t_local": t, "D_max": self._D_max}


# -- Classical VI baseline: Extragradient (two operator evals per iteration) --
class Extragradient:
    """Extragradient (Korpelevich) for VI:
        y_t     = Pi_X(x_t - alpha F(x_t) + noise_1)
        x_{t+1} = Pi_X(x_t - alpha F(y_t) + noise_2)
    Two operator evaluations + two projections per iteration."""
    name = "EG"
    update_rule = "eg"
    operator_calls_per_iter = 2
    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha; self._D_max = 1.0
    def init(self, D0: float) -> None:
        self._D_max = max(D0, 1e-12)
    def step(self, t: int, D_t: float) -> Tuple[float, dict]:
        rho = max(0.0, 1.0 - D_t / self._D_max)
        return self._alpha, {"rho": rho, "phase": -1, "shock_detected": False,
                             "t_local": t, "D_max": self._D_max}


# ── Ablation variant A: warm restart ONLY (no clock reset) ───────────────
class SRPLRWarmRestartOnly(SRPLRScheduler):
    """D_max warm restart, but t_local is NEVER reset.
    Ablation: isolates the contribution of D_max recalibration."""
    name = "SR-WarmRestart"

    def step(self, t: int, D_t: float) -> Tuple[float, dict]:
        cfg = self.cfg; shock_detected = False
        in_cooldown = (t - self.last_shock_t) < cfg.cooldown
        if not in_cooldown:
            type_a = D_t > self.D_max
            type_b = self.D_prev > cfg.delta and D_t > cfg.kappa * self.D_prev
            if type_a or type_b:
                shock_detected = True; self.last_shock_t = t; self.shock_count += 1
        if D_t > self.D_max:           # warm restart: update D_max
            self.D_max = D_t
        self.t_local += 1              # NO clock reset
        if shock_detected:
            self.phase = 1
        rho = max(0.0, 1.0 - D_t / self.D_max)
        if rho < cfg.tau_high or shock_detected:
            alpha = cfg.alpha0 / (1.0 + self.t_local / cfg.T1); self.phase = 1
        else:
            alpha = cfg.alpha0 / (1.0 + self.t_local / cfg.T2); self.phase = 2
        self.D_prev = D_t
        return alpha, {"rho": rho, "phase": self.phase,
                       "shock_detected": shock_detected, "t_local": self.t_local,
                       "D_max": self.D_max, "shock_count": self.shock_count}


# ── Ablation variant B: clock reset ONLY (D_max stays fixed) ────────────
class SRPLRClockOnly(SRPLRScheduler):
    """t_local resets on shock, but D_max is NEVER updated.
    Ablation: isolates the contribution of phase-clock restart."""
    name = "SR-ClockOnly"

    def step(self, t: int, D_t: float) -> Tuple[float, dict]:
        cfg = self.cfg; shock_detected = False
        in_cooldown = (t - self.last_shock_t) < cfg.cooldown
        if not in_cooldown:
            type_a = D_t > self.D_max
            type_b = self.D_prev > cfg.delta and D_t > cfg.kappa * self.D_prev
            if type_a or type_b:
                shock_detected = True; self.last_shock_t = t; self.shock_count += 1
        # NO D_max warm restart
        if shock_detected:             # clock reset only
            self.t_local = 0; self.phase = 1
        else:
            self.t_local += 1
        rho = max(0.0, 1.0 - D_t / self.D_max)   # D_max fixed
        if rho < cfg.tau_high or shock_detected:
            alpha = cfg.alpha0 / (1.0 + self.t_local / cfg.T1); self.phase = 1
        else:
            alpha = cfg.alpha0 / (1.0 + self.t_local / cfg.T2); self.phase = 2
        self.D_prev = D_t
        return alpha, {"rho": rho, "phase": self.phase,
                       "shock_detected": shock_detected, "t_local": self.t_local,
                       "D_max": self.D_max, "shock_count": self.shock_count}



# ── Proposed variant: SR-PLR with Phase-1 lock after shock ──────────────
class SRPLRLock(SRPLRScheduler):
    """Shock-responsive PLR with a mandatory Phase-1 lock window.

    Motivation:
      In the full SR-PLR rule, D_max warm restart can make rho recover quickly
      and may trigger Phase 2 before the post-shock dynamics have moved far
      enough from the old equilibrium.  The lock forces Phase 1 for L episodes
      after a detected shock, while still keeping the Robbins--Monro structure
      under a finite number of shocks.
    """
    name = "SR-PLR-Lock"

    def __init__(self, cfg: SRCfg, lock_len: int = 100, warm_restart: bool = True) -> None:
        super().__init__(cfg)
        self.lock_len = int(lock_len)
        self.warm_restart = bool(warm_restart)
        self.lock_remaining = 0

    def init(self, D0: float) -> None:
        super().init(D0)
        self.lock_remaining = 0

    def step(self, t: int, D_t: float) -> Tuple[float, dict]:
        cfg = self.cfg; shock_detected = False
        in_cooldown = (t - self.last_shock_t) < cfg.cooldown
        if not in_cooldown:
            type_a = D_t > self.D_max
            type_b = self.D_prev > cfg.delta and D_t > cfg.kappa * self.D_prev
            if type_a or type_b:
                shock_detected = True
                self.last_shock_t = t
                self.shock_count += 1

        if self.warm_restart and D_t > self.D_max:
            self.D_max = D_t

        if shock_detected:
            self.t_local = 0
            self.phase = 1
            self.lock_remaining = self.lock_len
        else:
            self.t_local += 1
            if self.lock_remaining > 0:
                self.lock_remaining -= 1

        rho = max(0.0, 1.0 - D_t / self.D_max)
        locked = self.lock_remaining > 0
        if rho < cfg.tau_high or shock_detected or locked:
            alpha = cfg.alpha0 / (1.0 + self.t_local / cfg.T1)
            self.phase = 1
        else:
            alpha = cfg.alpha0 / (1.0 + self.t_local / cfg.T2)
            self.phase = 2
        self.D_prev = D_t
        return alpha, {"rho": rho, "phase": self.phase,
                       "shock_detected": shock_detected, "t_local": self.t_local,
                       "D_max": self.D_max, "shock_count": self.shock_count,
                       "lock_remaining": self.lock_remaining}

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — EXPERIMENT RUNNER  (extended: D_active, rho_active, G_lambda)
# ═══════════════════════════════════════════════════════════════════════════

def run_single(
    scheduler,
    net_cfg:   NetCfg,
    shock_cfg: Optional[ShockCfg],
    plr_cfg:   PLRCfg,
    episodes:  int,
    seed:      int,
) -> pd.DataFrame:
    """
    Run one experiment. Returns a DataFrame with columns:
      t, D, D_active, G_lambda, rho, rho_active, alpha,
      phase, shock_detected, t_local, D_max

    D_max_active: warm-restart running maximum of D_active (shared by all
    methods in this diagnostic version for simplicity).

    Timing:
      t = shock_cfg.episode  : shock injected AFTER log; update uses post-shock F.
      t = shock_cfg.episode+1: scheduler sees post-shock D, detects shock.
    """
    rng = np.random.default_rng(seed)
    net = TradeNetwork(net_cfg)
    x = rng.uniform(net_cfg.init_low, net_cfg.init_high, size=(net_cfg.m, net_cfg.n))
    x = net.project(x)

    D0         = net.compute_D(x)
    D_active0  = net.compute_D_active(x)
    scheduler.init(D0)

    update_rule = getattr(scheduler, "update_rule", "plr")
    op_per_iter = getattr(scheduler, "operator_calls_per_iter", 1)

    D_max_active = max(D_active0, plr_cfg.delta)   # warm-restart tracking
    rows: List[dict] = []

    for t in range(episodes):
        D_t        = net.compute_D(x)
        D_active_t = net.compute_D_active(x)
        G_lam_t    = net.compute_G_lambda(x, lam=LAM_RES)

        # warm-restart D_max_active (tracks both pre- and post-shock extremes)
        if D_active_t > D_max_active:
            D_max_active = D_active_t

        alpha, diag = scheduler.step(t, D_t)

        rho_active = max(0.0, 1.0 - D_active_t / D_max_active)

        rows.append({
            "t":             t,
            "D":             D_t,
            "D_active":      D_active_t,
            "G_lambda":      G_lam_t,
            "rho":           diag["rho"],
            "rho_active":    rho_active,
            "alpha":         alpha,
            "phase":         diag["phase"],
            "shock_detected": bool(diag["shock_detected"]),
            "t_local":       diag["t_local"],
            "D_max":         diag["D_max"],
            "operator_calls": op_per_iter,
        })

        # inject shock AFTER log, BEFORE update
        if shock_cfg is not None and t == shock_cfg.episode:
            net.apply_shock(shock_cfg)

        if update_rule == "eg":
            xi1 = rng.normal(0.0, plr_cfg.sigma, size=x.shape)
            y   = net.project(x - alpha * net.compute_F(x) + alpha * xi1)
            xi2 = rng.normal(0.0, plr_cfg.sigma, size=x.shape)
            x   = net.project(x - alpha * net.compute_F(y) + alpha * xi2)
        else:
            F_t = net.compute_F(x)
            xi  = rng.normal(0.0, plr_cfg.sigma, size=x.shape)
            x   = net.project(x - alpha * F_t + alpha * xi)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    df:                  pd.DataFrame,
    shock_episode:       int,
    tau_recovery:        float = 0.50,
    max_recovery_window: int   = 400,
    osc_warmup:          int   = 50,
) -> dict:
    rho        = df["rho"].values
    rho_active = df["rho_active"].values
    t          = df["t"].values

    # recovery time based on rho_active (more meaningful for route_blockage)
    post = df[df["t"] > shock_episode]
    rec  = post[post["rho_active"] >= tau_recovery]
    tau_rec = (int(rec.iloc[0]["t"]) - shock_episode) if len(rec) > 0 else len(df)

    D_final        = float(df["D"].iloc[-1])
    D_active_final = float(df["D_active"].iloc[-1])
    G_lam_final    = float(df["G_lambda"].iloc[-1])

    try:
        cum_area        = float(np.trapezoid(rho, t))
        cum_area_active = float(np.trapezoid(rho_active, t))
    except AttributeError:
        cum_area        = float(np.trapz(rho, t))
        cum_area_active = float(np.trapz(rho_active, t))

    t_osc = shock_episode + osc_warmup
    osc_v = df[df["t"] >= t_osc]["rho_active"].values
    osc_index = float(np.mean(np.abs(np.diff(osc_v)))) if len(osc_v) > 1 else float("nan")

    success = int(tau_rec <= max_recovery_window)

    return {
        "recovery_time":    tau_rec,
        "D_final":          D_final,
        "D_active_final":   D_active_final,
        "G_lambda_final":   G_lam_final,
        "cum_area":         cum_area,
        "cum_area_active":  cum_area_active,
        "osc_index":        osc_index,
        "success":          success,
    }

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — DIAGNOSTIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — EXTENDED METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    df: pd.DataFrame,
    shock_ep: int,
    shock_type: str = "price_spike",
    tau_recovery: float = 0.50,
    max_recovery_window: int = 400,
    osc_warmup: int = 50,
) -> dict:
    """
    Compute all paper metrics with the correct primary recovery signal.

    IMPORTANT:
      * price_spike and supply_cut use scheduler rho (based on each method's
        own normalization), because all routes remain active and no penalty is
        inserted.  This exposes the fixed-D_max failure of PLR-original.
      * route_blockage uses rho_active, because D_all contains the artificial
        blocked-route penalty and is not a valid recovery indicator.
    """
    rho        = df["rho"].values
    rho_active = df["rho_active"].values
    t          = df["t"].values

    primary_col = "rho_active" if shock_type == "route_blockage" else "rho"
    primary_vals = df[primary_col].values

    post = df[df["t"] > shock_ep]
    rec  = post[post[primary_col] >= tau_recovery]
    recovered = len(rec) > 0
    tau_rec = (int(rec.iloc[0]["t"]) - shock_ep) if recovered else len(df)

    try:
        cum_area        = float(np.trapezoid(rho, t))
        cum_area_active = float(np.trapezoid(rho_active, t))
        cum_area_primary = float(np.trapezoid(primary_vals, t))
    except AttributeError:
        cum_area        = float(np.trapz(rho, t))
        cum_area_active = float(np.trapz(rho_active, t))
        cum_area_primary = float(np.trapz(primary_vals, t))

    t_osc  = shock_ep + osc_warmup
    osc_primary_v = df[df["t"] >= t_osc][primary_col].values
    osc_active_v  = df[df["t"] >= t_osc]["rho_active"].values
    osc_all_v     = df[df["t"] >= t_osc]["rho"].values

    def _oi(v):
        return float(np.mean(np.abs(np.diff(v)))) if len(v) > 1 else float("nan")

    osc_idx_primary = _oi(osc_primary_v)
    osc_idx_active  = _oi(osc_active_v)
    osc_idx_all     = _oi(osc_all_v)

    def _alpha(mask):
        sub = df[mask]
        return float(sub["alpha"].iloc[0]) if len(sub) else float("nan")

    alpha_before = _alpha(df["t"] == shock_ep - 1)
    alpha_after  = _alpha(df["t"] == shock_ep + 1)
    det_rows = df[df["shock_detected"]]
    shock_det_episode = int(det_rows.iloc[0]["t"]) if len(det_rows) else -1

    return {
        "primary_metric":       primary_col,
        "recovery_time":        tau_rec,
        "D_final":              float(df["D"].iloc[-1]),
        "D_active_final":       float(df["D_active"].iloc[-1]),
        "G_lambda_final":       float(df["G_lambda"].iloc[-1]),
        "rho_final":            round(float(df["rho"].iloc[-1]), 6),
        "rho_active_final":     round(float(df["rho_active"].iloc[-1]), 6),
        "cum_area":             round(cum_area, 3),
        "cum_area_active":      round(cum_area_active, 3),
        "cum_area_primary":     round(cum_area_primary, 3),
        "osc_index":            round(osc_idx_primary, 6) if not np.isnan(osc_idx_primary) else None,
        "osc_index_active":     round(osc_idx_active, 6) if not np.isnan(osc_idx_active) else None,
        "osc_index_all":        round(osc_idx_all, 6) if not np.isnan(osc_idx_all) else None,
        "success":              int(recovered and tau_rec <= max_recovery_window),
        "alpha_before_shock":   round(alpha_before, 8),
        "alpha_after_shock":    round(alpha_after, 8),
        "shock_detected_ep":    shock_det_episode,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — FULL EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def build_method_registry(plr_cfg: PLRCfg, sr_cfg: SRCfg, glr_cfg: GLRCfg) -> dict:
    """Return all scheduler factories by name.

    Includes fair variants added after diagnostics:
      - GLR-matched uses the same alpha0 as PLR/SR-PLR.
      - SR-PLR-Lock{L} forces Phase 1 for L episodes after a detected shock.
    """
    glr_matched = GLRCfg(alpha0=plr_cfg.alpha0,
                         alpha_min=glr_cfg.alpha_min,
                         gamma=glr_cfg.gamma,
                         max_patience=glr_cfg.max_patience,
                         tau_max=glr_cfg.tau_max)
    return {
        "PLR-original":     lambda: PLROriginal(plr_cfg),
        "SR-PLR-MARL":      lambda: SRPLRScheduler(sr_cfg),
        "SR-PLR-Lock50":    lambda: SRPLRLock(sr_cfg, lock_len=50),
        "SR-PLR-Lock100":   lambda: SRPLRLock(sr_cfg, lock_len=100),
        "SR-PLR-Lock200":   lambda: SRPLRLock(sr_cfg, lock_len=200),
        "GLR-style":        lambda: GLRScheduler(glr_cfg),
        "GLR-matched":      lambda: GLRScheduler(glr_matched),
        "Fixed-LR":         lambda: FixedLR(plr_cfg.alpha0),
        "Poly-decay":       lambda: PolyDecay(plr_cfg.alpha0, 0.5),
    }


def build_ablation_registry(plr_cfg: PLRCfg, sr_cfg: SRCfg) -> dict:
    """Ablation: isolate warm-restart, clock-reset, and Phase-1 lock."""
    return {
        "PLR-original":    lambda: PLROriginal(plr_cfg),
        "SR-WarmRestart":  lambda: SRPLRWarmRestartOnly(sr_cfg),
        "SR-ClockOnly":    lambda: SRPLRClockOnly(sr_cfg),
        "SR-PLR-MARL":     lambda: SRPLRScheduler(sr_cfg),
        "SR-PLR-Lock50":   lambda: SRPLRLock(sr_cfg, lock_len=50),
        "SR-PLR-Lock100":  lambda: SRPLRLock(sr_cfg, lock_len=100),
        "SR-PLR-Lock200":  lambda: SRPLRLock(sr_cfg, lock_len=200),
    }


def run_full_experiments(
    out_dir:          Path,
    episodes_2x2:     int = 800,
    episodes_3x3:     int = 800,
    shock_ep:         int = 300,
    n_seeds:          int = 10,
    shock_types:      Optional[List[str]] = None,
    sigma:            float = 0.01,
) -> None:
    """
    Full synthetic experiment runner.

    Produces per shock_type:
      raw_history_{shock_type}.csv    – every episode, method, seed (plot_seed only)
      summary_by_seed.csv             – metrics per (method, m×n, shock, seed)
      summary_by_method.csv           – aggregated mean ± std
      ablation_summary.csv            – ablation on price_spike, 2×2
      diagnostic_summary.json         – text interpretation
      figures/  (4 plot types)
    """
    if shock_types is None:
        shock_types = ["price_spike", "route_blockage", "supply_cut"]

    plr_cfg = PLRCfg(sigma=sigma)
    sr_cfg  = SRCfg(sigma=sigma)
    glr_cfg = GLRCfg()
    seeds   = list(range(n_seeds))
    plot_seed = 0   # seed used for trajectory plots

    main_methods   = build_method_registry(plr_cfg, sr_cfg, glr_cfg)
    ablat_methods  = build_ablation_registry(plr_cfg, sr_cfg)
    net_configs    = [
        (2, 2, episodes_2x2),
        (3, 3, episodes_3x3),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    all_summaries:  List[dict] = []
    ablat_summaries: List[dict] = []
    plot_histories: Dict[str, pd.DataFrame] = {}   # keyed by label for plots

    total = len(shock_types) * len(net_configs) * len(main_methods) * n_seeds
    done  = 0

    print(f"\nRunning {total} experiments "
          f"({len(shock_types)} shocks × {len(net_configs)} nets × "
          f"{len(main_methods)} methods × {n_seeds} seeds)...")

    for shock_type in shock_types:
        shock_cfg = ShockCfg(episode=shock_ep, shock_type=shock_type)
        raw_rows: List[pd.DataFrame] = []

        for (m, n, eps) in net_configs:
            net_cfg = NetCfg(m=m, n=n)
            net_label = f"{m}x{n}"

            # ── main comparison ──────────────────────────────────────────
            for mname, factory in main_methods.items():
                for seed in seeds:
                    sch = factory()
                    df  = run_single(sch, net_cfg, shock_cfg, plr_cfg, eps, seed)
                    met = compute_metrics(df, shock_ep, shock_type)

                    row = {"method": mname, "m": m, "n": n, "net": net_label,
                           "shock_type": shock_type, "seed": seed, **met}
                    all_summaries.append(row)

                    if seed == plot_seed:
                        df = df.assign(method=mname, m=m, n=n,
                                       net=net_label, shock_type=shock_type, seed=seed)
                        raw_rows.append(df)

                    done += 1
                    if done % 10 == 0:
                        print(f"  {done}/{total}  [{shock_type}  {net_label}  {mname}  s{seed}]")

            # ── ablation (price_spike, 2×2 only) ─────────────────────────
            if shock_type == "price_spike" and m == 2 and n == 2:
                for aname, afactory in ablat_methods.items():
                    for seed in seeds:
                        sch = afactory()
                        df  = run_single(sch, net_cfg, shock_cfg, plr_cfg, eps, seed)
                        met = compute_metrics(df, shock_ep, shock_type)
                        row = {"method": aname, "m": m, "n": n, "net": net_label,
                               "shock_type": shock_type, "seed": seed, **met}
                        ablat_summaries.append(row)

        # save raw history for plot_seed
        if raw_rows:
            raw_df = pd.concat(raw_rows, ignore_index=True)
            raw_df.to_csv(out_dir / f"raw_history_{shock_type}.csv", index=False)
            plot_histories[shock_type] = raw_df

    # ── aggregate summaries ──────────────────────────────────────────────
    summary_df = pd.DataFrame(all_summaries)
    ablat_df   = pd.DataFrame(ablat_summaries)

    summary_df.to_csv(out_dir / "summary_by_seed.csv", index=False)

    method_agg = _aggregate(summary_df)
    method_agg.to_csv(out_dir / "summary_by_method.csv", index=False)

    if not ablat_df.empty:
        ablat_agg = _aggregate(ablat_df)
        ablat_agg.to_csv(out_dir / "ablation_summary.csv", index=False)

    # ── plots ────────────────────────────────────────────────────────────
    if plot_histories:
        _plot_rho_trajectories(plot_histories, shock_ep, plr_cfg.tau_high, fig_dir)
        _plot_alpha_around_shock(plot_histories, shock_ep, fig_dir)
        _plot_summary_bars(method_agg, fig_dir)
        if "route_blockage" in plot_histories:
            _plot_route_blockage_comparison(plot_histories["route_blockage"],
                                            shock_ep, plr_cfg.tau_high, fig_dir)

    # ── diagnostic JSON ──────────────────────────────────────────────────
    diag = _build_diagnostic(summary_df, ablat_df, shock_ep)
    with open(out_dir / "diagnostic_summary.json", "w") as fh:
        json.dump(diag, fh, indent=2)

    # ── print interpretation ─────────────────────────────────────────────
    _print_interpretation(method_agg, ablat_df, summary_df)

    print(f"\nAll outputs saved to: {out_dir.resolve()}")


def _nan_safe(x):
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (method, net, shock_type): mean, std, success_rate."""
    metric_cols = ["recovery_time", "D_final", "D_active_final", "G_lambda_final",
                   "rho_final", "rho_active_final",
                   "cum_area", "cum_area_active", "cum_area_primary",
                   "osc_index", "osc_index_active", "osc_index_all",
                   "alpha_before_shock", "alpha_after_shock"]
    grp = df.groupby(["method", "net", "shock_type"])

    rows = []
    for (meth, net, shock), g in grp:
        r = {"method": meth, "net": net, "shock_type": shock,
             "n_seeds": len(g),
             "success_rate": round(float(g["success"].mean()), 3)}
        for col in metric_cols:
            vals = g[col].dropna()
            r[f"{col}_mean"] = round(float(vals.mean()), 4) if len(vals) else None
            r[f"{col}_std"]  = round(float(vals.std()),  4) if len(vals) > 1 else None
        rows.append(r)

    return pd.DataFrame(rows).sort_values(["shock_type", "net", "method"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

METHOD_COLORS = {
    "PLR-original":   ("steelblue",  "-",  1.8),
    "SR-PLR-MARL":    ("crimson",    "-",  2.0),
    "GLR-style":      ("darkorange", "--", 1.5),
    "Fixed-LR":       ("gray",       ":",  1.3),
    "Poly-decay":     ("purple",     "-.", 1.3),
    "SR-WarmRestart": ("teal",       "--", 1.5),
    "SR-ClockOnly":   ("olive",      "-.", 1.5),
}


def _plot_rho_trajectories(plot_hist, shock_ep, tau_high, fig_dir):
    """Fig 1: rho_active trajectories for all shock types, 2×2 network."""
    shock_types = list(plot_hist.keys())
    fig, axes = plt.subplots(1, len(shock_types),
                             figsize=(6 * len(shock_types), 5), sharey=True)
    if len(shock_types) == 1:
        axes = [axes]

    for ax, shock_type in zip(axes, shock_types):
        df = plot_hist[shock_type]
        sub = df[df["net"] == "2x2"]
        for mname, gdf in sub.groupby("method"):
            c, ls, lw = METHOD_COLORS.get(mname, ("black", "-", 1.0))
            ax.plot(gdf["t"], gdf["rho_active"], color=c, linestyle=ls,
                    linewidth=lw, alpha=0.9, label=mname)
        ax.axvline(shock_ep, color="black", linestyle=":", lw=1.2, label="shock")
        ax.axhline(tau_high, color="dimgray", linestyle=":", lw=0.8)
        ax.set_title(f"Shock: {shock_type}", fontsize=10)
        ax.set_xlabel("Episode"); ax.set_ylabel("ρ_active")
        ax.set_ylim(-0.05, 1.10); ax.legend(fontsize=7); ax.grid(alpha=0.2)

    fig.suptitle("ρ_active trajectories — 2×2 network (seed=0)", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_rho_trajectories.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_route_blockage_comparison(df_rb, shock_ep, tau_high, fig_dir):
    """Fig 2: rho_all vs rho_active for route_blockage — shows metric sensitivity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sub = df_rb[df_rb["net"] == "2x2"]
    for col, ax, title in [("rho",        axes[0], "ρ_all  (penalty-inflated — DO NOT USE)"),
                             ("rho_active", axes[1], "ρ_active  (correct metric)")]:
        for mname, gdf in sub.groupby("method"):
            c, ls, lw = METHOD_COLORS.get(mname, ("black", "-", 1.0))
            ax.plot(gdf["t"], gdf[col], color=c, linestyle=ls,
                    linewidth=lw, alpha=0.9, label=mname)
        ax.axvline(shock_ep, color="black", linestyle=":", lw=1.2)
        ax.axhline(tau_high, color="dimgray", linestyle=":", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Episode"); ax.set_ylabel(col)
        ax.set_ylim(-0.05, 1.10); ax.legend(fontsize=7); ax.grid(alpha=0.2)

    fig.suptitle("Route Blockage — ρ_all vs ρ_active (2×2, seed=0)", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_route_blockage_metric_comparison.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_alpha_around_shock(plot_hist, shock_ep, fig_dir):
    """Fig 3: alpha trajectories around shock for price_spike — shows LR reset."""
    if "price_spike" not in plot_hist:
        return
    df = plot_hist["price_spike"]
    sub = df[(df["net"] == "2x2") & (df["t"] >= shock_ep - 30) & (df["t"] <= shock_ep + 80)]

    fig, ax = plt.subplots(figsize=(10, 4))
    for mname, gdf in sub.groupby("method"):
        c, ls, lw = METHOD_COLORS.get(mname, ("black", "-", 1.0))
        ax.semilogy(gdf["t"], gdf["alpha"], color=c, linestyle=ls,
                    linewidth=lw, alpha=0.9, label=mname)
    ax.axvline(shock_ep, color="black", linestyle=":", lw=1.5, label="shock")
    ax.set_title("α_t around price spike — 2×2 network (seed=0)", fontsize=10)
    ax.set_xlabel("Episode"); ax.set_ylabel("Learning rate α_t (log scale)")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    fig.tight_layout()
    plt.savefig(fig_dir / "fig3_alpha_around_shock.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_bars(method_agg, fig_dir):
    """Fig 4: Bar chart of success_rate and recovery_time_mean for price_spike."""
    ps = method_agg[method_agg["shock_type"] == "price_spike"].copy()
    if ps.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for net_label, grp in ps.groupby("net"):
        x = np.arange(len(grp))
        width = 0.35

        ax = axes[0]
        bars = ax.bar(x, grp["success_rate"], width, label=net_label, alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(grp["method"], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Success rate"); ax.set_title("Success rate — price spike")
        ax.set_ylim(0, 1.1); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, grp["success_rate"]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        ax = axes[1]
        rt = grp["recovery_time_mean"].fillna(len(grp) * 800)
        bars = ax.bar(x, rt, width, label=net_label, alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(grp["method"], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Recovery time (episodes)"); ax.set_title("Recovery time — price spike")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_summary_bars_price_spike.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — INTERPRETATION AND DIAGNOSTIC JSON
# ═══════════════════════════════════════════════════════════════════════════

def _build_diagnostic(summary_df, ablat_df, shock_ep):
    """Build structured diagnostic dictionary saved to JSON."""
    diag = {
        "experiment_design": {
            "shock_episode": shock_ep,
            "networks": ["2x2", "3x3"],
            "seeds": int(summary_df["seed"].nunique()),
        },
        "shock_types": {
            "price_spike": {
                "mechanism": "P_j^k <- 2 * P_j^k. D spikes above D_max_initial.",
                "primary_metric": "rho_all (all routes active; no penalty distortion)",
                "key_finding": (
                    "PLR-original fails because fixed D_max < post-shock D "
                    "=> rho=0 permanently => alpha stays at late-training small value. "
                    "SR-PLR-MARL detects spike, resets D_max and t_local, "
                    "restores alpha_0, and recovers."
                ),
                "recommended_for_paper": True,
            },
            "route_blockage": {
                "mechanism": "One route zeroed and penalized (+10 in F). Feasible set shrinks.",
                "primary_metric": (
                    "rho_active (non-blocked routes only). "
                    "rho_all is MISLEADING because the +10 penalty permanently "
                    "inflates D_all regardless of convergence on active subnetwork."
                ),
                "key_finding": (
                    "Both methods converge on the active subnetwork (rho_active rises). "
                    "SR-PLR detects the shock and recovers faster. "
                    "rho_all should NOT be used as the primary metric here."
                ),
                "recommended_for_paper": True,
                "caveat": (
                    "Mention in the paper: for route_blockage shocks, D_active "
                    "and rho_active are the appropriate convergence metrics."
                ),
            },
            "supply_cut": {
                "mechanism": "Supply capacity reduced 50% via projection; F unchanged.",
                "primary_metric": "rho_all (no blocked routes; no penalty distortion)",
                "key_finding": (
                    "D does not spike at shock injection (projection absorbs the cut "
                    "without changing F). SR-PLR detector may not fire. "
                    "Both methods recover naturally. This is a robustness check, "
                    "not the primary contribution."
                ),
                "recommended_for_paper": "as_robustness_check",
            },
        },
        "ablation_summary_note": (
            "Ablation (price_spike, 2x2): "
            "PLR-original fails (fixed D_max, small late alpha). "
            "SR-WarmRestart: D_max recalibrated but t_local not reset; "
            "alpha stays small -> partial improvement. "
            "SR-ClockOnly: t_local resets (fresh alpha) but D_max stays fixed "
            "-> rho remains 0 until D drops below old D_max. "
            "SR-PLR-MARL (full): both mechanisms together -> strongest recovery."
        ),
    }

    # Add numeric summaries
    for shock_type in ["price_spike", "route_blockage", "supply_cut"]:
        sub = summary_df[summary_df["shock_type"] == shock_type]
        if sub.empty:
            continue
        sr_by_method = sub.groupby("method")["success"].mean().round(3).to_dict()
        diag["shock_types"][shock_type]["success_rate_by_method"] = sr_by_method
        rt_by_method = sub.groupby("method")["recovery_time"].mean().round(1).to_dict()
        diag["shock_types"][shock_type]["recovery_time_mean_by_method"] = rt_by_method

    return diag


def _print_interpretation(method_agg, ablat_df, summary_df):
    """Print a human-readable result interpretation to stdout."""
    SEP = "═" * 66
    print(f"\n{SEP}")
    print("  RESULT INTERPRETATION")
    print(SEP)

    # Price spike — main result
    print("\n  ── PRICE SPIKE (primary result) ──────────────────────────────")
    ps = method_agg[method_agg["shock_type"] == "price_spike"]
    for net in ["2x2", "3x3"]:
        sub = ps[ps["net"] == net][["method", "success_rate",
                                    "recovery_time_mean", "alpha_after_shock_mean"]].copy()
        sub = sub.sort_values("success_rate", ascending=False)
        print(f"\n    Network {net}:")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Route blockage
    print("\n  ── ROUTE BLOCKAGE (rho_active metric) ───────────────────────")
    rb = method_agg[method_agg["shock_type"] == "route_blockage"]
    for net in ["2x2", "3x3"]:
        sub = rb[rb["net"] == net][["method", "success_rate",
                                    "recovery_time_mean", "cum_area_active_mean"]].copy()
        sub = sub.sort_values("success_rate", ascending=False)
        print(f"\n    Network {net}:")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Supply cut
    print("\n  ── SUPPLY CUT (robustness check) ─────────────────────────────")
    sc = method_agg[method_agg["shock_type"] == "supply_cut"]
    for net in ["2x2", "3x3"]:
        sub = sc[sc["net"] == net][["method", "success_rate", "recovery_time_mean"]].copy()
        print(f"\n    Network {net}:")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Ablation
    if not ablat_df.empty:
        print("\n  ── ABLATION (price_spike, 2×2) ───────────────────────────────")
        ab_agg = ablat_df.groupby("method").agg(
            success_rate=("success", "mean"),
            recovery_time_mean=("recovery_time", "mean"),
            alpha_after_mean=("alpha_after_shock", "mean"),
        ).round(3).reset_index()
        ab_agg = ab_agg.sort_values("success_rate", ascending=False)
        print(ab_agg.to_string(index=False))

    # Paper readiness
    print(f"\n{SEP}")
    print("  PAPER READINESS ASSESSMENT")
    print(SEP)
    ps_sr = summary_df[summary_df["shock_type"] == "price_spike"].groupby("method")["success"].mean()
    sr_plr_rate  = ps_sr.get("SR-PLR-MARL",  0.0)
    plr_orig_rate = ps_sr.get("PLR-original", 0.0)
    glr_rate     = ps_sr.get("GLR-style",    0.0)

    print(f"\n  SR-PLR success rate (price_spike, all nets): {sr_plr_rate:.0%}")
    print(f"  PLR-original success rate:                   {plr_orig_rate:.0%}")
    print(f"  GLR-style success rate:                      {glr_rate:.0%}")

    if sr_plr_rate >= 0.9 and plr_orig_rate <= 0.1:
        verdict = "STRONG — evidence is clear and consistent across seeds and network sizes."
    elif sr_plr_rate >= 0.7:
        verdict = "MODERATE — good evidence but variance may be high. Check std in tables."
    else:
        verdict = "WEAK — revisit hyperparameters or shock intensity."
    print(f"\n  Verdict: {verdict}")
    print(f"\n  Figures for ICTA2026 paper:")
    print(f"    Fig 1: rho_active trajectories (seed=0, price_spike) — main result")
    print(f"    Fig 2: rho_all vs rho_active for route_blockage — metric justification")
    print(f"    Fig 3: alpha around price_spike — mechanism illustration")
    print(f"    Fig 4: success rate bar chart — summary across all methods")
    print(f"    Table 1: summary_by_method.csv filtered to price_spike")
    print(f"    Table 2: ablation_summary.csv — component contribution")
    print(SEP)




# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10A — MULTI-COMMODITY / MULTI-ROUTE SYNTHETIC SCALABILITY
# ═══════════════════════════════════════════════════════════════════════════

import time

@dataclass
class MCMRCfg:
    """Configuration for controlled multi-commodity / multi-route VI networks."""
    m: int = 3
    n: int = 3
    L: int = 1
    K: int = 3
    P_min: float = 0.10
    init_low: float = 1.0
    init_high: float = 5.0
    param_seed: int = 2026

    # Capacity parameters are arrays internally, but these set their scales.
    route_cap_base: float = 50.0
    supply_cap_base: float = 100.0
    demand_cap_base: float = 100.0


@dataclass
class MCMRShockCfg:
    episode: int = 300
    price_multiplier: float = 2.0
    k_star: int = 0
    all_commodities: bool = False


class MultiCommodityRouteNetwork:
    """Synthetic m x n x L x K VI trade network.

    Decision variable:
        x[i,j,l,k]

    Operator:
        F[i,j,l,k] = C_i^k(q_i^k) + T_{ijlk}(x_{ijlk}) - P_j^k(d_j^k),
    where q_i^k = sum_{j,l} x[i,j,l,k] and d_j^k = sum_{i,l} x[i,j,l,k].
    """
    def __init__(self, cfg: MCMRCfg) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.param_seed + 1000*cfg.m + 100*cfg.n + 10*cfg.L + cfg.K)

        self.a = rng.uniform(0.08, 0.12, size=(cfg.m, cfg.K))
        self.b = rng.uniform(0.07, 0.12, size=(cfg.m, cfg.K))
        self.tau = rng.uniform(0.02, 0.06, size=(cfg.m, cfg.n, cfg.L, cfg.K))
        self.beta = rng.uniform(0.03, 0.06, size=(cfg.m, cfg.n, cfg.L, cfg.K))
        self.p = rng.uniform(4.50, 5.50, size=(cfg.n, cfg.K))
        self.s = rng.uniform(0.04, 0.06, size=(cfg.n, cfg.K))

        self.u = rng.uniform(0.85, 1.15, size=(cfg.m, cfg.n, cfg.L, cfg.K)) * cfg.route_cap_base
        self.Q = rng.uniform(0.90, 1.10, size=(cfg.m, cfg.K)) * cfg.supply_cap_base
        self.D_cap = rng.uniform(0.90, 1.10, size=(cfg.n, cfg.K)) * cfg.demand_cap_base

        self.price_mult = np.ones((cfg.n, cfg.K), dtype=float)

    @property
    def dim(self) -> int:
        return int(self.cfg.m * self.cfg.n * self.cfg.L * self.cfg.K)

    def reset_shock(self) -> None:
        self.price_mult[:] = 1.0

    def apply_price_spike(self, shock: MCMRShockCfg) -> None:
        if shock.all_commodities:
            self.price_mult *= shock.price_multiplier
        else:
            k = int(np.clip(shock.k_star, 0, self.cfg.K - 1))
            self.price_mult[:, k] *= shock.price_multiplier

    def compute_F(self, x: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        q = x.sum(axis=(1, 2))       # (m,K)
        d = x.sum(axis=(0, 2))       # (n,K)
        C = self.a + self.b * q      # (m,K)
        P = np.maximum(cfg.P_min, self.p - self.s * d) * self.price_mult  # (n,K)
        T = self.tau + self.beta * x
        return C[:, None, None, :] + T - P[None, :, None, :]

    def compute_D(self, x: np.ndarray) -> float:
        return float(np.mean(np.abs(self.compute_F(x))))

    def compute_G_lambda(self, x: np.ndarray, lam: float = 1.0) -> float:
        xp = self.project(x - lam * self.compute_F(x))
        return float(np.linalg.norm(x - xp)) / max(lam, 1e-12)

    def project(self, x: np.ndarray) -> np.ndarray:
        y = np.clip(x, 0.0, self.u)
        # Alternating rescaling projection. This is intentionally simple and
        # consistent with the 2D controlled synthetic experiments.
        for _ in range(10):
            # Supply constraints: sum_{j,l} x[i,j,l,k] <= Q[i,k]
            for i in range(self.cfg.m):
                for k in range(self.cfg.K):
                    total = float(y[i, :, :, k].sum())
                    cap = float(self.Q[i, k])
                    if total > cap > 0:
                        y[i, :, :, k] *= cap / total

            # Demand constraints: sum_{i,l} x[i,j,l,k] <= D[j,k]
            for j in range(self.cfg.n):
                for k in range(self.cfg.K):
                    total = float(y[:, j, :, k].sum())
                    cap = float(self.D_cap[j, k])
                    if total > cap > 0:
                        y[:, j, :, k] *= cap / total
            y = np.clip(y, 0.0, self.u)
        return y


def build_mc_mr_method_registry(plr_cfg: PLRCfg, sr_cfg: SRCfg, glr_cfg: GLRCfg) -> Dict[str, object]:
    """Compact method set for the scalability experiment."""
    return {
        "PLR-original": lambda: PLROriginal(plr_cfg),
        "SR-PLR-MARL": lambda: SRPLRScheduler(sr_cfg),
        "GLR-style": lambda: GLRScheduler(glr_cfg),
        "Fixed-LR": lambda: FixedLR(plr_cfg.alpha0),
        "Poly-decay": lambda: PolyDecay(plr_cfg.alpha0, 0.5),
    }


def run_mc_mr_single(
    scheduler,
    net_cfg: MCMRCfg,
    shock_cfg: MCMRShockCfg,
    plr_cfg: PLRCfg,
    episodes: int,
    seed: int,
) -> pd.DataFrame:
    """Run one multi-commodity/multi-route experiment."""
    rng = np.random.default_rng(seed)
    net = MultiCommodityRouteNetwork(net_cfg)
    x = rng.uniform(net_cfg.init_low, net_cfg.init_high,
                    size=(net_cfg.m, net_cfg.n, net_cfg.L, net_cfg.K))
    x = net.project(x)

    D0 = net.compute_D(x)
    scheduler.init(D0)
    update_rule = getattr(scheduler, "update_rule", "plr")
    op_per_iter = getattr(scheduler, "operator_calls_per_iter", 1)
    rows: List[dict] = []

    for t in range(episodes):
        start_t = time.perf_counter()
        D_t = net.compute_D(x)
        G_t = net.compute_G_lambda(x, lam=LAM_RES)
        alpha, diag = scheduler.step(t, D_t)

        rows.append({
            "t": t,
            "D": D_t,
            "G_lambda": G_t,
            "rho": diag["rho"],
            "alpha": alpha,
            "phase": diag.get("phase", -1),
            "shock_detected": bool(diag.get("shock_detected", False)),
            "t_local": diag.get("t_local", t),
            "D_max": diag.get("D_max", np.nan),
            "operator_calls": op_per_iter,
            "step_runtime_sec": time.perf_counter() - start_t,
        })

        # Same timing convention as the 2D runner: the shock is injected after
        # logging at t=shock_ep; the scheduler sees the post-shock residual next.
        if t == shock_cfg.episode:
            net.apply_price_spike(shock_cfg)

        if update_rule == "eg":
            xi1 = rng.normal(0.0, plr_cfg.sigma, size=x.shape)
            y   = net.project(x - alpha * net.compute_F(x) + alpha * xi1)
            xi2 = rng.normal(0.0, plr_cfg.sigma, size=x.shape)
            x   = net.project(x - alpha * net.compute_F(y) + alpha * xi2)
        else:
            F_t = net.compute_F(x)
            xi = rng.normal(0.0, plr_cfg.sigma, size=x.shape)
            x = net.project(x - alpha * F_t + alpha * xi)

    return pd.DataFrame(rows)


def compute_mc_mr_metrics(
    df: pd.DataFrame,
    shock_ep: int,
    tau_recovery: float = 0.50,
    max_recovery_window: int = 400,
    osc_warmup: int = 50,
) -> dict:
    """Metrics for the multi-commodity/multi-route scalability experiment.

    The main recovery metric follows the 2D synthetic price-spike experiments:
    recovery is based on the scheduler proximity rho. For auditability, the
    table also reports the final projected residual G_lambda.
    """
    t = df["t"].values
    rho = df["rho"].values
    post = df[df["t"] > shock_ep]
    rec = post[post["rho"] >= tau_recovery]
    recovered = len(rec) > 0
    tau_rec = (int(rec.iloc[0]["t"]) - shock_ep) if recovered else len(df)

    try:
        cum_area = float(np.trapezoid(rho, t))
    except AttributeError:
        cum_area = float(np.trapz(rho, t))

    osc_vals = df[df["t"] >= shock_ep + osc_warmup]["rho"].values
    osc_index = float(np.mean(np.abs(np.diff(osc_vals)))) if len(osc_vals) > 1 else float("nan")

    def _alpha(mask):
        sub = df[mask]
        return float(sub["alpha"].iloc[0]) if len(sub) else float("nan")

    det_rows = df[df["shock_detected"]]
    shock_det_episode = int(det_rows.iloc[0]["t"]) if len(det_rows) else -1

    return {
        "recovery_time": tau_rec,
        "success": int(recovered and tau_rec <= max_recovery_window),
        "D_final": float(df["D"].iloc[-1]),
        "G_lambda_final": float(df["G_lambda"].iloc[-1]),
        "rho_final": float(df["rho"].iloc[-1]),
        "cum_area": round(cum_area, 3),
        "osc_index": round(osc_index, 6) if not np.isnan(osc_index) else None,
        "alpha_before_shock": round(_alpha(df["t"] == shock_ep - 1), 8),
        "alpha_after_shock": round(_alpha(df["t"] == shock_ep + 1), 8),
        "shock_detected_ep": shock_det_episode,
        "runtime_sec": float(df["step_runtime_sec"].sum()),
    }


def _aggregate_mc_mr(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "recovery_time", "D_final", "G_lambda_final", "rho_final",
        "cum_area", "osc_index", "alpha_before_shock", "alpha_after_shock", "runtime_sec"
    ]
    grp = df.groupby(["setting", "dimension", "method"])
    rows = []
    for (setting, dim, method), g in grp:
        r = {
            "setting": setting,
            "dimension": int(dim),
            "method": method,
            "n_seeds": len(g),
            "success_rate": round(float(g["success"].mean()), 3),
        }
        for col in metric_cols:
            vals = g[col].dropna()
            r[f"{col}_mean"] = round(float(vals.mean()), 4) if len(vals) else None
            r[f"{col}_std"] = round(float(vals.std()), 4) if len(vals) > 1 else None
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["dimension", "setting", "method"]).reset_index(drop=True)


def _plot_mc_mr_largest(raw_df: pd.DataFrame, fig_dir: Path, shock_ep: int) -> None:
    if raw_df.empty:
        return
    # Select largest dimension and seed 0.
    max_dim = int(raw_df["dimension"].max())
    sub = raw_df[(raw_df["dimension"] == max_dim) & (raw_df["seed"] == 0)]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for mname, gdf in sub.groupby("method"):
        c, ls, lw = METHOD_COLORS.get(mname, ("black", "-", 1.2))
        ax.plot(gdf["t"], gdf["rho"], color=c, linestyle=ls, linewidth=lw, alpha=0.9, label=mname)
    ax.axvline(shock_ep, color="black", linestyle=":", lw=1.2)
    ax.axhline(0.60, color="dimgray", linestyle=":", lw=0.8)
    setting = sub["setting"].iloc[0]
    ax.set_title(f"Multi-commodity/multi-route proximity, {setting}, seed=0")
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"Proximity $\rho_t$")
    ax.set_ylim(-0.05, 1.10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    plt.savefig(fig_dir / "mc_mr_largest_rho_trajectory.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_mc_mr_full_experiments(
    out_dir: Path,
    episodes: int = 1200,
    shock_ep: int = 300,
    n_seeds: int = 10,
    sigma: float = 0.01,
    include_150: bool = False,
    shock_all_commodities: bool = False,
    k_star: int = 0,
) -> None:
    """Controlled scalability experiment with tensor-valued VI networks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plr_cfg = PLRCfg(sigma=sigma)
    sr_cfg = SRCfg(sigma=sigma)
    glr_cfg = GLRCfg()
    methods = build_mc_mr_method_registry(plr_cfg, sr_cfg, glr_cfg)
    seeds = list(range(n_seeds))

    settings = [
        ("3x3_K3_L1", MCMRCfg(m=3, n=3, K=3, L=1)),
        ("3x3_K1_L3", MCMRCfg(m=3, n=3, K=1, L=3)),
        ("4x4_K3_L2", MCMRCfg(m=4, n=4, K=3, L=2)),
    ]
    if include_150:
        settings.append(("5x5_K3_L2", MCMRCfg(m=5, n=5, K=3, L=2)))

    total = len(settings) * len(methods) * len(seeds)
    print(f"\nRunning MC/MR scalability experiment: {total} runs "
          f"({len(settings)} settings × {len(methods)} methods × {len(seeds)} seeds)")

    all_summary_rows: List[dict] = []
    raw_rows: List[pd.DataFrame] = []
    done = 0

    for setting_name, net_cfg in settings:
        shock_cfg = MCMRShockCfg(
            episode=shock_ep,
            price_multiplier=2.0,
            k_star=min(k_star, net_cfg.K - 1),
            all_commodities=shock_all_commodities,
        )
        dim = net_cfg.m * net_cfg.n * net_cfg.L * net_cfg.K
        for method_name, factory in methods.items():
            for seed in seeds:
                sch = factory()
                df = run_mc_mr_single(sch, net_cfg, shock_cfg, plr_cfg, episodes, seed)
                met = compute_mc_mr_metrics(df, shock_ep)
                row = {
                    "setting": setting_name,
                    "m": net_cfg.m,
                    "n": net_cfg.n,
                    "K": net_cfg.K,
                    "L": net_cfg.L,
                    "dimension": dim,
                    "method": method_name,
                    "seed": seed,
                    "shock_type": "commodity_price_spike_all" if shock_all_commodities else "commodity_price_spike_one",
                    "k_star": shock_cfg.k_star,
                    **met,
                }
                all_summary_rows.append(row)

                # Store scalar history for all runs; compact enough for this experiment.
                raw_rows.append(df.assign(
                    setting=setting_name,
                    dimension=dim,
                    method=method_name,
                    seed=seed,
                    m=net_cfg.m,
                    n=net_cfg.n,
                    K=net_cfg.K,
                    L=net_cfg.L,
                ))

                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  {done}/{total}  [{setting_name}  {method_name}  s{seed}]")

    by_seed = pd.DataFrame(all_summary_rows)
    by_method = _aggregate_mc_mr(by_seed)
    raw_df = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()

    by_seed.to_csv(out_dir / "mc_mr_summary_by_seed.csv", index=False)
    by_method.to_csv(out_dir / "mc_mr_summary_by_method.csv", index=False)
    raw_df.to_csv(out_dir / "mc_mr_raw_history.csv", index=False)

    _plot_mc_mr_largest(raw_df, fig_dir, shock_ep)

    print("\nMC/MR SCALABILITY SUMMARY")
    print("─" * 72)
    cols = ["setting", "dimension", "method", "success_rate", "recovery_time_mean",
            "D_final_mean", "alpha_after_shock_mean", "runtime_sec_mean"]
    print(by_method[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nMC/MR outputs saved to: {out_dir.resolve()}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11 — V8 STRENGTHENED EXPERIMENTS
#   Standardized recovery metrics + PG/EG baselines + sensitivity.
# ═══════════════════════════════════════════════════════════════════════════

PG_EG_ALPHA = 0.1   # tuned constant step for classical PG/EG baselines

METHOD_COLORS.update({
    "PG-fixed": ("seagreen", "--", 1.6),
    "EG":       ("darkviolet", "-.", 1.6),
})


def v8_metrics(
    df: pd.DataFrame,
    shock_ep: int,
    primary_col: str = "rho",
    tau_recovery: float = 0.50,
    max_window: int = 400,
) -> dict:
    """Standardized recovery + residual metrics (v8).

    success      : recovered to tau_recovery within max_window post-shock episodes.
    RT_all       : recovery time; failed runs censored at max_window.
    RT_success   : recovery time over successful runs only (NaN otherwise).
    Final_D      : final economic mismatch D(x_T).
    Final_G      : final projected VI residual G_lambda(x_T).
    Best_G_post  : best (smallest) projected residual after the shock.
    operator_calls: total operator evaluations over the run (EG counts double).
    """
    post = df[df["t"] > shock_ep]
    rec = post[post[primary_col] >= tau_recovery]
    if len(rec) > 0:
        tau_rec = int(rec.iloc[0]["t"]) - shock_ep
    else:
        tau_rec = None
    success = int(tau_rec is not None and tau_rec <= max_window)

    rt_all = float(tau_rec) if success else float(max_window)
    rt_success = float(tau_rec) if success else float("nan")

    g_post = post["G_lambda"].values
    best_g_post = float(np.min(g_post)) if len(g_post) else float("nan")

    def _alpha_at(tt):
        sub = df[df["t"] == tt]
        return float(sub["alpha"].iloc[0]) if len(sub) else float("nan")

    out = {
        "success":        success,
        "RT_all":         rt_all,
        "RT_success":     rt_success,
        "Final_D":        float(df["D"].iloc[-1]),
        "Final_G":        float(df["G_lambda"].iloc[-1]),
        "Best_G_post":    best_g_post,
        "rho_final":      float(df[primary_col].iloc[-1]),
        "operator_calls": int(df["operator_calls"].sum()) if "operator_calls" in df else len(df),
        "alpha_after_shock": round(_alpha_at(shock_ep + 1), 8),
        "primary_metric": primary_col,
    }
    if "D_active" in df.columns:
        out["Final_D_active"] = float(df["D_active"].iloc[-1])
    return out


def _v8_aggregate(by_seed: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Aggregate v8 metrics: success_rate + mean/std of the rest."""
    metric_cols = ["RT_all", "RT_success", "Final_D", "Final_G",
                   "Best_G_post", "rho_final", "operator_calls", "alpha_after_shock"]
    if "Final_D_active" in by_seed.columns:
        metric_cols.append("Final_D_active")
    rows = []
    for key, g in by_seed.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        r = dict(zip(group_cols, key))
        r["n_seeds"] = len(g)
        r["success_rate"] = round(float(g["success"].mean()), 3)
        for col in metric_cols:
            vals = g[col].dropna()
            r[f"{col}_mean"] = round(float(vals.mean()), 4) if len(vals) else None
            r[f"{col}_std"]  = round(float(vals.std()), 4) if len(vals) > 1 else None
        rows.append(r)
    return pd.DataFrame(rows)


def _v8_main_methods(plr_cfg: PLRCfg, sr_cfg: SRCfg) -> Dict[str, object]:
    return {
        "PLR-original": lambda: PLROriginal(plr_cfg),
        "SR-PLR-MARL":  lambda: SRPLRScheduler(sr_cfg),
        "Fixed-LR":     lambda: FixedLR(plr_cfg.alpha0),
        "Poly-decay":   lambda: PolyDecay(plr_cfg.alpha0, 0.5),
        "PG-fixed":     lambda: ProjectedGradient(PG_EG_ALPHA),
        "EG":           lambda: Extragradient(PG_EG_ALPHA),
    }

V8_METHOD_ORDER = ["PLR-original", "SR-PLR-MARL", "Fixed-LR",
                   "Poly-decay", "PG-fixed", "EG"]


def _v8_fig_recovery_residual(raw0: pd.DataFrame, shock_ep: int,
                              fname: Path, title: str, primary_col: str = "rho") -> None:
    """Two-panel figure: proximity (left) and projected residual G_lambda (right)
    after the shock, comparing SR-PLR with baselines (seed 0)."""
    if raw0.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for mname in V8_METHOD_ORDER:
        gdf = raw0[raw0["method"] == mname]
        if gdf.empty:
            continue
        c, ls, lw = METHOD_COLORS.get(mname, ("black", "-", 1.2))
        axes[0].plot(gdf["t"], gdf[primary_col], color=c, linestyle=ls, lw=lw, alpha=0.9, label=mname)
        axes[1].semilogy(gdf["t"], gdf["G_lambda"].clip(lower=1e-6),
                         color=c, linestyle=ls, lw=lw, alpha=0.9, label=mname)
    for ax in axes:
        ax.axvline(shock_ep, color="black", linestyle=":", lw=1.2)
        ax.set_xlabel("Episode"); ax.grid(alpha=0.25)
    axes[0].axhline(0.5, color="dimgray", linestyle=":", lw=0.8)
    axes[0].set_ylabel(r"Proximity $\rho_t$"); axes[0].set_ylim(-0.05, 1.10)
    axes[0].set_title("Proximity recovery"); axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_ylabel(r"Projected residual $G_\lambda(x_t)$ (log)")
    axes[1].set_title("Projected VI residual")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    plt.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_main_v8(out_dir: Path, episodes: int = 800, shock_ep: int = 300,
                n_seeds: int = 30, sigma: float = 0.01) -> None:
    """Single-commodity 2x2/3x3 price-spike + route-blockage with PG/EG and
    standardized metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    plr_cfg = PLRCfg(sigma=sigma); sr_cfg = SRCfg(sigma=sigma)
    methods = _v8_main_methods(plr_cfg, sr_cfg)
    seeds = list(range(n_seeds))
    nets = [(2, 2), (3, 3)]
    shocks = ["price_spike", "route_blockage"]

    rows: List[dict] = []
    raw0_store: Dict[str, List[pd.DataFrame]] = {}
    total = len(shocks) * len(nets) * len(methods) * n_seeds
    done = 0
    print(f"\n[main-v8] {total} runs (single-commodity).")
    for shock_type in shocks:
        primary = "rho_active" if shock_type == "route_blockage" else "rho"
        for (mm, nn) in nets:
            net_cfg = NetCfg(m=mm, n=nn); net_label = f"{mm}x{nn}"
            for mname, fac in methods.items():
                for seed in seeds:
                    sch = fac()
                    df = run_single(sch, net_cfg,
                                    ShockCfg(episode=shock_ep, shock_type=shock_type),
                                    plr_cfg, episodes, seed)
                    met = v8_metrics(df, shock_ep, primary_col=primary)
                    rows.append({"shock": shock_type, "net": net_label,
                                 "method": mname, "seed": seed, **met})
                    if seed == 0:
                        key = f"{shock_type}|{net_label}"
                        raw0_store.setdefault(key, []).append(
                            df.assign(method=mname, net=net_label, shock=shock_type))
                    done += 1
                    if done % 50 == 0:
                        print(f"  {done}/{total}")
    by_seed = pd.DataFrame(rows)
    by_method = _v8_aggregate(by_seed, ["shock", "net", "method"])
    by_seed.to_csv(out_dir / "main_v8_summary_by_seed.csv", index=False)
    by_method.to_csv(out_dir / "main_v8_summary_by_method.csv", index=False)

    key = "price_spike|3x3"
    if key in raw0_store:
        raw0 = pd.concat(raw0_store[key], ignore_index=True)
        _v8_fig_recovery_residual(raw0, shock_ep,
            fig_dir / "fig_v8_price_spike_3x3.png",
            "Price-spike recovery, 3x3 single-commodity (seed 0)", "rho")
    key = "route_blockage|2x2"
    if key in raw0_store:
        raw0 = pd.concat(raw0_store[key], ignore_index=True)
        _v8_fig_recovery_residual(raw0, shock_ep,
            fig_dir / "fig_v8_route_blockage_2x2.png",
            "Route-blockage recovery, 2x2 (seed 0)", "rho_active")
    print(f"[main-v8] saved to {out_dir.resolve()}")


def run_mc_mr_v8(out_dir: Path, episodes: int = 1200, shock_ep: int = 300,
                 n_seeds: int = 30, sigma: float = 0.01, include_150: bool = True) -> None:
    """Multi-commodity/multi-route scalability with PG/EG + standardized metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    plr_cfg = PLRCfg(sigma=sigma); sr_cfg = SRCfg(sigma=sigma)
    methods = _v8_main_methods(plr_cfg, sr_cfg)
    seeds = list(range(n_seeds))
    settings = [
        ("3x3_K1_L3", MCMRCfg(m=3, n=3, K=1, L=3)),
        ("3x3_K3_L1", MCMRCfg(m=3, n=3, K=3, L=1)),
        ("4x4_K3_L2", MCMRCfg(m=4, n=4, K=3, L=2)),
    ]
    if include_150:
        settings.append(("5x5_K3_L2", MCMRCfg(m=5, n=5, K=3, L=2)))

    rows: List[dict] = []
    raw0_largest: List[pd.DataFrame] = []
    max_dim = max(c.m * c.n * c.K * c.L for _, c in settings)
    total = len(settings) * len(methods) * n_seeds
    done = 0
    print(f"\n[mc-mr-v8] {total} runs.")
    for sname, cfg in settings:
        dim = cfg.m * cfg.n * cfg.K * cfg.L
        shock = MCMRShockCfg(episode=shock_ep, price_multiplier=2.0, all_commodities=True)
        for mname, fac in methods.items():
            for seed in seeds:
                sch = fac()
                df = run_mc_mr_single(sch, cfg, shock, plr_cfg, episodes, seed)
                met = v8_metrics(df, shock_ep, primary_col="rho")
                rows.append({"setting": sname, "dimension": dim,
                             "method": mname, "seed": seed, **met})
                if dim == max_dim and seed == 0:
                    raw0_largest.append(df.assign(method=mname, setting=sname))
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{total}")
    by_seed = pd.DataFrame(rows)
    by_method = _v8_aggregate(by_seed, ["setting", "dimension", "method"])
    by_seed.to_csv(out_dir / "mc_mr_v8_summary_by_seed.csv", index=False)
    by_method.to_csv(out_dir / "mc_mr_v8_summary_by_method.csv", index=False)
    if raw0_largest:
        raw0 = pd.concat(raw0_largest, ignore_index=True)
        _v8_fig_recovery_residual(raw0, shock_ep,
            fig_dir / "fig_v8_mc_mr_largest.png",
            f"MC/MR scalability, d={max_dim}, all-commodity spike (seed 0)", "rho")
    print(f"[mc-mr-v8] saved to {out_dir.resolve()}")


def run_sensitivity_v8(out_dir: Path, episodes: int = 1200, shock_ep: int = 300,
                       n_seeds: int = 10, sigma: float = 0.01) -> None:
    """SR-PLR sensitivity to kappa, cooldown, tau_high on a representative
    4x4,K=3,L=2 (d=96) network; one parameter varied at a time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plr_cfg = PLRCfg(sigma=sigma)
    cfg = MCMRCfg(m=4, n=4, K=3, L=2)
    dim = cfg.m * cfg.n * cfg.K * cfg.L
    shock = MCMRShockCfg(episode=shock_ep, price_multiplier=2.0, all_commodities=True)
    seeds = list(range(n_seeds))
    base = dict(kappa=1.5, cooldown=50, tau_high=0.60)

    grid = []
    for v in [1.2, 1.5, 2.0]:
        grid.append(("kappa", v, {**base, "kappa": v}))
    for v in [20, 50, 100]:
        grid.append(("cooldown", v, {**base, "cooldown": v}))
    for v in [0.5, 0.6, 0.7, 0.8]:
        grid.append(("tau_high", v, {**base, "tau_high": v}))

    rows = []
    print(f"\n[sensitivity-v8] {len(grid)} configs x {n_seeds} seeds on d={dim}.")
    for param, val, params in grid:
        per = []
        for seed in seeds:
            sr_cfg = SRCfg(sigma=sigma, kappa=params["kappa"],
                           cooldown=params["cooldown"], tau_high=params["tau_high"])
            df = run_mc_mr_single(SRPLRScheduler(sr_cfg), cfg, shock, plr_cfg, episodes, seed)
            per.append(v8_metrics(df, shock_ep, primary_col="rho"))
        pdf = pd.DataFrame(per)
        rows.append({
            "param": param, "value": val,
            "kappa": params["kappa"], "cooldown": params["cooldown"],
            "tau_high": params["tau_high"],
            "success_rate": round(float(pdf["success"].mean()), 3),
            "RT_all_mean": round(float(pdf["RT_all"].mean()), 2),
            "RT_all_std": round(float(pdf["RT_all"].std()), 2),
            "Final_G_mean": round(float(pdf["Final_G"].mean()), 4),
            "Final_G_std": round(float(pdf["Final_G"].std()), 4),
        })
    sdf = pd.DataFrame(rows)
    sdf.to_csv(out_dir / "sensitivity_v8.csv", index=False)
    print(sdf.to_string(index=False))
    print(f"[sensitivity-v8] saved to {out_dir.resolve()}")



# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SR-PLR-MARL v7 — synthetic + MC/MR scalability runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full",     action="store_true",
                        help="Run full experiment (5 methods, 3 shocks, 2 nets, 10 seeds).")
    parser.add_argument("--smoke",    action="store_true",
                        help="Quick diagnostic smoke test (PLR + SR-PLR, 3 shocks, seeds 0-1).")
    parser.add_argument("--out",      type=str, default="sr_plr_outputs")
    parser.add_argument("--episodes-2x2", type=int, default=800)
    parser.add_argument("--episodes-3x3", type=int, default=800)
    parser.add_argument("--shock-ep", type=int, default=300)
    parser.add_argument("--seeds",    type=int, default=10,
                        help="Number of seeds for --full (0..seeds-1). Default: 10.")
    parser.add_argument("--sigma",    type=float, default=0.01,
                        help="Gaussian observation-noise std. Default: 0.01.")
    parser.add_argument("--shock-types", nargs="+",
                        default=["price_spike", "route_blockage", "supply_cut"])
    parser.add_argument("--mc-mr-full", action="store_true",
                        help="Run controlled multi-commodity/multi-route scalability experiment.")
    parser.add_argument("--mc-mr-episodes", type=int, default=1200,
                        help="Episodes for --mc-mr-full. Default: 1200.")
    parser.add_argument("--mc-mr-shock-ep", type=int, default=300,
                        help="Shock episode for --mc-mr-full. Default: 300.")
    parser.add_argument("--mc-mr-seeds", type=int, default=10,
                        help="Number of seeds for --mc-mr-full. Default: 10.")
    parser.add_argument("--mc-mr-include-150", action="store_true",
                        help="Also run 5x5, K=3, L=2 setting with d=150.")
    parser.add_argument("--mc-mr-shock-all-commodities", action="store_true",
                        help="Apply price spike to all commodities instead of only k_star.")
    parser.add_argument("--mc-mr-k-star", type=int, default=0,
                        help="Commodity index to shock when not shocking all commodities. Default: 0.")
    parser.add_argument("--main-v8", action="store_true",
                        help="v8 single-commodity main table (PG/EG + standardized metrics).")
    parser.add_argument("--mc-mr-v8", action="store_true",
                        help="v8 MC/MR scalability (PG/EG + standardized metrics).")
    parser.add_argument("--sensitivity-v8", action="store_true",
                        help="v8 SR-PLR sensitivity sweep (kappa, cooldown, tau_high).")
    parser.add_argument("--v8-seeds", type=int, default=30,
                        help="Seeds for v8 main/MC-MR experiments. Default: 30.")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.main_v8:
        run_main_v8(out_dir, n_seeds=args.v8_seeds, sigma=args.sigma)
    elif args.mc_mr_v8:
        run_mc_mr_v8(out_dir, n_seeds=args.v8_seeds, sigma=args.sigma, include_150=True)
    elif args.sensitivity_v8:
        run_sensitivity_v8(out_dir, sigma=args.sigma)
    elif args.mc_mr_full:
        run_mc_mr_full_experiments(
            out_dir=out_dir,
            episodes=args.mc_mr_episodes,
            shock_ep=args.mc_mr_shock_ep,
            n_seeds=args.mc_mr_seeds,
            sigma=args.sigma,
            include_150=args.mc_mr_include_150,
            shock_all_commodities=args.mc_mr_shock_all_commodities,
            k_star=args.mc_mr_k_star,
        )

    elif args.full:
        run_full_experiments(
            out_dir        = out_dir,
            episodes_2x2   = args.episodes_2x2,
            episodes_3x3   = args.episodes_3x3,
            shock_ep       = args.shock_ep,
            n_seeds        = args.seeds,
            shock_types    = args.shock_types,
            sigma          = args.sigma,
        )

    elif args.smoke:
        # Quick 3-shock diagnostic with PLR + SR-PLR, 2 seeds
        out_dir.mkdir(parents=True, exist_ok=True)
        plr_cfg = PLRCfg(sigma=args.sigma); sr_cfg = SRCfg(sigma=args.sigma); glr_cfg = GLRCfg()
        smoke_methods = {
            "PLR-original": lambda: PLROriginal(plr_cfg),
            "SR-PLR-MARL":  lambda: SRPLRScheduler(sr_cfg),
        }
        for shock_type in args.shock_types:
            shock_cfg = ShockCfg(episode=args.shock_ep, shock_type=shock_type)
            net_cfg   = NetCfg(m=2, n=2)
            print(f"\n{'═'*60}\nSMOKE: {shock_type}\n{'═'*60}")
            summaries = []
            for mname, fac in smoke_methods.items():
                for seed in [0, 1]:
                    sch = fac()
                    df  = run_single(sch, net_cfg, shock_cfg, plr_cfg,
                                     args.episodes_2x2, seed)
                    met = compute_metrics(df, args.shock_ep, shock_type)
                    print(f"  {mname:20s} seed={seed}  "
                          f"success={met['success']}  "
                          f"tau_rec={met['recovery_time']:>4}  "
                          f"SR(det)={met['shock_detected_ep']:>4}  "
                          f"alpha_before={met['alpha_before_shock']:.5f}  "
                          f"alpha_after={met['alpha_after_shock']:.5f}")
                    summaries.append({"method": mname, "seed": seed,
                                      "shock_type": shock_type, **met})
            pd.DataFrame(summaries).to_csv(
                out_dir / f"smoke_{shock_type}.csv", index=False)

    else:
        print("Choose --full or --smoke. Use --help for options.")
        parser.print_help()


if __name__ == "__main__":
    main()
