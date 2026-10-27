#!/usr/bin/env python3
# ============================================================================
#  MC/MR MULTI-INSTANCE ROBUSTNESS RUNNER  (ICTA2026 revision)
# ----------------------------------------------------------------------------
#  Purpose
#  -------
#  The canonical Table 3 in the manuscript draws ONE random operator instance
#  per network size (MCMRCfg.param_seed = 2026) and then varies only the initial
#  point x0 and the observation noise across the 30 run-seeds. A reviewer can
#  fairly ask whether the reported behaviour is robust across DIFFERENT operator
#  draws, or specific to one lucky instance.
#
#  This script re-uses the *exact* engine of sr_plr_marl_v8_mc_mr.py and sweeps
#  several operator instances (param_seed) x several run-seeds. It reports:
#    (a) pooled mean +/- std over all (instance x run-seed) pairs, and
#    (b) the ACROSS-INSTANCE statistics: mean and std of the per-instance means
#        of Final_G / RT_all / success  --> the robustness-across-instances
#        number the reviewer wants.
#
#  Config matches the paper: T2 = 100, episodes = 1200, shock at ep 300,
#  all-commodity price spike (x2), sigma = 0.01, LAM_RES = 1.0.
#
#  Usage
#  -----
#    python sr_plr_marl_v8_multiinstance.py --out outputs_mc_mr_multiinstance \
#           --param-seeds 2026 7 13 42 101 --run-seeds 10 \
#           --methods PLR-original SR-PLR-MARL Fixed-LR PG-fixed EG --include-150
#
#  A quick smoke run:
#    python sr_plr_marl_v8_multiinstance.py --smoke
# ============================================================================
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# --- import the existing engine (same directory) ---------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_plr_marl_v8_mc_mr as eng   # noqa: E402

# Network settings identical to the paper's Table 3.
PAPER_SETTINGS = [
    ("3x3_K1_L3", dict(m=3, n=3, K=1, L=3)),
    ("3x3_K3_L1", dict(m=3, n=3, K=3, L=1)),
    ("4x4_K3_L2", dict(m=4, n=4, K=3, L=2)),
    ("5x5_K3_L2", dict(m=5, n=5, K=3, L=2)),   # d = 150
]


def build_methods(method_names: List[str], plr_cfg, sr_cfg) -> Dict[str, object]:
    """Subset of the canonical v8 method registry, preserving its factories."""
    full = eng._v8_main_methods(plr_cfg, sr_cfg)
    missing = [m for m in method_names if m not in full]
    if missing:
        raise SystemExit(f"Unknown method(s): {missing}. Available: {list(full)}")
    return {m: full[m] for m in method_names}


def _done_keys(by_run_path: Path) -> set:
    """Set of (param_seed, setting, method, run_seed) already in the CSV."""
    if not by_run_path.exists():
        return set()
    df = pd.read_csv(by_run_path)
    return {(int(r.param_seed), str(r.setting), str(r.method), int(r.run_seed))
            for r in df.itertuples()}


def run_grid(param_seeds: List[int], run_seeds: int, method_names: List[str],
             settings, episodes: int, shock_ep: int, sigma: float,
             t2: float, by_run_path: Path, time_budget: float = 1e9) -> bool:
    """Resumable grid runner with a wall-clock budget.

    Appends new rows to `by_run_path`, skipping (param_seed, setting, method,
    run_seed) tuples already present. Returns True iff the full grid is complete.
    """
    plr_cfg = eng.PLRCfg(sigma=sigma, T2=t2)
    sr_cfg  = eng.SRCfg(sigma=sigma, T2=t2)
    methods = build_methods(method_names, plr_cfg, sr_cfg)

    done_keys = _done_keys(by_run_path)
    total = len(param_seeds) * len(settings) * len(methods) * run_seeds
    print(f"[multi-instance] grid={total} runs, already done={len(done_keys)}, "
          f"T2={t2}, budget={time_budget:0.0f}s")

    batch: List[dict] = []

    def flush():
        if not batch:
            return
        df = pd.DataFrame(batch)
        header = not by_run_path.exists()
        df.to_csv(by_run_path, mode="a", header=header, index=False)
        batch.clear()

    t0 = time.perf_counter()
    ran = 0
    for pseed in param_seeds:
        for sname, dims in settings:
            cfg = eng.MCMRCfg(param_seed=pseed, **dims)
            dim = dims["m"] * dims["n"] * dims["K"] * dims["L"]
            shock = eng.MCMRShockCfg(episode=shock_ep, price_multiplier=2.0,
                                     all_commodities=True)
            for mname, fac in methods.items():
                for rseed in range(run_seeds):
                    key = (int(pseed), sname, mname, int(rseed))
                    if key in done_keys:
                        continue
                    if time.perf_counter() - t0 > time_budget:
                        flush()
                        nd = len(done_keys) + ran
                        print(f"[multi-instance] budget hit: {nd}/{total} done "
                              f"this+prior; re-run to continue.")
                        return nd >= total
                    sch = fac()
                    df = eng.run_mc_mr_single(sch, cfg, shock, plr_cfg,
                                              episodes, rseed)
                    met = eng.v8_metrics(df, shock_ep, primary_col="rho")
                    batch.append({"setting": sname, "dimension": dim,
                                  "method": mname, "param_seed": int(pseed),
                                  "run_seed": int(rseed), **met})
                    ran += 1
                    if len(batch) >= 15:
                        flush()
    flush()
    nd = len(done_keys) + ran
    print(f"[multi-instance] complete: {nd}/{total} runs.")
    return True


def summarize(by_run: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-instance means, across-instance summary)."""
    metrics = ["Final_G", "RT_all", "Best_G_post", "success", "Final_D"]

    # (1) Per-instance mean over run-seeds.
    by_inst = (by_run.groupby(["setting", "dimension", "method", "param_seed"])
                      [metrics].mean().reset_index())

    # (2) Across-instance summary: mean / std of the per-instance means, plus
    #     a pooled std over all runs for reference.
    g = by_inst.groupby(["setting", "dimension", "method"])
    summ = g[metrics].agg(["mean", "std", "min", "max"]).reset_index()
    # flatten columns
    summ.columns = ["__".join(c).rstrip("__") for c in summ.columns.to_flat_index()]

    pooled = (by_run.groupby(["setting", "dimension", "method"])
                    [metrics].agg(["mean", "std"]).reset_index())
    pooled.columns = ["__".join(c).rstrip("__") for c in pooled.columns.to_flat_index()]
    pooled = pooled.rename(columns={f"{m}__mean": f"{m}__pooled_mean" for m in metrics})
    pooled = pooled.rename(columns={f"{m}__std": f"{m}__pooled_std" for m in metrics})

    summ = summ.merge(pooled, on=["setting", "dimension", "method"])
    n_inst = by_inst.groupby(["setting", "dimension", "method"]).size().reset_index(name="n_instances")
    summ = summ.merge(n_inst, on=["setting", "dimension", "method"])
    return by_inst, summ


def print_headline(by_inst: pd.DataFrame) -> None:
    """Compact console view: Final_G across instances per (setting, method)."""
    print("\n================ ACROSS-INSTANCE Final_G (mean of per-instance means) "
          "================")
    pv = (by_inst.groupby(["dimension", "setting", "method"])["Final_G"]
                 .agg(["mean", "std", "min", "max", "count"]).reset_index()
                 .sort_values(["dimension", "method"]))
    with pd.option_context("display.width", 140, "display.max_rows", None):
        print(pv.to_string(index=False,
              formatters={"mean": "{:0.3f}".format, "std": "{:0.3f}".format,
                          "min": "{:0.3f}".format, "max": "{:0.3f}".format}))
    print("\n================ ACROSS-INSTANCE success-rate & RT_all ================")
    pv2 = (by_inst.groupby(["dimension", "method"])[["success", "RT_all"]]
                  .agg(["mean", "std"]).reset_index())
    pv2.columns = ["__".join(c).rstrip("__") for c in pv2.columns.to_flat_index()]
    with pd.option_context("display.width", 140, "display.max_rows", None):
        print(pv2.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=str, default="outputs_mc_mr_multiinstance")
    ap.add_argument("--param-seeds", type=int, nargs="+",
                    default=[2026, 7, 13, 42, 101],
                    help="Operator-instance seeds (2026 = the paper's instance).")
    ap.add_argument("--run-seeds", type=int, default=10,
                    help="Run-seeds (x0 + noise) per instance.")
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["PLR-original", "SR-PLR-MARL", "Fixed-LR",
                             "PG-fixed", "EG"])
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--shock-ep", type=int, default=300)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--t2", type=float, default=100.0,
                    help="Phase-2 horizon; 100 matches the paper's Table 3.")
    ap.add_argument("--include-150", action="store_true",
                    help="Include the 5x5,K=3,L=2 (d=150) setting.")
    ap.add_argument("--time-budget", type=float, default=1e9,
                    help="Wall-clock seconds per invocation (resumable). Use ~40 "
                         "when the shell caps each call at 45s.")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run for timing: 2 instances, 2 methods, 3 seeds, "
                         "smallest 2 settings.")
    args = ap.parse_args()

    if args.smoke:
        param_seeds = [2026, 7]
        run_seeds = 3
        methods = ["PLR-original", "SR-PLR-MARL"]
        settings = PAPER_SETTINGS[:2]
    else:
        param_seeds = args.param_seeds
        run_seeds = args.run_seeds
        methods = args.methods
        settings = PAPER_SETTINGS if args.include_150 else PAPER_SETTINGS[:3]

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    by_run_path = out_dir / "mc_mr_multiinstance_by_run.csv"
    complete = run_grid(param_seeds, run_seeds, methods, settings,
                        args.episodes, args.shock_ep, args.sigma, args.t2,
                        by_run_path, time_budget=args.time_budget)

    if not complete:
        return  # partial progress saved; re-invoke to continue.

    by_run = pd.read_csv(by_run_path)
    by_inst, summ = summarize(by_run)
    by_inst.to_csv(out_dir / "mc_mr_multiinstance_by_instance.csv", index=False)
    summ.to_csv(out_dir / "mc_mr_multiinstance_summary.csv", index=False)

    print_headline(by_inst)
    print(f"\n[multi-instance] saved to {out_dir.resolve()}")
    print(" - mc_mr_multiinstance_by_run.csv      (every run)")
    print(" - mc_mr_multiinstance_by_instance.csv (per-instance means)")
    print(" - mc_mr_multiinstance_summary.csv     (across-instance mean/std/min/max)")


if __name__ == "__main__":
    main()
