#!/usr/bin/env python3
# ============================================================================
#  SUPPLY-CUT runner (single-commodity, T2=100) -- ICTA2026 revision
#  Fills the missing supply_cut shock at the canonical config so the shock-type
#  table can report all three shocks consistently. price_spike and
#  route_blockage already exist in outputs_v8_t2_100/main/.
#  Resumable: each call runs up to --time-budget seconds and appends to CSV.
# ============================================================================
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from typing import List
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sr_plr_marl_v8_mc_mr as eng   # noqa: E402

NETS = [(2, 2), (3, 3)]
METHODS = ["PLR-original", "SR-PLR-MARL", "Fixed-LR", "PG-fixed", "EG"]


def done_keys(p: Path) -> set:
    if not p.exists():
        return set()
    d = pd.read_csv(p)
    return {(str(r.net), str(r.method), int(r.seed)) for r in d.itertuples()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs_supply_cut_v8")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--shock-ep", type=int, default=300)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--t2", type=float, default=100.0)
    ap.add_argument("--frac", type=float, default=0.50)
    ap.add_argument("--time-budget", type=float, default=40.0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    by_run = out / "supply_cut_by_run.csv"
    plr_cfg = eng.PLRCfg(sigma=args.sigma, T2=args.t2)
    sr_cfg = eng.SRCfg(sigma=args.sigma, T2=args.t2)
    methods = eng._v8_main_methods(plr_cfg, sr_cfg)
    methods = {m: methods[m] for m in METHODS}

    have = done_keys(by_run)
    grid = len(NETS) * len(methods) * args.seeds
    print(f"[supply_cut] grid={grid}, done={len(have)}, T2={args.t2}, budget={args.time_budget}s")

    batch: List[dict] = []
    def flush():
        if batch:
            pd.DataFrame(batch).to_csv(by_run, mode="a",
                                       header=not by_run.exists(), index=False)
            batch.clear()

    t0 = time.perf_counter(); ran = 0
    for (mm, nn) in NETS:
        net_cfg = eng.NetCfg(m=mm, n=nn); label = f"{mm}x{nn}"
        for mname, fac in methods.items():
            for seed in range(args.seeds):
                if (label, mname, seed) in have:
                    continue
                if time.perf_counter() - t0 > args.time_budget:
                    flush()
                    print(f"[supply_cut] budget hit: {len(have)+ran}/{grid}; re-run.")
                    return
                sch = fac()
                shock = eng.ShockCfg(episode=args.shock_ep, shock_type="supply_cut",
                                     supply_cut_frac=args.frac)
                df = eng.run_single(sch, net_cfg, shock, plr_cfg, args.episodes, seed)
                met = eng.v8_metrics(df, args.shock_ep, primary_col="rho")
                batch.append({"shock": "supply_cut", "net": label,
                              "method": mname, "seed": seed, **met})
                ran += 1
                if len(batch) >= 15:
                    flush()
    flush()
    full = pd.read_csv(by_run)
    agg = (full.groupby(["shock", "net", "method"])
               .agg(n=("seed", "nunique"),
                    success=("success", "mean"),
                    RT_all=("RT_all", "mean"),
                    Final_G_mean=("Final_G", "mean"),
                    Final_G_std=("Final_G", "std"),
                    Best_G_post_mean=("Best_G_post", "mean"))
               .reset_index())
    agg.to_csv(out / "supply_cut_summary_by_method.csv", index=False)
    print("[supply_cut] COMPLETE")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
