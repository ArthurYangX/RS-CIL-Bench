#!/usr/bin/env python3
"""Overnight automated pipeline: hyperparameter sweep → best config → full paper re-run.

Usage:
    python overnight_sweep.py --machines "host1:port1:pw1,host2:port2:pw2,..."

Phase 1: Hyperparameter sweep (4 machines in parallel)
Phase 2: Re-run all 6-order × 3-seed paper experiments with best config
"""
import argparse
import subprocess
import json
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

CONDA_INIT = 'eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda activate jc'
WORK_DIR = "/root/autodl-tmp/jc"
RESULTS_DIR = "/root/autodl-tmp/results/sweep"

# Base command (common across all experiments)
BASE_CMD = (
    "python -u cmcd_lora_experiment_gmm.py --mode marathon "
    "--lora_rank 4 --lidar_rank 8 --warmup_tasks 3 "
    "--lambda_domain_kd 0.5 --domain_conditioned_reuse "
    "--proto_components 2 --proto_score_mode mixture"
)

# Phase 1: Sweep configurations
SWEEP_CONFIGS = [
    # Round 1: find best lambda + same_domain_only
    {"name": "gmm_sd_lam1.0",   "args": "--lambda_gmm_ortho 1.0 --gmm_same_domain_only",  "seed": 0, "order": "MTH"},
    {"name": "gmm_sd_lam2.0",   "args": "--lambda_gmm_ortho 2.0 --gmm_same_domain_only",  "seed": 0, "order": "MTH"},
    {"name": "gmm_sd_lam0.5",   "args": "--lambda_gmm_ortho 0.5 --gmm_same_domain_only",  "seed": 0, "order": "MTH"},
    {"name": "gmm_sd_lam5.0",   "args": "--lambda_gmm_ortho 5.0 --gmm_same_domain_only",  "seed": 0, "order": "MTH"},
    # Round 2: compare with all-domain + combinations
    {"name": "gmm_all_lam0.5",  "args": "--lambda_gmm_ortho 0.5",                          "seed": 0, "order": "MTH"},
    {"name": "gmm_sd_lam1.0_sgkd0.5", "args": "--lambda_gmm_ortho 1.0 --gmm_same_domain_only --lambda_sgkd 0.5", "seed": 0, "order": "MTH"},
    {"name": "gmm_sd_lam1.0_proto2",  "args": "--lambda_gmm_ortho 1.0 --gmm_same_domain_only --lambda_proto 2.0", "seed": 0, "order": "MTH"},
    {"name": "gmm_sd_lam3.0",   "args": "--lambda_gmm_ortho 3.0 --gmm_same_domain_only",  "seed": 0, "order": "MTH"},
]

# Phase 2: Full paper experiments (6 orders × 3 seeds)
ORDERS = ["MTH", "MHT", "TMH", "THM", "HMT", "HTM"]
SEEDS = [0, 1, 2]


def ssh_cmd(host, port, password, cmd, timeout=3000):
    """Run command on remote machine via sshpass."""
    full_cmd = (
        f'sshpass -p "{password}" ssh -p {port} -o StrictHostKeyChecking=no '
        f'root@{host} "{cmd}"'
    )
    try:
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1


def deploy_code(machine):
    """Ensure gmm experiment code is on the machine."""
    host, port, pw = machine
    # Check if code exists
    out, rc = ssh_cmd(host, port, pw, f"ls {WORK_DIR}/cmcd_lora_experiment_gmm.py")
    if rc != 0:
        print(f"  [{host}:{port}] Code not found, needs deployment")
        return False
    return True


def run_experiment(machine, name, args, seed, order, log_dir):
    """Run a single experiment on a machine. Blocks until complete."""
    host, port, pw = machine
    machine_id = f"{host}:{port}"

    log_file = f"{log_dir}/{name}_seed{seed}_{order}.log"
    cmd = (
        f'{CONDA_INIT} && cd {WORK_DIR} && '
        f'mkdir -p {log_dir} && '
        f'{BASE_CMD} --seed {seed} --dataset_order {order} {args} '
        f'2>&1 | tee {log_file}'
    )

    print(f"  [{machine_id}] Starting: {name} seed={seed} order={order}")
    start = time.time()
    out, rc = ssh_cmd(host, port, pw, cmd, timeout=3600)
    elapsed = time.time() - start
    print(f"  [{machine_id}] Finished: {name} ({elapsed/60:.1f} min, rc={rc})")

    return {
        "name": name,
        "seed": seed,
        "order": order,
        "machine": machine_id,
        "elapsed_min": round(elapsed / 60, 1),
        "rc": rc,
        "log_file": log_file,
    }


def extract_result(machine, log_file):
    """Extract final CMCD-LoRA+SHINE accuracy from log."""
    host, port, pw = machine
    out, _ = ssh_cmd(host, port, pw,
                     f"grep 'CMCD-LoRA+SHINE' {log_file} | tail -1")
    if out:
        # Parse "Avg=XX.X% | MUUFL=XX.X%, Trento=XX.X%, Houston=XX.X%"
        try:
            avg = float(out.split("Avg=")[1].split("%")[0])
            return avg
        except:
            pass
    return None


def run_parallel(machines, experiments, log_dir):
    """Run experiments in parallel across machines."""
    results = []
    queue = list(experiments)

    while queue:
        batch = queue[:len(machines)]
        queue = queue[len(machines):]

        with ThreadPoolExecutor(max_workers=len(machines)) as executor:
            futures = {}
            for i, exp in enumerate(batch):
                m = machines[i % len(machines)]
                f = executor.submit(
                    run_experiment, m, exp["name"], exp["args"],
                    exp["seed"], exp["order"], log_dir
                )
                futures[f] = (m, exp)

            for f in as_completed(futures):
                m, exp = futures[f]
                r = f.result()
                acc = extract_result(m, r["log_file"])
                r["accuracy"] = acc
                results.append(r)
                if acc:
                    print(f"  >>> {r['name']}: {acc:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", required=True,
                        help="Comma-separated host:port:password")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip sweep, use provided best config")
    parser.add_argument("--best-args", default=None,
                        help="Best args string (skip sweep)")
    parser.add_argument("--sweep-only", action="store_true",
                        help="Only run sweep, don't run paper experiments")
    args = parser.parse_args()

    # Parse machines
    machines = []
    for m in args.machines.split(","):
        parts = m.strip().split(":")
        if len(parts) == 3:
            machines.append((parts[0], parts[1], parts[2]))
        elif len(parts) == 4:
            # host:port:password with : in password
            machines.append((parts[0], parts[1], ":".join(parts[2:])))

    n_machines = len(machines)
    print(f"═══ Overnight Pipeline: {n_machines} machines ═══")
    print(f"Machines: {[f'{h}:{p}' for h, p, _ in machines]}")

    # Verify all machines accessible
    for h, p, pw in machines:
        out, rc = ssh_cmd(h, p, pw, "echo ok && nvidia-smi --query-gpu=name --format=csv,noheader", timeout=15)
        print(f"  {h}:{p} → {'OK: ' + out if rc == 0 else 'FAILED'}")

    # ═══ PHASE 1: Sweep ═══
    if not args.skip_sweep:
        print(f"\n═══ PHASE 1: Hyperparameter Sweep ({len(SWEEP_CONFIGS)} configs) ═══")
        sweep_results = run_parallel(machines, SWEEP_CONFIGS, f"{RESULTS_DIR}/sweep")

        # Find best config
        valid = [r for r in sweep_results if r["accuracy"] is not None]
        if not valid:
            print("ERROR: No valid sweep results!")
            sys.exit(1)

        best = max(valid, key=lambda r: r["accuracy"])
        print(f"\n═══ SWEEP RESULT ═══")
        for r in sorted(valid, key=lambda r: -r["accuracy"]):
            print(f"  {r['name']}: {r['accuracy']:.1f}%")
        print(f"\n  BEST: {best['name']} = {best['accuracy']:.1f}%")

        # Extract best args
        best_config = next(c for c in SWEEP_CONFIGS if c["name"] == best["name"])
        best_args = best_config["args"]

        # Save sweep results
        with open("/tmp/sweep_results.json", "w") as f:
            json.dump({"results": valid, "best": best, "best_args": best_args}, f, indent=2)
        print(f"  Sweep results saved to /tmp/sweep_results.json")
    else:
        best_args = args.best_args or "--lambda_gmm_ortho 1.0 --gmm_same_domain_only"
        print(f"  Skipping sweep, using: {best_args}")

    if args.sweep_only:
        print("\n═══ Sweep only mode, stopping. ═══")
        return

    # ═══ PHASE 2: Full Paper Re-run ═══
    print(f"\n═══ PHASE 2: Full Paper Re-run (6 orders × 3 seeds = 18 experiments) ═══")
    print(f"  Best config: {best_args}")

    paper_experiments = []
    for order in ORDERS:
        for seed in SEEDS:
            paper_experiments.append({
                "name": f"paper_{order}_s{seed}",
                "args": best_args + f"",
                "seed": seed,
                "order": order,
            })

    paper_results = run_parallel(machines, paper_experiments, f"{RESULTS_DIR}/paper")

    # Summary
    print(f"\n═══ FINAL RESULTS ═══")
    print(f"Config: {best_args}")
    print(f"{'Order':<8} {'Seed 0':>8} {'Seed 1':>8} {'Seed 2':>8} {'Mean':>8}")
    print("-" * 42)
    order_means = {}
    for order in ORDERS:
        accs = []
        row = f"{order:<8}"
        for seed in SEEDS:
            r = next((r for r in paper_results
                      if r["order"] == order and r["seed"] == seed), None)
            acc = r["accuracy"] if r else None
            row += f" {acc:7.1f}%" if acc else " ??"
            if acc:
                accs.append(acc)
        mean = sum(accs) / len(accs) if accs else 0
        order_means[order] = mean
        row += f" {mean:7.1f}%"
        print(row)

    overall = sum(order_means.values()) / len(order_means) if order_means else 0
    print(f"\n  Overall mean: {overall:.1f}%")

    # Save
    with open("/tmp/paper_results.json", "w") as f:
        json.dump({
            "config": best_args,
            "results": paper_results,
            "order_means": order_means,
            "overall_mean": overall,
        }, f, indent=2)
    print(f"  Results saved to /tmp/paper_results.json")
    print(f"\n═══ OVERNIGHT PIPELINE COMPLETE ═══")


if __name__ == "__main__":
    main()
