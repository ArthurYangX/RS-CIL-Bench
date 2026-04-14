#!/usr/bin/env python3
"""Adaptive overnight pipeline: sweep → best config → paper re-run.

Each sweep round adapts based on previous results.
Runs 4 experiments in parallel on 4 GPUs.
"""
import subprocess
import json
import time
import os
import re
import sys
import signal
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ═══ CONFIG ═══
SSH_HOST = os.environ.get("SSH_HOST", "connect.westc.seetacloud.com")
SSH_PORT = os.environ.get("SSH_PORT", "46503")
SSH_PASS = os.environ.get("SSH_PASS", "J1AeQodNS/uj")
REMOTE_DIR = "/root/autodl-tmp/jc"
CONDA_INIT = 'eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda activate jc'
BASE_CMD = (
    "python -u cmcd_lora_experiment_gmm.py --mode marathon "
    "--lora_rank 4 --lidar_rank 8 --warmup_tasks 3 "
    "--domain_conditioned_reuse --proto_components 2 --proto_score_mode mixture"
)
LOCAL_RESULTS = "/Users/arthuryang/Desktop/research/HSI/experiment_data/sweep"
EXP_TIMEOUT = 5400  # 90 min per experiment (was 3600, increased for safety)


def ssh_run(cmd, timeout=300, retries=3):
    """Run command via SSH with retry logic."""
    full = (
        f"sshpass -p '{SSH_PASS}' ssh -p {SSH_PORT} "
        f"-o StrictHostKeyChecking=no -o ConnectTimeout=10 "
        f"root@{SSH_HOST} '{cmd}'"
    )
    for attempt in range(retries):
        try:
            r = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0 or r.stdout.strip():
                return r.stdout.strip()
            if attempt < retries - 1:
                print(f"    [SSH retry {attempt+1}/{retries}]")
                time.sleep(5)
        except subprocess.TimeoutExpired:
            if attempt < retries - 1:
                print(f"    [SSH timeout, retry {attempt+1}/{retries}]")
                time.sleep(5)
            else:
                return "TIMEOUT"
        except Exception as e:
            if attempt < retries - 1:
                print(f"    [SSH error: {e}, retry {attempt+1}/{retries}]")
                time.sleep(5)
            else:
                return f"ERROR: {e}"
    return ""


def kill_remote_experiments():
    """Kill all running experiments on the server."""
    print("\n[CLEANUP] Killing remote experiments...")
    ssh_run("pkill -f cmcd_lora_experiment_gmm.py || true", timeout=15, retries=1)
    print("[CLEANUP] Done")


# Register cleanup on exit
atexit.register(kill_remote_experiments)
signal.signal(signal.SIGINT, lambda s, f: (kill_remote_experiments(), sys.exit(1)))
signal.signal(signal.SIGTERM, lambda s, f: (kill_remote_experiments(), sys.exit(1)))


def check_disk_space():
    """Pre-flight disk space check."""
    out = ssh_run("df -h /root/autodl-tmp | tail -1")
    print(f"  Disk: {out}")
    if out:
        parts = out.split()
        for p in parts:
            if p.endswith('%'):
                usage = int(p.rstrip('%'))
                if usage > 90:
                    print(f"  WARNING: Disk usage {usage}% — may run out of space!")
                    return False
    return True


def run_exp(gpu, name, args, seed, order, results_base):
    """Run one experiment on one GPU. Returns dict with results."""
    log_dir = f"{results_base}/{name}"
    cmd = (
        f'{CONDA_INIT} && cd {REMOTE_DIR} && mkdir -p {log_dir} && '
        f'CUDA_VISIBLE_DEVICES={gpu} {BASE_CMD} '
        f'--seed {seed} --dataset_order {order} '
        f'--output_dir {log_dir}/ '
        f'{args} '
        f'2>&1 | tee {log_dir}/run.log'
    )
    start = time.time()
    print(f"  [GPU {gpu}] START: {name}")
    ssh_run(cmd, timeout=EXP_TIMEOUT, retries=1)
    elapsed = (time.time() - start) / 60

    # Extract accuracy
    out = ssh_run(f"grep 'CMCD-LoRA+SHINE' {log_dir}/run.log | tail -1")
    acc = None
    per_ds = {}
    if out and out != "TIMEOUT":
        m = re.search(r'Avg=(\d+\.\d+)%', out)
        if m:
            acc = float(m.group(1))
        for ds in ['MUUFL', 'Trento', 'Houston']:
            m2 = re.search(rf'{ds}=(\d+\.\d+)%', out)
            if m2:
                per_ds[ds] = float(m2.group(1))

    # Save summary
    summary = {
        "name": name, "args": args, "seed": seed, "order": order,
        "accuracy": acc, "per_ds": per_ds,
        "elapsed_min": round(elapsed, 1), "result_line": out or ""
    }
    ssh_run(f"cat > {log_dir}/summary.json << 'EOJSON'\n{json.dumps(summary, indent=2)}\nEOJSON")

    tag = f"{acc:.1f}%" if acc else "FAILED"
    print(f"  [GPU {gpu}] DONE: {name} = {tag} ({elapsed:.0f}min)")
    return summary


def run_parallel(configs, results_base):
    """Run up to 4 configs in parallel. Returns list of result dicts."""
    results = []
    for batch_start in range(0, len(configs), 4):
        batch = configs[batch_start:batch_start + 4]
        print(f"\n{'─'*60}")
        print(f"  Batch: {' | '.join(c['name'] for c in batch)}")
        print(f"{'─'*60}")

        with ThreadPoolExecutor(max_workers=len(batch)) as ex:
            futures = {}
            for i, c in enumerate(batch):
                f = ex.submit(
                    run_exp, i, c["name"], c["args"],
                    c.get("seed", 0), c.get("order", "MTH"), results_base
                )
                futures[f] = c
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    print(f"  [ERROR] Experiment failed: {e}")
                    results.append({
                        "name": futures[f]["name"], "accuracy": None,
                        "args": futures[f]["args"], "error": str(e)
                    })
    return results


def pick_best(results, n=1):
    valid = [r for r in results if r.get("accuracy") is not None]
    valid.sort(key=lambda r: -r["accuracy"])
    return valid[:n]


def print_results(results, title="Results"):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")
    valid = sorted(
        [r for r in results if r.get("accuracy") is not None],
        key=lambda r: -r["accuracy"]
    )
    for i, r in enumerate(valid):
        ds = r.get("per_ds", {})
        ds_str = f"M={ds.get('MUUFL','?')} T={ds.get('Trento','?')} H={ds.get('Houston','?')}"
        star = " ★" if i == 0 else ""
        print(f"  {r['name']:<35s} {r['accuracy']:5.1f}%  {ds_str}{star}")


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    results_base = f"/root/autodl-tmp/results/sweep_{ts}"
    all_results = []
    # Track tested lambdas to avoid duplicates
    tested_lams = set()

    print("╔═══════════════════════════════════════════════════════╗")
    print(f"║  Overnight Adaptive Pipeline — {ts}         ║")
    print("╚═══════════════════════════════════════════════════════╝")

    # Verify server
    print(ssh_run("nvidia-smi --query-gpu=index,name --format=csv,noheader"))
    ssh_run(f"mkdir -p {results_base}")

    # Pre-flight checks
    if not check_disk_space():
        resp = input("Continue anyway? [y/N] ")
        if resp.lower() != 'y':
            sys.exit(1)

    # ═══════════════════════════════════════
    # ROUND 1: Lambda coarse sweep (sd=True)
    # ═══════════════════════════════════════
    r1_lams = [0.5, 1.0, 2.0, 5.0]
    tested_lams.update(r1_lams)

    print("\n╔═ ROUND 1: Lambda coarse sweep (same_domain=True) ═╗")
    r1 = run_parallel([
        {"name": f"R1_sd_lam{l}", "args": f"--lambda_gmm_ortho {l} --gmm_same_domain_only --lambda_domain_kd 0.5"}
        for l in r1_lams
    ], results_base)
    all_results.extend(r1)
    print_results(r1, "Round 1")

    best_r1 = pick_best(r1)[0]
    best_lam = float(re.search(r'lambda_gmm_ortho (\S+)', best_r1["args"]).group(1))
    print(f"\n  → Best lambda: {best_lam} ({best_r1['accuracy']:.1f}%)")

    # ═══════════════════════════════════════
    # ROUND 2: Lambda fine sweep around best
    # ═══════════════════════════════════════
    candidates = sorted(set([
        round(best_lam * 0.3, 2),
        round(best_lam * 0.6, 2),
        round(best_lam * 1.5, 2),
        round(best_lam * 3.0, 2),
    ]) - tested_lams)[:4]
    tested_lams.update(candidates)

    if len(candidates) < 4:
        # Fill with boundary values not yet tested
        extras = [0.1, 0.3, 3.0, 10.0]
        for e in extras:
            if e not in tested_lams and len(candidates) < 4:
                candidates.append(e)
                tested_lams.add(e)

    print(f"\n╔═ ROUND 2: Lambda fine sweep: {candidates} ═╗")
    r2 = run_parallel([
        {"name": f"R2_sd_lam{l}", "args": f"--lambda_gmm_ortho {l} --gmm_same_domain_only --lambda_domain_kd 0.5"}
        for l in candidates
    ], results_base)
    all_results.extend(r2)
    print_results(r1 + r2, "Rounds 1+2")

    best_r12 = pick_best(r1 + r2)[0]
    best_lam = float(re.search(r'lambda_gmm_ortho (\S+)', best_r12["args"]).group(1))
    print(f"\n  → Best lambda: {best_lam} ({best_r12['accuracy']:.1f}%)")

    # ═══════════════════════════════════════
    # ROUND 3: SGKD + domain_kd combos
    # ═══════════════════════════════════════
    print(f"\n╔═ ROUND 3: Combos with lambda_gmm={best_lam} ═╗")
    r3 = run_parallel([
        {"name": "R3_sgkd0.5",
         "args": f"--lambda_gmm_ortho {best_lam} --gmm_same_domain_only --lambda_domain_kd 0.5 --lambda_sgkd 0.5"},
        {"name": "R3_sgkd1.0",
         "args": f"--lambda_gmm_ortho {best_lam} --gmm_same_domain_only --lambda_domain_kd 0.5 --lambda_sgkd 1.0"},
        {"name": "R3_dkd0",
         "args": f"--lambda_gmm_ortho {best_lam} --gmm_same_domain_only --lambda_domain_kd 0"},
        {"name": "R3_dkd1.0",
         "args": f"--lambda_gmm_ortho {best_lam} --gmm_same_domain_only --lambda_domain_kd 1.0"},
    ], results_base)
    all_results.extend(r3)
    print_results(r1 + r2 + r3, "Rounds 1-3")

    best_r123 = pick_best(r1 + r2 + r3)[0]
    best_args_r123 = best_r123["args"]
    print(f"\n  → Best: {best_r123['name']} = {best_r123['accuracy']:.1f}%")

    # ═══════════════════════════════════════
    # ROUND 4: LR + epochs
    # ═══════════════════════════════════════
    print(f"\n╔═ ROUND 4: LR + epochs ═╗")
    r4 = run_parallel([
        {"name": "R4_lr2.5e-4", "args": f"{best_args_r123} --lora_lr 0.00025"},
        {"name": "R4_lr1e-3",   "args": f"{best_args_r123} --lora_lr 0.001"},
        {"name": "R4_ep25",     "args": f"{best_args_r123} --lora_epochs 25"},
        {"name": "R4_ep100",    "args": f"{best_args_r123} --lora_epochs 100"},
    ], results_base)
    all_results.extend(r4)
    print_results(r1 + r2 + r3 + r4, "Rounds 1-4")

    best_r1234 = pick_best(r1 + r2 + r3 + r4)[0]
    print(f"\n  → Best: {best_r1234['name']} = {best_r1234['accuracy']:.1f}%")

    # Detect if lr or epochs improved
    best_lr = None
    best_ep = None
    for r in r4:
        if r.get("accuracy") and r["accuracy"] > best_r123["accuracy"]:
            if "lr" in r["name"]:
                m = re.search(r'lora_lr (\S+)', r["args"])
                if m: best_lr = m.group(1)
            if "ep" in r["name"]:
                m = re.search(r'lora_epochs (\S+)', r["args"])
                if m: best_ep = m.group(1)

    # ═══════════════════════════════════════
    # ROUND 5: Cross lr×epochs + sd=False
    # ═══════════════════════════════════════
    print(f"\n╔═ ROUND 5: Cross-validation ═╗")
    r5_configs = []

    if best_lr and best_ep:
        r5_configs.append({"name": "R5_lr_ep_cross",
                           "args": f"{best_args_r123} --lora_lr {best_lr} --lora_epochs {best_ep}"})
    if best_lr:
        r5_configs.append({"name": "R5_best_lr_ep25",
                           "args": f"{best_args_r123} --lora_lr {best_lr} --lora_epochs 25"})
    r5_configs.append({"name": "R5_lr2.5e-4_ep25",
                       "args": f"{best_args_r123} --lora_lr 0.00025 --lora_epochs 25"})
    r5_configs.append({"name": "R5_sd_false",
                       "args": best_args_r123.replace("--gmm_same_domain_only", "").strip()})

    r5 = run_parallel(r5_configs[:4], results_base)
    all_results.extend(r5)

    # ═══ FINAL SWEEP RESULT ═══
    print_results(all_results, "ALL SWEEP RESULTS")
    final_best = pick_best(all_results)[0]
    final_args = final_best["args"]

    print(f"\n{'★'*60}")
    print(f"  FINAL BEST: {final_best['name']} = {final_best['accuracy']:.1f}%")
    print(f"  Args: {final_args}")
    print(f"{'★'*60}")

    # Save sweep summary
    sweep_summary = {
        "best_name": final_best["name"],
        "best_accuracy": final_best["accuracy"],
        "best_args": final_args,
        "all_results": sorted(
            [{"name": r["name"], "accuracy": r.get("accuracy"), "args": r["args"],
              "per_ds": r.get("per_ds", {})}
             for r in all_results if r.get("accuracy")],
            key=lambda r: -(r["accuracy"] or 0)
        ),
        "timestamp": ts
    }
    ssh_run(f"cat > {results_base}/sweep_final.json << 'EOF'\n{json.dumps(sweep_summary, indent=2)}\nEOF")

    # ═══════════════════════════════════════
    # PHASE 2: Paper re-run
    # ═══════════════════════════════════════
    print(f"\n╔═══════════════════════════════════════════════════════╗")
    print(f"║  PHASE 2: Paper Re-run (6 orders × 3 seeds)          ║")
    print(f"║  Config: {final_args[:50]}...")
    print(f"╚═══════════════════════════════════════════════════════╝")

    # Check disk before Phase 2
    check_disk_space()

    paper_configs = []
    for order in ["MTH", "MHT", "TMH", "THM", "HMT", "HTM"]:
        for seed in [0, 1, 2]:
            paper_configs.append({
                "name": f"paper_{order}_s{seed}",
                "args": final_args,
                "seed": seed,
                "order": order,
            })

    paper_results = run_parallel(paper_configs, results_base)

    # ═══ FINAL TABLE ═══
    print(f"\n{'═'*60}")
    print(f"  PAPER RESULTS — {final_best['name']}")
    print(f"{'═'*60}")
    print(f"{'Order':<8} {'Seed0':>8} {'Seed1':>8} {'Seed2':>8} {'Mean':>8}")
    print("─" * 42)

    order_means = {}
    for order in ["MTH", "MHT", "TMH", "THM", "HMT", "HTM"]:
        accs = []
        row = f"{order:<8}"
        for seed in [0, 1, 2]:
            r = next((r for r in paper_results
                      if r.get("order") == order and r.get("seed") == seed), None)
            a = r["accuracy"] if r and r.get("accuracy") else None
            row += f" {a:7.1f}%" if a else "      ??"
            if a: accs.append(a)
        mean = sum(accs) / len(accs) if accs else 0
        order_means[order] = mean
        row += f" {mean:7.1f}%"
        print(row)

    overall = sum(order_means.values()) / len(order_means) if order_means else 0
    print(f"\n  Overall 6-order mean: {overall:.1f}%")

    # Save paper results
    paper_summary = {
        "config": final_args,
        "order_means": order_means,
        "overall_mean": round(overall, 1),
        "results": [{"name": r["name"], "accuracy": r.get("accuracy"),
                     "per_ds": r.get("per_ds", {}), "order": r.get("order"),
                     "seed": r.get("seed")}
                    for r in paper_results],
        "timestamp": ts
    }
    ssh_run(f"cat > {results_base}/paper_final.json << 'EOF'\n{json.dumps(paper_summary, indent=2)}\nEOF")

    # ═══ Pull to local ═══
    print(f"\nPulling results (json/csv/log only)...")
    os.makedirs(LOCAL_RESULTS, exist_ok=True)
    ssh_run(
        f"cd {results_base} && find . \\( -name '*.json' -o -name '*.csv' -o -name '*.log' -o -name '*.yaml' \\) "
        f"| tar czf /tmp/overnight.tar.gz -T -",
        timeout=120
    )
    subprocess.run(
        f"sshpass -p '{SSH_PASS}' scp -P {SSH_PORT} -o StrictHostKeyChecking=no "
        f"root@{SSH_HOST}:/tmp/overnight.tar.gz {LOCAL_RESULTS}/overnight_{ts}.tar.gz",
        shell=True, timeout=300
    )
    # Verify transfer
    local_file = f"{LOCAL_RESULTS}/overnight_{ts}.tar.gz"
    if os.path.exists(local_file) and os.path.getsize(local_file) > 1000:
        print(f"  Transfer OK: {os.path.getsize(local_file)} bytes")
    else:
        print(f"  WARNING: Transfer may have failed!")

    print(f"\n{'═'*60}")
    print(f"  PIPELINE COMPLETE: {datetime.now()}")
    print(f"  Sweep: 20 configs across 5 rounds")
    print(f"  Paper: 18 experiments (6 orders × 3 seeds)")
    print(f"  Best: {final_best['name']} = {final_best['accuracy']:.1f}%")
    print(f"  6-order mean: {overall:.1f}%")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
