#!/bin/bash
# Pilot experiments: data access optimization for CMCD-LoRA
# Run on GPU server when 4090 is available
#
# Prerequisites:
#   eval "$($CONDA_HOME/bin/conda shell.bash hook)" && conda activate jc
#   cd /root/autodl-tmp/jc
#
# IMPORTANT: Each pilot requires MANUAL CODE PATCHES to cmcd_lora_experiment.py
# before running. See each pilot's .py file for exact patch instructions.
# Back up the original file before patching.
#
# Violations reference:
#   1: LoRA intra-domain old-class replay (uses same-domain old training data)
#   2: Cross-domain prototype recomputation (unnecessary with domain-selective inference)
#   3: Same-domain prototype brute-force re-extraction (replaceable by Eq.6)
#   4: Cross-domain SHINE stats recomputation (unnecessary)

SEED=0
ORDER=MTH
COMMON="--mode marathon --seed $SEED --dataset_order $ORDER \
  --lora_rank 4 --lidar_rank 8 \
  --warmup_tasks 3 --lambda_domain_kd 0.5 \
  --domain_conditioned_reuse \
  --proto_components 2 --proto_score_mode mixture"

echo "============================================"
echo "  CMCD-LoRA Data Access Pilot Experiments"
echo "============================================"
echo ""
echo "Baseline (all-domain recompute): 84.5% (existing, no need to re-run)"
echo ""

echo "--- Pilot A: current-domain-only prototype/SHINE recompute ---"
echo "PATCH: line ~2601: for ds in [ds_name] instead of dataset_order"
echo "Eliminates: violations 2 (cross-domain proto) + 4 (cross-domain SHINE)"
echo "Run: python cmcd_lora_experiment.py $COMMON"
echo ""

echo "--- Pilot B: analytic DCR (Eq.6) for prototype correction ---"
echo "PATCH: integrate pilot_b_analytic_dcr.py into evaluation loop"
echo "       set pure_analytic=True for strictest test"
echo "Eliminates: violation 3 (same-domain proto re-extraction)"
echo "Run: python cmcd_lora_experiment.py $COMMON"
echo ""

echo "--- Pilot C: statistics-only exemplar-free (L1) ---"
echo "PATCHES: A + B + change LoRA data to current-task-only + enable proto_aug"
echo "  1. line ~2482: all_domain_ds = subset_by_classes(..., task_class_set)"
echo "  2. enable: --n_pseudo 16 --lambda_pseudo 0.5"
echo "Eliminates: ALL violations (1+2+3+4) — zero raw data access for old classes"
echo "Storage: 101 KB only"
echo "Run: python cmcd_lora_experiment.py $COMMON --n_pseudo 16 --lambda_pseudo 0.5"
echo ""

echo "--- Pilot D: feature-buffer mode (L2) ---"
echo "PATCHES: A + B + integrate pilot_d_feature_buffer.py"
echo "Eliminates: ALL violations — zero raw data access after warmup"
echo "Storage: ~5 MB features + 101 KB stats"
echo "Run: python cmcd_lora_experiment.py $COMMON"
echo ""

echo "============================================"
echo "  Expected results matrix"
echo "============================================"
echo ""
echo "Config          | Storage  | Raw data access  | Expected BA"
echo "----------------|----------|------------------|------------"
echo "Baseline (L3)   | 0 extra  | all domains      | 84.5%"
echo "Pilot A (L3-)   | 0 extra  | current dom only | ~84.5%"
echo "Pilot B (L3--)  | 0 extra  | current dom only | ~84.0-84.5%"
echo "Pilot C (L1)    | 101 KB   | NONE             | ~75-82%?"
echo "Pilot D (L2)    | ~5 MB    | NONE             | ~80-84%?"
echo "iCaRL           | 7.5 MB   | NONE             | 77.9%"
echo ""
echo "Key question: Can Pilot C/D beat iCaRL (77.9%) without any raw data access?"
