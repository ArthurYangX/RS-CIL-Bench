#!/bin/bash
# Pilot experiments: data access optimization for CMCD-LoRA
# Run on GPU server when 4090 is available
#
# Prerequisites:
#   eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda activate jc
#   cd /root/autodl-tmp/jc

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
echo "Patch: line ~2601: for ds in [ds_name] instead of dataset_order"
echo "Eliminates: cross-domain data access (violations 2+4)"
echo "Command: python cmcd_lora_experiment.py $COMMON --results_tag pilot_a"
echo ""

echo "--- Pilot B: analytic DCR (Eq.6) for prototype correction ---"
echo "Patch: integrate pilot_b_analytic_dcr.py"
echo "Eliminates: same-domain prototype re-extraction (violation 3)"
echo "Command: python cmcd_lora_experiment.py $COMMON --results_tag pilot_b"
echo ""

echo "--- Pilot C: statistics-only exemplar-free (L1) ---"
echo "Patches: A + B + change LoRA training to current-task-only + enable proto_aug"
echo "Eliminates: ALL raw data access for old classes (violations 1+2+3+4)"
echo "Storage: 101 KB only"
echo "Command: python cmcd_lora_experiment.py $COMMON --n_pseudo 16 --lambda_pseudo 0.5 --pilot_mode c --results_tag pilot_c"
echo ""

echo "--- Pilot D: feature-buffer mode (L2) ---"
echo "Patches: A + B + feature buffer for LoRA old-class replay"
echo "Eliminates: ALL raw data access after warmup (violations 1+2+3+4)"
echo "Storage: ~5 MB features + 101 KB stats"
echo "Command: python cmcd_lora_experiment.py $COMMON --pilot_mode d --results_tag pilot_d"
echo ""

echo "============================================"
echo "  Expected results matrix"
echo "============================================"
echo ""
echo "Config          | Storage  | Raw data access | Expected BA"
echo "----------------|----------|-----------------|------------"
echo "Baseline (L3)   | 0 extra  | all domains     | 84.5%"
echo "Pilot A (L3-)   | 0 extra  | current dom     | ~84.5%"
echo "Pilot B (L3--)  | 0 extra  | current dom     | ~84.0-84.5%"
echo "Pilot C (L1)    | 101 KB   | NONE            | ~75-82%?"
echo "Pilot D (L2)    | ~5 MB    | NONE            | ~80-84%?"
echo "iCaRL           | 7.5 MB   | NONE            | 77.9%"
echo ""
echo "Key question: Can Pilot C/D beat iCaRL (77.9%) without any raw data access?"
