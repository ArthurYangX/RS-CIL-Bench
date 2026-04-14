#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Overnight Experiment Runner
# Phase 1: Fine-grained sweep (4 rounds × 4 GPUs = 16 configs)
# Phase 2: Paper re-run (6 orders × 3 seeds = 18 experiments)
# Total: ~5-6 hours
# ═══════════════════════════════════════════════════════════════

SSH_HOST="connect.westc.seetacloud.com"
SSH_PORT="46503"
SSH_PASS="J1AeQodNS/uj"
REMOTE_DIR="/root/autodl-tmp/jc"
RESULTS_BASE="/root/autodl-tmp/results/sweep_$(date +%Y%m%d_%H%M)"
LOCAL_RESULTS="/Users/arthuryang/Desktop/research/HSI/experiment_data/sweep"

CONDA_INIT='eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda activate jc'
BASE_CMD="python -u cmcd_lora_experiment_gmm.py --mode marathon --lora_rank 4 --lidar_rank 8 --warmup_tasks 3 --domain_conditioned_reuse --proto_components 2 --proto_score_mode mixture"

ssh_run() {
    sshpass -p "$SSH_PASS" ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no root@"$SSH_HOST" "$1" 2>/dev/null
}

# Run one experiment on specific GPU, return accuracy
run_exp() {
    local GPU=$1 NAME=$2 EXTRA_ARGS=$3 SEED=$4 ORDER=$5
    local LOG_DIR="${RESULTS_BASE}/${NAME}"
    local START=$(date +%s)

    ssh_run "${CONDA_INIT} && cd ${REMOTE_DIR} && mkdir -p ${LOG_DIR} && \
        CUDA_VISIBLE_DEVICES=${GPU} ${BASE_CMD} \
        --seed ${SEED} --dataset_order ${ORDER} \
        --output_dir ${LOG_DIR}/ \
        ${EXTRA_ARGS} \
        2>&1 | tee ${LOG_DIR}/run.log"

    local END=$(date +%s)
    local ELAPSED=$(( (END - START) / 60 ))

    # Extract accuracy
    local ACC=$(ssh_run "grep 'CMCD-LoRA+SHINE' ${LOG_DIR}/run.log | tail -1 | sed 's/.*Avg=//' | sed 's/%.*//'")
    local FULL=$(ssh_run "grep 'CMCD-LoRA+SHINE' ${LOG_DIR}/run.log | tail -1")

    # Save summary
    ssh_run "cat > ${LOG_DIR}/summary.json << JSONEOF
{
  \"name\": \"${NAME}\",
  \"seed\": ${SEED},
  \"order\": \"${ORDER}\",
  \"args\": \"${EXTRA_ARGS}\",
  \"accuracy\": ${ACC:-0},
  \"result\": \"${FULL}\",
  \"elapsed_min\": ${ELAPSED}
}
JSONEOF"

    echo "[GPU ${GPU}] ${NAME}: ${ACC}% (${ELAPSED}min)"
}

# Run 4 experiments in parallel
run_batch() {
    echo ""
    echo "━━━ Batch: $1 | $5 | $9 | ${13} ━━━"

    run_exp 0 "$1" "$2" "$3" "$4" &
    run_exp 1 "$5" "$6" "$7" "$8" &
    run_exp 2 "$9" "${10}" "${11}" "${12}" &
    run_exp 3 "${13}" "${14}" "${15}" "${16}" &
    wait

    echo "━━━ Batch complete ━━━"
}

get_acc() {
    ssh_run "grep 'CMCD-LoRA+SHINE' ${RESULTS_BASE}/$1/run.log | tail -1 | sed 's/.*Avg=//' | sed 's/%.*//'"
}

# ═══════════════════════════════════════════════════════════════
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Overnight Pipeline: $(date)                         "
echo "║  Results: ${RESULTS_BASE}                            "
echo "╚══════════════════════════════════════════════════════╝"

ssh_run "nvidia-smi --query-gpu=index,name --format=csv,noheader"
ssh_run "mkdir -p ${RESULTS_BASE}"

# ═══ PHASE 1: SWEEP ═══
# Already have: same_domain=False, lam=1.0 → 78.8%
# Goal: find optimal lambda + same_domain + combinations

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  PHASE 1 ROUND 1: Lambda sweep (same_domain=True)   ║"
echo "╚══════════════════════════════════════════════════════╝"

run_batch \
    "R1_sd_lam0.5"  "--lambda_gmm_ortho 0.5 --gmm_same_domain_only --lambda_domain_kd 0.5"  0 MTH \
    "R1_sd_lam1.0"  "--lambda_gmm_ortho 1.0 --gmm_same_domain_only --lambda_domain_kd 0.5"  0 MTH \
    "R1_sd_lam2.0"  "--lambda_gmm_ortho 2.0 --gmm_same_domain_only --lambda_domain_kd 0.5"  0 MTH \
    "R1_sd_lam5.0"  "--lambda_gmm_ortho 5.0 --gmm_same_domain_only --lambda_domain_kd 0.5"  0 MTH

echo ""
echo "═══ Round 1 Results ═══"
for n in R1_sd_lam0.5 R1_sd_lam1.0 R1_sd_lam2.0 R1_sd_lam5.0; do
    echo "  $n: $(get_acc $n)%"
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  PHASE 1 ROUND 2: Finer lambda + boundaries         ║"
echo "╚══════════════════════════════════════════════════════╝"

run_batch \
    "R2_sd_lam0.1"  "--lambda_gmm_ortho 0.1 --gmm_same_domain_only --lambda_domain_kd 0.5"  0 MTH \
    "R2_sd_lam0.3"  "--lambda_gmm_ortho 0.3 --gmm_same_domain_only --lambda_domain_kd 0.5"  0 MTH \
    "R2_sd_lam3.0"  "--lambda_gmm_ortho 3.0 --gmm_same_domain_only --lambda_domain_kd 0.5"  0 MTH \
    "R2_sd_lam10"   "--lambda_gmm_ortho 10.0 --gmm_same_domain_only --lambda_domain_kd 0.5" 0 MTH

echo ""
echo "═══ Round 2 Results ═══"
for n in R2_sd_lam0.1 R2_sd_lam0.3 R2_sd_lam3.0 R2_sd_lam10; do
    echo "  $n: $(get_acc $n)%"
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  PHASE 1 ROUND 3: SGKD + combinations               ║"
echo "╚══════════════════════════════════════════════════════╝"

# Use lambda=1.0 as anchor (can adjust based on round 1-2 if needed)
run_batch \
    "R3_sd_lam1_sgkd0.5"  "--lambda_gmm_ortho 1.0 --gmm_same_domain_only --lambda_domain_kd 0.5 --lambda_sgkd 0.5"   0 MTH \
    "R3_sd_lam1_sgkd1.0"  "--lambda_gmm_ortho 1.0 --gmm_same_domain_only --lambda_domain_kd 0.5 --lambda_sgkd 1.0"   0 MTH \
    "R3_sd_lam2_sgkd0.5"  "--lambda_gmm_ortho 2.0 --gmm_same_domain_only --lambda_domain_kd 0.5 --lambda_sgkd 0.5"   0 MTH \
    "R3_sd_lam1_proto2"   "--lambda_gmm_ortho 1.0 --gmm_same_domain_only --lambda_domain_kd 0.5 --lambda_proto 2.0"   0 MTH

echo ""
echo "═══ Round 3 Results ═══"
for n in R3_sd_lam1_sgkd0.5 R3_sd_lam1_sgkd1.0 R3_sd_lam2_sgkd0.5 R3_sd_lam1_proto2; do
    echo "  $n: $(get_acc $n)%"
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  PHASE 1 ROUND 4: Verify top configs (seed 1+2)     ║"
echo "╚══════════════════════════════════════════════════════╝"

# Find best 2 configs from rounds 1-3
echo "Finding top configs..."
BEST1="" BEST1_ACC=0 BEST1_ARGS=""
BEST2="" BEST2_ACC=0 BEST2_ARGS=""
for dir in $(ssh_run "ls -d ${RESULTS_BASE}/R*/ 2>/dev/null"); do
    name=$(basename $dir)
    acc=$(get_acc $name)
    args=$(ssh_run "python3 -c \"import json; print(json.load(open('${dir}summary.json'))['args'])\"" 2>/dev/null)
    if [ -n "$acc" ] && (( $(echo "$acc > $BEST1_ACC" | bc -l 2>/dev/null || echo 0) )); then
        BEST2=$BEST1; BEST2_ACC=$BEST1_ACC; BEST2_ARGS=$BEST1_ARGS
        BEST1=$name; BEST1_ACC=$acc; BEST1_ARGS=$args
    elif [ -n "$acc" ] && (( $(echo "$acc > $BEST2_ACC" | bc -l 2>/dev/null || echo 0) )); then
        BEST2=$name; BEST2_ACC=$acc; BEST2_ARGS=$args
    fi
done

echo "  Top 1: $BEST1 = ${BEST1_ACC}% → $BEST1_ARGS"
echo "  Top 2: $BEST2 = ${BEST2_ACC}% → $BEST2_ARGS"

# Verify on seed 1 and 2
run_batch \
    "R4_best1_s1"  "${BEST1_ARGS}" 1 MTH \
    "R4_best1_s2"  "${BEST1_ARGS}" 2 MTH \
    "R4_best2_s1"  "${BEST2_ARGS}" 1 MTH \
    "R4_best2_s2"  "${BEST2_ARGS}" 2 MTH

echo ""
echo "═══ Round 4 Verification ═══"
echo "  Best1 ($BEST1): s0=$(get_acc $BEST1)% s1=$(get_acc R4_best1_s1)% s2=$(get_acc R4_best1_s2)%"
echo "  Best2 ($BEST2): s0=$(get_acc $BEST2)% s1=$(get_acc R4_best2_s1)% s2=$(get_acc R4_best2_s2)%"

# Pick final best (by 3-seed mean)
B1_MEAN=$(echo "scale=1; ($(get_acc $BEST1) + $(get_acc R4_best1_s1) + $(get_acc R4_best1_s2)) / 3" | bc)
B2_MEAN=$(echo "scale=1; ($(get_acc $BEST2) + $(get_acc R4_best2_s1) + $(get_acc R4_best2_s2)) / 3" | bc)
echo ""
echo "  Best1 3-seed mean: ${B1_MEAN}%"
echo "  Best2 3-seed mean: ${B2_MEAN}%"

if (( $(echo "$B1_MEAN >= $B2_MEAN" | bc -l) )); then
    FINAL_ARGS=$BEST1_ARGS
    FINAL_NAME=$BEST1
    FINAL_MEAN=$B1_MEAN
else
    FINAL_ARGS=$BEST2_ARGS
    FINAL_NAME=$BEST2
    FINAL_MEAN=$B2_MEAN
fi

echo ""
echo "★ FINAL CONFIG: $FINAL_NAME = ${FINAL_MEAN}%"
echo "  Args: $FINAL_ARGS"

# Save sweep summary
ssh_run "cat > ${RESULTS_BASE}/sweep_summary.json << SUMEOF
{
  \"best_config\": \"${FINAL_NAME}\",
  \"best_args\": \"${FINAL_ARGS}\",
  \"best_3seed_mean\": ${FINAL_MEAN},
  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
}
SUMEOF"

# ═══ PHASE 2: Paper Re-run ═══
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  PHASE 2: Full Paper Re-run (18 experiments)         ║"
echo "║  Config: ${FINAL_ARGS}"
echo "╚══════════════════════════════════════════════════════╝"

# Already have MTH s0/s1/s2 from sweep — skip those, run remaining 15
# Actually, re-run all 18 for clean results with output_dir consistency

run_batch \
    "paper_MTH_s0" "${FINAL_ARGS}" 0 MTH \
    "paper_MTH_s1" "${FINAL_ARGS}" 1 MTH \
    "paper_MTH_s2" "${FINAL_ARGS}" 2 MTH \
    "paper_MHT_s0" "${FINAL_ARGS}" 0 MHT

run_batch \
    "paper_MHT_s1" "${FINAL_ARGS}" 1 MHT \
    "paper_MHT_s2" "${FINAL_ARGS}" 2 MHT \
    "paper_TMH_s0" "${FINAL_ARGS}" 0 TMH \
    "paper_TMH_s1" "${FINAL_ARGS}" 1 TMH

run_batch \
    "paper_TMH_s2" "${FINAL_ARGS}" 2 TMH \
    "paper_THM_s0" "${FINAL_ARGS}" 0 THM \
    "paper_THM_s1" "${FINAL_ARGS}" 1 THM \
    "paper_THM_s2" "${FINAL_ARGS}" 2 THM

run_batch \
    "paper_HMT_s0" "${FINAL_ARGS}" 0 HMT \
    "paper_HMT_s1" "${FINAL_ARGS}" 1 HMT \
    "paper_HMT_s2" "${FINAL_ARGS}" 2 HMT \
    "paper_HTM_s0" "${FINAL_ARGS}" 0 HTM

# Last 2 experiments (only 2 GPUs needed, pad with duplicates)
run_batch \
    "paper_HTM_s1" "${FINAL_ARGS}" 1 HTM \
    "paper_HTM_s2" "${FINAL_ARGS}" 2 HTM \
    "paper_HTM_s2_dup" "${FINAL_ARGS}" 2 HTM \
    "paper_HTM_s1_dup" "${FINAL_ARGS}" 1 HTM

# ═══ FINAL SUMMARY TABLE ═══
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  FINAL RESULTS TABLE                                 ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "Config: ${FINAL_ARGS}"
echo ""
printf "%-8s %8s %8s %8s %8s\n" "Order" "Seed0" "Seed1" "Seed2" "Mean"
echo "─────────────────────────────────────────────"

OVERALL_SUM=0
OVERALL_N=0
for ORDER in MTH MHT TMH THM HMT HTM; do
    S0=$(get_acc "paper_${ORDER}_s0")
    S1=$(get_acc "paper_${ORDER}_s1")
    S2=$(get_acc "paper_${ORDER}_s2")
    if [ -n "$S0" ] && [ -n "$S1" ] && [ -n "$S2" ]; then
        MEAN=$(echo "scale=1; ($S0 + $S1 + $S2) / 3" | bc)
        OVERALL_SUM=$(echo "$OVERALL_SUM + $MEAN" | bc)
        OVERALL_N=$((OVERALL_N + 1))
    else
        MEAN="??"
    fi
    printf "%-8s %7s%% %7s%% %7s%% %7s%%\n" "$ORDER" "${S0:-??}" "${S1:-??}" "${S2:-??}" "$MEAN"
done

if [ $OVERALL_N -gt 0 ]; then
    OVERALL=$(echo "scale=1; $OVERALL_SUM / $OVERALL_N" | bc)
    echo ""
    echo "  Overall 6-order mean: ${OVERALL}%"
fi

# ═══ Pull to local ═══
echo ""
echo "Pulling results to local (excluding checkpoints and features)..."
mkdir -p "$LOCAL_RESULTS"

ssh_run "cd ${RESULTS_BASE} && find . \( -name '*.json' -o -name '*.csv' -o -name '*.log' -o -name '*.yaml' \) | tar czf /tmp/overnight_results.tar.gz -T -"
sshpass -p "$SSH_PASS" scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
    root@"$SSH_HOST":/tmp/overnight_results.tar.gz "$LOCAL_RESULTS/overnight_results.tar.gz" 2>/dev/null

cd "$LOCAL_RESULTS" && tar xzf overnight_results.tar.gz && rm overnight_results.tar.gz

echo ""
echo "═══════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE: $(date)"
echo "  Sweep: 16 configs (4 rounds × 4 GPUs)"
echo "  Paper: 18 experiments (6 orders × 3 seeds)"
echo "  Results: $LOCAL_RESULTS"
echo "═══════════════════════════════════════════════════"
