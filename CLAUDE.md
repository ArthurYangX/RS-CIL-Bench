# Project Instructions

## Remote Server

- SSH: `ssh -p <PORT> root@connect.westc.seetacloud.com`
- Password: (see AutoDL console, changes per session)
- GPU: 1x NVIDIA RTX 4090
- CPU: 16 cores
- Memory: 90GB

- Conda environment: `jc`  

- Activate before any Python command:
  `eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda activate jc`

- Code directory:
  `/root/autodl-tmp/`   

- Run long jobs in background with:
  `screen -dmS exp_${RANDOM} bash -lc 'COMMAND'`

---

## Experiment Execution Rules

- Always run experiments on the remote GPU server.
- Never run GPU-heavy jobs locally.

- Before launching experiments, briefly explain the plan.

- Prefer minimal local changes.
- Do not rewrite the entire training pipeline unless necessary.

- Run at most 2 experiments per iteration unless clearly justified.
- Prefer high-impact, low-cost experiments.

- All experiments must:
  - have a clear purpose
  - be logged
  - have a short summary after completion

---

## Code Safety Rules

- Do not delete core training code.
- Avoid large refactors unless explicitly required.
- Keep changes minimal and reversible.

---

## Research Goals

- Improve experimental credibility
- Strengthen ablations and baselines
- Ensure results are statistically reliable
- Avoid overclaiming

Focus on:
- ablation completeness
- fair comparison
- robustness (multi-seed if needed)

距离输入一个需求，自动迭代验证，最终吐出一个芯片
