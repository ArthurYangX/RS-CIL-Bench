"""Pilot C: Statistics-only exemplar-free mode (L1).

Goal: Replace ALL raw-data access with prototype-based pseudo replay.
      LoRA trains on current-task new classes + proto_aug pseudo features for old classes.
      Prototypes corrected via Eq.6 (Pilot B). SHINE only for current domain (Pilot A).

This is the most aggressive exemplar-free variant.
Route A failed (33.9%) because it had ZERO old-class replay.
Pilot C adds proto_aug pseudo replay — sampling from stored prototypes —
which provides old-class gradient signal without accessing raw data.

Storage: only 101 KB (prototypes + SHINE stats + LoRA adapters).

Changes needed in cmcd_lora_experiment.py:
1. LoRA training data: ONLY current task's new classes (not all domain classes)
2. Enable proto_aug for old-class pseudo replay: --n_pseudo 16 --lambda_pseudo 0.5
3. Prototype update: Eq.6 analytic correction (Pilot B), not re-extraction
4. SHINE stats: only current domain (Pilot A)
5. Cross-domain: zero data access

Run command (after applying Pilot A + B patches):
  python cmcd_lora_experiment.py --mode marathon --seed 0 --dataset_order MTH \\
    --lora_rank 4 --lidar_rank 8 \\
    --warmup_tasks 3 --lambda_domain_kd 0.5 \\
    --domain_conditioned_reuse \\
    --proto_components 2 --proto_score_mode mixture \\
    --n_pseudo 16 --lambda_pseudo 0.5 \\
    --pilot_mode c  # custom flag to activate current-task-only training data

Key code change for LoRA training data (line ~2478):
  # ORIGINAL:
  # ds_seen_cls = set(c for c in seen_classes if class_to_domain.get(c) == ds_name)
  # all_domain_ds = subset_by_classes(train_padded[ds_name], ds_seen_cls)

  # PILOT C:
  # Only use current task's NEW classes for real data
  # Old classes get pseudo replay via proto_aug (--n_pseudo 16 --lambda_pseudo 0.5)
  all_domain_ds = subset_by_classes(train_padded[ds_name], task_class_set)

Combined with:
  - Pilot A: prototype/SHINE only recomputed for current domain
  - Pilot B: Eq.6 analytic DCR for prototype correction
  - proto_aug: n_pseudo=16 per old class, lambda_pseudo=0.5

Expected result:
  - Route A (no replay at all): 33.9% (failed)
  - Pilot C (proto_aug replay): hopefully 75-82%?
  - Full method (raw data replay): 84.5%

Even if Pilot C loses a few pp vs full, it's a valid exemplar-free mode
that stores only 101 KB and never accesses any training sample.
"""
