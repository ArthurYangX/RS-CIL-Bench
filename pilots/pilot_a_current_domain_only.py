"""Pilot A: Only recompute prototypes/SHINE for the current domain.

Theory: With domain-selective inference, other domains' adapters haven't changed,
so their prototypes don't need updating.

Apply this patch to cmcd_lora_experiment.py around line 2601.
"""

# ============================================================
# ORIGINAL CODE (line ~2601):
# ============================================================
# for ds in dataset_order:
#     ds_cls_seen = [c for c in seen_classes if class_to_domain.get(c) == ds]
#     if not ds_cls_seen:
#         continue
#     ds_subset = subset_by_classes(train_padded[ds], set(ds_cls_seen))
#     ...extract features, update prototypes, update SHINE stats...

# ============================================================
# PATCHED CODE:
# ============================================================
# Determine which domains need recomputation.
# - During warmup: backbone is changing globally, but only current domain has data
# - After warmup: only the current domain's adapter changed
# So in both cases, only current domain needs recomputation.

# recompute_domains = [ds_name]  # <-- THE KEY CHANGE
#
# for ds in recompute_domains:
#     ds_cls_seen = [c for c in seen_classes if class_to_domain.get(c) == ds]
#     if not ds_cls_seen:
#         continue
#     ds_subset = subset_by_classes(train_padded[ds], set(ds_cls_seen))
#     ...rest unchanged...

# ============================================================
# EXPECTED RESULT: identical to full recomputation (< 0.1pp diff)
# because domain-selective inference ensures other domains' feature
# pipelines are unchanged.
# ============================================================

# For evaluation metrics, we still need all domains' prototypes
# and SHINE stats — but they should already be cached from their
# last update. No re-extraction needed for frozen domains.
# The evaluation loop (test set) does NOT need train data access.
