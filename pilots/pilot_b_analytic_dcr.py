"""Pilot B: Analytic DCR (Eq. 6) — correct same-domain prototypes via matrix
multiplication instead of re-extracting features from training data.

Theory (Lemma 1): For single-layer linear LoRA,
  corrected_proto = (I + alpha * W_up @ W_down) @ old_proto
is exact. Multi-block introduces approximation error.

This pilot tests whether Eq. 6 is a viable replacement for the current
implicit DCR (full re-extraction from training data).
"""

import torch
import torch.nn.functional as F

BRANCHES = ("spec", "hsi_spa", "lid_spa")


def analytic_dcr_correction(
    prototype_store,
    anchor_prototype_store,
    model,
    domain_name,
    class_to_domain,
    seen_classes,
    device,
):
    """Apply Eq. 6 to correct same-domain prototypes after adapter training.

    For each class c in the given domain:
      - Spectral prototype: no correction (spectral branch is frozen)
      - HSI-spatial prototype: apply accumulated HSI LoRA transformation
      - LiDAR-spatial prototype: apply accumulated LiDAR LoRA transformation

    The transformation for a multi-block LoRA is the sequential composition:
      h -> LoRA_block0(LoRA_block1(...(LoRA_blockN(h))...))

    For a SINGLE block, this is exact: (I + alpha * W_up @ W_down) @ h
    For MULTIPLE blocks, we compose them sequentially as an approximation.
    """
    domain_classes = [
        c for c in seen_classes if class_to_domain.get(c) == domain_name
    ]
    if not domain_classes:
        return

    # Get the domain's adapter key
    hsi_bank = model.hsi_lora_bank
    lid_bank = model.lidar_lora_bank

    # Find which adapter this domain uses
    hsi_adapter_key = None
    lid_adapter_key = None
    if hasattr(hsi_bank, 'domain_to_adapter'):
        hsi_adapter_key = hsi_bank.domain_to_adapter.get(domain_name)
        lid_adapter_key = lid_bank.domain_to_adapter.get(domain_name)

    if hsi_adapter_key is None or lid_adapter_key is None:
        # No adapter for this domain (e.g., warmup domain) — no correction needed
        return

    # Build the composite transformation matrix for each branch
    # For multi-block LoRA: T = (I + a*U_N*D_N) @ ... @ (I + a*U_1*D_1) @ (I + a*U_0*D_0)
    # Applied to the GAP-pooled feature vector (not 2D feature map)
    #
    # IMPORTANT: LoRA is applied to 2D feature maps (B,C,H,W) via 1x1 conv.
    # After GAP, the effect on the pooled vector is the same as the per-pixel
    # linear transformation (since 1x1 conv is equivalent to per-pixel linear).

    def build_transform_matrix(lora_bank, adapter_key, dim):
        """Build the composite linear transform from multi-block LoRA."""
        T = torch.eye(dim, device=device)
        if adapter_key not in lora_bank.task_loras:
            return T
        loras = lora_bank.task_loras[adapter_key]
        for block_idx in range(len(loras)):
            lora = loras[block_idx]
            if lora.rank <= 0:
                continue
            D = lora.get_down_matrix(detach=True).to(device)  # (rank, dim)
            U = lora.get_up_matrix(detach=True).to(device)    # (dim, rank)
            alpha = lora.scale
            # (I + alpha * U @ D)
            block_T = torch.eye(dim, device=device) + alpha * (U @ D)
            T = block_T @ T  # sequential composition
        return T

    dim = 64  # feature dimension per branch
    T_hsi = build_transform_matrix(hsi_bank, hsi_adapter_key, dim)
    T_lid = build_transform_matrix(lid_bank, lid_adapter_key, dim)

    # Apply correction to each class's prototype
    for c in domain_classes:
        if c not in prototype_store.prototypes:
            continue

        # Correct spatial prototypes
        for b, T_mat in [("hsi_spa", T_hsi), ("lid_spa", T_lid)]:
            if b in prototype_store.prototypes[c]:
                old_proto = prototype_store.prototypes[c][b].to(device)
                new_proto = T_mat @ old_proto
                prototype_store.prototypes[c][b] = new_proto.cpu()

            # Also correct component means if using GMM
            if c in prototype_store.component_means:
                if b in prototype_store.component_means[c]:
                    old_means = prototype_store.component_means[c][b].to(device)
                    # old_means: (K, dim)
                    new_means = (T_mat @ old_means.t()).t()
                    prototype_store.component_means[c][b] = new_means.cpu()

        # Spectral prototype: no correction (frozen branch)
        # Spectral component means: no correction

    # Anchor prototypes: spectral only, no correction needed


def apply_pilot_b(
    prototype_store,
    anchor_prototype_store,
    model,
    ds_name,
    class_to_domain,
    seen_classes,
    device,
    domain_stats,
    train_padded,
    eval_batch_size,
    domain_selective,
):
    """Replacement for the full recomputation loop (Pilot B).

    Instead of re-extracting features from ALL training data:
    1. Apply Eq.6 analytic correction to current-domain spatial prototypes
    2. Only re-extract current-domain data for SHINE stats update
       (SHINE stats cannot be analytically corrected — they need actual features)
    3. Skip all cross-domain data access entirely
    """
    from torch.utils.data import DataLoader

    # Step 1: Analytic DCR correction for current domain prototypes
    analytic_dcr_correction(
        prototype_store,
        anchor_prototype_store,
        model,
        ds_name,
        class_to_domain,
        seen_classes,
        device,
    )

    # Step 2: Re-extract current domain for SHINE stats only
    # (SHINE needs actual feature statistics, not corrected prototypes)
    ds_cls_seen = [c for c in seen_classes if class_to_domain.get(c) == ds_name]
    if ds_cls_seen:
        from anchor_lora_experiment import (
            subset_by_classes, extract_features, compute_domain_stats, BRANCHES
        )
        ds_subset = subset_by_classes(train_padded[ds_name], set(ds_cls_seen))
        if ds_subset is not None:
            ds_loader = DataLoader(
                ds_subset, batch_size=eval_batch_size,
                shuffle=False, drop_last=False
            )
            ds_task_ids = (
                model.get_domain_task_ids(ds_name)
                if domain_selective else None
            )
            ds_feats, ds_labels = extract_features(
                model, ds_loader, device, active_task_ids=ds_task_ids
            )
            domain_stats[ds_name] = compute_domain_stats(
                ds_feats, ds_labels, set(ds_cls_seen)
            )

            # ALSO update current-domain prototypes from real features
            # (more accurate than Eq.6 for multi-block LoRA)
            for c in ds_cls_seen:
                mask_c = ds_labels == c
                if mask_c.sum() > 0:
                    class_feats = {
                        b: ds_feats[b][mask_c]
                        for b in BRANCHES if ds_feats[b] is not None
                    }
                    prototype_store.update(c, class_feats)

    # Step 3: Cross-domain — do nothing (prototypes and stats cached from before)


# ============================================================
# INTEGRATION NOTE:
#
# In cmcd_lora_experiment.py, replace the evaluation loop
# (lines ~2595-2678) with a call to this function for the
# current domain, followed by the existing test-set evaluation.
#
# The test-set evaluation does NOT need training data access —
# it only uses the prototype_store and domain_stats that we
# already updated above.
# ============================================================
