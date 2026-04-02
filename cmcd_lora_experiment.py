#!/usr/bin/env python3
"""CMCD-LoRA: Domain-Conditioned SD-LoRA + Domain-Aware Feature KD.

Combines two ideas on top of the AnchorLoRA baseline:
  1. Domain-Conditioned SD-LoRA Reuse:
     - Tasks in the SAME domain reuse a single LoRA adapter (shared adapter within domain)
     - Tasks in a DIFFERENT domain allocate a new adapter
     - Replaces the broken divergence-based decision in SDLoRABank
  2. Domain-Aware Feature KD from a Teacher Bank:
     - Stores best checkpoint per domain as a teacher
     - During LoRA training, distills features from domain-specific teacher
     - Uses importance-weighted cosine distance (inter-class variance per dimension)

Usage:
  python cmcd_lora_experiment.py --mode marathon --seed 0 --dataset_order MTH
  python cmcd_lora_experiment.py --mode marathon --seed 0 --dataset_order MTH --domain_conditioned_reuse
  python cmcd_lora_experiment.py --mode marathon --seed 0 --dataset_order MTH --lambda_domain_kd 0.5
  python cmcd_lora_experiment.py --mode marathon --seed 0 --dataset_order MTH --domain_conditioned_reuse --lambda_domain_kd 0.5
"""
import sys
import os
import json
import argparse
import random
import copy
import math
import time

os.chdir("/root/autodl-tmp/jc")
sys.path.insert(0, "/root/autodl-tmp/jc")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from collections import defaultdict

from datasets.my_dataset import get_loader
from config.paths import get_dataname_to_path
from networks.s2cm_net import S2CMNet
from networks.network import LLL_Net

from anchor_lora_experiment import (
    UNIFIED_HSI_BANDS,
    UNIFIED_LIDAR_CHANS,
    BRANCHES,
    PaddedDataset,
    collate_fn,
    subset_by_classes,
    compute_domain_stats,
    apply_shine,
    shine_prototypes,
    PrototypeStore,
    cosine_ncm_logits,
    predict_ncm,
    predict_shine,
    compute_metrics,
    extract_features,
    AnchorLoRAModel,
    TaskLoRABank,
    LoRAAdapter2D,
    SpectralDriftMonitor,
    LiDARChannelAdapter,
    train_warmup_task,
    train_lora_task,
    spectral_anchor_kd_loss,
    prototype_consistency_loss,
    proto_aug_features,
)

# Try importing from anchor_lora_experiment; define fallbacks if missing
try:
    from anchor_lora_experiment import resolve_bootstrap_checkpoint
except ImportError:
    def resolve_bootstrap_checkpoint(dataset_order, seed):
        """Fallback: return the default checkpoint path."""
        return "/root/autodl-tmp/results/s2cm/marathon_task0.pth"

try:
    from anchor_lora_experiment import DEFAULT_RESULTS_ROOT
except ImportError:
    DEFAULT_RESULTS_ROOT = "/root/autodl-tmp/results/s2cm"


# ======================================================================
# SPECTRAL DRIFT GATE (CMDA core mechanism)
# ======================================================================
class SpectralDriftGate(nn.Module):
    """Cross-Modal Drift-Gated Adaptation (CMDA).

    Uses frozen spectral features as a zero-cost drift detector to modulate
    LoRA adaptation strength per sample.

    Physics prior: spectral features reflect material absorption (stable across
    geographic domains), so spectral "surprise" indicates domain shift.

    gate(x) = sigmoid(w * drift_score + b)
    where drift_score = min_c ||z_spec - proto_c|| (distance to nearest old prototype)

    High drift → gate ≈ 1 → more LoRA adaptation (plasticity)
    Low drift  → gate ≈ 0 → more frozen backbone (stability)
    """

    def __init__(self, embed_dim):
        super().__init__()
        # Init so gate has wide dynamic range:
        # drift=0 → sigmoid(5*0-2.5)=0.08, drift=1 → sigmoid(5*1-2.5)=0.92
        self.w = nn.Parameter(torch.tensor(5.0))
        self.b = nn.Parameter(torch.tensor(-2.5))
        self.embed_dim = embed_dim
        # Registered buffer for old-class spectral prototypes
        self._proto_mat = None  # (n_old_classes, embed_dim)
        self._proto_norm = None

    def update_prototypes(self, prototype_store, old_classes):
        """Update the spectral prototype matrix from stored prototypes."""
        protos = []
        for c in sorted(old_classes):
            if c in prototype_store.prototypes and "spec" in prototype_store.prototypes[c]:
                protos.append(prototype_store.prototypes[c]["spec"])
        if protos:
            self._proto_mat = torch.stack(protos)  # (K, D)
            self._proto_norm = F.normalize(self._proto_mat, dim=1)
        else:
            self._proto_mat = None
            self._proto_norm = None

    def forward(self, z_spec):
        """Compute per-sample drift gate from spectral features.

        Args:
            z_spec: (B, D) frozen spectral features

        Returns:
            gate: (B, 1) values in [0, 1]
        """
        if self._proto_mat is None:
            # No old prototypes yet → full adaptation
            return torch.ones(z_spec.shape[0], 1, device=z_spec.device)

        proto_norm = self._proto_norm.to(z_spec.device)
        z_norm = F.normalize(z_spec, dim=1)

        # Cosine similarity to nearest old prototype
        sim = z_norm @ proto_norm.t()  # (B, K)
        max_sim, _ = sim.max(dim=1)  # (B,) — similarity to closest old class

        # Drift score: 1 - max_sim (0 = identical to old class, 1 = fully novel)
        drift_score = 1.0 - max_sim  # (B,)

        # Learnable gate
        gate = torch.sigmoid(self.w * drift_score + self.b)  # (B,)
        return gate.unsqueeze(1)  # (B, 1)


# ======================================================================
# DOMAIN-CONDITIONED SD-LORA BANK
# ======================================================================
class DomainConditionedLoRABank(TaskLoRABank):
    """LoRA bank with domain-conditioned adapter reuse.

    Instead of the divergence-based decision of SDLoRABank, this uses a
    simple domain-conditioned rule:
      - If the new task is in the SAME domain as an existing task -> reuse that adapter
      - If the new task is in a DIFFERENT domain -> allocate a new adapter

    This is motivated by the observation that tasks within the same domain
    (e.g., consecutive splits of Houston) share similar spatial patterns,
    so reusing the LoRA adapter prevents unnecessary parameter growth and
    preserves intra-domain knowledge.
    """

    def __init__(self, num_blocks, dim, rank, modality_name="hsi"):
        super().__init__(num_blocks, dim, rank, modality_name)
        self.task_to_adapter = {}
        self.domain_to_adapter = {}

    def add_task(self, task_id, domain_name=None):
        """Add LoRA adapters for a new task, with domain-conditioned reuse.

        If domain_name is provided and a task in the same domain already has
        an adapter, reuse it. Otherwise, allocate a new adapter.
        """
        tid = str(task_id)

        if domain_name is not None and domain_name in self.domain_to_adapter:
            # Same domain -> reuse existing adapter
            reuse_key = self.domain_to_adapter[domain_name]
            self.task_to_adapter[tid] = reuse_key
            print(f"    [{self.modality_name}] Task {task_id} reuses adapter "
                  f"'{reuse_key}' (domain={domain_name})")

            # Unfreeze the reused adapter so it can continue training
            if reuse_key in self.task_loras:
                for param in self.task_loras[reuse_key].parameters():
                    param.requires_grad = True
            return

        # Different domain or first task -> new adapter
        adapter_key = tid
        loras = nn.ModuleList([
            LoRAAdapter2D(self.dim, self.rank) for _ in range(self.num_blocks)
        ])
        self.task_loras[adapter_key] = loras
        self.task_to_adapter[tid] = adapter_key

        if domain_name is not None:
            self.domain_to_adapter[domain_name] = adapter_key

        print(f"    [{self.modality_name}] Task {task_id} NEW adapter "
              f"'{adapter_key}' (domain={domain_name})")

    def freeze_task(self, task_id):
        """Freeze the adapter used by this task and update subspace basis."""
        tid = str(task_id)
        adapter_key = self.task_to_adapter.get(tid, tid)

        if adapter_key not in self.task_loras:
            return

        for param in self.task_loras[adapter_key].parameters():
            param.requires_grad = False

        # Update subspace basis
        for block_idx, lora in enumerate(self.task_loras[adapter_key]):
            down_mat = lora.get_down_matrix(detach=True)
            if block_idx not in self.subspace_bases:
                self.subspace_bases[block_idx] = down_mat
            else:
                existing = self.subspace_bases[block_idx]
                if existing.shape[0] < (len(self.task_loras) * self.rank):
                    self.subspace_bases[block_idx] = torch.cat(
                        [existing, down_mat], dim=0
                    )

    def get_active_task_id(self):
        """Get the adapter key of the currently trainable LoRA."""
        for adapter_key in self.task_loras:
            for param in self.task_loras[adapter_key].parameters():
                if param.requires_grad:
                    return adapter_key
        task_ids = list(self.task_loras.keys())
        return task_ids[-1] if task_ids else None

    def forward_block(self, x, block_idx, active_task_ids=None):
        """Override: apply only UNIQUE adapters, not aliased duplicates.
        Sequential composition (same as base class) but deduplicates
        reused adapters to avoid applying the same adapter twice.
        """
        if active_task_ids is not None:
            seen_adapters = set()
            for tid in active_task_ids:
                actual_id = self.task_to_adapter.get(str(tid), str(tid))
                if actual_id not in seen_adapters and actual_id in self.task_loras:
                    seen_adapters.add(actual_id)
                    x = self.task_loras[actual_id][block_idx](x)
        else:
            seen_adapters = set()
            for tid in self.task_loras:
                actual_id = self.task_to_adapter.get(tid, tid)
                if actual_id not in seen_adapters:
                    seen_adapters.add(actual_id)
                    x = self.task_loras[actual_id][block_idx](x)
        return x


# ======================================================================
# DOMAIN-AWARE TEACHER BANK (for feature KD)
# ======================================================================
class DomainTeacherBank:
    """Stores best model checkpoint per domain for cascade KD."""

    def __init__(self):
        self.teachers = {}
        self.universal_teacher = None

    def update(self, domain, model, metric):
        """Update teacher for a domain if metric improves."""
        if domain not in self.teachers or metric > self.teachers[domain][1]:
            self.teachers[domain] = (copy.deepcopy(model.state_dict()), metric)
            print(f"    Teacher bank: updated '{domain}' (metric={metric:.4f})")

    def set_universal(self, model):
        """Set the universal teacher (typically after task 0)."""
        self.universal_teacher = copy.deepcopy(model.state_dict())

    def get_teacher_state(self, domain):
        """Get teacher state dict for a domain, or universal fallback."""
        if domain in self.teachers:
            return self.teachers[domain][0]
        return self.universal_teacher

    def get_previous_domain_teacher(self, current_domain, domain_order):
        """Get teacher from the domain preceding current_domain in the order."""
        try:
            idx = domain_order.index(current_domain)
        except ValueError:
            return None
        if idx == 0:
            # First domain -> use universal teacher
            return self.universal_teacher
        prev_domain = domain_order[idx - 1]
        return self.get_teacher_state(prev_domain)


# ======================================================================
# IMPORTANCE-WEIGHTED FEATURE KD
# ======================================================================
def compute_importance_weights(prototype_store, seen_classes, branch_dims):
    """Compute per-dimension importance weights based on inter-class variance."""
    protos = prototype_store.all_protos()
    cls_ids = sorted(c for c in seen_classes if c in protos)

    total_dim = sum(branch_dims.values())
    if len(cls_ids) < 2:
        return torch.ones(total_dim)

    all_p = []
    for c in cls_ids:
        parts = []
        for b in BRANCHES:
            if b in protos[c]:
                parts.append(protos[c][b])
        if parts:
            all_p.append(torch.cat(parts))

    if not all_p:
        return torch.ones(total_dim)

    P = torch.stack(all_p)
    var = P.var(dim=0).clamp(min=1e-8)
    weights = var / var.mean()
    weights = weights.clamp(min=0.1, max=10.0)
    return weights


def domain_feature_kd_loss(student_feats, teacher_feats, importance_weights=None):
    """Importance-weighted cosine distance for feature-level KD.

    For each branch, computes:
        loss = sum_d w_d * (1 - cos_sim(student_d, teacher_d))

    where w_d is the importance weight for dimension d.
    """
    losses = []
    offset = 0
    for b in BRANCHES:
        if b not in student_feats or b not in teacher_feats:
            continue
        s = student_feats[b]
        t = teacher_feats[b].detach()

        if s is None or t is None:
            continue

        dim = s.shape[1]

        # Cosine distance per sample: 1 - cos_sim
        cos_sim = F.cosine_similarity(s, t, dim=1)  # (B,)
        cos_dist = 1.0 - cos_sim  # (B,)

        if importance_weights is not None:
            # Per-dimension weighted cosine distance
            # Compute element-wise weighted squared difference
            s_norm = F.normalize(s, dim=1)
            t_norm = F.normalize(t, dim=1)
            diff_sq = (s_norm - t_norm) ** 2  # (B, D)
            w = importance_weights[offset:offset + dim].to(s.device)
            w = w / (w.sum() + 1e-8) * dim  # normalize so mean weight = 1
            weighted_dist = (w.unsqueeze(0) * diff_sq).mean()
            losses.append(weighted_dist)
        else:
            losses.append(cos_dist.mean())

        offset += dim

    if not losses:
        device = next(
            (student_feats[b].device for b in BRANCHES
             if b in student_feats and student_feats[b] is not None),
            torch.device('cpu')
        )
        return torch.tensor(0.0, device=device)

    return sum(losses) / len(losses)


# ======================================================================
# MONKEY-PATCH: Domain-conditioned add_task for AnchorLoRAModel
# ======================================================================
def patch_model_for_domain_conditioned_reuse(model, hsi_rank, lidar_rank):
    """Replace model's TaskLoRABank instances with DomainConditionedLoRABank.

    This is done after model creation so we can reuse all of AnchorLoRAModel's
    other logic (forward, warm-up, etc.) without subclassing.
    """
    num_blocks = model.num_spatial_blocks
    embed_dim = model.backbone.embed_dim

    # Replace the banks
    model.hsi_lora_bank = DomainConditionedLoRABank(
        num_blocks, embed_dim, hsi_rank, modality_name="hsi"
    )
    model.lidar_lora_bank = DomainConditionedLoRABank(
        num_blocks, embed_dim, lidar_rank, modality_name="lidar"
    )

    # Patch begin_task to pass domain_name to add_task
    original_begin_task = model.begin_task.__func__

    def domain_aware_begin_task(self, task_id, is_warmup=False, domain_name=None):
        """Patched begin_task that passes domain_name to LoRA banks."""
        self.current_task = task_id

        if is_warmup:
            for p in self.backbone.parameters():
                p.requires_grad = True
            for p in self.lidar_channel_adapter.parameters():
                p.requires_grad = True
        else:
            self.freeze_backbone()
            # Domain-conditioned add_task
            self.hsi_lora_bank.add_task(task_id, domain_name=domain_name)
            self.lidar_lora_bank.add_task(task_id, domain_name=domain_name)
            device = next(self.backbone.parameters()).device
            self.hsi_lora_bank.to(device)
            self.lidar_lora_bank.to(device)

    import types
    model.begin_task = types.MethodType(domain_aware_begin_task, model)

    return model


def patch_model_with_drift_gate(model):
    """Add SpectralDriftGate to AnchorLoRAModel.

    Patches forward_features so that LoRA contributions are gated by
    spectral drift score per sample (CMDA mechanism).
    """
    embed_dim = model.backbone.embed_dim
    model.drift_gate = SpectralDriftGate(embed_dim)
    device = next(model.backbone.parameters()).device
    model.drift_gate.to(device)

    import types

    def gated_forward_features(self, x_hsi, x_lidar, return_aux=True,
                                active_task_ids=None, use_lora=None):
        """Forward with spectral drift-gated LoRA adaptation (CMDA)."""
        backbone = self.backbone
        apply_lora = use_lora if use_lora is not None else self.warmup_done

        # Spectral branch (frozen, stable)
        f_spec = backbone.spectral_branch(x_hsi)  # (B, D)

        # Compute drift gate from spectral features (CMDA core)
        if apply_lora and hasattr(self, 'drift_gate'):
            gate = self.drift_gate(f_spec.detach())  # (B, 1) — no grad to spectral
        else:
            gate = None

        # LiDAR channel adaptation
        x_lidar_adapted = self.lidar_channel_adapter(x_lidar)

        # Spatial projections
        h_map = backbone.hsi_spatial_proj(x_hsi)
        l_map = backbone.lidar_proj(x_lidar_adapted)

        h_spa_map = h_map
        l_spa_map = l_map

        for block_idx, block in enumerate(backbone.spatial_branch.blocks):
            h_out = block(h_spa_map)
            l_out = block(l_spa_map)
            if isinstance(h_out, tuple):
                h_out = h_out[0]
            if isinstance(l_out, tuple):
                l_out = l_out[0]

            # Apply drift-gated LoRA
            if apply_lora and len(self.hsi_lora_bank.task_loras) > 0:
                # Standard LoRA output
                h_lora = self.hsi_lora_bank.forward_block(
                    h_out, block_idx, active_task_ids=active_task_ids)
                l_lora = self.lidar_lora_bank.forward_block(
                    l_out, block_idx, active_task_ids=active_task_ids)

                if gate is not None:
                    # gate: (B, 1), need (B, 1, 1, 1) for spatial features
                    g = gate.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
                    # Gated: interpolate between frozen output and LoRA output
                    h_out = h_out + g * (h_lora - h_out)
                    l_out = l_out + g * (l_lora - l_out)
                else:
                    h_out = h_lora
                    l_out = l_lora

            h_spa_map = h_out
            l_spa_map = l_out

        # Cross fusion
        if backbone.cross_fusion is not None:
            h_spa_map, l_spa_map = backbone.cross_fusion(h_spa_map, l_spa_map)

        f_hsi_spa = backbone.spatial_branch.norm(h_spa_map.mean(dim=(2, 3)))
        f_lid_spa = backbone.spatial_branch.norm(l_spa_map.mean(dim=(2, 3)))

        if return_aux:
            return {
                'f_spec': f_spec,
                'f_hsi_spa': f_hsi_spa,
                'f_lid_spa': f_lid_spa,
                'drift_gate': gate,  # For logging
                'h_map': h_map.detach() if h_map is not None else None,
                'l_map': l_map.detach() if l_map is not None else None,
            }

        fused = torch.cat([f_spec, f_hsi_spa, f_lid_spa], dim=1)
        return backbone.fusion_proj(fused)

    model.forward_features = types.MethodType(gated_forward_features, model)
    print(f"  CMDA drift gate installed (2 learnable params: w, b)")
    return model


# ======================================================================
# TRAINING: LoRA PHASE with domain feature KD
# ======================================================================
def train_lora_task_with_domain_kd(
    model, train_loader, seen_classes, old_protos,
    prototype_store, class_to_domain, device,
    teacher_model=None, importance_weights=None,
    epochs=50, lr=5e-4,
    lambda_proto=1.0, lambda_kd=1.0,
    lambda_ortho=0.1, lambda_domain_kd=0.5,
    n_pseudo=8, lambda_pseudo=0.5, noise_scale=1.0,
):
    """Train LoRA adapters with domain-aware feature KD + prototype replay.

    Extends train_lora_task with an additional loss term:
      loss_domain_kd = importance-weighted cosine distance to teacher features.
    """
    model.train()
    lora_params = [p for p in model.parameters() if p.requires_grad]
    if not lora_params:
        print("    [LoRA] WARNING: No trainable parameters!")
        return

    n_trainable = sum(p.numel() for p in lora_params)
    print(f"    [LoRA+DKD] Trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    old_classes = set(old_protos.keys()) if old_protos else set()

    old_protos_snapshot = {}
    for c in old_classes:
        old_protos_snapshot[c] = {
            b: old_protos[c][b].clone()
            for b in BRANCHES if b in old_protos.get(c, {})
        }

    # Prepare teacher for domain KD
    if teacher_model is not None:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    has_domain_kd = (teacher_model is not None and lambda_domain_kd > 0)

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        model.train()

        for batch in train_loader:
            xh, xl, tgt = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
            )

            aux = model.forward_features(xh, xl, return_aux=True)
            feats = {b: aux[f'f_{b}'] for b in BRANCHES}

            # Build prototypes: in-batch + stored
            current_protos = {}
            for c in seen_classes:
                mask_c = tgt == c
                if mask_c.sum() > 0:
                    current_protos[c] = {
                        b: feats[b][mask_c].mean(0)
                        for b in BRANCHES if feats[b] is not None
                    }
                elif c in prototype_store.prototypes:
                    current_protos[c] = {
                        b: prototype_store.prototypes[c][b].to(device)
                        for b in BRANCHES if b in prototype_store.prototypes[c]
                    }

            available_classes = set(current_protos.keys())
            if len(available_classes) < 2:
                continue

            # CE loss
            logits = cosine_ncm_logits(feats, current_protos, available_classes)
            cids = sorted(available_classes)
            cid_to_idx = {c: i for i, c in enumerate(cids)}
            valid_mask = torch.tensor(
                [t.item() in cid_to_idx for t in tgt], device=device
            )
            if valid_mask.sum() == 0:
                continue
            mapped_targets = torch.tensor(
                [cid_to_idx.get(t.item(), 0) for t in tgt], device=device
            )
            loss_ce = F.cross_entropy(
                logits[valid_mask], mapped_targets[valid_mask]
            )

            # Prototype consistency
            if old_protos and len(old_classes) >= 2:
                loss_proto = prototype_consistency_loss(
                    old_protos, current_protos, old_classes, device=device
                )
            else:
                loss_proto = torch.tensor(0.0, device=device)

            # Orthogonal regularization
            loss_ortho_h = model.hsi_lora_bank.orthogonal_regularization()
            loss_ortho_l = model.lidar_lora_bank.orthogonal_regularization()
            loss_ortho = loss_ortho_h + loss_ortho_l

            # Spectral anchor KD
            if old_protos_snapshot and len(old_classes) >= 2:
                loss_kd = spectral_anchor_kd_loss(
                    feats, old_protos_snapshot, old_classes
                )
            else:
                loss_kd = torch.tensor(0.0, device=device)

            # Domain-aware feature KD
            loss_dkd = torch.tensor(0.0, device=device)
            if has_domain_kd:
                with torch.no_grad():
                    teacher_aux = teacher_model.forward_features(
                        xh, xl, return_aux=True
                    )
                teacher_feats = {b: teacher_aux[f'f_{b}'] for b in BRANCHES}
                loss_dkd = domain_feature_kd_loss(
                    feats, teacher_feats, importance_weights
                )

            # Prototype replay: generate pseudo-features for old classes
            # Key insight: pseudo features must be compared against CURRENT MODEL's
            # live prototypes (from batch), not stored prototypes — otherwise the
            # cosine similarity is trivially high and CE loss ≈ 0 (no gradient).
            #
            # We use the current batch's new-class features as negative anchors:
            # pseudo old-class features should be closer to their own prototype
            # than to any new-class prototype in the current batch.
            loss_pseudo_ce = torch.tensor(0.0, device=device)
            if n_pseudo > 0 and old_classes and prototype_store is not None:
                pf, pl = proto_aug_features(
                    prototype_store, old_classes, device,
                    n_pseudo=n_pseudo, noise_scale=noise_scale)
                if pf is not None and pl is not None:
                    # Build prototype bank with DETACHED old protos + LIVE new protos
                    # Old protos: detached (target), new protos: from current batch (contrastive anchor)
                    replay_protos = {}
                    for c in sorted(available_classes):
                        if c in old_classes and c in old_protos_snapshot:
                            # Old class: use snapshot (fixed target)
                            replay_protos[c] = {
                                b: old_protos_snapshot[c][b].to(device).detach()
                                for b in BRANCHES if b in old_protos_snapshot[c]
                            }
                        elif c in current_protos:
                            # New class: use LIVE batch proto (creates contrastive pressure)
                            replay_protos[c] = {
                                b: current_protos[c][b].detach()
                                for b in BRANCHES if b in current_protos[c]
                            }
                    if len(replay_protos) >= 2:
                        plogits = cosine_ncm_logits(pf, replay_protos, set(replay_protos.keys()))
                        if plogits is not None:
                            p_cids = sorted(replay_protos.keys())
                            p_c2i = {c: i for i, c in enumerate(p_cids)}
                            p_valid = torch.tensor(
                                [t.item() in p_c2i for t in pl], device=device)
                            if p_valid.sum() > 0:
                                p_mapped = torch.tensor(
                                    [p_c2i.get(t.item(), 0) for t in pl], device=device)
                                loss_pseudo_ce = F.cross_entropy(
                                    plogits[p_valid], p_mapped[p_valid])

            loss = (
                loss_ce
                + lambda_proto * loss_proto
                + lambda_kd * loss_kd
                + lambda_ortho * loss_ortho
                + lambda_domain_kd * loss_dkd
                + lambda_pseudo * loss_pseudo_ce
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if avg_loss < best_loss and avg_loss > 0:
            best_loss = avg_loss
            best_state = {
                'hsi_lora': copy.deepcopy(model.hsi_lora_bank.state_dict()),
                'lidar_lora': copy.deepcopy(model.lidar_lora_bank.state_dict()),
            }
            if hasattr(model, 'drift_gate'):
                best_state['drift_gate'] = copy.deepcopy(model.drift_gate.state_dict())

        if (epoch + 1) % 10 == 0:
            print(
                f"    [LoRA+DKD] Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
                f"(CE={loss_ce.item():.3f} proto={loss_proto.item():.3f} "
                f"ortho={loss_ortho.item():.3f} kd={loss_kd.item():.3f} "
                f"dkd={loss_dkd.item():.3f} pCE={loss_pseudo_ce.item():.3f})"
            )

    if best_state is not None:
        model.hsi_lora_bank.load_state_dict(best_state['hsi_lora'])
        model.lidar_lora_bank.load_state_dict(best_state['lidar_lora'])
        if 'drift_gate' in best_state and hasattr(model, 'drift_gate'):
            model.drift_gate.load_state_dict(best_state['drift_gate'])
    model.eval()


# ======================================================================
# MARATHON RUNNER (extends run_anchor_lora_marathon)
# ======================================================================
def run_cmcd_lora_marathon(net, device, args, dataset_order=None):
    """Run the full CIL marathon with CMCD-LoRA components.

    This is structurally similar to run_anchor_lora_marathon but adds:
      - Domain-conditioned SD-LoRA reuse (if --domain_conditioned_reuse)
      - Domain-aware feature KD from teacher bank (if --lambda_domain_kd > 0)
    """
    if dataset_order is None:
        dataset_order = ["MUUFL", "Trento", "Houston"]

    warmup_tasks = args.warmup_tasks
    hsi_rank = args.lora_rank
    lidar_rank = args.lora_rank * args.lidar_rank_mult
    domain_conditioned = getattr(args, 'domain_conditioned_reuse', False)
    lambda_domain_kd = getattr(args, 'lambda_domain_kd', 0.0)

    print(f"\n{'='*80}")
    print(f"CMCD-LoRA Marathon: warmup={warmup_tasks}, "
          f"HSI rank={hsi_rank}, LiDAR rank={lidar_rank}")
    print(f"Domain-conditioned reuse: {domain_conditioned}")
    print(f"Domain KD lambda: {lambda_domain_kd}")
    print(f"Dataset order: {' -> '.join(dataset_order)}")
    print(f"{'='*80}")

    paths = get_dataname_to_path()
    splits_map = {"Trento": [2, 2, 2], "Houston": [5, 5, 5], "MUUFL": [4, 4, 3]}

    offsets = {}
    off = 0
    for ds in dataset_order:
        offsets[ds] = off
        off += sum(splits_map[ds])

    # Load data
    test_padded, train_padded = {}, {}
    for ds in dataset_order:
        dp = os.path.expanduser(paths[ds])
        trn_loader, tst_loader, nc, _, _ = get_loader(
            dp, args.batch_size, 7, 3, is_shuffle=False, tsk_offset=0
        )
        use_lidar_adapter = getattr(args, 'lidar_adapter', False)
        pad_lidar = not use_lidar_adapter
        test_padded[ds] = PaddedDataset(
            tst_loader.dataset, class_offset=offsets[ds], pad_lidar=pad_lidar
        )
        train_padded[ds] = PaddedDataset(
            trn_loader.dataset, class_offset=offsets[ds], pad_lidar=pad_lidar
        )

    # Build task layout
    task_layout = []
    for ds in dataset_order:
        o = offsets[ds]
        local_off = 0
        for nc in splits_map[ds]:
            cls = list(range(o + local_off, o + local_off + nc))
            task_layout.append((ds, cls))
            local_off += nc

    max_tasks = getattr(args, 'max_tasks', None) or len(task_layout)
    task_layout = task_layout[:max_tasks]

    # Build class-to-domain mapping
    class_to_domain = {}
    for ds in dataset_order:
        o = offsets[ds]
        for c in range(o, o + sum(splits_map[ds])):
            class_to_domain[c] = ds

    # Create AnchorLoRA model
    backbone = net.model if hasattr(net, 'model') else net
    model = AnchorLoRAModel(
        backbone, hsi_rank=hsi_rank, lidar_rank=lidar_rank,
        warmup_end_task=warmup_tasks - 1
    )
    model.to(device)

    # Apply domain-conditioned LoRA bank if requested
    if domain_conditioned:
        model = patch_model_for_domain_conditioned_reuse(
            model, hsi_rank, lidar_rank
        )
        model.to(device)
        print("  Domain-conditioned LoRA banks installed")

    # Install CMDA drift gate if requested
    use_drift_gate = getattr(args, 'use_drift_gate', False)
    if use_drift_gate:
        model = patch_model_with_drift_gate(model)

    # Domain teacher bank for feature KD
    teacher_bank = DomainTeacherBank()
    branch_dims = None
    importance_weights = None

    # Stores
    prototype_store = PrototypeStore()
    seen_classes = set()
    old_classes = set()

    # Track methods for comparison
    # CMCD-LoRA / +SHINE: evaluated WITH LoRA features
    # Baseline / SHINE: evaluated WITHOUT LoRA features (frozen backbone only)
    methods = ["CMCD-LoRA", "CMCD-LoRA+SHINE", "Baseline", "SHINE"]
    results = {m: [] for m in methods}

    # Separate prototype stores for LoRA vs no-LoRA features
    baseline_prototype_store = PrototypeStore()

    domain_selective = getattr(args, 'domain_selective', False)

    for task_idx, (ds_name, cls_list) in enumerate(task_layout):
        old_classes = set(seen_classes)
        seen_classes.update(cls_list)

        is_warmup = task_idx < warmup_tasks

        # Register task-domain mapping
        model.register_task_domain(task_idx, ds_name)

        phase_str = '[WARMUP]' if is_warmup else '[LoRA+DKD]'
        print(f"\n{'='*70}")
        print(f"Task {task_idx}: {ds_name} cls {cls_list} | "
              f"#seen={len(seen_classes)} {phase_str}")
        print(f"{'='*70}")

        # Get training data for current task
        task_class_set = set(cls_list)
        task_train_ds = subset_by_classes(train_padded[ds_name], task_class_set)
        if task_train_ds is None:
            print(f"  WARNING: No training data for task {task_idx}")
            continue

        task_loader = DataLoader(
            task_train_ds, batch_size=args.batch_size,
            shuffle=True, drop_last=False
        )

        # Get old prototypes
        old_protos = {}
        for c in old_classes:
            if c in prototype_store.prototypes:
                old_protos[c] = {
                    b: prototype_store.prototypes[c][b].clone()
                    for b in BRANCHES if b in prototype_store.prototypes[c]
                }

        # ── Train ──
        if domain_conditioned:
            # Patched begin_task takes domain_name
            model.begin_task(task_idx, is_warmup=is_warmup, domain_name=ds_name)
        else:
            model.begin_task(task_idx, is_warmup=is_warmup)

        if is_warmup:
            print(f"  Training warm-up (full backbone, "
                  f"spectral lr x {args.spectral_lr_scale})")
            train_warmup_task(
                model, task_loader, seen_classes, old_protos, device,
                epochs=args.warmup_epochs, base_lr=args.warmup_lr,
                spectral_lr_scale=args.spectral_lr_scale,
                lambda_proto=args.lambda_proto,
            )
        else:
            print(f"  Training LoRA+DKD "
                  f"(HSI rank={hsi_rank}, LiDAR rank={lidar_rank})")

            # Update drift gate prototypes before training (CMDA)
            if use_drift_gate and hasattr(model, 'drift_gate'):
                model.drift_gate.update_prototypes(prototype_store, old_classes)
                n_old = len(old_classes)
                print(f"  CMDA drift gate: {n_old} old-class spectral prototypes loaded")

            # Build domain loader (all seen classes in current domain)
            ds_seen_cls = set(
                c for c in seen_classes if class_to_domain.get(c) == ds_name
            )
            all_domain_ds = subset_by_classes(train_padded[ds_name], ds_seen_cls)
            if all_domain_ds is not None:
                domain_loader = DataLoader(
                    all_domain_ds, batch_size=args.batch_size,
                    shuffle=True, drop_last=False
                )
            else:
                domain_loader = task_loader

            # Get domain teacher for feature KD
            # The teacher is a full AnchorLoRAModel snapshot including LoRA weights.
            # We use copy.deepcopy(model) so it has the same architecture,
            # then overwrite with the teacher bank's state dict.
            teacher_model = None
            if lambda_domain_kd > 0 and task_idx > 0:
                teacher_state = teacher_bank.get_previous_domain_teacher(
                    ds_name, dataset_order
                )
                if teacher_state is not None:
                    teacher_model = copy.deepcopy(model)
                    teacher_model.load_state_dict(teacher_state, strict=False)
                    teacher_model.eval()
                    for p in teacher_model.parameters():
                        p.requires_grad = False
                    print(f"  Domain teacher loaded for KD "
                          f"(LoRA params: {teacher_model.count_lora_params():,})")

            # Compute importance weights
            if task_idx > 0 and branch_dims is not None:
                importance_weights = compute_importance_weights(
                    prototype_store, old_classes, branch_dims
                )

            train_lora_task_with_domain_kd(
                model, domain_loader, seen_classes, old_protos,
                prototype_store, class_to_domain, device,
                teacher_model=teacher_model,
                importance_weights=importance_weights,
                epochs=args.lora_epochs, lr=args.lora_lr,
                lambda_proto=args.lambda_proto,
                lambda_kd=args.lambda_kd,
                lambda_ortho=args.lambda_ortho,
                lambda_domain_kd=lambda_domain_kd,
                n_pseudo=args.n_pseudo,
                lambda_pseudo=args.lambda_pseudo,
                noise_scale=args.noise_scale,
            )

            # Clean up teacher to free memory
            if teacher_model is not None:
                del teacher_model
                torch.cuda.empty_cache()

        model.end_task(task_idx, is_warmup=is_warmup)

        # ══════════════════════════════════════════════════════════
        # EVALUATION (same structure as anchor_lora_experiment)
        # ══════════════════════════════════════════════════════════
        model.eval()
        eval_batch_size = args.batch_size * 2

        # Step 1: Extract train features WITH LoRA and WITHOUT LoRA
        lora_train_feats = {}
        nolora_train_feats = {}
        domain_stats = {}
        baseline_domain_stats = {}

        for ds in dataset_order:
            ds_cls_seen = [c for c in seen_classes if class_to_domain.get(c) == ds]
            if not ds_cls_seen:
                continue
            ds_subset = subset_by_classes(train_padded[ds], set(ds_cls_seen))
            if ds_subset is None:
                continue
            ds_loader = DataLoader(
                ds_subset, batch_size=eval_batch_size,
                shuffle=False, drop_last=False
            )
            ds_task_ids = (
                model.get_domain_task_ids(ds) if domain_selective else None
            )
            # WITH LoRA
            ds_feats, ds_labels = extract_features(
                model, ds_loader, device, active_task_ids=ds_task_ids
            )
            lora_train_feats[ds] = (ds_feats, ds_labels)

            # WITHOUT LoRA (frozen backbone only)
            if model.warmup_done:
                ds_feats_nolora, ds_labels_nolora = extract_features(
                    model, ds_loader, device, use_lora=False
                )
            else:
                ds_feats_nolora, ds_labels_nolora = ds_feats, ds_labels
            nolora_train_feats[ds] = (ds_feats_nolora, ds_labels_nolora)

            # Update prototypes (LoRA features)
            for c in ds_cls_seen:
                mask_c = ds_labels == c
                if mask_c.sum() > 0:
                    class_feats = {
                        b: ds_feats[b][mask_c]
                        for b in BRANCHES if ds_feats[b] is not None
                    }
                    prototype_store.update(c, class_feats)

            # Update baseline prototypes (no-LoRA features)
            for c in ds_cls_seen:
                mask_c = ds_labels_nolora == c
                if mask_c.sum() > 0:
                    class_feats_bl = {
                        b: ds_feats_nolora[b][mask_c]
                        for b in BRANCHES if ds_feats_nolora[b] is not None
                    }
                    baseline_prototype_store.update(c, class_feats_bl)

            # Compute domain stats for both
            domain_stats[ds] = compute_domain_stats(
                ds_feats, ds_labels, set(ds_cls_seen)
            )
            baseline_domain_stats[ds] = compute_domain_stats(
                ds_feats_nolora, ds_labels_nolora, set(ds_cls_seen)
            )

            # Detect branch dims (once)
            if branch_dims is None:
                branch_dims = {
                    b: ds_feats[b].shape[1]
                    for b in BRANCHES
                    if ds_feats[b] is not None
                }

        all_protos = prototype_store.all_protos()
        baseline_protos = baseline_prototype_store.all_protos()

        # Update domain teacher bank
        if lambda_domain_kd > 0:
            for ds in dataset_order:
                ds_cls_seen = [
                    c for c in seen_classes if class_to_domain.get(c) == ds
                ]
                if not ds_cls_seen or ds not in lora_train_feats:
                    continue
                ds_feats, ds_labels = lora_train_feats[ds]
                preds = predict_ncm(
                    ds_feats, all_protos, set(ds_cls_seen)
                )
                acc = (preds == ds_labels).float().mean().item()
                teacher_bank.update(ds, model, acc)

            if task_idx == 0:
                teacher_bank.set_universal(model)

        # Step 2: Extract test features WITH and WITHOUT LoRA
        test_feats_lora = {}
        test_feats_nolora = {}
        for ds in dataset_order:
            eval_ds = subset_by_classes(test_padded[ds], seen_classes)
            if eval_ds is None:
                continue
            eval_loader = DataLoader(
                eval_ds, batch_size=eval_batch_size,
                shuffle=False, drop_last=False
            )
            ds_task_ids = (
                model.get_domain_task_ids(ds) if domain_selective else None
            )
            test_feats_lora[ds] = extract_features(
                model, eval_loader, device, active_task_ids=ds_task_ids
            )
            if model.warmup_done:
                test_feats_nolora[ds] = extract_features(
                    model, eval_loader, device, use_lora=False
                )
            else:
                test_feats_nolora[ds] = test_feats_lora[ds]

        # Step 3: Evaluate all methods with correct feature/prototype pairs
        for method in results.keys():
            # CMCD-LoRA uses LoRA features + LoRA prototypes
            # Baseline uses no-LoRA features + no-LoRA prototypes
            use_lora_feats = method.startswith("CMCD-LoRA")
            use_shine = "+SHINE" in method or method == "SHINE"

            feats_dict = test_feats_lora if use_lora_feats else test_feats_nolora
            protos_m = all_protos if use_lora_feats else baseline_protos
            stats_m = domain_stats if use_lora_feats else baseline_domain_stats

            all_preds, all_targets = [], []
            for ds in dataset_order:
                if ds not in feats_dict:
                    continue
                eval_feats, eval_labels = feats_dict[ds]

                if use_shine:
                    preds = predict_shine(
                        eval_feats, protos_m, seen_classes,
                        stats_m, ds, class_to_domain
                    )
                else:
                    preds = predict_ncm(eval_feats, protos_m, seen_classes)

                all_preds.append(preds)
                all_targets.append(eval_labels)

            if all_preds:
                all_preds_t = torch.cat(all_preds)
                all_targets_t = torch.cat(all_targets)
                metrics = compute_metrics(
                    all_preds_t, all_targets_t, seen_classes,
                    class_to_domain, dataset_order
                )
            else:
                metrics = {"balanced_acc": 0.0, "per_ds": {}, "per_class": {}}

            results[method].append({
                "task": task_idx, "dataset": ds_name,
                "n_seen": len(seen_classes),
                "avg_tag": metrics["balanced_acc"],
                "per_ds": metrics["per_ds"],
            })

        # Print results
        for m in results:
            if results[m]:
                r = results[m][-1]
                ds_str = ", ".join(
                    f"{k}={v*100:.1f}%" for k, v in r["per_ds"].items()
                )
                print(f"  {m:<22} Avg={r['avg_tag']*100:.1f}% | {ds_str}")

        # Print LoRA param count
        print(f"  LoRA params: {model.count_lora_params():,}")
        if domain_conditioned and hasattr(model.hsi_lora_bank, 'domain_to_adapter'):
            n_adapters_h = len(model.hsi_lora_bank.task_loras)
            n_adapters_l = len(model.lidar_lora_bank.task_loras)
            print(f"  Adapters: HSI={n_adapters_h}, LiDAR={n_adapters_l} "
                  f"(domain-conditioned)")

    # ── Final Summary ──
    print(f"\n{'='*80}")
    print(f"FINAL: {' -> '.join(dataset_order)} ({len(task_layout)} tasks)")
    print(f"Config: warmup={warmup_tasks}, HSI_rank={hsi_rank}, "
          f"LiDAR_rank={lidar_rank}")
    print(f"Domain-conditioned reuse: {domain_conditioned}")
    print(f"Domain KD lambda: {lambda_domain_kd}")
    print(f"Total LoRA params: {model.count_lora_params():,}")
    print(f"{'='*80}")

    header = f"{'Method':<24}"
    for ds in dataset_order:
        header += f" {ds:>8}"
    header += f" {'Avg TAg':>8}"
    print(header)
    print("-" * len(header))

    for m in results:
        if results[m]:
            r = results[m][-1]
            row = f"{m:<24}"
            for ds in dataset_order:
                row += f" {r['per_ds'].get(ds, 0)*100:>7.1f}%"
            row += f" {r['avg_tag']*100:>7.1f}%"
            print(row)

    # ── Forgetting Rate ──
    print(f"\n--- Forgetting Rate (per domain) ---")
    for m in results:
        if len(results[m]) < 2:
            continue
        forgetting_rates = {}
        for ds in dataset_order:
            best_acc = 0.0
            final_acc = 0.0
            for r in results[m]:
                acc = r["per_ds"].get(ds, 0)
                if acc > best_acc:
                    best_acc = acc
                final_acc = r["per_ds"].get(ds, 0)
            if best_acc > 0:
                forgetting_rates[ds] = (best_acc - final_acc) * 100
        if forgetting_rates:
            avg_forget = sum(forgetting_rates.values()) / len(forgetting_rates)
            parts = ", ".join(
                f"{k}={v:.1f}pp" for k, v in forgetting_rates.items()
            )
            print(f"  {m:<24} {parts} | Avg={avg_forget:.1f}pp")

    # Cross-method comparison: CMCD-LoRA vs Baseline (no LoRA)
    for base, enhanced in [("SHINE", "CMCD-LoRA+SHINE"),
                           ("Baseline", "CMCD-LoRA")]:
        if base in results and enhanced in results and results[base] and results[enhanced]:
            b_final = results[base][-1]
            e_final = results[enhanced][-1]
            delta = e_final["avg_tag"] - b_final["avg_tag"]
            print(f"\n*** {enhanced} vs {base}: {delta*100:+.1f}pp ***")
            for ds in dataset_order:
                dd = e_final["per_ds"].get(ds, 0) - b_final["per_ds"].get(ds, 0)
                print(f"  {ds}: {dd*100:+.1f}pp")

    return results


# ======================================================================
# MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="CMCD-LoRA: Domain-Conditioned SD-LoRA + Domain-Aware Feature KD"
    )
    parser.add_argument("--mode", default="marathon", choices=["marathon", "mvp"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument(
        "--output_dir",
        default="/root/autodl-tmp/results/s2cm/cmcd_lora/",
    )

    # Warm-up config
    parser.add_argument("--warmup_tasks", default=3, type=int,
                        help="Number of warm-up tasks")
    parser.add_argument("--warmup_epochs", default=30, type=int)
    parser.add_argument("--warmup_lr", default=1e-3, type=float)
    parser.add_argument("--spectral_lr_scale", default=0.1, type=float,
                        help="LR multiplier for spectral branch during warm-up")

    # LoRA config
    parser.add_argument("--lora_rank", default=4, type=int,
                        help="HSI spatial LoRA rank")
    parser.add_argument("--lidar_rank_mult", default=2, type=int,
                        help="LiDAR LoRA rank = lora_rank x this")
    parser.add_argument("--lora_epochs", default=50, type=int)
    parser.add_argument("--lora_lr", default=5e-4, type=float)

    # Loss weights (same as anchor_lora_experiment)
    parser.add_argument("--lambda_proto", default=1.0, type=float)
    parser.add_argument("--lambda_kd", default=0.0, type=float,
                        help="Spectral anchor KD weight (0 = disabled, matching "
                             "good baseline config)")
    parser.add_argument("--lambda_ortho", default=0.1, type=float)

    # CMCD-LoRA specific args
    parser.add_argument("--domain_conditioned_reuse", action="store_true",
                        help="Enable domain-conditioned SD-LoRA reuse: tasks in "
                             "same domain share adapter, different domain gets new")
    parser.add_argument("--lambda_domain_kd", default=0.5, type=float,
                        help="Weight for domain-aware feature KD loss "
                             "(0 = disabled)")
    parser.add_argument("--use_drift_gate", action="store_true",
                        help="Enable CMDA spectral drift gate for per-sample "
                             "adaptive LoRA strength")
    parser.add_argument("--n_pseudo", default=8, type=int,
                        help="Number of pseudo-features per old class for replay")
    parser.add_argument("--lambda_pseudo", default=0.5, type=float,
                        help="Weight for prototype replay CE loss")
    parser.add_argument("--noise_scale", default=1.0, type=float,
                        help="Noise scale for proto augmentation")

    # Other config (same as anchor_lora)
    parser.add_argument("--max_tasks", default=None, type=int)
    parser.add_argument("--lidar_adapter", action="store_true",
                        help="Use learned LiDAR channel adapter")
    parser.add_argument("--domain_selective", action="store_true",
                        help="Domain-selective LoRA at inference")
    parser.add_argument("--dataset_order", default="MTH", type=str,
                        help="Domain ordering: MTH=MUUFL->Trento->Houston, "
                             "THM=Trento->Houston->MUUFL, etc.")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse dataset ordering
    ORDER_MAP = {
        "THM": ["Trento", "Houston", "MUUFL"],
        "HMT": ["Houston", "MUUFL", "Trento"],
        "MTH": ["MUUFL", "Trento", "Houston"],
        "HTM": ["Houston", "Trento", "MUUFL"],
        "TMH": ["Trento", "MUUFL", "Houston"],
        "MHT": ["MUUFL", "Houston", "Trento"],
    }
    dataset_order = ORDER_MAP.get(args.dataset_order, ["MUUFL", "Trento", "Houston"])

    # Load pre-trained backbone
    ckpt_path = resolve_bootstrap_checkpoint(dataset_order, args.seed)
    print(f"Loading checkpoint: {ckpt_path}")

    backbone = S2CMNet(
        in_chans_hsi=UNIFIED_HSI_BANDS,
        in_chans_lidar=UNIFIED_LIDAR_CHANS,
        img_size=7,
        embed_dim=64,
    )
    net = LLL_Net(backbone, remove_existing_head=True)
    # Detect head size from checkpoint to avoid size mismatch
    ckpt_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    head_key = "heads.0.weight"
    if head_key in ckpt_state:
        n_head_classes = ckpt_state[head_key].shape[0]
    else:
        n_head_classes = 2
    net.add_head(n_head_classes)
    net.load_state_dict(ckpt_state, strict=False)
    net.to(device)
    net.eval()

    if args.mode == "mvp":
        args.max_tasks = 4

    results = run_cmcd_lora_marathon(net, device, args, dataset_order=dataset_order)

    # Save results
    def to_ser(obj):
        if isinstance(obj, dict):
            return {str(k): to_ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_ser(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, set):
            return sorted(list(obj))
        return obj

    order_str = args.dataset_order
    flags = []
    if args.domain_conditioned_reuse:
        flags.append("dcr")
    if args.lambda_domain_kd > 0:
        flags.append(f"dkd{args.lambda_domain_kd}")
    if args.lambda_kd == 0:
        flags.append("no_kd")
    flag_str = "_" + "_".join(flags) if flags else ""
    config_str = (
        f"w{args.warmup_tasks}_r{args.lora_rank}x{args.lidar_rank_mult}"
        f"_{order_str}{flag_str}"
    )
    out_path = os.path.join(
        args.output_dir, f"cmcd_lora_{config_str}_seed{args.seed}.json"
    )
    with open(out_path, "w") as f:
        json.dump({
            "config": to_ser(vars(args)),
            "dataset_order": dataset_order,
            "results": to_ser(results),
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
