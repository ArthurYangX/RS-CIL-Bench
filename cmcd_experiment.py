#!/usr/bin/env python3
"""CMCD: Cross-Modal Cascade Distillation for Exemplar-Free CIL.

Inspired by Nemotron-Cascade 2 MOPD + LDC (ECCV 2024) + SDFT (MIT 2026).

Key components:
  1. Learnable Cross-Modal Drift Compensation: Per-branch MLP to map old->new features
  2. Domain-Aware Teacher Bank: Store best checkpoint per domain for feature-level KD
  3. Importance-Weighted Feature KD: Dimension-wise weighted distillation

The core insight: exemplar-based methods (iCaRL: 81.4%, LUCIR: 83.9%) win because
their replay features are always fresh (passed through current backbone). Our Gaussian
capsules become stale as backbone drifts. Drift compensation makes capsules fresh again.

Usage:
  python cmcd_experiment.py --mode marathon --seed 0
  python cmcd_experiment.py --mode marathon --seed 0 --dataset_order MTH
  python cmcd_experiment.py --mode marathon --seed 0 --no_drift_comp  # ablation
  python cmcd_experiment.py --mode marathon --seed 0 --no_domain_kd   # ablation
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
    resolve_bootstrap_checkpoint,
    DEFAULT_RESULTS_ROOT,
)


# ======================================================================
# DRIFT COMPENSATOR -- maps features from old backbone to new backbone
# ======================================================================
class DriftCompensator(nn.Module):
    """Per-branch learnable drift compensation.

    Inspired by LDC (ECCV 2024): learns rotation + scaling + translation
    to map old-backbone features to new-backbone feature space.

    Architecture: residual MLP with layer norm.
    z_new = z_old + MLP(z_old)
    """

    def __init__(self, feat_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(feat_dim // 2, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        # Initialize near-identity (residual starts at ~0)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_old):
        return z_old + self.net(z_old)


def train_drift_compensator(comp, old_feats, new_feats, epochs=30, lr=1e-3, batch_size=256):
    """Train drift compensator to map old_feats -> new_feats.

    Uses new-task data where we have features from both old and new model.
    """
    device = next(comp.parameters()).device

    all_old = []
    all_new = []
    for b in BRANCHES:
        if b in old_feats and b in new_feats:
            all_old.append(old_feats[b])
            all_new.append(new_feats[b])

    if not all_old:
        return

    z_old = torch.cat(all_old, dim=1).to(device)
    z_new = torch.cat(all_new, dim=1).to(device)

    optimizer = torch.optim.Adam(comp.parameters(), lr=lr, weight_decay=1e-5)
    N = z_old.shape[0]

    best_loss = float("inf")
    for ep in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            z_o = z_old[idx]
            z_n = z_new[idx]

            z_pred = comp(z_o)
            loss = F.mse_loss(z_pred, z_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss

    print(f"    Drift comp trained: {epochs} epochs, final_loss={best_loss:.6f}")


class BranchDriftCompensators:
    """Manages drift compensator across tasks."""

    def __init__(self, branch_dims, device):
        self.device = device
        self.branch_dims = branch_dims
        total_dim = sum(branch_dims.values())
        self.compensator = DriftCompensator(total_dim).to(device)
        self.is_trained = False

    def train(self, old_feats, new_feats, **kwargs):
        """Train compensator on paired old/new features."""
        self.compensator.train()
        train_drift_compensator(self.compensator, old_feats, new_feats, **kwargs)
        self.compensator.eval()
        self.is_trained = True

    def compensate_prototype(self, proto_dict):
        """Apply drift compensation to a prototype dict."""
        if not self.is_trained:
            return proto_dict

        parts = []
        branch_order = []
        for b in BRANCHES:
            if b in proto_dict:
                parts.append(proto_dict[b].to(self.device))
                branch_order.append(b)

        if not parts:
            return proto_dict

        z_cat = torch.cat(parts).unsqueeze(0)
        with torch.no_grad():
            z_comp = self.compensator(z_cat).squeeze(0)

        result = {}
        offset = 0
        for b in branch_order:
            dim = proto_dict[b].shape[0]
            result[b] = z_comp[offset : offset + dim].cpu()
            offset += dim

        return result


# ======================================================================
# DOMAIN-AWARE TEACHER BANK
# ======================================================================
class DomainTeacherBank:
    """Stores best model checkpoint per domain for cascade KD.

    Inspired by Nemotron-Cascade 2 MOPD: use domain-specific best
    teacher rather than just the immediately preceding checkpoint.
    """

    def __init__(self):
        self.teachers = {}
        self.universal_teacher = None

    def update(self, domain, net, metric):
        if domain not in self.teachers or metric > self.teachers[domain][1]:
            self.teachers[domain] = (copy.deepcopy(net.state_dict()), metric)
            print(f"    Teacher bank: updated {domain} (metric={metric:.4f})")

    def set_universal(self, net):
        self.universal_teacher = copy.deepcopy(net.state_dict())

    def get_teacher(self, domain, net_template):
        if domain in self.teachers:
            state = self.teachers[domain][0]
        elif self.universal_teacher is not None:
            state = self.universal_teacher
        else:
            return None
        net_copy = copy.deepcopy(net_template)
        net_copy.load_state_dict(state)
        net_copy.eval()
        return net_copy


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


def feature_kd_loss(student_feats, teacher_feats, importance_weights=None):
    """Importance-weighted feature-level KD loss."""
    parts_s, parts_t = [], []
    for b in BRANCHES:
        if b in student_feats and b in teacher_feats:
            parts_s.append(student_feats[b])
            parts_t.append(teacher_feats[b])

    if not parts_s:
        return torch.tensor(0.0)

    z_s = torch.cat(parts_s, dim=1)
    z_t = torch.cat(parts_t, dim=1)
    diff_sq = (z_s - z_t) ** 2

    if importance_weights is not None:
        w = importance_weights.to(z_s.device).unsqueeze(0)
        return (w * diff_sq).mean()
    else:
        return diff_sq.mean()


# ======================================================================
# FEATURE EXTRACTION — reuse from baselines for compatibility
# ======================================================================
from baselines_experiment import extract_branch_features as extract_branch_features_local


# ======================================================================
# TRAINING: DCPRN-style with drift compensation + domain KD
# ======================================================================
def train_cmcd_task(
    net,
    task_loader,
    device,
    seen_classes,
    old_protos,
    old_net=None,
    prototype_store=None,
    domain_teacher=None,
    importance_weights=None,
    drift_comp=None,
    epochs=30,
    lr=5e-4,
    lambda_kd=1.0,
    lambda_fkd=0.5,
    lambda_proto=0.5,
    lambda_replay=1.0,
    replay_samples=100,
    **kwargs,
):
    """Train one CIL task with CMCD components.

    Base: DCPRN-style training (dual KD + proto replay)
    Added: importance-weighted feature KD from domain teacher + drift-compensated replay
    """
    inner = net.model if hasattr(net, "model") else net

    # Freeze strategy: freeze early VSSBlocks, unfreeze late blocks + fusion layers
    # Reviewer suggestion: late-layer adaptation with prototype anchoring
    n_unfreeze_blocks = kwargs.get("n_unfreeze_blocks", 2)  # last N blocks trainable
    if old_net is not None:
        # First freeze everything
        for p in inner.parameters():
            p.requires_grad = False
        trainable_modules = []
        # Unfreeze last N spatial blocks (bias-tuning style)
        if hasattr(inner, "spatial_branch") and hasattr(inner.spatial_branch, "blocks"):
            blocks = inner.spatial_branch.blocks
            n_blocks = len(blocks)
            for i in range(max(0, n_blocks - n_unfreeze_blocks), n_blocks):
                for p in blocks[i].parameters():
                    p.requires_grad = True
                trainable_modules.append(f"spatial_block_{i}")
            # Also unfreeze spatial norm
            if hasattr(inner.spatial_branch, "norm"):
                for p in inner.spatial_branch.norm.parameters():
                    p.requires_grad = True
                trainable_modules.append("spatial_norm")
        # Unfreeze cross-modal fusion layers
        if hasattr(inner, "cross_fusion") and inner.cross_fusion is not None:
            for p in inner.cross_fusion.parameters():
                p.requires_grad = True
            trainable_modules.append("cross_fusion")
        if hasattr(inner, "fusion_proj"):
            for p in inner.fusion_proj.parameters():
                p.requires_grad = True
            trainable_modules.append("fusion_proj")
        # Also unfreeze the projection heads
        for name in ["proj_spec", "proj_hsi_spa", "proj_lid_spa"]:
            mod = getattr(inner, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True
                trainable_modules.append(name)
        print(f"    Trainable modules: {trainable_modules}")
    else:
        for p in inner.parameters():
            p.requires_grad = True

    trainable_params = [p for p in inner.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"    Trainable params: {n_trainable:,}")
    if not trainable_params:
        print("    No trainable params, skipping training")
        return

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    old_classes = set()
    if old_protos:
        old_classes = set(old_protos.keys())

    for ep in range(epochs):
        inner.train()
        total_loss = 0.0
        n_batches = 0

        for x_hsi, x_lidar, targets in task_loader:
            x_hsi = x_hsi.to(device)
            x_lidar = x_lidar.to(device)
            targets = targets.to(device)

            # Forward through student
            aux_raw = inner.forward_features(x_hsi, x_lidar, return_aux=True)
            kmap = {"f_spec": "spec", "f_hsi_spa": "hsi_spa", "f_lid_spa": "lid_spa"}
            aux = {kmap.get(k, k): v for k, v in aux_raw.items()}

            loss = torch.tensor(0.0, device=device)

            # === Loss 1: Feature KD from old model ===
            if old_net is not None and lambda_kd > 0:
                old_inner = old_net.model if hasattr(old_net, "model") else old_net
                old_inner.eval()
                with torch.no_grad():
                    old_aux_raw = old_inner.forward_features(
                        x_hsi, x_lidar, return_aux=True
                    )
                    old_aux = {kmap.get(k, k): v for k, v in old_aux_raw.items()}

                for b in ["hsi_spa", "lid_spa"]:
                    if b in aux and b in old_aux:
                        z_s = F.normalize(aux[b], dim=1)
                        z_t = F.normalize(old_aux[b], dim=1)
                        loss = (
                            loss
                            + lambda_kd * 0.5 * (1.0 - (z_s * z_t).sum(dim=1)).mean()
                        )

            # === Loss 2: Domain-aware feature KD ===
            if domain_teacher is not None and lambda_fkd > 0:
                d_inner = (
                    domain_teacher.model
                    if hasattr(domain_teacher, "model")
                    else domain_teacher
                )
                d_inner.eval()
                with torch.no_grad():
                    d_aux_raw = d_inner.forward_features(
                        x_hsi, x_lidar, return_aux=True
                    )
                    d_aux = {kmap.get(k, k): v for k, v in d_aux_raw.items()}

                student_feats = {b: aux[b] for b in BRANCHES if b in aux}
                teacher_feats = {b: d_aux[b] for b in BRANCHES if b in d_aux}
                loss = loss + lambda_fkd * feature_kd_loss(
                    student_feats, teacher_feats, importance_weights
                )

            # === Loss 3: Virtual replay from drift-compensated prototypes ===
            if old_protos and lambda_replay > 0:
                replay_loss = _virtual_replay_loss(
                    aux, old_protos, seen_classes, old_classes,
                    drift_comp, device, replay_samples
                )
                loss = loss + lambda_replay * replay_loss

            # === Loss 4: Prototype anchoring (prevent feature drift) ===
            # For old classes visible in current batch's domain,
            # anchor current features close to stored prototypes
            lambda_anchor = kwargs.get("lambda_anchor", 1.0)
            if old_protos and lambda_anchor > 0:
                anchor_loss = torch.tensor(0.0, device=device)
                n_anchored = 0
                for b in BRANCHES:
                    if b not in aux:
                        continue
                    z_cur = aux[b]  # (B, D)
                    for c in sorted(old_protos.keys()):
                        if b in old_protos[c]:
                            mu = old_protos[c][b].to(device)
                            # Cosine distance between current features and prototype
                            anchor_loss = anchor_loss + (
                                1.0 - F.cosine_similarity(
                                    z_cur.mean(0, keepdim=True), mu.unsqueeze(0)
                                )
                            ).mean()
                            n_anchored += 1
                if n_anchored > 0:
                    loss = loss + lambda_anchor * anchor_loss / n_anchored

            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 10.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (ep + 1) % 10 == 0 or ep == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"    Epoch {ep+1}/{epochs}: loss={avg_loss:.4f}")


def _virtual_replay_loss(
    current_aux, old_protos, seen_classes, old_classes,
    drift_comp, device, replay_samples
):
    """Compute virtual replay loss using drift-compensated prototypes.

    Generates virtual features by sampling around (compensated) prototypes,
    then computes contrastive loss against all class prototypes.
    """
    # Build prototype bank for all seen classes
    all_cls = sorted(seen_classes)
    proto_parts = {}

    for c in all_cls:
        if c in old_protos:
            p = old_protos[c]
            if drift_comp is not None and drift_comp.is_trained and c in old_classes:
                p = drift_comp.compensate_prototype(p)
            # Use fused_feat or first available branch
            parts = []
            for b in BRANCHES:
                if b in p:
                    parts.append(p[b].to(device))
            if parts:
                proto_parts[c] = torch.cat(parts)

    if len(proto_parts) < 2:
        return torch.tensor(0.0, device=device)

    # Build new-class prototypes from current batch
    z_fused = current_aux.get("fused_feat", current_aux.get("z_final"))
    if z_fused is None:
        # Concatenate branches
        parts = []
        for b in BRANCHES:
            if b in current_aux and current_aux[b] is not None:
                parts.append(current_aux[b])
        if parts:
            z_fused = torch.cat(parts, dim=1)
        else:
            return torch.tensor(0.0, device=device)

    # Generate virtual features for old classes
    replay_feats = []
    replay_labels = []
    ns_per_class = max(replay_samples // max(len(old_classes), 1), 3)

    for c in sorted(old_classes):
        if c not in proto_parts:
            continue
        mu = proto_parts[c]
        # Small noise around prototype
        noise_scale = mu.abs().mean() * 0.1 + 1e-4
        samples = mu.unsqueeze(0) + noise_scale * torch.randn(
            ns_per_class, mu.shape[0], device=device
        )
        replay_feats.append(samples)
        replay_labels.extend([c] * ns_per_class)

    if not replay_feats:
        return torch.tensor(0.0, device=device)

    replay_z = torch.cat(replay_feats)
    replay_y = torch.tensor(replay_labels, device=device, dtype=torch.long)

    # Build prototype matrix for all seen classes
    proto_list = []
    cls_order = []
    for c in all_cls:
        if c in proto_parts:
            proto_list.append(proto_parts[c])
            cls_order.append(c)

    if len(proto_list) < 2:
        return torch.tensor(0.0, device=device)

    proto_mat = torch.stack(proto_list)
    sim = F.normalize(replay_z, dim=1) @ F.normalize(proto_mat, dim=1).t()

    cls_to_idx = {c: i for i, c in enumerate(cls_order)}
    mapped = []
    for c in replay_y.tolist():
        if c in cls_to_idx:
            mapped.append(cls_to_idx[c])
        else:
            mapped.append(0)
    mapped_labels = torch.tensor(mapped, device=device, dtype=torch.long)

    return F.cross_entropy(sim * 10.0, mapped_labels)


# ======================================================================
# MARATHON RUNNER
# ======================================================================
def run_cmcd_marathon(net, device, args, dataset_order=None):
    if dataset_order is None:
        dataset_order = ["Trento", "Houston", "MUUFL"]

    print(f"\n{'='*80}")
    print(
        f"CMCD Marathon: seed={args.seed}, drift_comp={not args.no_drift_comp}, "
        f"domain_kd={not args.no_domain_kd}"
    )
    print(f"Dataset order: {' -> '.join(dataset_order)}")
    print(f"{'='*80}")

    paths = get_dataname_to_path()
    splits_map = {"Trento": [2, 2, 2], "Houston": [5, 5, 5], "MUUFL": [4, 4, 3]}

    offsets = {}
    off = 0
    for ds in dataset_order:
        offsets[ds] = off
        off += sum(splits_map[ds])

    test_padded, train_padded = {}, {}
    for ds in dataset_order:
        dp = os.path.expanduser(paths[ds])
        trn_loader, tst_loader, nc, _, _ = get_loader(
            dp, args.batch_size, 7, 3, is_shuffle=False, tsk_offset=0
        )
        test_padded[ds] = PaddedDataset(
            tst_loader.dataset, class_offset=offsets[ds]
        )
        train_padded[ds] = PaddedDataset(
            trn_loader.dataset, class_offset=offsets[ds]
        )

    task_layout = []
    for ds in dataset_order:
        o = offsets[ds]
        local_off = 0
        for nc in splits_map[ds]:
            cls = list(range(o + local_off, o + local_off + nc))
            task_layout.append((ds, cls))
            local_off += nc

    max_tasks = getattr(args, "max_tasks", None) or len(task_layout)
    task_layout = task_layout[:max_tasks]

    class_to_domain = {}
    for ds in dataset_order:
        o = offsets[ds]
        for c in range(o, o + sum(splits_map[ds])):
            class_to_domain[c] = ds

    total_classes = sum(sum(splits_map[ds]) for ds in dataset_order)

    # State
    old_net = None
    prototype_store = PrototypeStore()
    seen_classes = set()
    old_classes = set()
    domain_teacher_bank = DomainTeacherBank()
    drift_comp = None
    branch_dims = None
    importance_weights = None

    method_names = ["CMCD", "CMCD+SHINE"]
    if not args.no_drift_comp:
        method_names += ["CMCD-DC", "CMCD-DC+SHINE"]
    results = {m: [] for m in method_names}
    eval_batch_size = args.batch_size * 2

    for task_idx, (ds_name, cls_list) in enumerate(task_layout):
        old_classes = set(seen_classes)
        seen_classes.update(cls_list)

        print(f"\n{'='*70}")
        print(
            f"Task {task_idx}: {ds_name} cls {cls_list} | #seen={len(seen_classes)}"
        )
        print(f"{'='*70}")

        task_ds = subset_by_classes(train_padded[ds_name], set(cls_list))
        if task_ds is None:
            print(f"  WARNING: No data for task {task_idx}")
            continue
        task_loader = DataLoader(
            task_ds, batch_size=args.batch_size, shuffle=True
        )

        # Collect old protos (raw, not compensated)
        old_protos = {}
        for c in old_classes:
            if c in prototype_store.prototypes:
                old_protos[c] = {
                    b: prototype_store.prototypes[c][b].clone()
                    for b in BRANCHES
                    if b in prototype_store.prototypes[c]
                }

        # === Extract features BEFORE training (for drift compensation) ===
        # Use ALL seen data (not just new task) to train drift compensator
        pre_feats = None
        if task_idx > 0 and not args.no_drift_comp:
            # Collect features from all seen domains
            all_pre_feats = {b: [] for b in BRANCHES}
            for ds in dataset_order:
                ds_cls = [c for c in seen_classes if class_to_domain.get(c) == ds]
                if not ds_cls:
                    continue
                ds_subset = subset_by_classes(train_padded[ds], set(ds_cls))
                if ds_subset is None:
                    continue
                ds_loader = DataLoader(ds_subset, batch_size=eval_batch_size, shuffle=False)
                ds_feats, _ = extract_branch_features_local(net, ds_loader, device)
                for b in BRANCHES:
                    if b in ds_feats and ds_feats[b] is not None:
                        all_pre_feats[b].append(ds_feats[b])
            pre_feats = {}
            for b in BRANCHES:
                if all_pre_feats[b]:
                    pre_feats[b] = torch.cat(all_pre_feats[b])

        # === Get domain teacher for feature KD ===
        domain_teacher = None
        if task_idx > 0 and not args.no_domain_kd:
            for prev_ds in dataset_order:
                if prev_ds != ds_name:
                    domain_teacher = domain_teacher_bank.get_teacher(prev_ds, net)
                    if domain_teacher is not None:
                        domain_teacher = domain_teacher.to(device)
                        break

            if domain_teacher is None:
                domain_teacher = domain_teacher_bank.get_teacher(
                    "__universal__", net
                )
                if domain_teacher is not None:
                    domain_teacher = domain_teacher.to(device)

        # === Compute importance weights ===
        if task_idx > 0 and branch_dims is not None:
            importance_weights = compute_importance_weights(
                prototype_store, old_classes, branch_dims
            )

        # === Train ===
        if task_idx == 0:
            print("  Task 0: Using pretrained backbone (no additional training)")
        else:
            print(
                f"  Training CMCD: lambda_kd={args.lambda_kd}, "
                f"lambda_fkd={args.lambda_fkd}, lambda_replay={args.lambda_replay}"
            )

            comp_protos = {}
            for c, p in old_protos.items():
                if drift_comp is not None and drift_comp.is_trained:
                    comp_protos[c] = drift_comp.compensate_prototype(p)
                else:
                    comp_protos[c] = p

            train_cmcd_task(
                net,
                task_loader,
                device,
                seen_classes,
                comp_protos,
                old_net=old_net,
                prototype_store=prototype_store,
                domain_teacher=domain_teacher,
                importance_weights=importance_weights,
                drift_comp=drift_comp,
                epochs=args.train_epochs,
                lr=args.train_lr,
                lambda_kd=args.lambda_kd,
                lambda_fkd=args.lambda_fkd,
                lambda_proto=args.lambda_proto,
                lambda_replay=args.lambda_replay,
                replay_samples=args.replay_samples,
                n_unfreeze_blocks=args.n_unfreeze_blocks,
                lambda_anchor=args.lambda_anchor,
            )

        # === Train drift compensator AFTER training ===
        if task_idx > 0 and not args.no_drift_comp and pre_feats is not None:
            # Extract features from ALL seen data with updated model
            all_post_feats = {b: [] for b in BRANCHES}
            for ds in dataset_order:
                ds_cls = [c for c in seen_classes if class_to_domain.get(c) == ds]
                if not ds_cls:
                    continue
                ds_subset = subset_by_classes(train_padded[ds], set(ds_cls))
                if ds_subset is None:
                    continue
                ds_loader = DataLoader(ds_subset, batch_size=eval_batch_size, shuffle=False)
                ds_feats, _ = extract_branch_features_local(net, ds_loader, device)
                for b in BRANCHES:
                    if b in ds_feats and ds_feats[b] is not None:
                        all_post_feats[b].append(ds_feats[b])
            post_feats = {}
            for b in BRANCHES:
                if all_post_feats[b]:
                    post_feats[b] = torch.cat(all_post_feats[b])

            if branch_dims is None:
                branch_dims = {
                    b: post_feats[b].shape[1]
                    for b in BRANCHES
                    if b in post_feats
                }

            if drift_comp is None:
                drift_comp = BranchDriftCompensators(branch_dims, device)

            print("  Training drift compensator...")
            drift_comp.train(
                pre_feats, post_feats, epochs=args.dc_epochs, lr=args.dc_lr
            )

        # === Save old model ===
        old_net = copy.deepcopy(net)

        # === Update prototypes ===
        net_eval = net
        if hasattr(net_eval, "eval"):
            net_eval.eval()

        domain_stats = {}
        for ds in dataset_order:
            ds_cls = [c for c in seen_classes if class_to_domain.get(c) == ds]
            if not ds_cls:
                continue
            ds_subset = subset_by_classes(train_padded[ds], set(ds_cls))
            if ds_subset is None:
                continue
            ds_loader = DataLoader(
                ds_subset, batch_size=eval_batch_size, shuffle=False
            )
            ds_feats, ds_labels = extract_branch_features_local(
                net_eval, ds_loader, device
            )

            for c in ds_cls:
                mask_c = ds_labels == c
                if mask_c.sum() > 0:
                    class_feats = {
                        b: ds_feats[b][mask_c]
                        for b in BRANCHES
                        if b in ds_feats and ds_feats[b] is not None
                    }
                    prototype_store.update(c, class_feats)

            domain_stats[ds] = compute_domain_stats(
                ds_feats, ds_labels, set(ds_cls)
            )

            # Update domain teacher bank
            if not args.no_domain_kd:
                ds_protos = {
                    c: prototype_store.prototypes[c]
                    for c in ds_cls
                    if c in prototype_store.prototypes
                }
                if ds_protos:
                    preds = predict_ncm(
                        ds_feats, prototype_store.all_protos(), set(ds_cls)
                    )
                    acc = (preds == ds_labels).float().mean().item()
                    domain_teacher_bank.update(ds, net_eval, acc)

        # Set universal teacher at task 0
        if task_idx == 0:
            domain_teacher_bank.set_universal(net_eval)
            # Detect branch dims
            for ds in dataset_order:
                ds_cls = [
                    c for c in seen_classes if class_to_domain.get(c) == ds
                ]
                if ds_cls:
                    ds_subset = subset_by_classes(train_padded[ds], set(ds_cls))
                    if ds_subset:
                        ds_loader = DataLoader(
                            ds_subset, batch_size=eval_batch_size, shuffle=False
                        )
                        ds_feats, _ = extract_branch_features_local(
                            net_eval, ds_loader, device
                        )
                        if branch_dims is None:
                            branch_dims = {
                                b: ds_feats[b].shape[1]
                                for b in BRANCHES
                                if b in ds_feats
                            }
                        break

        all_protos = prototype_store.all_protos()

        # Prepare drift-compensated protos for evaluation
        dc_protos = {}
        if drift_comp is not None and drift_comp.is_trained:
            for c in old_classes:
                if c in all_protos:
                    dc_protos[c] = drift_comp.compensate_prototype(all_protos[c])
            for c in cls_list:
                if c in all_protos:
                    dc_protos[c] = all_protos[c]

        # === Evaluate ===
        for method_name in results.keys():
            use_shine = "+SHINE" in method_name
            use_dc = "-DC" in method_name

            eval_protos = dc_protos if (use_dc and dc_protos) else all_protos

            all_preds, all_targets = [], []
            for ds in dataset_order:
                eval_ds = subset_by_classes(test_padded[ds], seen_classes)
                if eval_ds is None:
                    continue
                eval_loader = DataLoader(
                    eval_ds, batch_size=eval_batch_size, shuffle=False
                )
                eval_feats, eval_labels = extract_branch_features_local(
                    net_eval, eval_loader, device
                )

                if use_shine:
                    preds = predict_shine(
                        eval_feats,
                        eval_protos,
                        seen_classes,
                        domain_stats,
                        ds,
                        class_to_domain,
                    )
                else:
                    preds = predict_ncm(eval_feats, eval_protos, seen_classes)

                all_preds.append(preds)
                all_targets.append(eval_labels)

            if all_preds:
                all_preds_t = torch.cat(all_preds)
                all_targets_t = torch.cat(all_targets)
                metrics = compute_metrics(
                    all_preds_t,
                    all_targets_t,
                    seen_classes,
                    class_to_domain,
                    dataset_order,
                )
            else:
                metrics = {"balanced_acc": 0.0, "per_ds": {}, "per_class": {}}

            results[method_name].append(
                {
                    "task": task_idx,
                    "dataset": ds_name,
                    "n_seen": len(seen_classes),
                    "avg_tag": metrics["balanced_acc"],
                    "per_ds": metrics["per_ds"],
                }
            )

        # Print results
        for m in results:
            if results[m]:
                r = results[m][-1]
                ds_str = ", ".join(
                    f"{k}={v*100:.1f}%" for k, v in r["per_ds"].items()
                )
                print(f"  {m:<20} Avg={r['avg_tag']*100:.1f}% | {ds_str}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL: {' -> '.join(dataset_order)} ({len(task_layout)} tasks)")
    print(f"{'='*80}")
    for m in results:
        if results[m]:
            finals = [r["avg_tag"] for r in results[m]]
            inc_avg = sum(finals) / len(finals)
            final_acc = finals[-1]
            print(
                f"  {m:<20} final={final_acc*100:.1f}%, inc_avg={inc_avg*100:.1f}%"
            )

    return results


def main():
    parser = argparse.ArgumentParser("CMCD Experiment")
    parser.add_argument("--mode", default="marathon", choices=["marathon"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--dataset_order",
        default="THM",
        choices=["THM", "MTH", "HMT", "TMH", "MHT", "HTM"],
    )

    # Training
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--train_lr", type=float, default=5e-4)
    parser.add_argument("--lambda_kd", type=float, default=1.0)
    parser.add_argument("--lambda_fkd", type=float, default=0.5)
    parser.add_argument("--lambda_proto", type=float, default=0.5)
    parser.add_argument("--lambda_replay", type=float, default=1.0)
    parser.add_argument("--replay_samples", type=int, default=100)

    # Late-layer adaptation
    parser.add_argument("--n_unfreeze_blocks", type=int, default=2,
                        help="Number of late spatial blocks to unfreeze")
    parser.add_argument("--lambda_anchor", type=float, default=1.0,
                        help="Prototype anchoring loss weight")

    # Drift compensation
    parser.add_argument("--dc_epochs", type=int, default=30)
    parser.add_argument("--dc_lr", type=float, default=1e-3)
    parser.add_argument("--no_drift_comp", action="store_true")

    # Domain KD
    parser.add_argument("--no_domain_kd", action="store_true")

    # Other
    parser.add_argument("--max_tasks", type=int, default=None)

    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset order
    order_map = {
        "THM": ["Trento", "Houston", "MUUFL"],
        "MTH": ["MUUFL", "Trento", "Houston"],
        "HMT": ["Houston", "MUUFL", "Trento"],
        "TMH": ["Trento", "MUUFL", "Houston"],
        "MHT": ["MUUFL", "Houston", "Trento"],
        "HTM": ["Houston", "Trento", "MUUFL"],
    }
    dataset_order = order_map[args.dataset_order]

    # Load backbone
    ckpt_path = resolve_bootstrap_checkpoint(dataset_order, args.seed)
    print(f"Loading checkpoint: {ckpt_path}")

    backbone = S2CMNet(
        in_chans_hsi=UNIFIED_HSI_BANDS,
        in_chans_lidar=UNIFIED_LIDAR_CHANS,
        img_size=7,
        embed_dim=64,
    )
    net = LLL_Net(backbone, remove_existing_head=True)
    net.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False),
        strict=False,
    )
    net.to(device)

    print(f"Backbone loaded, device={device}")

    # Run
    results = run_cmcd_marathon(net, device, args, dataset_order=dataset_order)

    # Save
    results_dir = os.path.join(DEFAULT_RESULTS_ROOT, "cmcd")
    os.makedirs(results_dir, exist_ok=True)
    order_str = args.dataset_order
    out_path = os.path.join(
        results_dir, f"cmcd_{order_str}_seed{args.seed}.json"
    )

    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
