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

# Resolve project root dynamically (works both locally and on server)
_PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(_PROJECT_ROOT):
    os.chdir(_PROJECT_ROOT)
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
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
    compute_classifier_logits,
    cosine_ncm_logits,
    predict_ncm,
    predict_shine,
    resolve_branch_weights,
    compute_metrics,
    extract_features,
    AnchorLoRAModel,
    TaskLoRABank,
    LoRAAdapter2D,
    LoRAAdapter1D,
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


RUNTIME_DOMAINS = ("MUUFL", "Trento", "Houston")
DOMAIN_ALIASES = {
    "muufl": "MUUFL",
    "trento": "Trento",
    "houston": "Houston",
    "houston2013": "Houston",
}
BRANCH_WEIGHT_ALIASES = {
    "spec": "spec",
    "hsi": "hsi_spa",
    "hsi_spa": "hsi_spa",
    "lid": "lid_spa",
    "lidar": "lid_spa",
    "lid_spa": "lid_spa",
}


def canonicalize_domain_name(domain_name):
    """Map common domain aliases to the canonical runtime domain name."""
    if domain_name is None:
        return None
    token = "".join(
        ch for ch in str(domain_name).strip().lower()
        if ch.isalnum()
    )
    return DOMAIN_ALIASES.get(token)


def parse_domain_csv(domain_csv, allowed_domains=None, arg_name="imbalance_domains"):
    """Parse a comma-separated domain list into validated runtime domain names."""
    if not domain_csv:
        return set()

    allowed = None
    if allowed_domains is not None:
        allowed = set()
        for domain_name in allowed_domains:
            canonical = canonicalize_domain_name(domain_name)
            if canonical is None:
                raise ValueError(
                    f"{arg_name}: unsupported allowed domain '{domain_name}'"
                )
            allowed.add(canonical)

    parsed = set()
    unknown = []
    for raw_item in str(domain_csv).split(","):
        item = raw_item.strip()
        if not item:
            continue
        canonical = canonicalize_domain_name(item)
        if canonical is None or (allowed is not None and canonical not in allowed):
            unknown.append(item)
            continue
        parsed.add(canonical)

    if unknown:
        allowed_list = sorted(allowed) if allowed is not None else list(RUNTIME_DOMAINS)
        raise ValueError(
            f"{arg_name}: unsupported domain(s) {unknown}. "
            f"Allowed: {allowed_list}"
        )
    return parsed


def parse_branch_weights(branch_weights_csv, arg_name="--branch_weights"):
    """Parse branch fusion weights into canonical branch names."""
    if branch_weights_csv is None:
        return None
    if isinstance(branch_weights_csv, dict):
        return resolve_branch_weights(branch_weights_csv)

    raw = str(branch_weights_csv).strip()
    if not raw:
        return None

    parsed = {}
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"{arg_name}: expected entries like 'spec=1.2', got '{item}'"
            )
        key, value = item.split("=", 1)
        branch = BRANCH_WEIGHT_ALIASES.get(key.strip().lower())
        if branch is None:
            allowed = ", ".join(sorted(BRANCH_WEIGHT_ALIASES))
            raise ValueError(
                f"{arg_name}: unsupported branch '{key.strip()}'; allowed keys: {allowed}"
            )
        if branch in parsed:
            raise ValueError(f"{arg_name}: branch '{branch}' specified more than once")
        try:
            parsed[branch] = float(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"{arg_name}: invalid numeric weight for '{branch}': {value.strip()!r}"
            ) from exc
    return resolve_branch_weights(parsed)


def format_branch_weights(branch_weights):
    resolved = resolve_branch_weights(branch_weights)
    return ",".join(f"{branch}={resolved[branch]:.3f}" for branch in BRANCHES)


def branch_weights_are_default(branch_weights, atol=1e-12):
    resolved = resolve_branch_weights(branch_weights)
    return all(abs(resolved[branch] - 1.0) <= atol for branch in BRANCHES)


def format_domain_tag(domains):
    """Format a stable lowercase tag for filenames."""
    if not domains:
        return "none"
    return "-".join(sorted(domain.lower() for domain in domains))


def resolve_balanced_sampler_alphas(args):
    """Resolve warmup/LoRA sampler alphas with legacy fallback support."""
    legacy_alpha = float(getattr(args, "balanced_sampler_alpha", 0.0) or 0.0)
    warmup_alpha = getattr(args, "warmup_balanced_sampler_alpha", None)
    lora_alpha = getattr(args, "lora_balanced_sampler_alpha", None)
    if warmup_alpha is None:
        warmup_alpha = legacy_alpha
    if lora_alpha is None:
        lora_alpha = legacy_alpha
    return float(warmup_alpha), float(lora_alpha)


def resolve_phase_imbalance_domains(
    imbalance_domains,
    task_layout,
    warmup_tasks,
    use_cb_loss,
    warmup_sampler_alpha,
    lora_sampler_alpha,
    allow_partial_sampler_noop=False,
):
    """Validate that imbalance mitigation targets at least one active phase."""
    warmup_domains = {ds_name for ds_name, _ in task_layout[:warmup_tasks]}
    lora_domains = {ds_name for ds_name, _ in task_layout[warmup_tasks:]}
    warmup_active = set(imbalance_domains) & warmup_domains
    lora_active = set(imbalance_domains) & lora_domains

    if use_cb_loss and not warmup_active:
        raise ValueError(
            "use_cb_loss was enabled, but none of the requested imbalance "
            f"domains {sorted(imbalance_domains)} appear in warmup tasks. "
            f"Warmup domains: {sorted(warmup_domains)}"
        )
    if warmup_sampler_alpha > 0 and not warmup_active:
        if not (allow_partial_sampler_noop and lora_active):
            raise ValueError(
                "warmup balanced sampler was enabled, but none of the requested "
                f"imbalance domains {sorted(imbalance_domains)} appear in warmup "
                f"tasks. Warmup domains: {sorted(warmup_domains)}"
            )
    if lora_sampler_alpha > 0 and not lora_active:
        if not (allow_partial_sampler_noop and warmup_active):
            raise ValueError(
                "LoRA balanced sampler was enabled, but none of the requested "
                f"imbalance domains {sorted(imbalance_domains)} appear in LoRA "
                f"tasks. LoRA domains: {sorted(lora_domains)}"
            )

    return warmup_active, lora_active


def get_dataset_labels(dataset):
    """Extract integer class labels from a dataset or TensorDataset."""
    if isinstance(dataset, TensorDataset):
        if len(dataset.tensors) < 3:
            raise ValueError("Expected TensorDataset with labels at tensor index 2")
        labels = dataset.tensors[2]
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    elif hasattr(dataset, "targets"):
        labels = dataset.targets
    else:
        raise ValueError(f"Unsupported dataset type for label extraction: {type(dataset)}")

    if torch.is_tensor(labels):
        return labels.detach().cpu().long()
    return torch.as_tensor(np.array(labels), dtype=torch.long)


def build_balanced_loader(
    dataset,
    batch_size,
    balance_alpha=0.0,
    enable_balance=False,
):
    """Create a train loader with optional class-balanced sampling."""
    if dataset is None:
        return None

    if not enable_balance or balance_alpha <= 0:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

    labels = get_dataset_labels(dataset)
    if labels.numel() == 0:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

    unique_labels, counts = torch.unique(labels, return_counts=True)
    count_map = {
        int(cls.item()): float(cnt.item())
        for cls, cnt in zip(unique_labels, counts)
    }
    sample_weights = torch.tensor(
        [
            1.0 / (count_map[int(lbl.item())] ** balance_alpha)
            for lbl in labels
        ],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=int(labels.numel()),
        replacement=True,
    )
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, drop_last=False
    )


class ExemplarStore:
    """Per-class exemplar memory for upper-bound replay ablations."""

    def __init__(self, n_per_class=20):
        self.n_per_class = int(n_per_class)
        self.exemplars = {}  # {class_id: (xh_cpu, xl_cpu)}

    def update_exemplars(self, class_id, xh_all, xl_all, feats, max_store=None):
        """Select raw exemplars by greedy herding on current features."""
        if xh_all is None or xl_all is None or xh_all.shape[0] == 0:
            return

        n_keep = int(max_store or self.n_per_class)
        n_keep = min(n_keep, xh_all.shape[0])
        if n_keep <= 0:
            return

        feat_parts = [feats[b] for b in BRANCHES if feats.get(b) is not None]
        if not feat_parts:
            return

        f_cat = torch.cat(feat_parts, dim=1).detach().cpu()
        mean = f_cat.mean(0)

        selected = []
        selected_sum = torch.zeros_like(mean)
        remaining = set(range(f_cat.shape[0]))
        for _ in range(n_keep):
            best_idx, best_dist = -1, float("inf")
            target = (len(selected) + 1) * mean - selected_sum
            for idx in remaining:
                dist = (f_cat[idx] - target).norm().item()
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx < 0:
                break
            selected.append(best_idx)
            selected_sum += f_cat[best_idx]
            remaining.remove(best_idx)

        if not selected:
            return

        sel = torch.tensor(selected, dtype=torch.long)
        xh_cpu = xh_all.detach().cpu().index_select(0, sel)
        xl_cpu = xl_all.detach().cpu().index_select(0, sel)
        self.exemplars[int(class_id)] = (xh_cpu, xl_cpu)

    def get_exemplar_loader(self, class_ids, batch_size=64):
        """Create a replay loader from stored exemplars."""
        xh_list, xl_list, lb_list = [], [], []
        for class_id in class_ids:
            if class_id not in self.exemplars:
                continue
            xh, xl = self.exemplars[class_id]
            xh_list.append(xh)
            xl_list.append(xl)
            lb_list.append(torch.full((xh.shape[0],), class_id, dtype=torch.long))

        if not xh_list:
            return None

        ds = TensorDataset(
            torch.cat(xh_list, dim=0),
            torch.cat(xl_list, dim=0),
            torch.cat(lb_list, dim=0),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    def memory_usage(self):
        return sum(xh.shape[0] for xh, _ in self.exemplars.values())


def collect_dataset_tensors_and_features(
    model,
    dataset,
    device,
    batch_size,
    active_task_ids=None,
):
    """Collect raw tensors plus current features for exemplar herding."""
    if dataset is None:
        return None, None, None, None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()

    xh_parts, xl_parts, label_parts = [], [], []
    feat_parts = {b: [] for b in BRANCHES}

    with torch.no_grad():
        for batch in loader:
            xh = batch[0].to(device)
            xl = batch[1].to(device)
            tgt = batch[2].cpu()

            aux = model.forward_features(
                xh,
                xl,
                return_aux=True,
                active_task_ids=active_task_ids,
            )

            xh_parts.append(xh.cpu())
            xl_parts.append(xl.cpu())
            label_parts.append(tgt)
            for branch_name, feat_key in (
                ("spec", "f_spec"),
                ("hsi_spa", "f_hsi_spa"),
                ("lid_spa", "f_lid_spa"),
            ):
                if feat_key in aux:
                    feat_parts[branch_name].append(aux[feat_key].cpu())

    if not xh_parts:
        return None, None, None, None

    feats = {}
    for branch_name in BRANCHES:
        feats[branch_name] = (
            torch.cat(feat_parts[branch_name], dim=0)
            if feat_parts[branch_name] else None
        )

    return (
        torch.cat(xh_parts, dim=0),
        torch.cat(xl_parts, dim=0),
        feats,
        torch.cat(label_parts, dim=0),
    )


def update_exemplar_memory(
    model,
    exemplar_store,
    dataset,
    class_ids,
    device,
    batch_size,
    active_task_ids=None,
):
    """Refresh exemplar memory for the provided classes using current features."""
    if exemplar_store is None or dataset is None or not class_ids:
        return 0

    xh_all, xl_all, feats_all, labels = collect_dataset_tensors_and_features(
        model,
        dataset,
        device,
        batch_size=batch_size,
        active_task_ids=active_task_ids,
    )
    if xh_all is None or feats_all is None or labels is None:
        return 0

    n_updated = 0
    for class_id in sorted(class_ids):
        mask_c = labels == class_id
        if mask_c.sum() == 0:
            continue
        class_feats = {
            b: feats_all[b][mask_c]
            for b in BRANCHES if feats_all.get(b) is not None
        }
        exemplar_store.update_exemplars(
            class_id,
            xh_all[mask_c],
            xl_all[mask_c],
            class_feats,
        )
        n_updated += 1

    return n_updated


# ======================================================================
# ANALYTIC RLS HEAD (Spectral-Constrained REAL, Idea #4)
# ======================================================================
class AnalyticRLSHead:
    """Ridge Regression classifier for exemplar-free CIL.

    After each task, re-solves W from ALL seen classes' current features
    (re-extracted through the current model). This avoids the drift problem
    of cumulative RLS where R/P mix different feature spaces.

    Feature input: spectral-constrained decomposition:
        f = f_spec ⊕ (f_hsi_spa - proj_spec) ⊕ (f_lid_spa - proj_spec)

    Rebuild strategy (not incremental):
        Each task: extract ALL seen class features → R = X^T X, P = X^T Y → W = (R+λI)^{-1} P
        Features are always in the current model's space → no drift.
    """

    def __init__(self, n_classes, ridge_lambda=1.0, spectral_constrained=True):
        self.n_classes = n_classes
        self.ridge_lambda = ridge_lambda
        self.spectral_constrained = spectral_constrained
        self.feat_dim = None
        self.W = None
        self._class_seen = set()

    def _build_features(self, feats):
        """Build features for ridge regression.

        If spectral_constrained=True: f_spec ⊕ spatial_residuals (orthogonal to spec)
        If spectral_constrained=False: L2-normalized concat of all branches
        """
        f_spec = feats.get("spec")
        f_hsi = feats.get("hsi_spa")
        f_lid = feats.get("lid_spa")

        if self.spectral_constrained and f_spec is not None:
            parts = [f_spec]
            spec_norm = F.normalize(f_spec, dim=1)
            if f_hsi is not None:
                proj_h = (f_hsi * spec_norm).sum(dim=1, keepdim=True) * spec_norm
                parts.append(f_hsi - proj_h)
            if f_lid is not None:
                proj_l = (f_lid * spec_norm).sum(dim=1, keepdim=True) * spec_norm
                parts.append(f_lid - proj_l)
        else:
            # Raw concat with per-branch L2 normalization to equalize scale
            parts = []
            for b in ["spec", "hsi_spa", "lid_spa"]:
                fb = feats.get(b)
                if fb is not None:
                    parts.append(F.normalize(fb, dim=1))

        if not parts:
            return None
        return torch.cat(parts, dim=1)

    def rebuild(self, all_feats_by_domain, all_labels_by_domain):
        """Full re-solve from all seen classes' current features.

        Args:
            all_feats_by_domain: dict {domain: {branch: tensor(N, D)}}
            all_labels_by_domain: dict {domain: tensor(N,)}
        """
        # Concatenate all features and labels
        all_f = []
        all_y = []
        for domain in all_feats_by_domain:
            feats = all_feats_by_domain[domain]
            labels = all_labels_by_domain[domain]
            f = self._build_features(feats)
            if f is not None:
                all_f.append(f.detach().cpu().float())
                all_y.append(labels.cpu())

        if not all_f:
            return

        X = torch.cat(all_f, dim=0)  # (N_total, D)
        labels = torch.cat(all_y, dim=0)  # (N_total,)

        self.feat_dim = X.shape[1]
        N = X.shape[0]
        n_cls = max(self.n_classes, int(labels.max().item()) + 1)
        self.n_classes = n_cls

        # Track seen classes
        for c in labels.tolist():
            self._class_seen.add(int(c))

        # One-hot encode
        Y = torch.zeros(N, n_cls)
        for i, c in enumerate(labels.tolist()):
            Y[i, int(c)] = 1.0

        # Solve: W = (X^T X + λI)^{-1} X^T Y
        R = X.t() @ X
        P = X.t() @ Y
        reg = self.ridge_lambda * torch.eye(self.feat_dim)
        try:
            self.W = torch.linalg.solve(R + reg, P)
        except Exception:
            self.W = torch.linalg.pinv(R + reg) @ P

    def predict(self, feats):
        """Predict class labels."""
        f = self._build_features(feats)
        if f is None or self.W is None:
            return torch.zeros(0, dtype=torch.long)

        f = f.detach().cpu().float()
        logits = f @ self.W

        # Mask unseen classes
        mask = torch.full((self.n_classes,), -1e9)
        for c in self._class_seen:
            if c < self.n_classes:
                mask[c] = 0.0
        logits = logits + mask.unsqueeze(0)

        return logits.argmax(dim=1)


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

    def __init__(self, embed_dim, floor=0.0):
        super().__init__()
        # Init so gate has wide dynamic range:
        # drift=0 → sigmoid(5*0-2.5)=0.08, drift=1 → sigmoid(5*1-2.5)=0.92
        self.w = nn.Parameter(torch.tensor(5.0))
        self.b = nn.Parameter(torch.tensor(-2.5))
        self.embed_dim = embed_dim
        self.floor = float(floor)
        # Registered buffer for old-class spectral prototypes
        self._proto_mat = None  # (n_old_classes, embed_dim)
        self._proto_norm = None

    def set_floor(self, floor):
        """Set a lower bound on the effective gate strength."""
        floor = float(floor)
        if floor < 0.0 or floor >= 1.0:
            raise ValueError(f"drift gate floor must be in [0, 1), got {floor}")
        self.floor = floor

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

    def compute_raw_gate(self, z_spec):
        """Compute the raw per-sample drift gate from spectral features.

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

    def apply_floor(self, gate):
        """Apply a lower bound to preserve minimum LoRA strength."""
        if self.floor <= 0.0:
            return gate
        return self.floor + (1.0 - self.floor) * gate

    def forward(self, z_spec):
        """Compute the effective gate after applying the optional floor."""
        return self.apply_floor(self.compute_raw_gate(z_spec))


def refresh_drift_gate_prototypes(
    model,
    prototype_store,
    candidate_classes,
    class_to_domain=None,
    target_domain=None,
    cross_domain_only=True,
):
    """Refresh drift-gate prototypes for a target domain.

    Default behavior is intentionally cross-domain only: when adapting or
    evaluating a domain, we compare against prototypes from *other* domains.
    This makes the gate reflect domain shift rather than suppressing adaptation
    simply because a new class resembles an older class from the same domain.

    Returns:
        Number of class prototypes loaded into the gate.
    """
    if not hasattr(model, "drift_gate"):
        return 0

    selected_classes = sorted(
        c for c in candidate_classes if c in prototype_store.prototypes
    )

    if cross_domain_only and class_to_domain is not None and target_domain is not None:
        selected_classes = [
            c for c in selected_classes
            if class_to_domain.get(c) != target_domain
        ]

    model.drift_gate.update_prototypes(prototype_store, selected_classes)
    return len(selected_classes)


def _set_module_requires_grad(module, requires_grad):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad


def _infer_model_device(model):
    backbone_param = next(model.backbone.parameters(), None)
    if backbone_param is not None:
        return backbone_param.device
    any_param = next(model.parameters(), None)
    if any_param is not None:
        return any_param.device
    return torch.device("cpu")


def get_anchor_spec_feature(aux):
    """Return the stable spectral anchor feature from a forward aux dict."""
    return aux.get("f_spec_anchor", aux.get("f_spec"))


def spectral_tether_loss(spec_cls, spec_anchor):
    """Keep the adaptive spectral track close to the frozen anchor."""
    if spec_cls is None or spec_anchor is None:
        device = (
            spec_cls.device if spec_cls is not None else
            spec_anchor.device if spec_anchor is not None else
            "cpu"
        )
        return torch.tensor(0.0, device=device)
    return (1.0 - F.cosine_similarity(spec_cls, spec_anchor.detach(), dim=1)).mean()


def _sample_proto_branch(store, class_id, branch_name, device, noise_scale):
    return store.sample_branch(
        class_id,
        branch_name,
        device,
        noise_scale=noise_scale,
        n_samples=1,
        normalize=False,
    )


def proto_aug_features_dualtrack(
    prototype_store,
    anchor_prototype_store,
    old_classes,
    device,
    n_pseudo=8,
    noise_scale=1.0,
):
    """Sample pseudo-features with separate student/teacher spectral tracks."""
    student_feats = {b: [] for b in BRANCHES}
    teacher_spec_feats = []

    for c in sorted(old_classes):
        if c not in prototype_store.prototypes:
            continue
        if (
            c not in anchor_prototype_store.prototypes
            or "spec" not in anchor_prototype_store.prototypes[c]
        ):
            continue
        for _ in range(n_pseudo):
            sample_ok = True
            sample = {}
            for b in BRANCHES:
                feat_b = _sample_proto_branch(
                    prototype_store, c, b, device, noise_scale
                )
                if feat_b is None:
                    sample_ok = False
                    break
                sample[b] = feat_b
            if not sample_ok:
                continue

            teacher_spec = _sample_proto_branch(
                anchor_prototype_store, c, "spec", device, noise_scale
            )
            if teacher_spec is None:
                continue

            for b in BRANCHES:
                student_feats[b].append(sample[b])
            teacher_spec_feats.append(teacher_spec)

    if not teacher_spec_feats:
        return None, None

    for b in BRANCHES:
        student_feats[b] = (
            torch.stack(student_feats[b], dim=0) if student_feats[b] else None
        )
    return student_feats, torch.stack(teacher_spec_feats, dim=0)


def spectral_dualtrack_kd_loss(
    student_feats,
    teacher_spec_feats,
    student_protos,
    anchor_protos,
    old_classes,
    temperature=2.0,
    kd_tau=0.0,
):
    """Anchor-teacher KD with adaptive student spectral features."""
    valid_classes = [
        c for c in sorted(old_classes)
        if c in anchor_protos
        and "spec" in anchor_protos[c]
        and c in student_protos
    ]
    if len(valid_classes) < 2 or teacher_spec_feats is None:
        device = (
            teacher_spec_feats.device
            if teacher_spec_feats is not None else
            student_feats["spec"].device if student_feats.get("spec") is not None else
            "cpu"
        )
        return torch.tensor(0.0, device=device)

    device = teacher_spec_feats.device
    spec_proto_mat = torch.stack([
        F.normalize(anchor_protos[c]["spec"].to(device).unsqueeze(0), dim=1).squeeze(0)
        for c in valid_classes
    ])
    teacher_logits = (
        F.normalize(teacher_spec_feats.detach(), dim=1) @ spec_proto_mat.t()
    )

    student_scores = None
    for b in BRANCHES:
        feat_b = student_feats.get(b)
        if feat_b is None:
            continue
        proto_list = []
        for c in valid_classes:
            if b not in student_protos.get(c, {}):
                proto_list = []
                break
            proto_list.append(
                F.normalize(student_protos[c][b].to(device).unsqueeze(0), dim=1).squeeze(0)
            )
        if not proto_list:
            continue
        proto_mat = torch.stack(proto_list)
        sim = F.normalize(feat_b, dim=1) @ proto_mat.t()
        student_scores = sim if student_scores is None else student_scores + sim

    if student_scores is None:
        return torch.tensor(0.0, device=device)

    kd_weights = None
    if kd_tau > 0.0:
        teacher_conf = F.softmax(teacher_logits / temperature, dim=1).max(dim=1).values
        kd_weights = (
            (teacher_conf - kd_tau) / (1.0 - kd_tau + 1e-8)
        ).clamp(min=0.0, max=1.0)
        if kd_weights.sum() < 1e-8:
            return torch.tensor(0.0, device=device)

    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_scores / temperature, dim=1)
    if kd_weights is not None:
        kl_per = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(1)
        return (kd_weights * kl_per).sum() / kd_weights.sum() * (temperature ** 2)
    return (
        F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        * (temperature ** 2)
    )


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
        # Keep one frozen basis snapshot per unique adapter. Reused adapters are
        # updated in place instead of being appended repeatedly.
        self.adapter_basis_snapshots = {}

    def _rebuild_subspace_basis(self, block_idx):
        """Rebuild the aggregated basis for one block from unique adapter snapshots."""
        block_bases = []
        for adapter_key, block_snaps in self.adapter_basis_snapshots.items():
            if block_idx in block_snaps:
                block_bases.append(block_snaps[block_idx])

        if block_bases:
            self.subspace_bases[block_idx] = torch.cat(block_bases, dim=0)
        else:
            self.subspace_bases.pop(block_idx, None)

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

        # Update the frozen snapshot for this unique adapter. Reused adapters
        # overwrite their previous snapshot instead of inflating the basis.
        adapter_snaps = self.adapter_basis_snapshots.setdefault(adapter_key, {})
        for block_idx, lora in enumerate(self.task_loras[adapter_key]):
            down = lora.get_down_matrix(detach=True)
            if down is not None:
                adapter_snaps[block_idx] = down
                self._rebuild_subspace_basis(block_idx)

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

    def orthogonal_regularization(self):
        """Penalize overlap with OTHER adapters only.

        When a domain reuses an adapter across tasks, continuing to train that
        adapter should not be penalized against its own previous snapshot. The
        regularizer should only keep different-domain adapters decorrelated.
        """
        active_adapter = self.get_active_task_id()
        if active_adapter is None:
            device = next(self.parameters()).device if list(self.parameters()) else "cpu"
            return torch.tensor(0.0, device=device)

        losses = []
        for block_idx, lora in enumerate(self.task_loras[active_adapter]):
            other_bases = []
            for adapter_key, block_snaps in self.adapter_basis_snapshots.items():
                if adapter_key == active_adapter:
                    continue
                if block_idx in block_snaps:
                    if hasattr(lora, "down") and lora.down is not None:
                        other_bases.append(block_snaps[block_idx].to(
                            lora.down.weight.device
                        ))

            if not other_bases:
                continue

            new_down = lora.get_down_matrix(detach=False)
            if new_down is None:
                continue
            old_basis = torch.cat(other_bases, dim=0)
            overlap = new_down @ old_basis.T
            losses.append((overlap ** 2).sum())

        if not losses:
            device = next(self.parameters()).device if list(self.parameters()) else "cpu"
            return torch.tensor(0.0, device=device)
        return sum(losses) / len(losses)


# ======================================================================
# DOMAIN-AWARE TEACHER BANK (for feature KD)
# ======================================================================
class DomainTeacherBank:
    """Stores the latest model snapshot per domain for cascade KD.

    Using the latest domain snapshot is important here: within-domain metrics are
    measured on different class sets over time, so the numerically "best" score is
    often an early, incomplete checkpoint. For cross-domain KD we want the final
    hand-off representation of the previous domain, not the earliest easy slice.
    """

    def __init__(self):
        self.teachers = {}
        self.universal_teacher = None

    def update(self, domain, model, metric):
        """Cache the latest teacher snapshot for a domain."""
        self.teachers[domain] = (snapshot_teacher_model(model), metric)
        print(f"    Teacher bank: cached latest '{domain}' (metric={metric:.4f})")

    def set_universal(self, model):
        """Set the universal teacher (typically after task 0)."""
        self.universal_teacher = snapshot_teacher_model(model)

    def get_teacher_snapshot(self, domain):
        """Get teacher snapshot for a domain, or universal fallback."""
        if domain in self.teachers:
            return self.teachers[domain][0]
        return self.universal_teacher

    def get_previous_domain_snapshot(self, current_domain, domain_order):
        """Get (prev_domain, teacher_snapshot) for the domain preceding current_domain."""
        try:
            idx = domain_order.index(current_domain)
        except ValueError:
            return None, None
        if idx == 0:
            # First domain -> use universal teacher
            return None, self.universal_teacher
        prev_domain = domain_order[idx - 1]
        return prev_domain, self.get_teacher_snapshot(prev_domain)


def _snapshot_lora_bank(bank):
    snapshot = {
        "task_lora_keys": list(bank.task_loras.keys()),
        "subspace_bases": {
            int(block_idx): basis.clone()
            for block_idx, basis in getattr(bank, "subspace_bases", {}).items()
        },
        "adapter_basis_snapshots": {
            adapter_key: {
                int(block_idx): basis.clone()
                for block_idx, basis in block_snaps.items()
            }
            for adapter_key, block_snaps in getattr(
                bank, "adapter_basis_snapshots", {}
            ).items()
        },
    }
    if hasattr(bank, "task_to_adapter"):
        snapshot["task_to_adapter"] = copy.deepcopy(bank.task_to_adapter)
    if hasattr(bank, "domain_to_adapter"):
        snapshot["domain_to_adapter"] = copy.deepcopy(bank.domain_to_adapter)
    return snapshot


def snapshot_teacher_model(model):
    """Capture the train-time state needed to reconstruct a clean teacher model."""
    snapshot = {
        "state_dict": copy.deepcopy(model.state_dict()),
        "warmup_done": bool(getattr(model, "warmup_done", False)),
        "current_task": int(getattr(model, "current_task", -1)),
        "task_to_domain": copy.deepcopy(getattr(model, "task_to_domain", {})),
        "domain_to_tasks": copy.deepcopy(getattr(model, "domain_to_tasks", {})),
        "hsi_bank": _snapshot_lora_bank(model.hsi_lora_bank),
        "lidar_bank": _snapshot_lora_bank(model.lidar_lora_bank),
    }
    if hasattr(model, "drift_gate"):
        snapshot["drift_gate_runtime"] = {
            "proto_mat": None if model.drift_gate._proto_mat is None
            else model.drift_gate._proto_mat.clone(),
            "proto_norm": None if model.drift_gate._proto_norm is None
            else model.drift_gate._proto_norm.clone(),
        }
    return snapshot


def _restore_lora_bank_snapshot(bank, bank_snapshot):
    expected_keys = set(bank_snapshot.get("task_lora_keys", []))
    for key in list(bank.task_loras.keys()):
        if key not in expected_keys:
            del bank.task_loras[key]

    if hasattr(bank, "task_to_adapter"):
        bank.task_to_adapter = copy.deepcopy(
            bank_snapshot.get("task_to_adapter", {})
        )
    if hasattr(bank, "domain_to_adapter"):
        bank.domain_to_adapter = copy.deepcopy(
            bank_snapshot.get("domain_to_adapter", {})
        )

    if hasattr(bank, "adapter_basis_snapshots"):
        bank.adapter_basis_snapshots = {
            adapter_key: {
                int(block_idx): basis.clone()
                for block_idx, basis in block_snaps.items()
            }
            for adapter_key, block_snaps in bank_snapshot.get(
                "adapter_basis_snapshots", {}
            ).items()
        }
        if bank.adapter_basis_snapshots:
            bank.subspace_bases = {}
            block_ids = sorted({
                int(block_idx)
                for block_snaps in bank.adapter_basis_snapshots.values()
                for block_idx in block_snaps.keys()
            })
            for block_idx in block_ids:
                bank._rebuild_subspace_basis(block_idx)
            return

    bank.subspace_bases = {
        int(block_idx): basis.clone()
        for block_idx, basis in bank_snapshot.get("subspace_bases", {}).items()
    }


def build_teacher_model_from_snapshot(template_model, teacher_snapshot):
    """Rebuild a teacher model without leaking newly-added adapters from the student."""
    teacher_model = copy.deepcopy(template_model)

    teacher_model.warmup_done = teacher_snapshot.get(
        "warmup_done", getattr(teacher_model, "warmup_done", False)
    )
    teacher_model.current_task = teacher_snapshot.get(
        "current_task", getattr(teacher_model, "current_task", -1)
    )
    teacher_model.task_to_domain = copy.deepcopy(
        teacher_snapshot.get("task_to_domain", {})
    )
    teacher_model.domain_to_tasks = copy.deepcopy(
        teacher_snapshot.get("domain_to_tasks", {})
    )

    _restore_lora_bank_snapshot(
        teacher_model.hsi_lora_bank, teacher_snapshot.get("hsi_bank", {})
    )
    _restore_lora_bank_snapshot(
        teacher_model.lidar_lora_bank, teacher_snapshot.get("lidar_bank", {})
    )

    teacher_model.load_state_dict(
        teacher_snapshot.get("state_dict", {}), strict=False
    )
    if hasattr(teacher_model, "drift_gate"):
        drift_runtime = teacher_snapshot.get("drift_gate_runtime", {})
        teacher_model.drift_gate._proto_mat = (
            None if drift_runtime.get("proto_mat") is None
            else drift_runtime["proto_mat"].clone()
        )
        teacher_model.drift_gate._proto_norm = (
            None if drift_runtime.get("proto_norm") is None
            else drift_runtime["proto_norm"].clone()
        )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    return teacher_model


def resolve_domain_selective_routing(args, domain_conditioned):
    """Enable domain-selective routing by default when adapters are domain-scoped."""
    if getattr(args, "disable_domain_selective", False):
        return False
    if getattr(args, "domain_selective", False):
        return True
    return bool(domain_conditioned)


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


def patch_model_with_structural_gate(model, init_alpha=-2.0):
    """Add per-block × per-branch structural gates to AnchorLoRAModel.

    8 learnable scalars (4 blocks × 2 modalities) that control LoRA
    adaptation strength at each network position:

        h_out = h_frozen + g_i * (h_lora - h_frozen)

    Implementation: wraps each LoRA bank's forward_block to apply the gate,
    instead of rewriting forward_features. This means structural gate is
    compatible with ANY forward_features variant (original, drift-gated, etc.)
    and fixes the Bug #2 overlap issue.
    """
    n_blocks = model.num_spatial_blocks
    device = next(model.backbone.parameters()).device

    # 4 gates per modality, init conservatively
    model.structural_gate_hsi = nn.Parameter(
        torch.full((n_blocks,), init_alpha, device=device)
    )
    model.structural_gate_lid = nn.Parameter(
        torch.full((n_blocks,), init_alpha, device=device)
    )

    import types

    # Wrap forward_block on each LoRA bank to apply structural gate.
    # forward_block is called as bank.forward_block(x, block_idx, ...)
    # where bank is the TaskLoRABank/DomainConditionedLoRABank instance.
    # We replace it with a plain function (not a method), so no self is passed.
    _orig_hsi_fb = model.hsi_lora_bank.forward_block
    _orig_lid_fb = model.lidar_lora_bank.forward_block

    def gated_hsi_forward_block(x, block_idx, active_task_ids=None):
        x_frozen = x
        x_lora = _orig_hsi_fb(x, block_idx, active_task_ids=active_task_ids)
        g = torch.sigmoid(model.structural_gate_hsi[block_idx])
        return x_frozen + g * (x_lora - x_frozen)

    def gated_lid_forward_block(x, block_idx, active_task_ids=None):
        x_frozen = x
        x_lora = _orig_lid_fb(x, block_idx, active_task_ids=active_task_ids)
        g = torch.sigmoid(model.structural_gate_lid[block_idx])
        return x_frozen + g * (x_lora - x_frozen)

    model.hsi_lora_bank.forward_block = gated_hsi_forward_block
    model.lidar_lora_bank.forward_block = gated_lid_forward_block

    g_vals = [f"h{i}={torch.sigmoid(model.structural_gate_hsi[i]).item():.2f}" for i in range(n_blocks)]
    g_vals += [f"l{i}={torch.sigmoid(model.structural_gate_lid[i]).item():.2f}" for i in range(n_blocks)]
    print(f"  Structural gate installed: {n_blocks} blocks × 2 modalities = {2*n_blocks} params")
    print(f"    Init: {', '.join(g_vals)}")
    print(f"    (Applied via forward_block wrapping — compatible with all forward variants)")

    return model


def patch_model_with_drift_gate(model, floor=0.0):
    """Add SpectralDriftGate to AnchorLoRAModel.

    Patches forward_features so that LoRA contributions are gated by
    spectral drift score per sample (CMDA mechanism).
    """
    embed_dim = model.backbone.embed_dim
    model.drift_gate = SpectralDriftGate(embed_dim, floor=floor)
    model.drift_gate.to(_infer_model_device(model))

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
            raw_gate = self.drift_gate.compute_raw_gate(
                f_spec.detach()
            )  # (B, 1) — no grad to spectral
            gate = self.drift_gate.apply_floor(raw_gate)
        else:
            raw_gate = None
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
                'drift_gate_raw': raw_gate,
                'drift_gate': gate,  # For logging
                'h_map': h_map.detach() if h_map is not None else None,
                'l_map': l_map.detach() if l_map is not None else None,
            }

        fused = torch.cat([f_spec, f_hsi_spa, f_lid_spa], dim=1)
        return backbone.fusion_proj(fused)

    model.forward_features = types.MethodType(gated_forward_features, model)
    print(
        f"  CMDA drift gate installed (2 learnable params: w, b; "
        f"floor={float(floor):.2f})"
    )
    return model


def patch_model_with_spectral_dualtrack(model, rank=2, scale=0.25):
    """Add a small adaptive spectral residual while keeping a frozen anchor track."""
    embed_dim = model.backbone.embed_dim
    model.spec_adapter = LoRAAdapter1D(embed_dim, rank, scale=scale)
    model.spec_adapter.to(_infer_model_device(model))
    model.use_spectral_dualtrack = True
    model.spec_adapter_rank = rank
    model.spec_adapter_scale = scale
    _set_module_requires_grad(model.spec_adapter, False)

    import types

    original_forward_features = model.forward_features.__func__
    original_begin_task = model.begin_task.__func__
    original_count_lora_params = model.count_lora_params.__func__

    def dualtrack_forward_features(
        self,
        x_hsi,
        x_lidar,
        return_aux=True,
        active_task_ids=None,
        use_lora=None,
    ):
        aux = original_forward_features(
            self,
            x_hsi,
            x_lidar,
            return_aux=True,
            active_task_ids=active_task_ids,
            use_lora=use_lora,
        )

        apply_adapter = use_lora if use_lora is not None else self.warmup_done
        f_spec_anchor = get_anchor_spec_feature(aux)
        if apply_adapter and hasattr(self, "spec_adapter"):
            f_spec_cls = self.spec_adapter(f_spec_anchor)
        else:
            f_spec_cls = f_spec_anchor

        aux["f_spec_anchor"] = f_spec_anchor
        aux["f_spec"] = f_spec_cls

        if return_aux:
            return aux

        fused = torch.cat([aux["f_spec"], aux["f_hsi_spa"], aux["f_lid_spa"]], dim=1)
        return self.backbone.fusion_proj(fused)

    def dualtrack_begin_task(self, task_id, *args, **kwargs):
        result = original_begin_task(self, task_id, *args, **kwargs)
        is_warmup = kwargs.get("is_warmup")
        if is_warmup is None and args:
            is_warmup = args[0]
        _set_module_requires_grad(self.spec_adapter, not bool(is_warmup))
        return result

    def dualtrack_count_lora_params(self):
        total = original_count_lora_params(self)
        if hasattr(self, "spec_adapter"):
            total += sum(p.numel() for p in self.spec_adapter.parameters())
        return total

    model.forward_features = types.MethodType(dualtrack_forward_features, model)
    model.begin_task = types.MethodType(dualtrack_begin_task, model)
    model.count_lora_params = types.MethodType(dualtrack_count_lora_params, model)
    print(
        f"  Spectral dual-track installed "
        f"(rank={rank}, scale={scale:.2f}, frozen anchor + adaptive residual)"
    )
    return model


def extract_features(model, loader, device, active_task_ids=None, use_lora=None):
    """Extract evaluation features, preserving both adaptive and anchor spectral tracks."""
    model.eval()
    feats = {b: [] for b in BRANCHES}
    anchor_specs = []
    labels = []
    kmap = {"f_spec": "spec", "f_hsi_spa": "hsi_spa", "f_lid_spa": "lid_spa"}
    with torch.no_grad():
        for batch in loader:
            xh, xl, tgt = batch[0].to(device), batch[1].to(device), batch[2]
            aux = model.forward_features(
                xh,
                xl,
                return_aux=True,
                active_task_ids=active_task_ids,
                use_lora=use_lora,
            )
            for k, s in kmap.items():
                if k in aux:
                    feats[s].append(aux[k].cpu())
            anchor_specs.append(get_anchor_spec_feature(aux).cpu())
            labels.append(tgt)

    for b in BRANCHES:
        feats[b] = torch.cat(feats[b], 0) if feats[b] else None
    feats["spec_anchor"] = (
        torch.cat(anchor_specs, 0) if anchor_specs else None
    )
    return feats, torch.cat(labels, 0)


# ======================================================================
# TRAINING: LoRA PHASE with domain feature KD
# ======================================================================
def build_current_prototypes(
    feats, targets, seen_classes, old_classes, prototype_store, device
):
    """Build NCM prototypes with stable old-class handling.

    Old classes always use the stored prototype. Only classes introduced by the
    current task are allowed to use in-batch means.
    """
    current_protos = {}
    new_task_classes = set(seen_classes) - set(old_classes)

    for c in seen_classes:
        if c in new_task_classes:
            mask_c = targets == c
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
        elif c in prototype_store.prototypes:
            current_protos[c] = {
                b: prototype_store.prototypes[c][b].to(device)
                for b in BRANCHES if b in prototype_store.prototypes[c]
            }

    return current_protos


def build_current_proto_stds(
    feats, targets, seen_classes, old_classes, prototype_store, device
):
    """Build diagonal std estimates aligned with current CE prototypes."""
    current_stds = {}
    new_task_classes = set(seen_classes) - set(old_classes)

    for c in seen_classes:
        if c in new_task_classes:
            mask_c = targets == c
            if mask_c.sum() > 0:
                current_stds[c] = {}
                for b in BRANCHES:
                    if feats[b] is None:
                        continue
                    feats_b = feats[b][mask_c]
                    if feats_b.shape[0] > 1:
                        current_stds[c][b] = feats_b.std(0, correction=0).clamp(min=1e-4)
                    elif (
                        c in prototype_store.stds
                        and b in prototype_store.stds[c]
                    ):
                        current_stds[c][b] = prototype_store.stds[c][b].to(device)
                    else:
                        current_stds[c][b] = torch.ones_like(feats_b[0]) * 0.05
            elif c in prototype_store.stds:
                current_stds[c] = {
                    b: prototype_store.stds[c][b].to(device)
                    for b in BRANCHES if b in prototype_store.stds[c]
                }
        elif c in prototype_store.stds:
            current_stds[c] = {
                b: prototype_store.stds[c][b].to(device)
                for b in BRANCHES if b in prototype_store.stds[c]
            }

    return current_stds


def build_proto_consistency_prototypes(
    feats, targets, old_classes, prototype_store, device
):
    """Build old-class prototypes for the geometry-preservation loss.

    Unlike CE prototypes, this view should use current student features when an
    old class appears in the batch; otherwise it falls back to the stored mean.
    This keeps `L_proto` informative for same-domain replay batches without
    destabilizing the classifier with noisy old-class batch means.
    """
    current_old_protos = {}
    for c in old_classes:
        mask_c = targets == c
        if mask_c.sum() > 0:
            current_old_protos[c] = {
                b: feats[b][mask_c].mean(0)
                for b in BRANCHES if feats[b] is not None
            }
        elif c in prototype_store.prototypes:
            current_old_protos[c] = {
                b: prototype_store.prototypes[c][b].to(device)
                for b in BRANCHES if b in prototype_store.prototypes[c]
            }
    return current_old_protos


def prototype_replay_ce_loss(
    current_protos,
    current_proto_stds,
    available_classes,
    prototype_store,
    old_classes,
    device,
    n_pseudo=8,
    noise_scale=1.0,
    classifier_mode="cosine",
    proto_score_mode="single",
    branch_weights=None,
):
    """Replay old classes by sampling pseudo-features from stored prototypes.

    This is a prototype-space replay term: sampled old-class pseudo-features are
    classified against the current NCM prototype set. The pseudo-features are
    detached samples, so gradients flow through any current-task prototypes
    participating in the logits rather than through the replay samples.
    """
    if (
        n_pseudo <= 0
        or not old_classes
        or prototype_store is None
        or len(available_classes) < 2
    ):
        return torch.tensor(0.0, device=device)

    pf, pl = proto_aug_features(
        prototype_store,
        old_classes,
        device,
        n_pseudo=n_pseudo,
        noise_scale=noise_scale,
    )
    if pf is None or pl is None:
        return torch.tensor(0.0, device=device)

    plogits = compute_classifier_logits(
        pf,
        current_protos,
        available_classes,
        classifier_mode=classifier_mode,
        proto_score_mode=proto_score_mode,
        proto_stds=current_proto_stds,
        prototype_store=prototype_store,
        branch_weights=branch_weights,
    )
    if plogits is None:
        return torch.tensor(0.0, device=device)

    p_cids = sorted(available_classes)
    p_c2i = {c: i for i, c in enumerate(p_cids)}
    p_valid = torch.tensor(
        [t.item() in p_c2i for t in pl], device=device
    )
    if p_valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    p_mapped = torch.tensor(
        [p_c2i.get(t.item(), 0) for t in pl], device=device
    )
    return F.cross_entropy(plogits[p_valid], p_mapped[p_valid])


def train_lora_task_with_domain_kd(
    model, train_loader, seen_classes, old_protos,
    prototype_store, class_to_domain, device,
    logger=None, task_idx=0,
    old_anchor_protos=None, anchor_prototype_store=None,
    teacher_model=None, importance_weights=None,
    epochs=50, lr=5e-4,
    lambda_proto=1.0, lambda_kd=1.0,
    lambda_ortho=0.1, lambda_domain_kd=0.5,
    lambda_spec_tether=0.0,
    classifier_mode="cosine",
    student_task_ids=None, teacher_task_ids=None,
    kd_pseudo_per_class=16, kd_noise_scale=1.0,
    n_pseudo=0, lambda_pseudo=0.0, noise_scale=1.0,
    exemplar_loader=None,
    proto_score_mode="single",
    use_mixture_train_logits=False,
    branch_weights=None,
):
    """Train LoRA adapters with optional domain-aware feature KD.

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
    old_anchor_protos_snapshot = {}
    for c in old_classes:
        if c in (old_anchor_protos or {}) and "spec" in old_anchor_protos[c]:
            old_anchor_protos_snapshot[c] = {
                "spec": old_anchor_protos[c]["spec"].clone()
            }

    # Prepare teacher for domain KD
    if teacher_model is not None:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    has_domain_kd = (teacher_model is not None and lambda_domain_kd > 0)
    has_spectral_dualtrack = bool(getattr(model, "use_spectral_dualtrack", False))
    train_proto_score_mode = (
        proto_score_mode
        if use_mixture_train_logits and prototype_store is not None
        else "single"
    )
    best_loss = float('inf')
    best_state = None
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        model.train()
        exemplar_iter = iter(exemplar_loader) if exemplar_loader is not None else None

        for batch in train_loader:
            xh, xl, tgt = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
            )
            n_current = tgt.shape[0]

            if exemplar_iter is not None:
                try:
                    replay_batch = next(exemplar_iter)
                except StopIteration:
                    exemplar_iter = iter(exemplar_loader)
                    replay_batch = next(exemplar_iter)

                xh_replay = replay_batch[0].to(device)
                xl_replay = replay_batch[1].to(device)
                tgt_replay = replay_batch[2].to(device)
                xh = torch.cat([xh, xh_replay], dim=0)
                xl = torch.cat([xl, xl_replay], dim=0)
                tgt = torch.cat([tgt, tgt_replay], dim=0)

            aux = model.forward_features(
                xh, xl, return_aux=True, active_task_ids=student_task_ids
            )
            feats = {b: aux[f'f_{b}'] for b in BRANCHES}
            spec_anchor = get_anchor_spec_feature(aux)

            current_protos = build_current_prototypes(
                feats, tgt, seen_classes, old_classes, prototype_store, device
            )
            current_proto_stds = build_current_proto_stds(
                feats, tgt, seen_classes, old_classes, prototype_store, device
            )

            available_classes = set(current_protos.keys())
            if len(available_classes) < 2:
                continue

            # CE loss
            logits = compute_classifier_logits(
                feats,
                current_protos,
                available_classes,
                classifier_mode=classifier_mode,
                proto_score_mode=train_proto_score_mode,
                proto_stds=current_proto_stds,
                prototype_store=prototype_store,
                branch_weights=branch_weights,
            )
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
                current_old_protos = build_proto_consistency_prototypes(
                    feats, tgt, old_classes, prototype_store, device
                )
                loss_proto = prototype_consistency_loss(
                    old_protos, current_old_protos, old_classes, device=device
                )
            else:
                loss_proto = torch.tensor(0.0, device=device)

            # Orthogonal regularization
            loss_ortho_h = model.hsi_lora_bank.orthogonal_regularization()
            loss_ortho_l = model.lidar_lora_bank.orthogonal_regularization()
            loss_ortho = loss_ortho_h + loss_ortho_l

            if has_spectral_dualtrack and lambda_spec_tether > 0:
                loss_spec_tether = spectral_tether_loss(feats["spec"], spec_anchor)
            else:
                loss_spec_tether = torch.tensor(0.0, device=device)

            # Spectral anchor KD on old-class pseudo-features.
            # Using current-batch cross-domain samples makes the frozen spectral
            # teacher unreliable; replayed old prototypes keep KD on-distribution.
            if old_protos_snapshot and len(old_classes) >= 2:
                if (
                    has_spectral_dualtrack
                    and anchor_prototype_store is not None
                    and old_anchor_protos_snapshot
                ):
                    kd_pf_student, kd_teacher_spec = proto_aug_features_dualtrack(
                        prototype_store,
                        anchor_prototype_store,
                        old_classes,
                        device,
                        n_pseudo=kd_pseudo_per_class,
                        noise_scale=kd_noise_scale,
                    )
                    if kd_pf_student is not None and kd_teacher_spec is not None:
                        loss_kd = spectral_dualtrack_kd_loss(
                            kd_pf_student,
                            kd_teacher_spec,
                            old_protos_snapshot,
                            old_anchor_protos_snapshot,
                            old_classes,
                            kd_tau=0.0,
                        )
                    else:
                        loss_kd = torch.tensor(0.0, device=device)
                else:
                    kd_pf, _ = proto_aug_features(
                        prototype_store, old_classes, device,
                        n_pseudo=kd_pseudo_per_class, noise_scale=kd_noise_scale
                    )
                    if kd_pf is not None:
                        loss_kd = spectral_anchor_kd_loss(
                            kd_pf,
                            old_protos_snapshot,
                            old_classes,
                            kd_tau=0.0,
                            branch_weights=branch_weights,
                        )
                    else:
                        loss_kd = torch.tensor(0.0, device=device)
            else:
                loss_kd = torch.tensor(0.0, device=device)

            # Domain-aware feature KD
            loss_dkd = torch.tensor(0.0, device=device)
            if has_domain_kd:
                student_feats_dkd = {
                    b: feats[b][:n_current]
                    for b in BRANCHES if feats[b] is not None
                }
                with torch.no_grad():
                    teacher_aux = teacher_model.forward_features(
                        xh[:n_current],
                        xl[:n_current],
                        return_aux=True,
                        active_task_ids=teacher_task_ids,
                    )
                teacher_feats = {b: teacher_aux[f'f_{b}'] for b in BRANCHES}
                loss_dkd = domain_feature_kd_loss(
                    student_feats_dkd, teacher_feats, importance_weights
                )

            loss_pseudo_ce = prototype_replay_ce_loss(
                current_protos,
                current_proto_stds,
                available_classes,
                prototype_store,
                old_classes,
                device,
                n_pseudo=n_pseudo,
                noise_scale=noise_scale,
                classifier_mode=classifier_mode,
                proto_score_mode=train_proto_score_mode,
                branch_weights=branch_weights,
            )

            loss = (
                loss_ce
                + lambda_proto * loss_proto
                + lambda_kd * loss_kd
                + lambda_ortho * loss_ortho
                + lambda_pseudo * loss_pseudo_ce
                + lambda_spec_tether * loss_spec_tether
                + lambda_domain_kd * loss_dkd
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
            if hasattr(model, 'spec_adapter'):
                best_state['spec_adapter'] = copy.deepcopy(model.spec_adapter.state_dict())
            if hasattr(model, 'structural_gate_hsi'):
                best_state['sg_hsi'] = model.structural_gate_hsi.data.clone()
                best_state['sg_lid'] = model.structural_gate_lid.data.clone()

        if (epoch + 1) % 10 == 0:
            print(
                f"    [LoRA+DKD] Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
                f"(CE={loss_ce.item():.3f} proto={loss_proto.item():.3f} "
                f"ortho={loss_ortho.item():.3f} kd={loss_kd.item():.3f} "
                f"dkd={loss_dkd.item():.3f} preplay={loss_pseudo_ce.item():.3f} "
                f"specT={loss_spec_tether.item():.3f})"
            )
        # Log every epoch to train_log.csv
        if logger is not None:
            logger.log_epoch(task_idx, epoch, avg_loss,
                             lr=optimizer.param_groups[0]['lr'])

    if best_state is not None:
        model.hsi_lora_bank.load_state_dict(best_state['hsi_lora'])
        model.lidar_lora_bank.load_state_dict(best_state['lidar_lora'])
        if 'drift_gate' in best_state and hasattr(model, 'drift_gate'):
            model.drift_gate.load_state_dict(best_state['drift_gate'])
        if 'spec_adapter' in best_state and hasattr(model, 'spec_adapter'):
            model.spec_adapter.load_state_dict(best_state['spec_adapter'])
        if 'sg_hsi' in best_state and hasattr(model, 'structural_gate_hsi'):
            model.structural_gate_hsi.data.copy_(best_state['sg_hsi'])
            model.structural_gate_lid.data.copy_(best_state['sg_lid'])
    model.eval()


# ======================================================================
# MARATHON RUNNER (extends run_anchor_lora_marathon)
# ======================================================================
def run_cmcd_lora_marathon(net, device, args, dataset_order=None):
    # Initialize experiment logger (v2 protocol)
    from experiment_logger import ExperimentLogger
    _log_method = "CMCD-LoRA"
    if getattr(args, "domain_conditioned_reuse", False):
        _log_method += "+DCR"
    logger = ExperimentLogger(_log_method, args.seed, "/root/autodl-tmp/runs", args=args)
    # logger is passed directly to training functions, not via args

    """Run the full CIL marathon with CMCD-LoRA components.

    This is structurally similar to run_anchor_lora_marathon but adds:
      - Domain-conditioned SD-LoRA reuse (if --domain_conditioned_reuse)
      - Domain-aware feature KD from teacher bank (if --lambda_domain_kd > 0)
    """
    if dataset_order is None:
        dataset_order = ["MUUFL", "Trento", "Houston"]

    warmup_tasks = args.warmup_tasks

    # Ablation flags
    _spectral_lora_only = getattr(args, "spectral_lora_only", False)
    _shared_spatial = getattr(args, "shared_spatial_lora", False)
    _fc_head = getattr(args, "fc_head", False)

    if _spectral_lora_only:
        hsi_rank = 0
        lidar_rank = 0
        print("  [ABLATION] Spectral LoRA only: spatial LoRA disabled")
    else:
        hsi_rank = args.lora_rank
        lidar_rank = args.lidar_rank if args.lidar_rank is not None else args.lora_rank * args.lidar_rank_mult
    domain_conditioned = getattr(args, 'domain_conditioned_reuse', False)
    domain_selective = resolve_domain_selective_routing(args, domain_conditioned)
    lambda_domain_kd = getattr(args, 'lambda_domain_kd', 0.0)
    if _fc_head:
        classifier_mode = "fc"
        print("  [ABLATION] FC head: using FC classifier instead of prototypes")
    else:
        classifier_mode = getattr(args, "classifier_mode", "cosine")
    proto_components = int(getattr(args, "proto_components", 1) or 1)
    proto_score_mode = getattr(args, "proto_score_mode", "single")
    branch_weights = getattr(args, "branch_weights", None)
    use_mixture_train_logits = bool(
        getattr(args, "use_mixture_train_logits", False)
        and proto_score_mode == "mixture"
        and proto_components > 1
    )
    use_spectral_dualtrack = getattr(args, "use_spectral_dualtrack", False)
    n_exemplars = int(getattr(args, "n_exemplars", 0) or 0)
    exemplar_batch_size = int(
        getattr(args, "exemplar_batch_size", 0) or args.batch_size
    )
    exemplar_replay_all_old = bool(
        getattr(args, "exemplar_replay_all_old", False)
    )
    drift_gate_cross_domain_only = not getattr(
        args, "drift_gate_same_domain", False
    )
    drift_gate_floor = float(getattr(args, "drift_gate_floor", 0.0) or 0.0)
    imbalance_domains = parse_domain_csv(
        getattr(args, "imbalance_domains", "MUUFL"),
        allowed_domains=dataset_order,
    )
    warmup_sampler_alpha, lora_sampler_alpha = resolve_balanced_sampler_alphas(
        args
    )
    legacy_shared_sampler = (
        float(getattr(args, "balanced_sampler_alpha", 0.0) or 0.0) > 0
        and getattr(args, "warmup_balanced_sampler_alpha", None) is None
        and getattr(args, "lora_balanced_sampler_alpha", None) is None
    )
    use_cb_loss = bool(getattr(args, "use_cb_loss", False))

    print(f"\n{'='*80}")
    print(f"CMCD-LoRA Marathon: warmup={warmup_tasks}, "
          f"HSI rank={hsi_rank}, LiDAR rank={lidar_rank}")
    print(f"Domain-conditioned reuse: {domain_conditioned}")
    print(f"Domain-selective routing: {domain_selective}")
    print(f"Domain KD lambda: {lambda_domain_kd}")
    print(
        f"Classifier: mode={classifier_mode}, "
        f"proto_components={proto_components}, score={proto_score_mode}"
    )
    print(f"Branch fusion weights: {format_branch_weights(branch_weights)}")
    print(
        "Train logits: "
        + ("mixture-aware" if use_mixture_train_logits else "single-prototype")
    )
    if use_spectral_dualtrack:
        print(
            f"Spectral dual-track: rank={args.spec_adapter_rank}, "
            f"scale={args.spec_adapter_scale}, "
            f"tether={args.lambda_spec_tether}"
        )
    if n_exemplars > 0:
        replay_scope = "all-old" if exemplar_replay_all_old else "cross-domain-old"
        print(
            f"Exemplar replay: {n_exemplars}/class, "
            f"batch={exemplar_batch_size}, scope={replay_scope}"
        )
    if getattr(args, "use_drift_gate", False):
        gate_mode = (
            "cross-domain-only"
            if drift_gate_cross_domain_only else "legacy all-old-classes"
        )
        print(
            f"Drift gate prototypes: {gate_mode} | "
            f"floor={drift_gate_floor:.2f}"
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

    # Load data
    test_padded, train_padded = {}, {}
    for ds in dataset_order:
        dp = os.path.expanduser(paths[ds])
        trn_loader, tst_loader, nc, _, _ = get_loader(
            dp, args.batch_size, args.img_size, 3, is_shuffle=False, tsk_offset=0
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
    warmup_imbalance_domains, lora_imbalance_domains = (
        resolve_phase_imbalance_domains(
            imbalance_domains,
            task_layout,
            warmup_tasks,
            use_cb_loss=use_cb_loss,
            warmup_sampler_alpha=warmup_sampler_alpha,
            lora_sampler_alpha=lora_sampler_alpha,
            allow_partial_sampler_noop=legacy_shared_sampler,
        )
    )

    if use_cb_loss:
        print(f"Warmup CB loss domains: {sorted(warmup_imbalance_domains)}")
    if warmup_sampler_alpha > 0:
        print(
            f"Warmup balanced sampler alpha: {warmup_sampler_alpha} "
            f"for domains {sorted(warmup_imbalance_domains)}"
        )
    if lora_sampler_alpha > 0:
        print(
            f"LoRA balanced sampler alpha: {lora_sampler_alpha} "
            f"for domains {sorted(lora_imbalance_domains)}"
        )
    if legacy_shared_sampler and warmup_sampler_alpha > 0:
        if not warmup_imbalance_domains and lora_imbalance_domains:
            print(
                "Legacy balanced_sampler_alpha has no warmup overlap; "
                "it will apply only during LoRA."
            )
        if not lora_imbalance_domains and warmup_imbalance_domains:
            print(
                "Legacy balanced_sampler_alpha has no LoRA overlap; "
                "it will apply only during warmup."
            )

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

    # Ablation: shared spatial LoRA (before DCR patching)
    if _shared_spatial:
        model.lidar_lora_bank = model.hsi_lora_bank
        print("  [ABLATION] Shared spatial LoRA: LiDAR reuses HSI LoRA bank")

    # Apply domain-conditioned LoRA bank if requested
    if domain_conditioned:
        model = patch_model_for_domain_conditioned_reuse(
            model, hsi_rank, lidar_rank
        )
        model.to(device)
        print("  Domain-conditioned LoRA banks installed")
        if _shared_spatial:
            model.lidar_lora_bank = model.hsi_lora_bank
            print("  [ABLATION] Shared spatial LoRA: LiDAR reuses HSI DCR bank")

    # Install structural gate if requested (must be BEFORE drift gate)
    use_structural_gate = getattr(args, 'use_structural_gate', False)
    if use_structural_gate:
        sg_init = getattr(args, 'structural_gate_init', -2.0)
        model = patch_model_with_structural_gate(model, init_alpha=sg_init)

    # Install CMDA drift gate if requested
    use_drift_gate = getattr(args, 'use_drift_gate', False)
    if use_drift_gate:
        model = patch_model_with_drift_gate(model, floor=drift_gate_floor)
    if use_spectral_dualtrack:
        model = patch_model_with_spectral_dualtrack(
            model,
            rank=args.spec_adapter_rank,
            scale=args.spec_adapter_scale,
        )

    # Domain teacher bank for feature KD
    teacher_bank = DomainTeacherBank()
    branch_dims = None
    importance_weights = None

    # Stores
    prototype_store = PrototypeStore(n_components=proto_components)
    anchor_prototype_store = PrototypeStore(n_components=1)
    exemplar_store = ExemplarStore(n_per_class=n_exemplars) if n_exemplars > 0 else None
    seen_classes = set()
    old_classes = set()

    # Track methods for comparison
    # CMCD-LoRA / +SHINE: evaluated WITH LoRA features
    # Baseline / SHINE: evaluated WITHOUT LoRA features (frozen backbone only)
    methods = ["CMCD-LoRA", "CMCD-LoRA+SHINE", "Baseline", "SHINE"]
    use_real_head = getattr(args, 'use_real_head', False)
    if use_real_head:
        methods += ["REAL", "REAL+SHINE"]
    results = {m: [] for m in methods}

    # Analytic RLS head (if requested)
    real_head = None
    real_head_nolora = None

    # Separate prototype stores for LoRA vs no-LoRA features
    baseline_prototype_store = PrototypeStore(n_components=proto_components)

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
        if ds_name in warmup_imbalance_domains and warmup_sampler_alpha > 0:
            task_loader = build_balanced_loader(
                task_train_ds,
                batch_size=args.batch_size,
                balance_alpha=warmup_sampler_alpha,
                enable_balance=True,
            )
            print(
                f"  Balanced sampler enabled for {ds_name} task data "
                f"(alpha={warmup_sampler_alpha})"
            )

        # Get old prototypes
        old_protos = {}
        for c in old_classes:
            if c in prototype_store.prototypes:
                old_protos[c] = {
                    b: prototype_store.prototypes[c][b].clone()
                    for b in BRANCHES if b in prototype_store.prototypes[c]
                }
        old_anchor_protos = {}
        for c in old_classes:
            if c in anchor_prototype_store.prototypes and "spec" in anchor_prototype_store.prototypes[c]:
                old_anchor_protos[c] = {
                    "spec": anchor_prototype_store.prototypes[c]["spec"].clone()
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
                logger=logger, task_idx=task_idx,
                epochs=args.warmup_epochs, base_lr=args.warmup_lr,
                spectral_lr_scale=args.spectral_lr_scale,
                lambda_proto=args.lambda_proto,
                use_cb_loss=(use_cb_loss and ds_name in warmup_imbalance_domains),
                classifier_mode=classifier_mode,
                proto_score_mode=proto_score_mode,
                use_mixture_train_logits=use_mixture_train_logits,
            )
        else:
            print(f"  Training LoRA+DKD "
                  f"(HSI rank={hsi_rank}, LiDAR rank={lidar_rank})")

            # Update drift gate prototypes before training (CMDA)
            if use_drift_gate and hasattr(model, 'drift_gate'):
                n_gate = refresh_drift_gate_prototypes(
                    model,
                    anchor_prototype_store,
                    old_classes,
                    class_to_domain=class_to_domain,
                    target_domain=ds_name,
                    cross_domain_only=drift_gate_cross_domain_only,
                )
                scope = (
                    "cross-domain"
                    if drift_gate_cross_domain_only else "all-old"
                )
                print(
                    f"  CMDA drift gate: {n_gate} spectral prototypes loaded "
                    f"({scope})"
                )

            # Build domain loader (all seen classes in current domain)
            ds_seen_cls = set(
                c for c in seen_classes if class_to_domain.get(c) == ds_name
            )
            all_domain_ds = subset_by_classes(train_padded[ds_name], ds_seen_cls)
            if all_domain_ds is not None:
                domain_loader = build_balanced_loader(
                    all_domain_ds,
                    batch_size=args.batch_size,
                    balance_alpha=lora_sampler_alpha,
                    enable_balance=(
                        ds_name in lora_imbalance_domains
                        and lora_sampler_alpha > 0
                    ),
                )
                if ds_name in lora_imbalance_domains and lora_sampler_alpha > 0:
                    print(
                        f"  Balanced sampler enabled for {ds_name} domain replay "
                        f"(alpha={lora_sampler_alpha})"
                    )
            else:
                domain_loader = task_loader

            # Get domain teacher for feature KD
            # The teacher is a full AnchorLoRAModel snapshot including LoRA weights.
            # Rebuild from a stored snapshot so newly-added student adapters do not
            # leak into the teacher via strict=False loading.
            teacher_model = None
            teacher_task_ids = None
            if lambda_domain_kd > 0 and task_idx > 0:
                teacher_domain, teacher_snapshot = teacher_bank.get_previous_domain_snapshot(
                    ds_name, dataset_order
                )
                if teacher_snapshot is not None:
                    teacher_model = build_teacher_model_from_snapshot(
                        model, teacher_snapshot
                    )
                    if domain_selective and teacher_domain is not None:
                        teacher_task_ids = teacher_model.get_domain_task_ids(
                            teacher_domain
                        )
                    print(f"  Domain teacher loaded for KD "
                          f"(LoRA params: {teacher_model.count_lora_params():,})")

            # Compute importance weights
            if task_idx > 0 and branch_dims is not None:
                importance_weights = compute_importance_weights(
                    prototype_store, old_classes, branch_dims
                )

            student_task_ids = (
                model.get_domain_task_ids(ds_name) if domain_selective else None
            )
            exemplar_loader = None
            replay_classes = set()
            if exemplar_store is not None:
                if exemplar_replay_all_old:
                    replay_classes = set(old_classes)
                else:
                    replay_classes = {
                        c for c in old_classes
                        if class_to_domain.get(c) != ds_name
                    }
                replay_classes = {
                    c for c in replay_classes if c in exemplar_store.exemplars
                }
                exemplar_loader = exemplar_store.get_exemplar_loader(
                    sorted(replay_classes),
                    batch_size=exemplar_batch_size,
                )
                print(
                    f"  Exemplar memory: {exemplar_store.memory_usage()} samples | "
                    f"replay classes={len(replay_classes)}"
                )

            train_lora_task_with_domain_kd(
                model, domain_loader, seen_classes, old_protos,
                prototype_store, class_to_domain, device,
                logger=logger, task_idx=task_idx,
                old_anchor_protos=old_anchor_protos,
                anchor_prototype_store=anchor_prototype_store,
                teacher_model=teacher_model,
                importance_weights=importance_weights,
                epochs=args.lora_epochs, lr=args.lora_lr,
                lambda_proto=args.lambda_proto,
                lambda_kd=args.lambda_kd,
                lambda_ortho=args.lambda_ortho,
                lambda_domain_kd=lambda_domain_kd,
                lambda_spec_tether=args.lambda_spec_tether,
                classifier_mode=classifier_mode,
                student_task_ids=student_task_ids,
                teacher_task_ids=teacher_task_ids,
                n_pseudo=args.n_pseudo,
                lambda_pseudo=args.lambda_pseudo,
                noise_scale=args.noise_scale,
                exemplar_loader=exemplar_loader,
                proto_score_mode=proto_score_mode,
                use_mixture_train_logits=use_mixture_train_logits,
                branch_weights=branch_weights,
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
            if use_drift_gate and hasattr(model, 'drift_gate'):
                refresh_drift_gate_prototypes(
                    model,
                    anchor_prototype_store,
                    seen_classes,
                    class_to_domain=class_to_domain,
                    target_domain=ds,
                    cross_domain_only=drift_gate_cross_domain_only,
                )
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
                    anchor_feats = {
                        "spec": ds_feats["spec_anchor"][mask_c]
                        if ds_feats.get("spec_anchor") is not None
                        else ds_feats["spec"][mask_c]
                    }
                    anchor_prototype_store.update(c, anchor_feats)

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

        # Rebuild REAL head from ALL seen classes' current features (if enabled)
        if use_real_head:
            total_classes = sum(sum(splits_map[ds]) for ds in dataset_order)
            if real_head is None:
                spec_constrained = not getattr(args, 'real_raw_concat', False)
                real_head = AnalyticRLSHead(
                    total_classes, ridge_lambda=1.0,
                    spectral_constrained=spec_constrained)
                real_head_nolora = AnalyticRLSHead(
                    total_classes, ridge_lambda=1.0,
                    spectral_constrained=spec_constrained)
            # Full rebuild: use ALL seen classes' features in current space
            lora_feats_all = {}
            lora_labels_all = {}
            nolora_feats_all = {}
            nolora_labels_all = {}
            for ds in dataset_order:
                if ds in lora_train_feats:
                    lora_feats_all[ds], lora_labels_all[ds] = lora_train_feats[ds]
                if ds in nolora_train_feats:
                    nolora_feats_all[ds], nolora_labels_all[ds] = nolora_train_feats[ds]
            real_head.rebuild(lora_feats_all, lora_labels_all)
            real_head_nolora.rebuild(nolora_feats_all, nolora_labels_all)
            print(f"  REAL head rebuilt: {len(real_head._class_seen)} classes, "
                  f"feat_dim={real_head.feat_dim}")

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
                if use_drift_gate and hasattr(model, 'drift_gate'):
                    refresh_drift_gate_prototypes(
                        model,
                        anchor_prototype_store,
                        seen_classes,
                        class_to_domain=class_to_domain,
                        target_domain=ds,
                        cross_domain_only=drift_gate_cross_domain_only,
                    )
                ds_feats, ds_labels = lora_train_feats[ds]
                preds = predict_ncm(
                    ds_feats,
                    all_protos,
                    set(ds_cls_seen),
                    classifier_mode=classifier_mode,
                    proto_score_mode=proto_score_mode,
                    proto_stds=prototype_store.stds,
                    prototype_store=prototype_store,
                    branch_weights=branch_weights,
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
            if use_drift_gate and hasattr(model, 'drift_gate'):
                refresh_drift_gate_prototypes(
                    model,
                    anchor_prototype_store,
                    seen_classes,
                    class_to_domain=class_to_domain,
                    target_domain=ds,
                    cross_domain_only=drift_gate_cross_domain_only,
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
            store_m = prototype_store if use_lora_feats else baseline_prototype_store
            stds_m = store_m.stds

            all_preds, all_targets = [], []
            for ds in dataset_order:
                if ds not in feats_dict:
                    continue
                eval_feats, eval_labels = feats_dict[ds]

                # REAL head evaluation (analytic RLS classifier)
                if method in ("REAL", "REAL+SHINE"):
                    head = real_head if use_lora_feats else real_head_nolora
                    if head is not None and head.W is not None:
                        if method == "REAL+SHINE" and ds in stats_m and stats_m[ds] is not None:
                            shine_feats = apply_shine(eval_feats, stats_m[ds])
                            preds = head.predict(shine_feats)
                        else:
                            preds = head.predict(eval_feats)
                    else:
                        preds = torch.zeros(eval_labels.shape[0], dtype=torch.long)
                    all_preds.append(preds)
                    all_targets.append(eval_labels)
                    continue

                if use_shine:
                    preds = predict_shine(
                        eval_feats,
                        protos_m,
                        seen_classes,
                        stats_m,
                        ds,
                        class_to_domain,
                        classifier_mode=classifier_mode,
                        proto_score_mode=proto_score_mode,
                        proto_stds=stds_m,
                        prototype_store=store_m,
                        branch_weights=branch_weights,
                        shine_mode=getattr(args, 'shine_mode', 'standard'),
                    )
                else:
                    preds = predict_ncm(
                        eval_feats,
                        protos_m,
                        seen_classes,
                        classifier_mode=classifier_mode,
                        proto_score_mode=proto_score_mode,
                        proto_stds=stds_m,
                        prototype_store=store_m,
                        branch_weights=branch_weights,
                    )

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

            # Save +SHINE predictions for AA/Kappa computation
            if "+SHINE" in method and all_preds and args.output_dir:
                _shine_dir = os.path.join(args.output_dir, "shine_predictions")
                os.makedirs(_shine_dir, exist_ok=True)
                _shine_f = os.path.join(_shine_dir, f"preds_task{task_idx}.npz")
                np.savez_compressed(_shine_f,
                    gt=all_targets_t.numpy(), pred=all_preds_t.numpy())

            # ── Protocol logging ──
            if method == methods[0] and all_preds:
                gt_np = all_targets_t.numpy()
                pred_np = all_preds_t.numpy()
                # Log per-domain metrics using compute_metrics per_ds breakdown
                for _ds_name, _ds_acc in metrics.get("per_ds", {}).items():
                    logger.log_task_eval(task_idx, _ds_name, gt_np, pred_np, seen_classes)
                # Log summary
                logger.log_task_summary(task_idx, metrics["balanced_acc"])
                # Save artifacts
                # Build domain_ids from class_to_domain mapping
                _dom_ids = [class_to_domain.get(int(g), 'unknown') for g in gt_np]
                _dom_name_to_id = {ds: i for i, ds in enumerate(dataset_order)}
                _dom_ids_num = [_dom_name_to_id.get(d, -1) for d in _dom_ids]
                logger.save_predictions(task_idx, gt_np, pred_np,
                                        task_ids=np.full(len(gt_np), task_idx),
                                        domain_ids=np.array(_dom_ids_num))
                logger.save_confusion(task_idx, gt_np, pred_np, seen_classes)

                # Save features (branch-level, using model.forward_features)
                try:
                    _feat_dict = {}
                    for _ds_f in dataset_order:
                        _ds_sub = subset_by_classes(test_padded[_ds_f], seen_classes)
                        if _ds_sub is None:
                            continue
                        _fl = DataLoader(_ds_sub, batch_size=eval_batch_size, shuffle=False)
                        _f_spec, _f_hsi, _f_lid, _f_labels = [], [], [], []
                        with torch.no_grad():
                            for _b in _fl:
                                _xh, _xl, _tgt = _b[0].to(device), _b[1].to(device), _b[2]
                                _aux = model.forward_features(_xh, _xl, return_aux=True)
                                _f_spec.append(_aux['f_spec'].float().cpu())
                                _f_hsi.append(_aux['f_hsi_spa'].float().cpu())
                                _f_lid.append(_aux['f_lid_spa'].float().cpu())
                                _f_labels.append(_tgt)
                        if _f_spec:
                            _feat_dict[f"{_ds_f}_spec"] = torch.cat(_f_spec).numpy()
                            _feat_dict[f"{_ds_f}_hsi_spa"] = torch.cat(_f_hsi).numpy()
                            _feat_dict[f"{_ds_f}_lid_spa"] = torch.cat(_f_lid).numpy()
                            _feat_dict[f"{_ds_f}_labels"] = torch.cat(_f_labels).numpy()
                    if _feat_dict:
                        logger.save_features(task_idx, _feat_dict)
                except Exception as e:
                    print(f"  [Logger] save_features failed: {e}")

                # Save logits (re-extract with model forward)
                try:
                    _all_logits = []
                    for _ds_l in dataset_order:
                        _ds_sub_l = subset_by_classes(test_padded[_ds_l], seen_classes)
                        if _ds_sub_l is None:
                            continue
                        _ll = DataLoader(_ds_sub_l, batch_size=eval_batch_size, shuffle=False)
                        with torch.no_grad():
                            for _batch_l in _ll:
                                _xh = _batch_l[0].to(device)
                                _xl = _batch_l[1].to(device)
                                _aux = model.forward_features(_xh, _xl, return_aux=True)
                                _fused = torch.cat([_aux['f_spec'], _aux['f_hsi_spa'], _aux['f_lid_spa']], dim=1)
                                _all_logits.append(_fused.cpu())
                    if _all_logits:
                        logger.save_logits(task_idx, torch.cat(_all_logits))
                except Exception as e:
                    print(f"  [Logger] save_logits failed: {e}")

        # Print results
        for m in results:
            if results[m]:
                r = results[m][-1]
                ds_str = ", ".join(
                    f"{k}={v*100:.1f}%" for k, v in r["per_ds"].items()
                )
                print(f"  {m:<22} Avg={r['avg_tag']*100:.1f}% | {ds_str}")

        if exemplar_store is not None:
            exemplar_task_ids = (
                model.get_domain_task_ids(ds_name) if domain_selective else None
            )
            exemplar_ds = subset_by_classes(train_padded[ds_name], set(cls_list))
            updated_classes = update_exemplar_memory(
                model,
                exemplar_store,
                exemplar_ds,
                cls_list,
                device,
                batch_size=eval_batch_size,
                active_task_ids=exemplar_task_ids,
            )
            print(
                f"  Exemplars updated: {updated_classes} classes | "
                f"total stored={exemplar_store.memory_usage()}"
            )

        # Save checkpoint with runtime state
        logger.save_checkpoint(task_idx, model.state_dict(), extra={
            'warmup_done': getattr(model, 'warmup_done', False),
            'domain_to_tasks': getattr(model, '_domain_to_tasks', {}),
            'seen_classes': list(seen_classes),
            'task_idx': task_idx,
        })

        # Print LoRA param count
        print(f"  LoRA params: {model.count_lora_params():,}")
        if domain_conditioned and hasattr(model.hsi_lora_bank, 'domain_to_adapter'):
            n_adapters_h = len(model.hsi_lora_bank.task_loras)
            n_adapters_l = len(model.lidar_lora_bank.task_loras)
            print(f"  Adapters: HSI={n_adapters_h}, LiDAR={n_adapters_l} "
                  f"(domain-conditioned)")

        # Print structural gate values
        if hasattr(model, 'structural_gate_hsi'):
            n_b = model.num_spatial_blocks
            g_h = [f"{torch.sigmoid(model.structural_gate_hsi[i]).item():.3f}" for i in range(n_b)]
            g_l = [f"{torch.sigmoid(model.structural_gate_lid[i]).item():.3f}" for i in range(n_b)]
            print(f"  Structural gates: HSI=[{','.join(g_h)}] LiDAR=[{','.join(g_l)}]")

    # ── Final Summary ──
    print(f"\n{'='*80}")
    print(f"FINAL: {' -> '.join(dataset_order)} ({len(task_layout)} tasks)")
    print(f"Config: warmup={warmup_tasks}, HSI_rank={hsi_rank}, "
          f"LiDAR_rank={lidar_rank}")
    print(f"Domain-conditioned reuse: {domain_conditioned}")
    print(f"Domain-selective routing: {domain_selective}")
    print(f"Domain KD lambda: {lambda_domain_kd}")
    print(
        f"Classifier: mode={classifier_mode}, "
        f"proto_components={proto_components}, score={proto_score_mode}"
    )
    print(f"Branch fusion weights: {format_branch_weights(branch_weights)}")
    print(
        "Train logits: "
        + ("mixture-aware" if use_mixture_train_logits else "single-prototype")
    )
    if exemplar_store is not None:
        print(
            f"Exemplar replay memory: {exemplar_store.memory_usage()} samples "
            f"({n_exemplars}/class)"
        )
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


    # Finalize logger
    _final_summary = {}
    for m, r_list in results.items():
        if isinstance(r_list, list) and r_list:
            _final_summary[f'{m}_final_avg_tag'] = r_list[-1]['avg_tag']
            if 'final_avg_tag' not in _final_summary:
                _final_summary['final_avg_tag'] = r_list[-1]['avg_tag']
                _final_summary['per_ds'] = r_list[-1].get('per_ds', {})
    logger.finalize(extra_summary=_final_summary)

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
    parser.add_argument("--img_size", default=9, type=int, choices=[7,9,11,13,15],
                        help="Patch size (locked to 9 after sweep)")
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
    parser.add_argument("--lidar_rank", default=None, type=int,
                        help="Direct LiDAR LoRA rank (overrides lidar_rank_mult if set)")
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
    parser.add_argument("--classifier_mode", default="cosine", type=str,
                        choices=["cosine", "maha_diag"],
                        help="Classifier used for CE/eval logits")
    parser.add_argument("--proto_components", default=1, type=int,
                        help="Number of stored sub-prototypes per class/branch")
    parser.add_argument("--proto_score_mode", default="single", type=str,
                        choices=["single", "mixture"],
                        help="Use aggregate prototype scoring or log-sum-exp "
                             "mixture scoring over stored components")
    parser.add_argument(
        "--branch_weights",
        default=None,
        type=str,
        help="Comma-separated branch fusion weights, e.g. "
             "'spec=1.2,hsi_spa=1.0,lid_spa=0.9'. Short aliases "
             "'hsi' and 'lid' are also accepted.",
    )
    parser.add_argument("--shine_mode", default="standard", type=str,
                        choices=["standard", "l2norm", "power"],
                        help="SHINE normalization mode: "
                             "standard (z-score), l2norm (z-score+L2), "
                             "power (z-score+power+L2, FeCAM-style)")
    parser.add_argument("--use_mixture_train_logits", action="store_true",
                        help="Use stored mixture components during warmup/LoRA "
                             "CE and prototype replay, while keeping default "
                             "single-prototype training when disabled")
    # Ablation flags
    parser.add_argument("--no_shine", action="store_true",
                        help="Ablation: disable SHINE domain alignment")
    parser.add_argument("--no_lora", action="store_true",
                        help="Ablation: freeze spatial branches, no LoRA")
    parser.add_argument("--spectral_lora_only", action="store_true",
                        help="Ablation: only spectral LoRA, spatial frozen")
    parser.add_argument("--shared_spatial_lora", action="store_true",
                        help="Ablation: HSI+LiDAR share one spatial LoRA")
    parser.add_argument("--fc_head", action="store_true",
                        help="Ablation: FC classifier instead of prototype NCM")

    # CMCD-LoRA specific args
    parser.add_argument("--domain_conditioned_reuse", action="store_true",
                        help="Enable domain-conditioned SD-LoRA reuse: tasks in "
                             "same domain share adapter, different domain gets new")
    parser.add_argument("--lambda_domain_kd", default=0.5, type=float,
                        help="Weight for domain-aware feature KD loss "
                             "(0 = disabled)")
    parser.add_argument("--use_real_head", action="store_true",
                        help="Add analytic RLS classifier (REAL head) as additional "
                             "evaluation method. Does not modify training.")
    parser.add_argument("--real_raw_concat", action="store_true",
                        help="REAL head uses raw L2-normalized branch concat "
                             "instead of spectral-constrained decomposition")
    parser.add_argument("--use_structural_gate", action="store_true",
                        help="Per-block × per-branch structural gate for LoRA: "
                             "8 learnable scalars controlling adaptation strength")
    parser.add_argument("--structural_gate_init", default=-2.0, type=float,
                        help="Initial alpha for structural gates "
                             "(default -2.0, gate≈0.12)")
    parser.add_argument("--use_drift_gate", action="store_true",
                        help="Enable CMDA spectral drift gate for per-sample "
                             "adaptive LoRA strength")
    parser.add_argument("--drift_gate_same_domain", action="store_true",
                        help="Legacy CMDA behavior: compute drift against all "
                             "old classes, including the current domain. "
                             "Default uses cross-domain-only prototypes.")
    parser.add_argument("--drift_gate_floor", default=0.0, type=float,
                        help="Minimum effective gate strength: "
                             "g_eff = g0 + (1-g0)*gate")
    parser.add_argument("--use_spectral_dualtrack", action="store_true",
                        help="Keep a frozen spectral anchor for gate/KD and add "
                             "a small adaptive spectral residual for classification")
    parser.add_argument("--spec_adapter_rank", default=2, type=int,
                        help="Rank of the adaptive spectral residual adapter")
    parser.add_argument("--spec_adapter_scale", default=0.25, type=float,
                        help="Residual scale for the adaptive spectral adapter")
    parser.add_argument("--lambda_spec_tether", default=0.05, type=float,
                        help="Keep the adaptive spectral track close to the "
                             "frozen anchor during LoRA training")
    parser.add_argument("--n_pseudo", default=0, type=int,
                        help="Old-class prototype replay samples per class "
                             "during LoRA (0=disable)")
    parser.add_argument("--lambda_pseudo", default=0.0, type=float,
                        help="Weight for old-class prototype replay CE")
    parser.add_argument("--noise_scale", default=1.0, type=float,
                        help="Noise scale for prototype replay sampling")
    parser.add_argument("--n_exemplars", default=0, type=int,
                        help="Raw exemplar memory per class for upper-bound "
                             "replay ablations (0=disable)")
    parser.add_argument("--exemplar_batch_size", default=0, type=int,
                        help="Replay batch size from exemplar memory during "
                             "LoRA. Defaults to --batch_size when 0.")
    parser.add_argument("--exemplar_replay_all_old", action="store_true",
                        help="Replay all stored old classes. Default only "
                             "replays cross-domain old classes because the "
                             "current domain already uses real data.")
    parser.add_argument("--use_cb_loss", action="store_true", default=False,
                        help="Enable class-balanced CE during warmup for "
                            "imbalanced domains")
    parser.add_argument("--balanced_sampler_alpha", default=0.0, type=float,
                        help="Legacy sampler alpha applied to both warmup and "
                             "LoRA phases unless the phase-specific overrides "
                             "below are set")
    parser.add_argument("--warmup_balanced_sampler_alpha", default=None, type=float,
                        help="Warmup-only sampler alpha override. Defaults to "
                             "--balanced_sampler_alpha")
    parser.add_argument("--lora_balanced_sampler_alpha", default=None, type=float,
                        help="LoRA-only sampler alpha override. Defaults to "
                             "--balanced_sampler_alpha")
    parser.add_argument("--imbalance_domains", default="MUUFL", type=str,
                        help="Comma-separated domains that should use "
                             "imbalance mitigation. Aliases such as "
                             "'muufl' and 'Houston2013' are accepted")

    # Other config (same as anchor_lora)
    parser.add_argument("--max_tasks", default=None, type=int)
    parser.add_argument("--lidar_adapter", action="store_true",
                        help="Use learned LiDAR channel adapter")
    parser.add_argument("--domain_selective", action="store_true",
                        help="Force domain-selective LoRA routing")
    parser.add_argument("--disable_domain_selective", action="store_true",
                        help="Disable domain-selective routing even when "
                             "domain-conditioned reuse is enabled")
    parser.add_argument("--dataset_order", default="MTH", type=str,
                        help="Domain ordering: MTH=MUUFL->Trento->Houston, "
                             "THM=Trento->Houston->MUUFL, etc.")

    args = parser.parse_args()
    try:
        args.branch_weights = parse_branch_weights(args.branch_weights)
    except ValueError as exc:
        parser.error(str(exc))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Also set CUBLAS workspace for full determinism on Ampere+ GPUs
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
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
    args.imbalance_domains = ",".join(
        sorted(
            parse_domain_csv(
                args.imbalance_domains,
                allowed_domains=dataset_order,
                arg_name="--imbalance_domains",
            )
        )
    )

    # Load pre-trained backbone
    ckpt_path = resolve_bootstrap_checkpoint(dataset_order, args.seed)
    print(f"Loading checkpoint: {ckpt_path}")

    backbone = S2CMNet(
        in_chans_hsi=UNIFIED_HSI_BANDS,
        in_chans_lidar=UNIFIED_LIDAR_CHANS,
        img_size=args.img_size,
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
    args.proto_components = max(1, int(args.proto_components))
    if args.proto_components == 1:
        args.proto_score_mode = "single"
    args.use_mixture_train_logits = bool(
        args.use_mixture_train_logits and args.proto_score_mode == "mixture"
    )
    results = run_cmcd_lora_marathon(net, device, args, dataset_order=dataset_order)
    effective_domain_selective = resolve_domain_selective_routing(
        args, args.domain_conditioned_reuse
    )

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
    def _fmt_flag_float(value):
        return str(value).replace("-", "m").replace(".", "p")
    flags = []
    imbalance_domains = sorted(
        parse_domain_csv(
            args.imbalance_domains,
            allowed_domains=dataset_order,
            arg_name="--imbalance_domains",
        )
    )
    imbalance_tag = format_domain_tag(imbalance_domains)
    warmup_sampler_alpha, lora_sampler_alpha = resolve_balanced_sampler_alphas(
        args
    )
    if args.domain_conditioned_reuse:
        flags.append("dcr")
    if args.lambda_domain_kd > 0:
        flags.append(f"dkd{args.lambda_domain_kd}")
    if args.lambda_kd == 0:
        flags.append("no_kd")
    if getattr(args, "classifier_mode", "cosine") != "cosine":
        flags.append(f"clf_{args.classifier_mode}")
    if getattr(args, "proto_components", 1) > 1:
        flags.append(f"pc{int(args.proto_components)}")
    if getattr(args, "proto_score_mode", "single") != "single":
        flags.append(f"ps_{args.proto_score_mode}")
    if not branch_weights_are_default(getattr(args, "branch_weights", None)):
        flags.append(
            "bw"
            f"_s{_fmt_flag_float(args.branch_weights['spec'])}"
            f"_h{_fmt_flag_float(args.branch_weights['hsi_spa'])}"
            f"_l{_fmt_flag_float(args.branch_weights['lid_spa'])}"
        )
    if getattr(args, "use_mixture_train_logits", False):
        flags.append("mixtrain")
    if getattr(args, "n_pseudo", 0) > 0 and getattr(args, "lambda_pseudo", 0.0) > 0:
        flags.append(
            "pr"
            f"_n{args.n_pseudo}"
            f"_lp{_fmt_flag_float(args.lambda_pseudo)}"
            f"_ns{_fmt_flag_float(args.noise_scale)}"
        )
    if getattr(args, "n_exemplars", 0) > 0:
        flags.append(f"er_n{int(args.n_exemplars)}")
        if getattr(args, "exemplar_batch_size", 0) > 0:
            flags.append(f"erb{int(args.exemplar_batch_size)}")
        if getattr(args, "exemplar_replay_all_old", False):
            flags.append("erall")
    if getattr(args, "use_cb_loss", False):
        flags.append(f"cb_{imbalance_tag}")
    if warmup_sampler_alpha > 0:
        flags.append(
            f"wbsa{_fmt_flag_float(warmup_sampler_alpha)}_{imbalance_tag}"
        )
    if lora_sampler_alpha > 0:
        flags.append(
            f"lbsa{_fmt_flag_float(lora_sampler_alpha)}_{imbalance_tag}"
        )
    if resolve_domain_selective_routing(args, args.domain_conditioned_reuse):
        flags.append("ds")
    if getattr(args, "use_spectral_dualtrack", False):
        flags.append(
            "specdt"
            f"_r{args.spec_adapter_rank}"
            f"_s{_fmt_flag_float(args.spec_adapter_scale)}"
            f"_t{_fmt_flag_float(args.lambda_spec_tether)}"
        )
    if getattr(args, "shine_mode", "standard") != "standard":
        flags.append(f"shine_{args.shine_mode}")
    if getattr(args, "use_real_head", False):
        flags.append("real")
    if getattr(args, "use_structural_gate", False):
        flags.append(f"sg{_fmt_flag_float(args.structural_gate_init)}")
    if getattr(args, "use_drift_gate", False):
        if getattr(args, "drift_gate_floor", 0.0) > 0:
            flags.append(f"gf{_fmt_flag_float(args.drift_gate_floor)}")
        flags.append(
            "cmda_allold" if getattr(args, "drift_gate_same_domain", False)
            else "cmda_xdom"
        )
    flag_str = "_" + "_".join(flags) if flags else ""
    config_str = (
        f"w{args.warmup_tasks}_rH{args.lora_rank}_rL{args.lidar_rank if args.lidar_rank is not None else args.lora_rank * args.lidar_rank_mult}"
        f"_{order_str}{flag_str}"
    )
    out_path = os.path.join(
        args.output_dir, f"cmcd_lora_{config_str}_seed{args.seed}.json"
    )
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                **to_ser(vars(args)),
                "domain_selective_effective": effective_domain_selective,
            },
            "dataset_order": dataset_order,
            "results": to_ser(results),
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
