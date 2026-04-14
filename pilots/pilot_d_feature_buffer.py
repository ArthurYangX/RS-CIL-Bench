"""Pilot D: Feature-buffer mode (L2).

Goal: Store frozen-backbone GAP features after warmup. Use these features
      (instead of raw patches) for LoRA training on old classes.
      Never re-access raw training data after warmup.

Storage: ~5 MB (feature vectors) + 101 KB (prototypes/SHINE/adapters)
         Still less than iCaRL's 7.5 MB raw-patch buffer.

Key insight: After warmup, the backbone is FROZEN. So the frozen-backbone
features (without LoRA) are deterministic for each training sample.
We extract them once and store them.

For LoRA training, we have two sub-options:

  D1: Linear approximation
      Treat multi-block LoRA as a single linear transform T on GAP features.
      f_adapted = T @ f_frozen (differentiable, backprop through T to update LoRA).
      Pro: simple. Con: ignores inter-block nonlinearities.

  D2: Feature replay (simpler, recommended)
      Use stored frozen features as input to a lightweight training loop
      that only involves LoRA layers + loss computation.
      Still needs frozen VSSBlock weights in memory for intermediate computation.
      Pro: exact gradients. Con: needs model in memory (but not raw data).

Implementation below focuses on D2 (feature replay) as it's more practical.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class FeatureBufferDataset(Dataset):
    """Dataset that serves pre-extracted frozen-backbone features.

    Each item: (f_hsi_frozen, f_lid_frozen, f_spec, label)
    where f_hsi_frozen and f_lid_frozen are the frozen backbone spatial features
    BEFORE any LoRA is applied, and f_spec is the spectral feature (always frozen).
    """

    def __init__(self):
        self.features = {}  # {class_id: {spec: tensor, hsi_spa: tensor, lid_spa: tensor}}
        self.labels = []
        self._built = False
        self._all_feats = None
        self._all_labels = None

    def add_class(self, class_id, feats_dict):
        """Add frozen features for a class.

        Args:
            class_id: int
            feats_dict: {branch_name: tensor of shape (N, dim)}
        """
        self.features[class_id] = {
            b: feats_dict[b].cpu() for b in feats_dict if feats_dict[b] is not None
        }
        self._built = False

    def _build(self):
        """Flatten into tensors for DataLoader."""
        all_spec, all_hsi, all_lid, all_labels = [], [], [], []
        assert self.features, "Feature buffer is empty — add classes before building"
        for c, fd in self.features.items():
            # All branches must be present (enforced by extract_and_store_features)
            assert all(b in fd for b in ("spec", "hsi_spa", "lid_spa")), \
                f"Class {c} missing branches: {set(('spec','hsi_spa','lid_spa')) - set(fd.keys())}"
            n = fd["spec"].shape[0]
            all_spec.append(fd["spec"])
            all_hsi.append(fd["hsi_spa"])
            all_lid.append(fd["lid_spa"])
            all_labels.extend([c] * n)
        self._all_feats = {
            "spec": torch.cat(all_spec, 0),
            "hsi_spa": torch.cat(all_hsi, 0),
            "lid_spa": torch.cat(all_lid, 0),
        }
        self._all_labels = torch.tensor(all_labels, dtype=torch.long)
        self._built = True

    def __len__(self):
        if not self._built:
            self._build()
        return self._all_labels.shape[0]

    def __getitem__(self, idx):
        if not self._built:
            self._build()
        return (
            self._all_feats["spec"][idx],
            self._all_feats["hsi_spa"][idx],
            self._all_feats["lid_spa"][idx],
            self._all_labels[idx],
        )

    def memory_usage_bytes(self):
        """Total storage in bytes."""
        total = 0
        for c, fd in self.features.items():
            for b, t in fd.items():
                total += t.nelement() * t.element_size()
        return total

    def memory_usage_mb(self):
        return self.memory_usage_bytes() / (1024 * 1024)

    def get_classes(self):
        return sorted(self.features.keys())

    def subset(self, class_ids):
        """Return a new FeatureBufferDataset with only the specified classes."""
        sub = FeatureBufferDataset()
        for c in class_ids:
            if c in self.features:
                sub.features[c] = self.features[c]
        return sub


def extract_and_store_features(model, train_padded, dataset_order, offsets,
                               class_to_domain, seen_classes, device,
                               batch_size=128):
    """One-time extraction of frozen-backbone features after warmup.

    Returns a FeatureBufferDataset containing GAP-pooled features
    for all seen training samples, extracted WITHOUT LoRA.
    """
    from anchor_lora_experiment import (
        subset_by_classes, extract_features, BRANCHES, PaddedDataset
    )

    buffer = FeatureBufferDataset()

    for ds in dataset_order:
        ds_cls = [c for c in seen_classes if class_to_domain.get(c) == ds]
        if not ds_cls:
            continue
        ds_subset = subset_by_classes(train_padded[ds], set(ds_cls))
        if ds_subset is None:
            continue
        ds_loader = DataLoader(ds_subset, batch_size=batch_size,
                               shuffle=False, drop_last=False)

        # Extract WITHOUT LoRA (frozen backbone only)
        feats, labels = extract_features(model, ds_loader, device, use_lora=False)

        for c in ds_cls:
            mask = labels == c
            if mask.sum() > 0:
                buffer.add_class(c, {
                    b: feats[b][mask] for b in BRANCHES if feats[b] is not None
                })

    print(f"Feature buffer: {len(buffer)} samples, "
          f"{buffer.memory_usage_mb():.1f} MB")
    return buffer


def train_lora_with_feature_buffer(
    model, current_task_loader, feature_buffer, old_classes,
    seen_classes, prototype_store, device,
    epochs=50, lr=5e-4, lambda_proto=1.0, lambda_pseudo=0.5,
):
    """Train LoRA using real data for current task + feature buffer for old classes.

    For old classes, we load frozen features from the buffer and apply
    the current LoRA transform to get adapted features. Loss is computed
    in feature space.

    NOTE: This is a simplified version. Full integration would need to
    handle all the loss terms (CE, proto, ortho, DKD) from the original
    training function.
    """
    # This is a sketch — full implementation would mirror
    # train_lora_task_with_domain_kd but replace raw-data old-class
    # batches with feature-buffer batches.
    #
    # Key difference: for buffer batches, we skip the backbone forward pass
    # and directly apply LoRA transform to stored features.
    pass


# ============================================================
# INTEGRATION PLAN:
#
# 1. After warmup (task 2 ends):
#    feature_buffer = extract_and_store_features(
#        model, train_padded, dataset_order, ...)
#
# 2. After each subsequent task (task 3+):
#    # Update buffer with new classes (need raw data for NEW classes only)
#    new_cls_feats = extract_features(model, new_task_loader, use_lora=False)
#    feature_buffer.add_class(new_cls, new_cls_feats)
#
# 3. For LoRA training:
#    # Current task: use raw data (normal)
#    # Old classes: use feature_buffer (no raw data access)
#    old_buffer_loader = feature_buffer.subset(old_domain_classes)
#
# 4. For prototype update:
#    # Use Eq.6 analytic correction (Pilot B)
#    # OR re-extract from feature_buffer + current LoRA transform
#
# Storage budget:
#    6,430 samples × 3 branches × 64 dims × 4 bytes = 4.7 MB
#    vs iCaRL: 640 samples × 38 channels × 9 × 9 × 4 bytes = 7.5 MB
# ============================================================
