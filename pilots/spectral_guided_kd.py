"""Spectral-Guided KD: Use frozen spectral branch as teacher for spatial LoRA.

Exploits drift asymmetry: spectral branch is stable across domains,
so its class rankings are reliable supervision for spatial adaptation.

Unlike the existing spectral_anchor_kd_loss (which uses proto_aug pseudo features),
this operates on CURRENT BATCH real features → direct gradients to LoRA.
"""
import torch
import torch.nn.functional as F

BRANCHES = ("spec", "hsi_spa", "lid_spa")


def spectral_guided_kd_loss(
    feats,
    old_protos,
    old_classes,
    class_to_domain=None,
    current_domain=None,
    temperature=2.0,
    same_domain_only=True,
):
    """KD loss: spectral teacher guides spatial student on old-class rankings.

    For each sample in the current batch:
      Teacher: spectral-only cosine similarity against old-class prototypes
      Student: full 3-branch cosine similarity against old-class prototypes
      Loss: KL(student || teacher)

    Args:
        feats: dict {branch: (B, D) tensor} — current batch features from model
        old_protos: dict {class_id: {branch: (D,) tensor}} — stored prototypes
        old_classes: set of old class IDs
        class_to_domain: dict {class_id: domain_name}
        current_domain: str — current training domain
        temperature: float — KD temperature
        same_domain_only: bool — if True, only constrain same-domain old classes
    """
    # Filter old classes
    if same_domain_only and class_to_domain and current_domain:
        target_classes = sorted(
            c for c in old_classes
            if c in old_protos and class_to_domain.get(c) == current_domain
        )
    else:
        target_classes = sorted(c for c in old_classes if c in old_protos)

    if len(target_classes) < 2:
        device = feats["spec"].device if feats.get("spec") is not None else "cpu"
        return torch.tensor(0.0, device=device)

    device = feats["spec"].device

    # Verify all target classes have spectral prototypes
    valid_classes = [
        c for c in target_classes
        if "spec" in old_protos.get(c, {})
    ]
    if len(valid_classes) < 2:
        return torch.tensor(0.0, device=device)

    # Teacher: spectral-only logits (frozen branch, detach for teacher role)
    spec_proto_mat = torch.stack([
        F.normalize(old_protos[c]["spec"].to(device).unsqueeze(0), dim=1).squeeze(0)
        for c in valid_classes
    ])  # (n_old, D)
    teacher_logits = (
        F.normalize(feats["spec"].detach(), dim=1) @ spec_proto_mat.t()
    )  # (B, n_old) — detach: teacher is fixed

    # Student: full 3-branch logits (LoRA-adapted, has gradients)
    student_scores = None
    for b in BRANCHES:
        if feats[b] is None:
            continue
        proto_mat = torch.stack([
            F.normalize(old_protos[c][b].to(device).unsqueeze(0), dim=1).squeeze(0)
            for c in valid_classes
            if b in old_protos.get(c, {})
        ])
        if proto_mat.shape[0] != len(valid_classes):
            continue  # skip branch if not all classes have this branch
        sim = F.normalize(feats[b], dim=1) @ proto_mat.t()  # (B, n_old)
        student_scores = sim if student_scores is None else student_scores + sim

    if student_scores is None:
        return torch.tensor(0.0, device=device)

    # KL divergence: student should match teacher's old-class distribution
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_scores / temperature, dim=1)

    return F.kl_div(
        student_log_probs, teacher_probs, reduction="batchmean"
    ) * (temperature ** 2)
