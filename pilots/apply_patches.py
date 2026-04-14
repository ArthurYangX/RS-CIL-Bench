#!/usr/bin/env python3
"""Apply pilot patches to cmcd_lora_experiment.py on the GPU server.

Usage:
    python apply_patches.py [--pilot a|c|ac] [--dry-run]

Patches:
    Pilot A: Only recompute prototypes/SHINE for current domain (line ~2598)
    Pilot C: LoRA training uses only current-task new classes (line ~2478)
             (must also enable --n_pseudo and --lambda_pseudo at runtime)

    'ac' applies both A and C together (recommended for exemplar-free test).

This script modifies cmcd_lora_experiment.py IN PLACE.
Always back up the original file first!
"""
import argparse
import re
import shutil
import sys


def patch_pilot_a(code):
    """Pilot A: for ds in dataset_order → for ds in [ds_name]
    in the prototype/SHINE recomputation loop."""

    # Find the exact pattern
    old = "        for ds in dataset_order:\n            ds_cls_seen = [c for c in seen_classes if class_to_domain.get(c) == ds]"
    new = ("        # [PILOT A] Only recompute current domain (cross-domain features unchanged)\n"
           "        recompute_domains = [ds_name] if model.warmup_done else [ds_name]\n"
           "        for ds in recompute_domains:\n"
           "            ds_cls_seen = [c for c in seen_classes if class_to_domain.get(c) == ds]")

    # This pattern appears in the EVALUATION section (after training)
    # We need to only patch the one inside the evaluation block, not any other occurrence
    # The evaluation block starts after "# EVALUATION"
    eval_marker = "# EVALUATION (same structure as anchor_lora_experiment)"

    idx = code.find(eval_marker)
    if idx < 0:
        print("ERROR: Could not find EVALUATION marker")
        return code, False

    # Find the first occurrence of `old` after the eval marker
    search_start = idx
    pos = code.find(old, search_start)
    if pos < 0:
        # Maybe already patched?
        if "recompute_domains" in code[search_start:search_start+2000]:
            print("Pilot A: already applied (found recompute_domains)")
            return code, True
        print("ERROR: Could not find prototype recomputation loop after EVALUATION")
        return code, False

    code = code[:pos] + new + code[pos + len(old):]
    print("Pilot A: patched prototype recomputation loop")
    return code, True


def patch_pilot_c(code):
    """Pilot C: Change LoRA training data from all-domain-seen to current-task-only.

    Changes:
        ds_seen_cls = set(c for c in seen_classes if class_to_domain.get(c) == ds_name)
        all_domain_ds = subset_by_classes(train_padded[ds_name], ds_seen_cls)
    To:
        all_domain_ds = subset_by_classes(train_padded[ds_name], task_class_set)
    """

    old = ("            # Build domain loader (all seen classes in current domain)\n"
           "            ds_seen_cls = set(\n"
           "                c for c in seen_classes if class_to_domain.get(c) == ds_name\n"
           "            )\n"
           "            all_domain_ds = subset_by_classes(train_padded[ds_name], ds_seen_cls)")

    new = ("            # [PILOT C] Only use current task's NEW classes for LoRA training\n"
           "            # Old classes get pseudo replay via --n_pseudo / --lambda_pseudo\n"
           "            ds_seen_cls = set(\n"
           "                c for c in seen_classes if class_to_domain.get(c) == ds_name\n"
           "            )\n"
           "            all_domain_ds = subset_by_classes(train_padded[ds_name], task_class_set)")

    if old not in code:
        if "# [PILOT C]" in code:
            print("Pilot C: already applied")
            return code, True
        print("ERROR: Could not find domain loader construction")
        return code, False

    code = code.replace(old, new, 1)
    print("Pilot C: patched LoRA training data to current-task-only")
    return code, True


def main():
    parser = argparse.ArgumentParser(description="Apply pilot patches")
    parser.add_argument("--pilot", choices=["a", "c", "ac"], default="ac",
                        help="Which pilot(s) to apply")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without modifying the file")
    parser.add_argument("--file", default="/root/autodl-tmp/jc/cmcd_lora_experiment.py",
                        help="Path to cmcd_lora_experiment.py")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        original = f.read()

    code = original
    success = True

    if "a" in args.pilot:
        code, ok = patch_pilot_a(code)
        success = success and ok

    if "c" in args.pilot:
        code, ok = patch_pilot_c(code)
        success = success and ok

    if not success:
        print("\nSome patches failed. File NOT modified.")
        sys.exit(1)

    if args.dry_run:
        # Show diff
        import difflib
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            code.splitlines(keepends=True),
            fromfile="original",
            tofile="patched",
            n=3,
        )
        sys.stdout.writelines(diff)
        print("\n[DRY RUN] No changes made.")
    else:
        # Backup
        backup = args.file + ".backup_pre_pilot"
        shutil.copy2(args.file, backup)
        print(f"Backup saved to {backup}")

        with open(args.file, "w") as f:
            f.write(code)
        print(f"Patches applied to {args.file}")

    print(f"\nTo run Pilot {'A' if args.pilot == 'a' else 'C' if args.pilot == 'c' else 'A+C'}:")
    cmd = ("python cmcd_lora_experiment.py --mode marathon --seed 0 --dataset_order MTH "
           "--lora_rank 4 --lidar_rank 8 --warmup_tasks 3 --lambda_domain_kd 0.5 "
           "--domain_conditioned_reuse --proto_components 2 --proto_score_mode mixture")
    if "c" in args.pilot:
        cmd += " --n_pseudo 16 --lambda_pseudo 0.5"
    print(f"  {cmd}")


if __name__ == "__main__":
    main()
