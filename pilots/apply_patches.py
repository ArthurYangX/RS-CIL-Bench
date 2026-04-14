#!/usr/bin/env python3
"""Apply pilot patches to cmcd_lora_experiment.py on the GPU server.

Usage:
    python apply_patches.py --pilot a       # Pilot A only
    python apply_patches.py --pilot f       # First-Task Freeze only
    python apply_patches.py --pilot af      # A + Freeze (recommended)
    python apply_patches.py --dry-run       # Preview without modifying
"""
import argparse
import shutil
import sys


def patch_pilot_a(code):
    """Pilot A: for ds in dataset_order → for ds in [ds_name]"""
    old = "        for ds in dataset_order:\n            ds_cls_seen = [c for c in seen_classes if class_to_domain.get(c) == ds]"
    idx = code.find("# EVALUATION (same structure as anchor_lora_experiment)")
    if idx < 0:
        print("ERROR: Could not find EVALUATION marker")
        return code, False
    pos = code.find(old, idx)
    if pos < 0:
        if "recompute_domains" in code[idx:idx+2000]:
            print("Pilot A: already applied")
            return code, True
        print("ERROR: Could not find prototype loop")
        return code, False
    new = ("        # [PILOT A] Only recompute current domain\n"
           "        recompute_domains = [ds_name]\n"
           "        for ds in recompute_domains:\n"
           "            ds_cls_seen = [c for c in seen_classes if class_to_domain.get(c) == ds]")
    code = code[:pos] + new + code[pos + len(old):]
    print("Pilot A: OK")
    return code, True


def patch_first_task_freeze(code):
    """First-Task Freeze: don't unfreeze reused adapters + skip training."""

    # ── Change 1: Don't unfreeze reused adapter ──
    old1 = (
        "            # Unfreeze the reused adapter so it can continue training\n"
        "            if reuse_key in self.task_loras:\n"
        "                for param in self.task_loras[reuse_key].parameters():\n"
        "                    param.requires_grad = True\n"
        "            return"
    )
    new1 = (
        "            # [FIRST-TASK FREEZE] Keep adapter frozen\n"
        "            return"
    )
    if old1 not in code:
        if "[FIRST-TASK FREEZE]" in code:
            print("Freeze (adapter): already applied")
        else:
            print("ERROR: Could not find unfreeze block")
            return code, False
    else:
        code = code.replace(old1, new1, 1)
        print("Freeze (adapter): OK")

    # ── Change 2: Skip training block when frozen ──
    # Strategy: replace the entire else-branch body from "Training LoRA+DKD"
    # through "Clean up teacher" with a freeze-aware version.

    # Find the start marker
    start_marker = '            print(f"  Training LoRA+DKD "\n                  f"(HSI rank={hsi_rank}, LiDAR rank={lidar_rank})")'
    # Find the end marker
    end_marker = "            # Clean up teacher to free memory"

    start_pos = code.find(start_marker)
    end_pos = code.find(end_marker)

    if start_pos < 0 or end_pos < 0:
        if "_lora_trainable" in code:
            print("Freeze (training): already applied")
            return code, True
        print("ERROR: Could not find training block boundaries")
        return code, False

    # Extract the original training block (between markers)
    original_block = code[start_pos:end_pos]

    # Build the replacement: check trainable, then indent original block under if
    lines = original_block.split('\n')
    # First line is the print statement — replace with check + conditional print
    new_lines = [
        '            # [FIRST-TASK FREEZE] Check if LoRA has trainable params',
        '            _lora_trainable = sum(1 for p in model.hsi_lora_bank.parameters() if p.requires_grad) + \\',
        '                             sum(1 for p in model.lidar_lora_bank.parameters() if p.requires_grad)',
        '            if _lora_trainable == 0:',
        '                print(f"  [First-Task Freeze] Adapter frozen for {ds_name}, skipping training")',
        '            else:',
        '                print(f"  Training LoRA+DKD (HSI rank={hsi_rank}, LiDAR rank={lidar_rank})")',
        '',
    ]
    # Re-indent remaining lines (skip the first print line, already handled)
    for line in lines[1:]:
        if not line.strip():
            new_lines.append('')
        else:
            # Original indent is 12 spaces (3 levels).
            # Wrap under "if _lora_trainable > 0:" → add 4 more spaces
            # But we use the check before each major section instead.
            # Simpler: just guard the train call and data loading.
            new_lines.append(line)

    # Actually, the simplest reliable approach: just guard the train_lora call
    # and domain_loader with the _lora_trainable check, without re-indenting everything
    # The drift_gate, teacher, etc. setup is harmless if training is skipped.

    replacement = (
        '            # [FIRST-TASK FREEZE] Check if LoRA has trainable params\n'
        '            _lora_trainable = sum(1 for p in model.hsi_lora_bank.parameters() if p.requires_grad) + \\\n'
        '                             sum(1 for p in model.lidar_lora_bank.parameters() if p.requires_grad)\n'
        '            if _lora_trainable == 0:\n'
        '                print(f"  [First-Task Freeze] Adapter frozen for {ds_name}, skipping training")\n'
        '            else:\n'
        '                print(f"  Training LoRA+DKD (HSI rank={hsi_rank}, LiDAR rank={lidar_rank})")\n'
    )

    # Replace just the print line, keep everything else
    code = code[:start_pos] + replacement + code[start_pos + len(lines[0]) + 1:]

    # Now find and guard the domain_loader + train call
    # Guard domain_loader construction
    old_loader = "            # Build domain loader (all seen classes in current domain)"
    new_loader = "            if _lora_trainable == 0:\n                pass  # [FREEZE] skip data loading and training\n            elif True:  # normal training path\n                pass\n            # Build domain loader (all seen classes in current domain)"

    # Actually this is getting too complex. Simpler approach:
    # Just guard the train_lora_task_with_domain_kd call.
    # The domain_loader etc will be constructed but not used. Wasteful but CORRECT.

    # Find train_lora_task_with_domain_kd call
    train_call = "            train_lora_task_with_domain_kd("
    if train_call in code:
        code = code.replace(
            train_call,
            "            if _lora_trainable > 0:  # [FREEZE] skip when adapter frozen\n                train_lora_task_with_domain_kd(",
            1
        )
        # Re-indent the arguments (they follow on subsequent lines with 16-space indent)
        # Find the closing paren
        call_pos = code.find("                train_lora_task_with_domain_kd(")
        if call_pos > 0:
            # Find lines until closing "            )"
            end_search = call_pos
            while end_search < len(code):
                line_end = code.find('\n', end_search)
                if line_end < 0:
                    break
                line = code[end_search:line_end]
                stripped = line.strip()
                if stripped == ')':
                    # Found closing paren — add 4 spaces indent
                    code = code[:end_search] + '                ' + stripped + code[line_end:]
                    break
                elif end_search > call_pos and not stripped.startswith('train_lora'):
                    # Argument line — add 4 more spaces
                    old_line = code[end_search:line_end]
                    new_line = '    ' + old_line
                    code = code[:end_search] + new_line + code[line_end:]
                    line_end += 4  # account for added spaces
                end_search = line_end + 1

        print("Freeze (training): OK — train call guarded")

    # Also guard domain_loader to avoid loading old data
    old_build = "            # Build domain loader (all seen classes in current domain)\n            ds_seen_cls"
    new_build = "            # Build domain loader\n            if _lora_trainable == 0:\n                domain_loader = task_loader  # [FREEZE] placeholder, won't be used\n            ds_seen_cls"
    if old_build in code:
        code = code.replace(old_build, new_build, 1)
        print("Freeze (loader): OK — placeholder when frozen")

    return code, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", default="af")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--file", default="/root/autodl-tmp/jc/cmcd_lora_experiment.py")
    args = parser.parse_args()

    with open(args.file) as f:
        original = f.read()

    code = original
    ok_all = True

    if "f" in args.pilot:
        code, ok = patch_first_task_freeze(code)
        ok_all = ok_all and ok
    if "a" in args.pilot:
        code, ok = patch_pilot_a(code)
        ok_all = ok_all and ok

    if not ok_all:
        sys.exit(1)

    if args.dry_run:
        import difflib
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            code.splitlines(keepends=True),
            fromfile="original", tofile="patched", n=3)
        sys.stdout.writelines(diff)
        print("\n[DRY RUN]")
    else:
        shutil.copy2(args.file, args.file + ".backup_pre_pilot")
        with open(args.file, "w") as f:
            f.write(code)
        print(f"Applied to {args.file}")

    print("\nRun:")
    print("  python cmcd_lora_experiment.py --mode marathon --seed 0 --dataset_order MTH "
          "--lora_rank 4 --lidar_rank 8 --warmup_tasks 3 --lambda_domain_kd 0.5 "
          "--domain_conditioned_reuse --proto_components 2 --proto_score_mode mixture")


if __name__ == "__main__":
    main()
