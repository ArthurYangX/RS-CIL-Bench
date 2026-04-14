"""First-Task Freeze + Pilot A: Monkey-patch approach.

Instead of modifying cmcd_lora_experiment.py via string replacement,
this module patches the relevant functions at runtime.

Usage:
    # At the top of cmcd_lora_experiment.py (or in a wrapper script):
    import freeze_patch
    freeze_patch.apply()

    # Then run normally:
    # python cmcd_lora_experiment.py --mode marathon ...

Or use the wrapper:
    python freeze_patch.py --seed 0 --dataset_order MTH ...
"""
import sys
import os

# Ensure jc is on path
os.chdir("/root/autodl-tmp/jc")
sys.path.insert(0, "/root/autodl-tmp/jc")


def apply():
    """Apply First-Task Freeze + Pilot A patches via monkey-patching."""

    import cmcd_lora_experiment as cmcd

    # ── Patch 1: Don't unfreeze reused adapters ──
    OrigAddTask = cmcd.DomainConditionedLoRABank.add_task

    def frozen_add_task(self, task_id, domain_name=None):
        tid = str(task_id)
        if domain_name is not None and domain_name in self.domain_to_adapter:
            # Same domain → reuse adapter but KEEP FROZEN
            reuse_key = self.domain_to_adapter[domain_name]
            self.task_to_adapter[tid] = reuse_key
            print(f"    [{self.modality_name}] Task {task_id} reuses FROZEN adapter "
                  f"'{reuse_key}' (domain={domain_name}) [First-Task Freeze]")
            # Do NOT unfreeze — this is the key change
            return
        # New domain → allocate new adapter (unchanged)
        OrigAddTask(self, task_id, domain_name=domain_name)

    cmcd.DomainConditionedLoRABank.add_task = frozen_add_task
    print("[freeze_patch] Patched DomainConditionedLoRABank.add_task → First-Task Freeze")

    # ── Patch 2: Skip training when adapter is frozen ──
    orig_train = cmcd.train_lora_task_with_domain_kd

    def guarded_train(*args, **kwargs):
        model = args[0] if args else kwargs.get('model')
        if model is not None:
            # Check ALL trainable params, not just LoRA banks
            # (covers drift_gate, structural_gate, spec_adapter, etc.)
            n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
            if n_trainable == 0:
                print("    [First-Task Freeze] No trainable params, skipping training")
                return
        return orig_train(*args, **kwargs)

    cmcd.train_lora_task_with_domain_kd = guarded_train
    print("[freeze_patch] Patched train_lora_task_with_domain_kd → skip when frozen")

    # ── Patch 3 (Pilot A): Override the evaluation recomputation scope ──
    # This is harder to monkey-patch since it's inline in the marathon loop.
    # We'll set a flag that the marathon loop can check.
    cmcd._PILOT_A_CURRENT_DOMAIN_ONLY = True
    print("[freeze_patch] Set _PILOT_A_CURRENT_DOMAIN_ONLY = True")
    print("[freeze_patch] NOTE: Pilot A requires a manual one-line change in the marathon loop:")
    print("               line ~2601: for ds in dataset_order → for ds in [ds_name]")
    print("               OR: for ds in ([ds_name] if getattr(cmcd, '_PILOT_A_CURRENT_DOMAIN_ONLY', False) else dataset_order):")


def apply_pilot_a_inline(code_string):
    """Apply Pilot A via string replacement.

    Three changes:
    1. Initialize domain_stats before the task loop (persist across tasks)
    2. Don't reset domain_stats to empty each task
    3. Loop only over current domain for feature extraction
    """
    # Change 1: Add persistent domain_stats initialization before task loop
    loop_marker = "    for task_idx, (ds_name, cls_list) in enumerate(task_layout):"
    pos_loop_start = code_string.find(loop_marker)
    if pos_loop_start < 0:
        return code_string, False
    init_line = ("    # [PILOT A] Persistent domain stats across tasks\n"
                 "    domain_stats = {}\n"
                 "    baseline_domain_stats = {}\n\n"
                 + loop_marker)
    code_string = code_string[:pos_loop_start] + init_line + code_string[pos_loop_start + len(loop_marker):]

    # Change 2: Remove per-task reset of domain_stats
    eval_marker = "# EVALUATION (same structure as anchor_lora_experiment)"
    idx = code_string.find(eval_marker)
    if idx < 0:
        return code_string, False

    old_reset = ("        lora_train_feats = {}\n"
                 "        nolora_train_feats = {}\n"
                 "        domain_stats = {}\n"
                 "        baseline_domain_stats = {}")
    new_reset = ("        lora_train_feats = {}\n"
                 "        nolora_train_feats = {}\n"
                 "        # [PILOT A] domain_stats persists — only current domain updated below")

    pos_reset = code_string.find(old_reset, idx)
    if pos_reset < 0:
        return code_string, False
    code_string = code_string[:pos_reset] + new_reset + code_string[pos_reset + len(old_reset):]

    # Change 3: Loop only over current domain
    old_loop = "        for ds in dataset_order:\n            ds_cls_seen = [c for c in seen_classes if class_to_domain.get(c) == ds]"
    idx = code_string.find(eval_marker)
    pos_loop = code_string.find(old_loop, idx)
    if pos_loop < 0:
        return code_string, False
    new_loop = ("        # [PILOT A] Only recompute current domain\n"
                "        for ds in [ds_name]:\n"
                "            ds_cls_seen = [c for c in seen_classes if class_to_domain.get(c) == ds]")
    code_string = code_string[:pos_loop] + new_loop + code_string[pos_loop + len(old_loop):]

    return code_string, True


if __name__ == "__main__":
    # When run directly, apply patches and then forward to cmcd_lora_experiment
    print("=" * 60)
    print("  First-Task Freeze + Pilot A Wrapper")
    print("=" * 60)

    # Apply Pilot A via file modification (backup + restore after)
    target = "/root/autodl-tmp/jc/cmcd_lora_experiment.py"
    with open(target) as f:
        original_code = f.read()

    patched_code, ok = apply_pilot_a_inline(original_code)
    if ok:
        with open(target, "w") as f:
            f.write(patched_code)
        print("[wrapper] Applied Pilot A to cmcd_lora_experiment.py")
    else:
        print("[wrapper] WARNING: Could not apply Pilot A")

    # Register restoration for SIGTERM and atexit (SIGKILL still can't be caught)
    import signal
    import atexit

    def restore_file():
        with open(target, "w") as f:
            f.write(original_code)
        print("[wrapper] Restored original cmcd_lora_experiment.py")

    atexit.register(restore_file)
    signal.signal(signal.SIGTERM, lambda s, f: (restore_file(), sys.exit(1)))

    try:
        # Import and reload to pick up the Pilot A file change
        import importlib
        import cmcd_lora_experiment
        importlib.reload(cmcd_lora_experiment)

        # Apply monkey patches AFTER reload (reload resets class/function defs)
        apply()

        # Forward remaining args to cmcd_lora_experiment.main()
        sys.argv = [target] + sys.argv[1:]
        cmcd_lora_experiment.main()
    finally:
        restore_file()
