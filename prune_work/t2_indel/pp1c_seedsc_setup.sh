#!/bin/bash
# One-time setup of a SEPARATE worktree of ~/protpardelle-1c carrying the seed_self_cond patch (patches/0001-*.patch), so the shared checkout
# (branch t2-batch-opt, used by every other run) is not switched or modified. Idempotent. Run inside a SLURM/debug-cpu allocation.
set -e
R=$HOME/protpardelle-1c
W=$HOME/pp1c_seedsc
P=$(ls $HOME/of_t2indel/prune_work/t2_indel/patches/0001-*.patch)
[ -d "$W" ] || git -C "$R" worktree add -B t2-seed-selfcond-sc "$W" t2-batch-opt
if ! grep -q seed_self_cond "$W/src/protpardelle/core/models.py"; then git -C "$W" -c user.name=pp1c-seedsc -c user.email=noreply@invalid am "$P"; fi
[ -e "$W/model_params" ] || ln -s "$R/model_params" "$W/model_params"
grep -c seed_self_cond "$W/src/protpardelle/core/models.py"
git -C "$W" log --oneline -2
ls "$W/model_params" | head -3
