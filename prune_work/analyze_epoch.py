"""Epoch-aggregate analysis of Run C's replica-exchange selection and T4 promotion.

Answers, over EVERY step of the epoch rather than one step:
  - which rung the pTM selector actually picks (the ladder's usage histogram)
  - whether pTM can discriminate the ladder at all (conf_spread distribution)
  - selection quality: conf_picks_loss_argmin vs the 1/K random baseline
  - what promotion did across the epoch
  - val, split by the three populations and the circularity-free subset
"""
import glob
import statistics as st
import sys
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN = sys.argv[1] if len(sys.argv) > 1 else "/home/jupyter-chenxi/runs/runC_replica_exchange"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4

series = defaultdict(list)
for ver in sorted(glob.glob(f"{RUN}/lightning_logs/version_*")):
    for ev in sorted(glob.glob(f"{ver}/events.out.tfevents.*")):
        acc = EventAccumulator(ev, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            for e in acc.Scalars(tag):
                series[tag].append((e.step, e.value))

if not series:
    print("no scalars yet")
    sys.exit(0)


def stats(tag):
    v = [x for _, x in series.get(tag, [])]
    if not v:
        return None
    v_sorted = sorted(v)
    return {
        "n": len(v), "mean": st.fmean(v), "med": v_sorted[len(v) // 2],
        "p10": v_sorted[int(0.10 * (len(v) - 1))], "p90": v_sorted[int(0.90 * (len(v) - 1))],
        "min": v_sorted[0], "max": v_sorted[-1],
    }


print("=" * 92)
print("RUNG USAGE -- which noise level the pTM selector actually picks")
print("=" * 92)
rungs = [x for _, x in series.get("explore/selected_rung_step", [])]
if rungs:
    hist = defaultdict(int)
    for r in rungs:
        hist[int(round(r))] += 1
    n = len(rungs)
    print(f"  steps: {n}   uniform baseline would be {100.0 / K:.1f}% per rung")
    for r in sorted(hist):
        print(f"    rung {r}: {hist[r]:6d}  {100.0 * hist[r] / n:5.1f}%")
    taus = [x for _, x in series.get("explore/selected_tau_step", [])]
    if taus:
        th = defaultdict(int)
        for t in taus:
            th[round(t, 3)] += 1
        print(f"  tau picked: {dict(sorted(th.items()))}")
else:
    print("  (no selected_rung_step logged yet)")

print()
print("=" * 92)
print("CAN pTM DISCRIMINATE THE LADDER? conf_spread vs loss_spread, full epoch")
print("=" * 92)
for tag in ("explore/conf_spread_step", "explore/loss_spread_step"):
    s = stats(tag)
    if s:
        print(f"  {tag:38s} n={s['n']:5d}  mean={s['mean']:.4f}  med={s['med']:.4f}  "
              f"p10={s['p10']:.4f}  p90={s['p90']:.4f}  max={s['max']:.4f}")
cs, ls = stats("explore/conf_spread_step"), stats("explore/loss_spread_step")
if cs and ls and ls["mean"]:
    print(f"  ⇒ conf_spread / loss_spread (mean ratio) = {cs['mean'] / ls['mean']:.5f}")
    print("     A tiny ratio means the 4 samples differ in true loss but look alike to pTM,")
    print("     i.e. selection is close to random regardless of the oracle gap available.")

print()
print("=" * 92)
print(f"SELECTION QUALITY -- random baseline for K={K} is {1.0 / K:.3f}")
print("=" * 92)
print("  ⭐ QUOTE THE _epoch ROWS. The _step series is only every --log_every_n_steps (20), i.e. a")
print("     1-in-20 SAMPLE; the _epoch value is the mean over EVERY step of the epoch.")
for base in ("conf_picks_loss_argmin", "regret_vs_best", "loss_gain_vs_mean",
             "using_true_loss", "conf_spread", "loss_spread"):
    for suffix in ("_epoch", "_step"):
        s = stats(f"explore/{base}{suffix}")
        if s:
            mark = "<<" if suffix == "_epoch" else "  "
            print(f"  {mark} explore/{base}{suffix:6s} n={s['n']:5d}  mean={s['mean']:.4f}  "
                  f"med={s['med']:.4f}  last={series[f'explore/{base}{suffix}'][-1][1]:.4f}")
ss = stats("explore/conf_picks_loss_argmin_epoch") or stats("explore/conf_picks_loss_argmin_step")
if ss:
    which = "epoch" if stats("explore/conf_picks_loss_argmin_epoch") else "STEP-SAMPLE (weak)"
    lift = ss["mean"] - 1.0 / K
    print(f"  ⇒ [{which}] lift over random: {lift:+.4f}  ({ss['mean'] / (1.0 / K):.2f}x random)")
    if ss["n"] < 30:
        print(f"  ⛔ n={ss['n']} is TOO FEW to quote -- the standing rule (a 1-step 1.000 that became "
              f"0.28 over an epoch) applies. Wait for a full epoch.")
ut = stats("explore/using_true_loss_step")
if ut and ut["mean"] > 0:
    print(f"  ⚠️ using_true_loss mean={ut['mean']:.3f} -- part of this epoch selected on the TRUE "
          f"loss, so conf_picks is TAUTOLOGICAL there. Split by phase before quoting.")

print()
print("=" * 92)
print("T4 PROMOTION across the epoch")
print("=" * 92)
for tag in ("t4/promoted_per_step_step", "t4/promote_rate_step", "t4/pool_written",
            "t4/pool_dropped", "t4/tm_pred_step", "t4/tm_template_step",
            "t4/margin_step", "t4/has_template_step"):
    s = stats(tag)
    if s:
        print(f"  {tag:34s} n={s['n']:5d}  mean={s['mean']:9.4f}  med={s['med']:9.4f}  max={s['max']:9.4f}")
pw = series.get("t4/pool_written", [])
if pw:
    print(f"  ⇒ pool_written last value: {pw[-1][1]:.0f} (per rank, cumulative)")
pp = stats("t4/promoted_per_step_step")
if pp:
    print(f"  ⇒ promoted_per_step mean {pp['mean']:.3f} vs K={K}: "
          f"{'OK, promote-all' if abs(pp['mean'] - K) < 0.05 else '!! not all K entering'}")

print()
print("=" * 92)
print("VALIDATION -- overall, per population, and the circularity-free subset")
print("=" * 92)
val_tags = sorted(t for t in series if t.startswith("val/"))
if not val_tags:
    print("  (no validation event yet)")
else:
    base = ["lddt_ca", "recall_2A", "gdt_ts", "alignment_rmsd", "ptm_calibration_spearman"]
    sfx = ["", "_src_pda", "_src_easy", "_src_hard", "_nonneural", "_neural_gated",
           "_held_out", "_train_overlap"]
    print(f"  {'metric':26s}" + "".join(f"{s or 'ALL':>15}" for s in sfx))
    for b in base:
        row = f"  {b:26s}"
        for s in sfx:
            t = f"val/{b}{s}"
            row += f"{series[t][-1][1]:15.4f}" if t in series else f"{'-':>15}"
        print(row)
