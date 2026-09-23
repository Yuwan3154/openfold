"""Exact ALL-RANK epoch metrics under DDP: per-population validation means (val_pop/*) and the rank-0-only
t4/* and explore/* training scalars (t4_all/*, explore_all/*).

⛔⛔ Lightning 2.5.1 syncs epoch-level self.log keys one at a time in each rank's dict insertion order, and its
'mean' averages the int64 batch counter by truncating division. Keys created in a batch-dependent order were
CROSS-PAIRED between populations and truncated (RAW Phase M §6(3)); unsynced keys are rank 0's quarter only.
Here each path issues ONE collective on every rank whatever the rank saw: an all_gather_object of per-entry
records (validation) or an all_reduce(SUM) of float64 sums (training). Ratios are formed after the reduction.
"""

import math

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------------------- validation


def population_records(batch, metrics, source_names):
    """One (entry index, groups, {metric: value}) record per batch item, recycling dim already stripped.

    Group membership uses the old per-population tags' own rules (per item instead of item 0 of the batch):
    `is_train_overlap` -> train_overlap/held_out, `in_nonneural_subset` -> nonneural/neural_gated,
    `val_source` -> src_<name>; every entry is also in 'all'.
    """
    idxs = batch["batch_idx"].reshape(-1).tolist()
    flags = {k: batch[k].reshape(-1).tolist()
             for k in ("is_train_overlap", "in_nonneural_subset", "val_source") if k in batch}
    values = {k: v.reshape(-1).tolist() for k, v in metrics.items()}
    # a scalar metric would silently misalign every entry after the first
    assert all(len(v) == len(idxs) for v in list(flags.values()) + list(values.values())), (
        f"per-item length mismatch: idx={len(idxs)} "
        + " ".join(f"{k}={len(v)}" for k, v in {**flags, **values}.items()))
    records = []
    for i, idx in enumerate(idxs):
        groups = ["all"]
        if "is_train_overlap" in flags:
            groups.append("train_overlap" if flags["is_train_overlap"][i] else "held_out")
        if "in_nonneural_subset" in flags:
            groups.append("nonneural" if flags["in_nonneural_subset"][i] else "neural_gated")
        if "val_source" in flags:
            src = int(flags["val_source"][i])
            groups.append(f"src_{source_names.get(src, str(src))}")
        records.append((int(idx), tuple(groups), {k: float(v[i]) for k, v in values.items()}))
    return records


def gather_records(records, distributed):
    """Every rank's records, rank-major. The validation path's one collective: call it on every rank."""
    if not distributed:
        return [records]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, records)
    return gathered


def dedup_entries(gathered):
    """One record per entry index, keeping the LOWEST rank's copy, plus the number of copies dropped.

    DistributedSampler pads the last round with repeats of the first entries, which land on later ranks.
    """
    kept, n_dropped = {}, 0
    for rank_records in gathered:
        for rec in rank_records:
            if rec[0] in kept:
                n_dropped += 1
            else:
                kept[rec[0]] = rec
    return [kept[i] for i in sorted(kept)], n_dropped


def population_means(gathered, groups):
    """{metric}_{group} means over the deduplicated entries, with n_{group} and n_duplicates_dropped.

    A group with no entries gets n = 0 and no means (there is nothing to average).
    """
    records, n_dropped = dedup_entries(gathered)
    out = {"n_duplicates_dropped": float(n_dropped)}
    for g in groups:
        members = [r for r in records if g in r[1]]
        out[f"n_{g}"] = float(len(members))
        for m in (members[0][2] if members else ()):
            out[f"{m}_{g}"] = math.fsum(r[2][m] for r in members) / len(members)
    return out


# ------------------------------------------------------------------------------------------ training

_T4 = (
    "n_steps", "n_items", "n_templated",
    # the rank-0 t4/* per-step values, summed over steps
    "step_tm_pred", "step_has_template", "step_tm_template", "step_margin", "step_promote_rate",
    "n_promote_steps", "promoted",
    # per-item sums for the like-for-like versions
    "tm_template_templated", "tm_pred_templated", "tm_pred_untemplated", "promote_templated",
)
_EXPLORE = (
    "n_steps", "n_ladder_steps", "conf_picks_loss_argmin", "loss_spread", "loss_gain_vs_mean",
    "regret_vs_best", "conf_spread", "using_true_loss", "selected_rung", "selected_tau",
)
FIELDS = tuple("t4/" + k for k in _T4) + tuple("explore/" + k for k in _EXPLORE)


class TrainEpochSums:
    """Per-rank float64 running sums behind t4_all/* and explore_all/*; reset each epoch, reduced once."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = dict.fromkeys(FIELDS, 0.0)

    def add_t4_step(self, tm_pred, tm_template, has_template, promote):
        """One training step's template_gate_metrics outputs, each (B,)."""
        tp, tt, has, pr = torch.stack([tm_pred, tm_template, has_template, promote]).detach().double().cpu()
        s = self.sums
        # exactly train_openfold's t4/* expressions: n_t counts a step with no template as 1, so it logs 0
        n_t = max(float(has.sum()), 1.0)
        s["t4/n_steps"] += 1.0
        s["t4/n_items"] += float(len(tp))
        s["t4/n_templated"] += float(has.sum())
        s["t4/step_tm_pred"] += float(tp.mean())
        s["t4/step_has_template"] += float(has.mean())
        s["t4/step_tm_template"] += float((tt * has).sum()) / n_t
        s["t4/step_margin"] += float(((tp - tt) * has).sum()) / n_t
        s["t4/step_promote_rate"] += float(pr.sum()) / n_t
        s["t4/tm_template_templated"] += float((tt * has).sum())
        s["t4/tm_pred_templated"] += float((tp * has).sum())
        s["t4/tm_pred_untemplated"] += float((tp * (1.0 - has)).sum())
        s["t4/promote_templated"] += float(pr.sum())

    def add_t4_promoted(self, n):
        """A step that wrote promotions (t4/promoted_per_step's value)."""
        self.sums["t4/n_promote_steps"] += 1.0
        self.sums["t4/promoted"] += float(n)

    def add_explore_step(self, pick, best_loss, losses, confs, using_true_loss, tau=None):
        """One best-of-K step, with train_openfold's explore/* expressions; `tau` only under a noise ladder."""
        s = self.sums
        s["explore/n_steps"] += 1.0
        s["explore/conf_picks_loss_argmin"] += float(pick == best_loss)
        s["explore/loss_spread"] += float(max(losses) - min(losses))
        s["explore/loss_gain_vs_mean"] += float(sum(losses) / len(losses) - losses[pick])
        s["explore/regret_vs_best"] += float(losses[pick] - losses[best_loss])
        s["explore/conf_spread"] += float(max(confs) - min(confs))
        s["explore/using_true_loss"] += 1.0 if using_true_loss else 0.0
        if tau is not None:
            s["explore/n_ladder_steps"] += 1.0
            s["explore/selected_rung"] += float(pick)
            s["explore/selected_tau"] += float(tau)

    def all_reduce(self, distributed, device):
        """All ranks' sums. The training path's one collective: call it on every rank, every epoch."""
        t = torch.tensor([self.sums[k] for k in FIELDS], dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return dict(zip(FIELDS, t.cpu().tolist()))


def train_epoch_metrics(totals):
    """t4_all/* and explore_all/* from all-rank sums, with the counts behind every ratio.

    Same leaf names as the rank-0 tags = the same definitions over all ranks (t4_all/tm_template etc. still
    count a no-template step as 0). *_on_templated / *_on_untemplated are the like-for-like versions over
    template-bearing / template-free items. A family with no steps, or a ratio with a zero count, is omitted.
    """
    out = {}
    t4 = {k: totals["t4/" + k] for k in _T4}
    if t4["n_steps"] > 0:
        n, nt = t4["n_steps"], t4["n_templated"]
        out["t4_all/n_steps"] = n
        out["t4_all/n_items"] = t4["n_items"]
        out["t4_all/n_templated"] = nt
        for k in ("tm_pred", "has_template", "tm_template", "margin", "promote_rate"):
            out[f"t4_all/{k}"] = t4["step_" + k] / n
        if t4["n_promote_steps"] > 0:
            out["t4_all/n_promote_steps"] = t4["n_promote_steps"]
            out["t4_all/promoted_per_step"] = t4["promoted"] / t4["n_promote_steps"]
        if nt > 0:
            out["t4_all/tm_template_on_templated"] = t4["tm_template_templated"] / nt
            out["t4_all/tm_pred_on_templated"] = t4["tm_pred_templated"] / nt
            out["t4_all/margin_on_templated"] = (t4["tm_pred_templated"] - t4["tm_template_templated"]) / nt
            out["t4_all/promote_rate_on_templated"] = t4["promote_templated"] / nt
        if t4["n_items"] - nt > 0:
            out["t4_all/tm_pred_on_untemplated"] = t4["tm_pred_untemplated"] / (t4["n_items"] - nt)
    ex = {k: totals["explore/" + k] for k in _EXPLORE}
    if ex["n_steps"] > 0:
        out["explore_all/n_steps"] = ex["n_steps"]
        for k in ("conf_picks_loss_argmin", "loss_spread", "loss_gain_vs_mean", "regret_vs_best",
                  "conf_spread", "using_true_loss"):
            out[f"explore_all/{k}"] = ex[k] / ex["n_steps"]
        if ex["n_ladder_steps"] > 0:
            out["explore_all/n_ladder_steps"] = ex["n_ladder_steps"]
            out["explore_all/selected_rung"] = ex["selected_rung"] / ex["n_ladder_steps"]
            out["explore_all/selected_tau"] = ex["selected_tau"] / ex["n_ladder_steps"]
    return out
