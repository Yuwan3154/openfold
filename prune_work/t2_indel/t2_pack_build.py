"""Build the packed T2 template product from a loose template tree (design T2_PACKED_TEMPLATES_PLAN.md; user 10-11: both levels, float32, rescue chains and rewind provenance in the same pack).
Input : <templates>/<id[1:3]>/<id>.npz (+ <id>.metrics.csv, + <id>.rescue.metrics.csv for rescued chains; a rescued chain's npz comes from the rescue run).
Output: <out>/chain.NNN.pack + chain.idx.npz (level 1), tmpl.NNN.bin + tmpl.idx.npz (level 2), metrics_all.csv.gz, MANIFEST.json  (layout: t2_pack_reader.py).
Shards = contiguous id ranges of about --shard-gb of loose-file bytes; each shard is built independently (--procs workers), written to .partial, md5-verified by re-reading (level 1) and renamed; a shard whose meta file exists is skipped (resumable).
Provenance of a main-run chain = its schedule: metrics.csv written before --full-step-until = full schedule (500 total steps, rewind 250), later = half-step (250 total, rewind 125); rescued chains (rewind 100, 250 total steps).
Run: python t2_pack_build.py --templates T --out O [--level both] [--shard-gb 2] [--procs 8] [--list template_chains.txt] [--full-step-until 2026-10-10T19:34:00-04:00]
Append (T2_TOPUP_DIVERSITY_PLAN.md section 4): python t2_pack_build.py --append --out O --topup T --rounds P1 P2 R1 [--shard-gb 2] [--procs 8]
  Loose top-up rounds <T>/<round>/templates/<id[1:3]>/<id>.npz (+ <id>.metrics.csv, which exists also for chains without an npz) become NEW shards numbered after the existing ones (one round never shares a shard); the existing shards are never rewritten.
  Every template / metrics row is tagged source ('topup-production' for P rounds, 'topup-rewind100' for R rounds), round, rewind, num_steps (t2_topup.ROUNDS; 250 steps). The indexes are rebuilt from the per-shard meta files (a chain may span shards: the base shard + top-up shards), metrics_all.csv.gz gets the run rows of the rounds (column 'round'; empty for the first-production rows), MANIFEST.json gets an append-only 'appended' record per round.
  A round already recorded in the manifest with the same file list is skipped (re-run = no-op); a recorded round whose file list changed is an error.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd

from t2_pack_reader import layout
from t2_topup import NUM_STEPS, ROUNDS

PAD = bytes(8)
SOURCE = {"prod": "topup-production", "r100": "topup-rewind100"}


def plan_shards(files, shard_bytes):
    shards, cur, size = [], [], 0
    for f in files:
        cur.append(f)
        size += os.path.getsize(f)
        if size >= shard_bytes:
            shards.append(cur)
            cur, size = [], 0
    if cur:
        shards.append(cur)
    return shards


def stem(f):
    return os.path.basename(f)[:-4]


def fingerprint(files, rescued, tag=None):
    """Identity of a shard plan: its (stem, size, rescue flag) list (+ the round tag of an appended shard); a stale meta from another plan is never reused."""
    return hashlib.sha256(("\n".join(f"{stem(f)}:{os.path.getsize(f)}:{stem(f) in rescued}" for f in files) + (json.dumps(tag, sort_keys=True) if tag else "")).encode()).hexdigest()


def md5_stream(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def meta_ok(meta, final, fp):
    return os.path.isfile(meta) and os.path.isfile(final) and str(np.load(meta)["fp"]) == fp


def save_meta(meta, **kw):
    np.savez(meta + ".tmp.npz", **kw)
    os.replace(meta + ".tmp.npz", meta)


def build_chain_shard(task):
    k, files, out, rescued, tag = task
    final, meta = os.path.join(out, f"chain.{k:03d}.pack"), os.path.join(out, "meta", f"chain.{k:03d}.npz")
    fp = fingerprint(files, rescued, tag)
    if meta_ok(meta, final, fp):
        return k
    stems, offs, lens, md5s, off = [], [], [], [], 0
    with open(final + ".partial", "wb") as fh:
        for f in files:
            b = open(f, "rb").read()
            np.load(f)["n_templates"]   # a half-written npz (production still running) fails here instead of being packed
            fh.write(b)
            stems.append(stem(f)); offs.append(off); lens.append(len(b)); md5s.append(hashlib.md5(b).hexdigest())
            off += len(b)
        fh.flush(); os.fsync(fh.fileno())
    with open(final + ".partial", "rb") as fh:   # re-read and compare every slice with the md5 of the source bytes
        for s, o, n, m in zip(stems, offs, lens, md5s):
            fh.seek(o)
            assert hashlib.md5(fh.read(n)).hexdigest() == m, (s, "pack bytes differ from the source after the write")
    shard_md5 = md5_stream(final + ".partial")
    os.replace(final + ".partial", final)
    save_meta(meta, stems=np.array(stems), offsets=np.array(offs, np.int64), lengths=np.array(lens, np.int64), md5=np.array(md5s), shard_md5=np.array(shard_md5), fp=np.array(fp), round=np.array(tag["round"] if tag else "base"))
    return k


def build_tmpl_shard(task):
    k, files, out, rescued, tag = task
    final, meta = os.path.join(out, f"tmpl.{k:03d}.bin"), os.path.join(out, "meta", f"tmpl.{k:03d}.npz")
    fp = fingerprint(files, rescued, tag)
    if meta_ok(meta, final, fp):
        return k
    cols = {n: [] for n in ("chain", "draw", "off", "nbytes", "L", "A", "L_native", "tm_native", "tm_template", "rewind", "num_steps", "source", "model", "arm", "round")}
    ops_parts = []
    h, off = hashlib.md5(), 0
    with open(final + ".partial", "wb") as fh:
        for f in files:
            cid = stem(f)
            z = np.load(f, allow_pickle=False)
            n, ro, ao = int(z["n_templates"]), z["res_offsets"], z["atom_offsets"]
            aat, ri, oi, am, co = z["aatype"], z["residue_index"], z["orig_idx"], z["atom_mask"], z["coords"]
            ops = json.loads(str(z["ops_json"]))
            rw_all, draw_all, tmn, tmt, model_all, arm_all, Lnat = z["rewind"], z["draw"], z["tm_native"], z["tm_template"], z["model"], z["arm"], int(z["L_native"])   # read each npz member once
            for i in range(n):
                r0, r1, a0, a1 = int(ro[i]), int(ro[i + 1]), int(ao[i]), int(ao[i + 1])
                L, A = r1 - r0, a1 - a0
                lay = layout(L, A)
                rec = bytearray(lay["nbytes"])
                rec[lay["aatype"]:lay["aatype"] + L] = aat[r0:r1].astype(np.int8).tobytes()
                rec[lay["residue_index"]:lay["residue_index"] + 4 * L] = ri[r0:r1].astype(np.int32).tobytes()
                rec[lay["orig_idx"]:lay["orig_idx"] + 4 * L] = oi[r0:r1].astype(np.int32).tobytes()
                rec[lay["atom_mask"]:lay["atom_mask"] + 37 * L] = am[r0:r1].astype(np.uint8).tobytes()
                rec[lay["coords"]:lay["coords"] + 12 * A] = co[a0:a1].astype(np.float32).tobytes()
                fh.write(rec)
                h.update(rec)
                rw = int(rw_all[i])
                if tag:
                    assert rw == tag["rewind"], (cid, rw, tag["round"], "npz rewind differs from the rewind of its round")
                rewind, steps, src = (tag["rewind"], tag["num_steps"], tag["source"]) if tag else (rw, 500 if rw == 250 else 250, "rescue" if cid in rescued else "main")
                for name, v in (("chain", cid), ("draw", int(draw_all[i])), ("off", off), ("nbytes", len(rec)), ("L", L), ("A", A), ("L_native", Lnat), ("tm_native", float(tmn[i])),
                                ("tm_template", float(tmt[i])), ("rewind", rewind), ("num_steps", steps), ("source", src),
                                ("model", str(model_all[i])), ("arm", str(arm_all[i])), ("round", tag["round"] if tag else "base")):
                    cols[name].append(v)
                ops_parts.append(json.dumps(ops[i], separators=(",", ":")).encode())
                off += len(rec)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(final + ".partial", final)
    dt = {"draw": np.int16, "off": np.int64, "nbytes": np.int64, "L": np.int32, "A": np.int32, "L_native": np.int32, "tm_native": np.float32, "tm_template": np.float32, "rewind": np.int16, "num_steps": np.int16}
    save_meta(meta, **{n: np.array(v, dt.get(n)) if n in dt else np.array(v) for n, v in cols.items()}, ops_len=np.array([len(p) for p in ops_parts], np.int32),
              ops_blob=np.frombuffer(b"".join(ops_parts), np.uint8), shard_md5=np.array(h.hexdigest()), fp=np.array(fp))
    return k


def write_chain_idx(out, n):
    """level-1 index of shards 0..n-1 from their meta files (shards of a pack built before 'round' existed count as base); returns the shard md5 list"""
    z = [np.load(os.path.join(out, "meta", f"chain.{k:03d}.npz")) for k in range(n)]
    save_meta(os.path.join(out, "chain.idx.npz"), stems=np.concatenate([x["stems"] for x in z]), shard=np.concatenate([np.full(len(x["stems"]), k, np.int32) for k, x in enumerate(z)]),
              offsets=np.concatenate([x["offsets"] for x in z]), lengths=np.concatenate([x["lengths"] for x in z]), md5=np.concatenate([x["md5"] for x in z]),
              round=np.concatenate([np.full(len(x["stems"]), str(x["round"]) if "round" in x.files else "base") for x in z]))
    return [str(x["shard_md5"]) for x in z]


def write_tmpl_idx(out, n):
    """level-2 index of shards 0..n-1: rows regrouped so that each chain is contiguous (a chain of the base shard and of top-up shards keeps all its rows, first-appearance chain order); returns (shard md5 list, n rows)"""
    z = [np.load(os.path.join(out, "meta", f"tmpl.{k:03d}.npz")) for k in range(n)]
    col = {c: np.concatenate([x[c] for x in z]) for c in z[0].files if c not in ("shard_md5", "fp", "round")}
    col["round"] = np.concatenate([x["round"] if "round" in x.files else np.full(len(x["chain"]), "base") for x in z])
    col["ops_off"] = np.concatenate([[0], np.cumsum(col["ops_len"], dtype=np.int64)[:-1]])
    col["shard"] = np.concatenate([np.full(len(x["chain"]), k, np.int32) for k, x in enumerate(z)])
    codes, uniq = pd.factorize(col["chain"])
    order = np.argsort(codes, kind="stable")
    col = {c: v if c == "ops_blob" else v[order] for c, v in col.items()}
    count = np.bincount(codes)
    save_meta(os.path.join(out, "tmpl.idx.npz"), chain_stems=np.array([str(u) for u in uniq]), chain_first=np.concatenate([[0], np.cumsum(count, dtype=np.int64)[:-1]]), chain_count=count.astype(np.int32), **col)
    return [str(x["shard_md5"]) for x in z], int(len(order))


def round_tag(rd):
    return dict(round=rd, source=SOURCE[ROUNDS[rd]["proto"]], rewind=ROUNDS[rd]["rewind"], num_steps=NUM_STEPS)


def round_files(topup, rd):
    return sorted(f for f in glob.glob(os.path.join(topup, rd, "templates", "*", "*.npz")) if not f.endswith(".tmp.npz"))


def extend_metrics(out, topup, tags):
    """metrics_all.csv.gz + the run rows (passing or not) of the rounds in tags; rows of these rounds already present are replaced, so a half-finished append is repaired by re-running"""
    path = os.path.join(out, "metrics_all.csv.gz")
    df = pd.read_csv(path, float_precision="round_trip")
    if "round" not in df:
        df["round"] = pd.Series(np.nan, index=df.index, dtype=object)
    base = df[~df["round"].isin([t["round"] for t in tags])]
    frames = [base]
    n_new = {}
    for t in tags:
        fs = sorted(glob.glob(os.path.join(topup, t["round"], "templates", "*", "*.metrics.csv")))
        new = [pd.read_csv(f, float_precision="round_trip").assign(source=t["source"], round=t["round"], rewind=t["rewind"], num_steps=t["num_steps"]) for f in fs]
        assert all(set(x.columns) == set(df.columns) for x in new), (t["round"], "a top-up metrics.csv has other columns than metrics_all")
        frames += new
        n_new[t["round"]] = (len(fs), sum(len(x) for x in new))
    merged = pd.concat(frames, ignore_index=True)
    assert merged.iloc[:len(base)].equals(base.reset_index(drop=True)), "the existing metrics rows changed"
    assert list(merged.columns) == list(df.columns), ("metrics_all columns changed", list(merged.columns), list(df.columns))
    tmp = os.path.join(out, "metrics_all.tmp.csv.gz")
    merged.to_csv(tmp, index=False)
    if len(base) == len(df):   # first append of these rounds: the base rows must be byte-identical as text (a legacy file lacks the empty trailing 'round' field)
        old, new_t = gzip.open(path, "rt").read().splitlines(), gzip.open(tmp, "rt").read().splitlines()[:len(df) + 1]
        legacy = "round" not in old[0].split(",")
        assert [l[:-6] if legacy and i == 0 else l[:-1] if legacy else l for i, l in enumerate(new_t)] == old, "the existing metrics rows changed as text"
    os.replace(os.path.join(out, "metrics_all.tmp.csv.gz"), path)
    return n_new


def precheck_base(out, man, n_old):
    """before anything is written: every existing shard has its meta, its md5 equals the manifest and its file has exactly the size its meta implies"""
    assert len(man["chain_shard_md5"]) == len(man["tmpl_shard_md5"]) == n_old, "manifest shard lists do not match n_shards"
    for k in range(n_old):
        c, t = (np.load(os.path.join(out, "meta", f"{n}.{k:03d}.npz")) for n in ("chain", "tmpl"))
        assert str(c["shard_md5"]) == man["chain_shard_md5"][k] and str(t["shard_md5"]) == man["tmpl_shard_md5"][k], (k, "meta md5 differs from the manifest")
        assert os.path.getsize(os.path.join(out, f"chain.{k:03d}.pack")) == int(c["offsets"][-1] + c["lengths"][-1]), (k, "chain shard size differs from its meta")
        assert os.path.getsize(os.path.join(out, f"tmpl.{k:03d}.bin")) == int(t["off"][-1] + t["nbytes"][-1]), (k, "tmpl shard size differs from its meta")


def check_round(topup, rd):
    """a round is appended only when finished: it has files, every metrics.csv with a passing row has its npz and every npz has its metrics.csv (the chain-done marker)"""
    npz = {stem(f) for f in round_files(topup, rd)}
    mets = sorted(glob.glob(os.path.join(topup, rd, "templates", "*", "*.metrics.csv")))
    assert npz and mets, (rd, "no loose files under", topup)
    have = {os.path.basename(f)[:-len(".metrics.csv")] for f in mets}
    planned = {c for f in glob.glob(os.path.join(topup, rd, "ids_shard*.txt")) for c in open(f).read().split()}
    assert planned == have, (rd, "round not complete or not the planned chains: planned", len(planned), "with metrics.csv", len(have), sorted(planned ^ have)[:3])
    passing = {os.path.basename(f)[:-len(".metrics.csv")] for f in mets if pd.read_csv(f, usecols=["pass_all"]).pass_all.any()}
    assert npz <= have and passing <= npz, (rd, "unfinished round: npz without metrics.csv", sorted(npz - have)[:3], "passing chain without npz", sorted(passing - npz)[:3])


def backup_once(out, names):
    for n in names:
        if not os.path.exists(os.path.join(out, n + ".pre-append")):
            shutil.copy2(os.path.join(out, n), os.path.join(out, n + ".pre-append"))


def append(a):
    mpath = os.path.join(a.out, "MANIFEST.json")
    man = json.load(open(mpath))
    assert len(set(a.rounds)) == len(a.rounds), ("duplicate rounds in --rounds", a.rounds)
    done = {e["round"]: e for e in man.get("appended", [])}
    n_old = n_next = man["n_shards"]
    todo = []
    for rd in a.rounds:
        tag, files = round_tag(rd), round_files(a.topup, rd)
        fp = fingerprint(files, set(), tag)
        if rd in done:
            assert done[rd]["fp"] == fp, (rd, "round already appended with another file list; the pack is append-only")
            continue
        check_round(a.topup, rd)
        shards = plan_shards(files, a.shard_gb * 1e9)
        todo.append((rd, tag, files, shards, n_next, fp))
        n_next += len(shards)
    if not todo:
        print("nothing to append: every round is already in the manifest")
        return
    precheck_base(a.out, man, n_old)
    backup_once(a.out, ["chain.idx.npz", "tmpl.idx.npz", "metrics_all.csv.gz", "MANIFEST.json"])
    tasks = [(k0 + j, s, a.out, set(), tag) for _, tag, _, shards, k0, _ in todo for j, s in enumerate(shards)]
    with Pool(a.procs) as pool:
        pool.map(build_chain_shard, tasks)
        pool.map(build_tmpl_shard, tasks)
    n_new = extend_metrics(a.out, a.topup, [t[1] for t in todo])
    chain_md5 = write_chain_idx(a.out, n_next)
    tmpl_md5, n_rows = write_tmpl_idx(a.out, n_next)
    assert chain_md5[:n_old] == man["chain_shard_md5"] and tmpl_md5[:n_old] == man["tmpl_shard_md5"], "existing shards differ from the manifest"
    rnd = np.load(os.path.join(a.out, "tmpl.idx.npz"))["round"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    for rd, tag, files, shards, k0, fp in todo:
        ks = range(k0, k0 + len(shards))
        man.setdefault("appended", []).append(dict(tag, fp=fp, created=datetime.now().astimezone().isoformat(timespec="seconds"), commit=commit, topup_dir=os.path.abspath(a.topup), shards=[k0, k0 + len(shards)],
                                                   n_chains=len(files), n_templates=int((rnd == rd).sum()), metrics_chains=n_new[rd][0], metrics_rows=n_new[rd][1], chain_shard_md5=[chain_md5[k] for k in ks], tmpl_shard_md5=[tmpl_md5[k] for k in ks]))
    man.update(n_base_shards=man.get("n_base_shards", n_old), n_shards=n_next, chain_shard_md5=chain_md5, tmpl_shard_md5=tmpl_md5)
    json.dump(man, open(mpath + ".tmp", "w"), indent=1)
    os.replace(mpath + ".tmp", mpath)
    print(f"appended {[t[0] for t in todo]}: shards {n_old}..{n_next - 1}, {n_rows} template rows in the index, metrics rows {n_new}")


def metrics_all(templates, ids, full_until, out):
    cut = datetime.fromisoformat(full_until).timestamp()
    frames = []
    for cid in ids:
        main = os.path.join(templates, cid[1:3], cid + ".metrics.csv")
        resc = os.path.join(templates, cid[1:3], cid + ".rescue.metrics.csv")
        if os.path.isfile(main):
            m = pd.read_csv(main)
            full = os.path.getmtime(main) < cut
            frames.append(m.assign(source="main", rewind=250 if full else 125, num_steps=500 if full else 250))
        if os.path.isfile(resc):
            frames.append(pd.read_csv(resc).assign(source="rescue", rewind=100, num_steps=250))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(out, "metrics_all.csv.gz"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="both", choices=["1", "2", "both"])
    ap.add_argument("--shard-gb", type=float, default=2.0)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--list", default=None, help="chain list: its sha256 and size go into the manifest")
    ap.add_argument("--full-step-until", default="2026-10-10T19:34:00-04:00")
    ap.add_argument("--append", action="store_true", help="append loose top-up rounds (--topup, --rounds) to the existing pack in --out")
    ap.add_argument("--topup", help="--append: directory holding <round>/templates/")
    ap.add_argument("--rounds", nargs="+", choices=sorted(ROUNDS), help="--append: rounds to append")
    a = ap.parse_args()
    if a.append:
        if not (a.topup and a.rounds):
            ap.error("--append needs --topup and --rounds")
        append(a)
        return
    if not a.templates:
        ap.error("--templates is required without --append")
    os.makedirs(os.path.join(a.out, "meta"), exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(a.templates, "*", "*.npz")) if not f.endswith(".tmp.npz"))
    ids = [stem(f) for f in files]
    rescued = {i for i in ids if os.path.isfile(os.path.join(a.templates, i[1:3], i + ".rescue.metrics.csv"))}
    shards = plan_shards(files, a.shard_gb * 1e9)
    print(f"{len(files)} chain npz -> {len(shards)} shards of ~{a.shard_gb} GB; {len(rescued)} rescue-run chains")
    with Pool(a.procs) as pool:
        if a.level in ("1", "both"):
            pool.map(build_chain_shard, [(k, s, a.out, rescued, None) for k, s in enumerate(shards)])
        if a.level in ("2", "both"):
            pool.map(build_tmpl_shard, [(k, s, a.out, rescued, None) for k, s in enumerate(shards)])
    man = dict(created=datetime.now().astimezone().isoformat(timespec="seconds"), n_chains=len(files), n_shards=len(shards), shard_gb=a.shard_gb, full_step_until=a.full_step_until,
               rescue_chains=len(rescued), templates_dir=os.path.abspath(a.templates), commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip(),
               schedule_rule="main-run chains: metrics.csv mtime < full_step_until -> 500 total steps, rewind 250; later -> 250 total, rewind 125; rescue run -> 250 total, rewind 100", coords="float32 (exact)")
    if a.list:
        man["list_sha256"] = hashlib.sha256(open(a.list, "rb").read()).hexdigest()
        man["list_chains"] = sum(1 for _ in open(a.list))
    if a.level in ("1", "both"):
        man["chain_shard_md5"] = write_chain_idx(a.out, len(shards))
    if a.level in ("2", "both"):
        man["tmpl_shard_md5"], man["n_templates"] = write_tmpl_idx(a.out, len(shards))
    csv_ids = sorted(os.path.basename(f)[:-len(".metrics.csv")] for f in glob.glob(os.path.join(a.templates, "*", "*.metrics.csv")) if not f.endswith(".rescue.metrics.csv"))
    df = metrics_all(a.templates, csv_ids, a.full_step_until, a.out)
    man["metrics_chains"] = len(csv_ids)
    man["metrics_rows"] = int(len(df))
    json.dump(man, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if not k.endswith("md5")}, indent=1))


if __name__ == "__main__":
    main()
