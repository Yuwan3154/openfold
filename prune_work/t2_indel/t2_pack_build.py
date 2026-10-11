"""Build the packed T2 template product from a loose template tree (design T2_PACKED_TEMPLATES_PLAN.md; user 10-11: both levels, float32, rescue chains and rewind provenance in the same pack).
Input : <templates>/<id[1:3]>/<id>.npz (+ <id>.metrics.csv, + <id>.rescue.metrics.csv for rescued chains; a rescued chain's npz comes from the rescue run).
Output: <out>/chain.NNN.pack + chain.idx.npz (level 1), tmpl.NNN.bin + tmpl.idx.npz (level 2), metrics_all.csv.gz, MANIFEST.json  (layout: t2_pack_reader.py).
Shards = contiguous id ranges of about --shard-gb of loose-file bytes; each shard is built independently (--procs workers), written to .partial, md5-verified by re-reading (level 1) and renamed; a shard whose meta file exists is skipped (resumable).
Provenance of a main-run chain = its schedule: metrics.csv written before --full-step-until = full schedule (500 total steps, rewind 250), later = half-step (250 total, rewind 125); rescued chains (rewind 100, 250 total steps).
Run: python t2_pack_build.py --templates T --out O [--level both] [--shard-gb 2] [--procs 8] [--list template_chains.txt] [--full-step-until 2026-10-10T19:34:00-04:00]
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd

from t2_pack_reader import layout

PAD = bytes(8)


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


def fingerprint(files, rescued):
    """Identity of a shard plan: its (stem, size, rescue flag) list; a stale meta from another plan is never reused."""
    return hashlib.sha256("\n".join(f"{stem(f)}:{os.path.getsize(f)}:{stem(f) in rescued}" for f in files).encode()).hexdigest()


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
    k, files, out, rescued = task
    final, meta = os.path.join(out, f"chain.{k:03d}.pack"), os.path.join(out, "meta", f"chain.{k:03d}.npz")
    fp = fingerprint(files, rescued)
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
    save_meta(meta, stems=np.array(stems), offsets=np.array(offs, np.int64), lengths=np.array(lens, np.int64), md5=np.array(md5s), shard_md5=np.array(shard_md5), fp=np.array(fp))
    return k


def build_tmpl_shard(task):
    k, files, out, rescued = task
    final, meta = os.path.join(out, f"tmpl.{k:03d}.bin"), os.path.join(out, "meta", f"tmpl.{k:03d}.npz")
    fp = fingerprint(files, rescued)
    if meta_ok(meta, final, fp):
        return k
    cols = {n: [] for n in ("chain", "draw", "off", "nbytes", "L", "A", "L_native", "tm_native", "tm_template", "rewind", "num_steps", "source", "model", "arm")}
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
                for name, v in (("chain", cid), ("draw", int(draw_all[i])), ("off", off), ("nbytes", len(rec)), ("L", L), ("A", A), ("L_native", Lnat), ("tm_native", float(tmn[i])),
                                ("tm_template", float(tmt[i])), ("rewind", rw), ("num_steps", 500 if rw == 250 else 250), ("source", "rescue" if cid in rescued else "main"),
                                ("model", str(model_all[i])), ("arm", str(arm_all[i]))):
                    cols[name].append(v)
                ops_parts.append(json.dumps(ops[i], separators=(",", ":")).encode())
                off += len(rec)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(final + ".partial", final)
    dt = {"draw": np.int16, "off": np.int64, "nbytes": np.int64, "L": np.int32, "A": np.int32, "L_native": np.int32, "tm_native": np.float32, "tm_template": np.float32, "rewind": np.int16, "num_steps": np.int16}
    save_meta(meta, **{n: np.array(v, dt.get(n)) if n in dt else np.array(v) for n, v in cols.items()}, ops_len=np.array([len(p) for p in ops_parts], np.int32),
              ops_blob=np.frombuffer(b"".join(ops_parts), np.uint8), shard_md5=np.array(h.hexdigest()), fp=np.array(fp))
    return k


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
    ap.add_argument("--templates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="both", choices=["1", "2", "both"])
    ap.add_argument("--shard-gb", type=float, default=2.0)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--list", default=None, help="chain list: its sha256 and size go into the manifest")
    ap.add_argument("--full-step-until", default="2026-10-10T19:34:00-04:00")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "meta"), exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(a.templates, "*", "*.npz")) if not f.endswith(".tmp.npz"))
    ids = [stem(f) for f in files]
    rescued = {i for i in ids if os.path.isfile(os.path.join(a.templates, i[1:3], i + ".rescue.metrics.csv"))}
    shards = plan_shards(files, a.shard_gb * 1e9)
    print(f"{len(files)} chain npz -> {len(shards)} shards of ~{a.shard_gb} GB; {len(rescued)} rescue-run chains")
    with Pool(a.procs) as pool:
        if a.level in ("1", "both"):
            pool.map(build_chain_shard, [(k, s, a.out, rescued) for k, s in enumerate(shards)])
        if a.level in ("2", "both"):
            pool.map(build_tmpl_shard, [(k, s, a.out, rescued) for k, s in enumerate(shards)])
    man = dict(created=datetime.now().astimezone().isoformat(timespec="seconds"), n_chains=len(files), n_shards=len(shards), shard_gb=a.shard_gb, full_step_until=a.full_step_until,
               rescue_chains=len(rescued), templates_dir=os.path.abspath(a.templates), commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip(),
               schedule_rule="main-run chains: metrics.csv mtime < full_step_until -> 500 total steps, rewind 250; later -> 250 total, rewind 125; rescue run -> 250 total, rewind 100", coords="float32 (exact)")
    if a.list:
        man["list_sha256"] = hashlib.sha256(open(a.list, "rb").read()).hexdigest()
        man["list_chains"] = sum(1 for _ in open(a.list))
    if a.level in ("1", "both"):
        z = [np.load(os.path.join(a.out, "meta", f"chain.{k:03d}.npz")) for k in range(len(shards))]
        np.savez(os.path.join(a.out, "chain.idx.npz"), stems=np.concatenate([x["stems"] for x in z]), shard=np.concatenate([np.full(len(x["stems"]), k, np.int32) for k, x in enumerate(z)]),
                 offsets=np.concatenate([x["offsets"] for x in z]), lengths=np.concatenate([x["lengths"] for x in z]), md5=np.concatenate([x["md5"] for x in z]))
        man["chain_shard_md5"] = [str(x["shard_md5"]) for x in z]
    if a.level in ("2", "both"):
        z = [np.load(os.path.join(a.out, "meta", f"tmpl.{k:03d}.npz")) for k in range(len(shards))]
        col = {n: np.concatenate([x[n] for x in z]) for n in z[0].files if n not in ("shard_md5", "fp")}
        col["ops_off"] = np.concatenate([[0], np.cumsum(col["ops_len"], dtype=np.int64)[:-1]])
        col["shard"] = np.concatenate([np.full(len(x["chain"]), k, np.int32) for k, x in enumerate(z)])
        stems, first, count = [], [], []
        for r, c in enumerate(col["chain"]):
            if not stems or stems[-1] != c:
                stems.append(str(c)); first.append(r); count.append(0)
            count[-1] += 1
        np.savez(os.path.join(a.out, "tmpl.idx.npz"), chain_stems=np.array(stems), chain_first=np.array(first, np.int64), chain_count=np.array(count, np.int32), **col)
        man["n_templates"] = int(len(col["chain"]))
        man["tmpl_shard_md5"] = [str(x["shard_md5"]) for x in z]
    csv_ids = sorted(os.path.basename(f)[:-len(".metrics.csv")] for f in glob.glob(os.path.join(a.templates, "*", "*.metrics.csv")) if not f.endswith(".rescue.metrics.csv"))
    df = metrics_all(a.templates, csv_ids, a.full_step_until, a.out)
    man["metrics_chains"] = len(csv_ids)
    man["metrics_rows"] = int(len(df))
    json.dump(man, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if not k.endswith("md5")}, indent=1))


if __name__ == "__main__":
    main()
