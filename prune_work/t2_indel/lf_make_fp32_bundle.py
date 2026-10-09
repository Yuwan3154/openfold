"""Build an fp32 copy of localfold's int5 AF2 bundle from DeepMind's params_model_1_ptm.npz (same tensor ids/shapes; every int5 tensor replaced by the original float32 array).
Run: python lf_make_fp32_bundle.py SRC_BUNDLE_DIR PARAMS_NPZ OUT_BUNDLE_DIR"""
import json
import os
import shutil
import sys

import numpy as np

src, npz, out = sys.argv[1:4]
z = np.load(npz, allow_pickle=True)
keys = list(z.keys())
assert os.path.realpath(out) != os.path.realpath(src), "OUT must differ from SRC"
if os.path.exists(out):
    shutil.rmtree(out)
shutil.copytree(src, out)
m = json.load(open(os.path.join(src, "manifest.json")))
T = m["tensors"]
SEC = {"evoformerStack": "evoformer/evoformer_iteration/", "extraMsaStack": "evoformer/extra_msa_stack/", "embedding": "evoformer/", "templateEmbedding": "evoformer/template_embedding/",
       "structureModule": "structure_module/", "confidenceHeads": "", "templateSingle": "evoformer/"}


def leaves(d, path=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from leaves(v, path + (k,))
        else:
            yield path, k, v


mapped, bad = {}, []
for sec, marker in SEC.items():
    for path, name, tid in leaves(m[sec]["parameters"]):
        mod = "/".join(path)
        cands = [k for k in keys if k.endswith("/" + mod + "//" + name) and marker in k]
        if sec == "embedding":
            cands = [k for k in cands if "evoformer_iteration" not in k and "extra_msa_stack" not in k and "template_embedding" not in k]
        if sec == "templateEmbedding":
            cands = [k for k in cands if k == "alphafold/alphafold_iteration/evoformer/template_embedding/" + mod + "//" + name]
        if sec == "confidenceHeads":
            head = {"predictedLddt": "predicted_lddt_head", "predictedAlignedError": "predicted_aligned_error_head"}[path[0]]
            cands = [k for k in keys if k == "alphafold/alphafold_iteration/" + head + "/" + "/".join(path[1:]) + "//" + name]
        if len(cands) != 1:
            bad.append((sec, mod, name, tid, cands))
            continue
        a = np.asarray(z[cands[0]], np.float32)
        if list(a.shape) != T[tid]["shape"]:
            bad.append((sec, mod, name, tid, "shape %s vs %s" % (a.shape, T[tid]["shape"])))
            continue
        mapped[tid] = a
print("mapped", len(mapped), "of", len(T), "tensors; unmapped leaves", len(bad))
assert not bad, bad[:15]
# validate against the 58 float32 tensors already in the bundle
chk = 0
for tid, a in mapped.items():
    e = T[tid]
    if e["dtype"] == "float32":
        raw = np.fromfile(os.path.join(src, e["file"]), dtype=np.float32, count=int(np.prod(e["shape"])), offset=e["byteOffset"]).reshape(e["shape"])
        assert np.array_equal(raw, a), tid
        chk += 1
print("float32 tensors verified bit-equal to the npz:", chk)
assert not os.path.exists(os.path.join(src, "weights-09.f32.bin")) and m["bundle"]["shards"] == 9
off = 0
with open(os.path.join(out, "weights-09.f32.bin"), "wb") as f:
    for tid, a in mapped.items():
        if T[tid]["dtype"] == "float32":
            continue
        f.write(a.tobytes())
        T[tid] = {"file": "weights-09.f32.bin", "shape": list(a.shape), "byteOffset": off, "dtype": "float32"}
        off += a.nbytes
n_int5_left = sum(1 for e in T.values() if e["dtype"] == "int5")
print("int5 tensors left:", n_int5_left, "fp32 bytes written:", off)
assert n_int5_left == 0
m["bundle"]["shards"] = m["bundle"]["shards"] + 1
json.dump(m, open(os.path.join(out, "manifest.json"), "w"))
