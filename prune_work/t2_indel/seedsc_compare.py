"""Check the seed_self_cond patch on ONE item: (a) flag OFF output must equal the reference sweep produced by the stock checkout (same seed, model,
rewinds): bit-identical coordinates; (b) flag ON must differ, with finite coordinates. Prints per-rung max/mean absolute differences (A).
Run: python seedsc_compare.py --ref REF.npz --off OFF.npz --on ON.npz
"""
import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--off", required=True)
    p.add_argument("--on", required=True)
    a = p.parse_args()
    ref, off, on = (np.load(f) for f in (a.ref, a.off, a.on))
    assert ref["coords"].shape == off["coords"].shape == on["coords"].shape, (ref["coords"].shape, off["coords"].shape, on["coords"].shape)
    assert np.isfinite(on["coords"]).all() and np.isfinite(off["coords"]).all()
    d_off = np.abs(ref["coords"] - off["coords"])
    d_on = np.abs(off["coords"] - on["coords"])
    print(f"flag OFF vs stock reference: max abs diff {d_off.max():.3e} A, identical={bool(np.array_equal(ref['coords'], off['coords']))}")
    print(f"flag ON  vs flag OFF: max {d_on.max():.3f} A, mean {d_on.mean():.4f} A over {d_on.size} values; flag stored: {bool(on['seed_self_cond'])}")
    assert d_off.max() == 0.0, "flag OFF is not bit-identical to the stock checkout"
    assert d_on.max() > 0.0, "flag ON had no effect"
    print("PASS")


if __name__ == "__main__":
    main()
