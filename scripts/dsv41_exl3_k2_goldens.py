"""K=2 golden planes for src/kernels/exl3/manifest.json, computed as the extractor computed the K=3 ones (exl3_ref);
proves itself by reproducing the K=3 planes' four digests first."""
import hashlib, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import exl3_ref as R
man = json.load(open(sys.argv[1]))
def splitmix_plane(seed, n_shorts):
    out = np.empty(n_shorts, np.uint16); s = seed & 0xFFFFFFFFFFFFFFFF; i = 0
    M = (1 << 64) - 1
    while i < n_shorts:
        s = (s + 0x9E3779B97F4A7C15) & M; z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
        z = z ^ (z >> 31)
        for j in range(4):
            if i + j < n_shorts: out[i + j] = (z >> (16 * j)) & 0xFFFF
        i += 4
    return out.view(np.int16)
def plane(seed, in_dim, out_dim, K):
    ni, nj = in_dim // 16, out_dim // 16
    code = splitmix_plane(seed, ni * nj * 16 * K)
    tiles = code.reshape(ni * nj, 16 * K)
    states = R.dq_states(tiles, K)                        # [T, 256] by code position
    perm = R.tensor_core_perm()
    w = np.empty((in_dim, out_dim), np.uint16); st = np.empty((in_dim, out_dim), np.uint16)
    vals = R.decode_mul1(np.arange(65536, dtype=np.uint32)).view(np.uint16)   # the mul1 table (f16 bits)
    pos_r, pos_c = perm // 16, perm % 16
    for ti in range(ni):
        for tj in range(nj):
            s = states[ti * nj + tj]
            w[ti*16 + pos_r, tj*16 + pos_c] = vals[s]; st[ti*16 + pos_r, tj*16 + pos_c] = s
    seen = np.zeros(65536, bool); rows = 0
    while seen.sum() < 65536:
        seen[st[rows]] = True; rows += 1
    rows = -(-rows // 64) * 64          # the extractor's row count: the covering prefix rounded up to 64 rows
    h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    return {"K": K, "code_sha256": h(code), "codebook": "mul1", "onehot_rows": rows, "onehot_rows_sha256": h(w[:rows]),
            "prng": man["golden"]["planes"]["gate_up"]["prng"], "projection": [in_dim, out_dim], "seed": seed,
            "states_covered_by_onehot_rows": 65536, "w_hat_sha256": h(w)}
tbl = R.decode_mul1(np.arange(65536, dtype=np.uint32)).view(np.uint16)
assert hashlib.sha256(tbl.tobytes()).hexdigest() == man["golden"]["mul1_table"]["sha256"], "mul1 table digest"
for name in ("gate_up", "down"):
    g = man["golden"]["planes"][name]
    got = plane(g["seed"], g["projection"][0], g["projection"][1], 3)
    assert got == g, (name, got, g)
print("K=3 planes reproduced (code, w_hat, onehot rows + sha): OK")
out = {"gate_up_k2": plane(1592590339, 5120, 2304, 2), "down_k2": plane(1592590340, 2304, 5120, 2)}
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[2], "w"), indent=1)
