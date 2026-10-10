#!/usr/bin/env python3
"""glm_moe_dsa_mtp_goldens.py: the GLM-5.3 MTP draft lane's fixture (src/glm_moe_dsa_mtp.zig).

The tiny model of `glm_moe_dsa_goldens.py` (the same config, seed and quantization) with an MTP layer at index
`num_hidden_layers`: random BF16 weights, the routed experts dense BF16 (the lane's test bank kind; the served lane binds
the EXL3 bank), saved in the release's names into `<pack>/mtp/mtp-residents.safetensors` (`self_attn.kv_b_proj` as the
release ships it; the routed experts as `mlp.switch_mlp.{gate,up,down}_proj.weight` [E, out, in]).

The MTP math (transformers `MtpLayer` / `MtpModel`, vLLM `DeepSeekMultiTokenPredictorLayer`):
  pair p = (token p+1, the target's final-normed hidden at p):
  h = layer( eh_proj( cat[ enorm(embed(token p+1)), hnorm(hidden p) ] ) ), logits = lm_head( shared_head.norm(h) )
  with the target's embedding and lm_head. A pair's KV and indexer key go to the MTP layer's own cache at position p.
  Depth k > 1 applies the same layer to (the draft k-1, shared_head.norm(h) of depth k-1) at position p + k - 1;
  `index_share_for_mtp_iteration`: such a step runs no indexer and attends the keys the first step's last row
  selected (every committed pair's key while they fit `index_topk`); it appends nothing to the cache.

The reference's attention and MoE are the checkpoint's bundled `glm_moe_dsa.py`; the routed experts take each routed
expert's matrices by id and multiply them in a batch, as the lane's dense bank does (`DenseSwitchGLU`). The verify
forward runs the reference's projections and MoE over all its rows and the attention one row at a time, row i over the
keys up to its own position with its own selection, as a serial step would (`rowwise`): on MLX's CPU its logits equal
the serial decode's bit for bit, so exact acceptance emits the serial decode's tokens.

Cases: two prompts (one the indexer bypasses at first, one that selects) x depth 1..5 x two schedules: `mtp` (the
drafts are the MTP layer's argmax) and `forced` (in round r the first r mod (depth + 1) drafts are the serial decode's
tokens, the rest the MTP layer's argmax: every accepted count, the MTP layer's catch-up over several accepted rows, the
target's rollback of every rejected count). Per round: t1, the forced count, the drafts and every draft step's logits,
the accepted count and the next token; per case the serial greedy decode (the tokens the rounds must emit).

  scripts/glm_moe_dsa_mtp_goldens.py --reference <glm_moe_dsa.py> --converter <convert_glm_bank.py> --out <dir>

Writes <dir>/snapshot, <dir>/pack (the target pack), <dir>/pack/mtp/mtp-residents.safetensors and
<dir>/pack/mtp-goldens.json."""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map_with_path
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import CacheList, KVCache

from glm_moe_dsa_goldens import CONFIG, QUANT, build, load_reference

MTP_CONFIG = {"num_nextn_predict_layers": 1, "index_share_for_mtp_iteration": True}
N_TOKENS = 14
PROMPTS = {"short": [3, 140, 77, 21], "long": [9, 200, 31, 88, 5, 172, 64, 119, 45, 230, 17, 96]}


class DenseSwitchGLU(nn.Module):
    """SwitchGLU over dense weights as the lane's dense bank computes it: each routed expert's matrices taken by id
    and multiplied in a batch (MLX's CPU gather_mm takes float32 only)."""

    def __init__(self, sw):
        super().__init__()
        self.gate = sw.gate_proj.weight
        self.up = sw.up_proj.weight
        self.down = sw.down_proj.weight

    def __call__(self, x, indices):
        k = indices.shape[-1]
        lead = x.shape[:-1]
        x = x.reshape(-1, 1, 1, x.shape[-1])
        flat = indices.reshape(-1)

        def proj(w, v):
            ws = mx.take(w, flat, axis=0)
            ws = ws.reshape(-1, k, ws.shape[1], ws.shape[2])
            return v @ ws.swapaxes(-1, -2)

        h = nn.silu(proj(self.gate, x)) * proj(self.up, x)
        y = proj(self.down, h)
        return y.reshape(*lead, k, y.shape[-1])


class Mtp(nn.Module):
    def __init__(self, ref, args, layer_idx):
        super().__init__()
        h = args.hidden_size
        self.enorm = nn.RMSNorm(h, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(h, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * h, h, bias=False)
        self.block = ref.DeepseekV32DecoderLayer(args, layer_idx)
        self.shared_head_norm = nn.RMSNorm(h, eps=args.rms_norm_eps)


def build_mtp(ref, seed):
    cfg = dict(CONFIG, num_hidden_layers=CONFIG["num_hidden_layers"] + 1,
               indexer_types=CONFIG["indexer_types"] + ["full"], mlp_layer_types=CONFIG["mlp_layer_types"] + ["sparse"])
    args = ref.ModelArgs.from_dict(cfg)
    mx.random.seed(seed)
    m = Mtp(ref, args, CONFIG["num_hidden_layers"])

    def init(path, x):
        if path.endswith("e_score_correction_bias"):
            return mx.random.uniform(-0.05, 0.05, x.shape)
        if path.endswith("mlp.gate.weight"):
            return mx.random.normal(x.shape) * 0.2
        if "norm" in path:
            return 1 + 0.1 * mx.random.normal(x.shape) if path.endswith("weight") else 0.1 * mx.random.normal(x.shape)
        return x

    m.update(tree_map_with_path(init, m.parameters()))
    m.update(tree_map_with_path(lambda p, x: x if p.endswith("e_score_correction_bias") else x.astype(mx.bfloat16),
                                m.parameters()))
    # The release ships kv_b_proj; the reference's sanitize splits it (the unquantized case). Build it from the
    # projections and split it back the reference's way, as views of kv_b_proj (the lane multiplies through views of
    # the published tensor; the reference's sanitize makes them contiguous: the same values).
    at = m.block.self_attn
    nope, vd = CONFIG["qk_nope_head_dim"], CONFIG["v_head_dim"]
    kv_b = mx.concatenate([at.embed_q.weight.swapaxes(-1, -2), at.unembed_out.weight], axis=1)
    kv_b = kv_b.reshape(-1, CONFIG["kv_lora_rank"])
    v = kv_b.reshape(CONFIG["num_attention_heads"], nope + vd, -1)
    at.embed_q.weight = v[:, :nope, :].swapaxes(-1, -2)
    at.unembed_out.weight = v[:, nope:, :]
    m.block.mlp.switch_mlp = DenseSwitchGLU(m.block.mlp.switch_mlp)
    mx.eval(m.parameters())
    return m, kv_b


def mtp_tensors(m, kv_b):
    """The MTP layer's tensors in the release's names (BF16; the router's bias as stored)."""
    p = "model.layers.%d." % CONFIG["num_hidden_layers"]
    b = m.block
    at = b.self_attn
    ix = at.indexer
    t = {"eh_proj.weight": m.eh_proj.weight, "enorm.weight": m.enorm.weight, "hnorm.weight": m.hnorm.weight,
         "shared_head.norm.weight": m.shared_head_norm.weight,
         "input_layernorm.weight": b.input_layernorm.weight,
         "post_attention_layernorm.weight": b.post_attention_layernorm.weight,
         "self_attn.q_a_proj.weight": at.q_a_proj.weight, "self_attn.q_a_layernorm.weight": at.q_a_layernorm.weight,
         "self_attn.q_b_proj.weight": at.q_b_proj.weight,
         "self_attn.kv_a_proj_with_mqa.weight": at.kv_a_proj_with_mqa.weight,
         "self_attn.kv_a_layernorm.weight": at.kv_a_layernorm.weight, "self_attn.kv_b_proj.weight": kv_b,
         "self_attn.o_proj.weight": at.o_proj.weight,
         "self_attn.indexer.wq_b.weight": ix.wq_b.weight, "self_attn.indexer.wk.weight": ix.wk.weight,
         "self_attn.indexer.k_norm.weight": ix.k_norm.weight, "self_attn.indexer.k_norm.bias": ix.k_norm.bias,
         "self_attn.indexer.weights_proj.weight": ix.weights_proj.weight,
         "mlp.gate.weight": b.mlp.gate.weight, "mlp.gate.e_score_correction_bias": b.mlp.gate.e_score_correction_bias,
         "mlp.shared_experts.gate_proj.weight": b.mlp.shared_experts.gate_proj.weight,
         "mlp.shared_experts.up_proj.weight": b.mlp.shared_experts.up_proj.weight,
         "mlp.shared_experts.down_proj.weight": b.mlp.shared_experts.down_proj.weight,
         "mlp.switch_mlp.gate_proj.weight": b.mlp.switch_mlp.gate, "mlp.switch_mlp.up_proj.weight": b.mlp.switch_mlp.up,
         "mlp.switch_mlp.down_proj.weight": b.mlp.switch_mlp.down}
    return {p + k: v for k, v in t.items()}


def mtp_input(model, m, ids, hidden):
    e = model.model.embed_tokens(mx.array(ids)[None])
    return m.eh_proj(mx.concatenate([m.enorm(e), m.hnorm(hidden)], axis=-1))


def mtp_append(m, x, cache):
    """The pairs' keys appended to the MTP cache (the latent, the rope key, the indexer key): all a pair leaves
    behind; only the last pending pair's output is ever read."""
    at = m.block.self_attn
    ix = at.indexer
    a = m.block.input_layernorm(x)
    B, L, _ = a.shape
    start = cache[0].offset
    ckv, k_pe = mx.split(at.kv_a_proj_with_mqa(a), [at.kv_lora_rank], axis=-1)
    k_pe = at.rope(k_pe.reshape(B, L, 1, at.qk_rope_head_dim).transpose(0, 2, 1, 3), start)
    cache[0].update_and_fetch(mx.expand_dims(at.kv_a_layernorm(ckv), axis=1), k_pe)
    k = ix.rope(ix.k_norm(ix.wk(a)).reshape(B, 1, L, ix.head_dim), offset=start)
    cache[1].update_and_fetch(k, mx.zeros([B, 1, L, 0]))


def select_last(ix, a, qr, cache, pos):
    """The indexer's selection for one row at `pos` over every index key (None while they fit index_topk)."""
    keys = cache[1].keys[..., :cache[1].offset, :]
    if keys.shape[2] <= ix.index_topk:
        return None
    q = ix.rope(ix.wq_b(qr).reshape(1, 1, ix.n_heads, ix.head_dim).swapaxes(1, 2), offset=pos)
    sc = q.astype(mx.float32) @ keys.astype(mx.float32).swapaxes(-1, -2)
    w = (ix.weights_proj(a).astype(mx.float32) * (ix.n_heads ** -0.5 * ix.softmax_scale)).swapaxes(-1, -2)[..., None]
    sc = (mx.maximum(sc, 0) * w).sum(axis=1, keepdims=True)
    return mx.argpartition(sc, kth=-ix.index_topk, axis=-1)[..., -ix.index_topk:]


def attn_shared(at, x, cache, topk, offset):
    """The MTP attention of one row at `offset` over the cache's keys as they stand (no append), with the selection
    `topk` (None: every key): the reference's L == 1 branch."""
    B, L, _ = x.shape
    qr = at.q_a_layernorm(at.q_a_proj(x))
    q = at.q_b_proj(qr).reshape(B, L, at.num_heads, at.q_head_dim).transpose(0, 2, 1, 3)
    q_nope, q_pe = mx.split(q, [at.qk_nope_head_dim], axis=-1)
    q_pe = at.rope(q_pe, offset)
    kv = cache[0]
    kv_latent, k_pe = kv.keys[..., :kv.offset, :], kv.values[..., :kv.offset, :]
    if topk is not None:
        idx = topk[:, :, 0, :, None]
        kv_latent = mx.take_along_axis(kv_latent, mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)), axis=2)
        k_pe = mx.take_along_axis(k_pe, mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)), axis=2)
    pe_scores = (q_pe * at.scale) @ k_pe.swapaxes(-1, -2)
    q_nope = at.embed_q(q_nope)
    out = mx.fast.scaled_dot_product_attention(q_nope, kv_latent, kv_latent, scale=at.scale, mask=pe_scores)
    out = at.unembed_out(out)
    return at.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


def mtp_row(m, x, cache, topk, pos):
    b = m.block
    h = x + attn_shared(b.self_attn, b.input_layernorm(x), cache, topk, pos)
    return h + b.mlp(b.post_attention_layernorm(h))


def mtp_first(model, m, ids, hidden, cache):
    """A round's first step: every pending pair appended, then the last pair's output and its own selection."""
    x = mtp_input(model, m, ids, hidden)
    mtp_append(m, x, cache)
    last = x[:, -1:]
    pos = cache[0].offset - 1
    b = m.block
    at = b.self_attn
    a = b.input_layernorm(last)
    topk = select_last(at.indexer, a, at.q_a_layernorm(at.q_a_proj(a)), cache, pos)
    return mtp_row(m, last, cache, topk, pos), topk


def mtp_reference(model, m, ids, hidden, cache):
    """The same pairs through the reference's decoder layer itself (its own cache): the mirror's check."""
    x = mtp_input(model, m, ids, hidden)
    mask = create_attention_mask(x, cache[0], return_array=True)
    out, _ = m.block(x, mask, cache)
    return out[:, -1:]


def mtp_step(model, m, tok, prev, cache, topk, pos):
    """A draft step past the first (index_share_for_mtp_iteration): its output [1, 1, H] at `pos`, over the cache as
    the first step left it, with the first step's selection; nothing appended."""
    return mtp_row(m, mtp_input(model, m, [tok], prev), cache, topk, pos)


def rowwise(ref):
    """The reference's attention with the verify's per-row form, on while `rowwise.on`: the KV of every row appended,
    then row i through the L == 1 branch over the keys [0, start + i], its selection from its own indexer row (a full
    layer) or the full layer's (a shared one)."""
    att = ref.DeepseekV32Attention
    orig = att.__call__

    def select(ix, x, qr, kfull, start, L):
        b = x.shape[0]
        q = ix.wq_b(qr).reshape(b, L, ix.n_heads, ix.head_dim).swapaxes(1, 2)
        q = ix.rope(q, offset=start)
        w = ix.weights_proj(x).astype(mx.float32) * (ix.n_heads ** -0.5 * ix.softmax_scale)
        w = w.swapaxes(-1, -2)[..., None]
        out = []
        for i in range(L):
            nb = start + i + 1
            if nb <= ix.index_topk:
                out.append(None)
                continue
            sc = q[:, :, i:i + 1].astype(mx.float32) @ kfull[..., :nb, :].astype(mx.float32).swapaxes(-1, -2)
            sc = (mx.maximum(sc, 0) * w[:, :, i:i + 1]).sum(axis=1, keepdims=True)
            out.append(mx.argpartition(sc, kth=-ix.index_topk, axis=-1)[..., -ix.index_topk:])
        return out

    def call(self, x, mask=None, cache=None, prev_topk_indices=None):
        B, L, _ = x.shape
        if not call.on or L == 1:
            return orig(self, x, mask, cache, prev_topk_indices)
        start = cache[0].offset
        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr).reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        ckv, k_pe = mx.split(self.kv_a_proj_with_mqa(x), [self.kv_lora_rank], axis=-1)
        k_pe = self.rope(k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3), start)
        q_pe = self.rope(q_pe, start)
        lat, pe = cache[0].update_and_fetch(mx.expand_dims(self.kv_a_layernorm(ckv), axis=1), k_pe)
        if self.indexer is not None:
            ix = self.indexer
            k = ix.rope(ix.k_norm(ix.wk(x)).reshape(B, 1, L, ix.head_dim), offset=start)
            kfull, _ = cache[1].update_and_fetch(k, mx.zeros([B, 1, L, 0]))
            topks = select(ix, x, qr, kfull, start, L)
        else:
            topks = prev_topk_indices
        outs = []
        for i in range(L):
            kl, kp = lat[..., :start + i + 1, :], pe[..., :start + i + 1, :]
            t = topks[i]
            if t is not None:
                idx = t[:, :, 0, :, None]
                kl = mx.take_along_axis(kl, mx.broadcast_to(idx, idx.shape[:-1] + (kl.shape[-1],)), axis=2)
                kp = mx.take_along_axis(kp, mx.broadcast_to(idx, idx.shape[:-1] + (kp.shape[-1],)), axis=2)
            sc = (q_pe[:, :, i:i + 1] * self.scale) @ kp.swapaxes(-1, -2)
            o = mx.fast.scaled_dot_product_attention(self.embed_q(q_nope[:, :, i:i + 1]), kl, kl, scale=self.scale,
                                                     mask=sc)
            outs.append(self.unembed_out(o))
        out = mx.concatenate(outs, axis=2).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out), topks

    call.on = False
    att.__call__ = call
    return call


def trim(cache, n):
    for c in cache:
        for sub in c.caches:
            sub.trim(n)


def argmax(row):
    return int(mx.argmax(row.astype(mx.float32), axis=-1).item())


def serial(model, prompt, n):
    cache = model.make_cache()
    toks = [argmax(model(mx.array(prompt)[None], cache)[0, -1])]
    while len(toks) < n:
        toks.append(argmax(model(mx.array([toks[-1]])[None], cache)[0, -1]))
    return toks


def rounds(model, m, rw, prompt, depth, n, oracle, forced):
    """A greedy decode by MTP rounds with exact acceptance, as the lane runs it; with `forced`, round r's first
    r mod (depth + 1) drafts are the oracle's (the serial decode)."""
    cache = model.make_cache()
    mcache = CacheList(KVCache(), KVCache())
    hid = model.model(mx.array(prompt)[None], cache)
    t1 = argmax(model.lm_head(hid[:, -1]))
    rcache = CacheList(KVCache(), KVCache())
    if len(prompt) > 1:
        mtp_append(m, mtp_input(model, m, prompt[1:], hid[:, :-1]), mcache)
        mtp_reference(model, m, prompt[1:], hid[:, :-1], rcache)
    pend_h, pend_t = hid[:, -1:], []
    out, emitted = [], []
    while len(emitted) < n:
        f = len(out) % (depth + 1) if forced else 0
        y, topk = mtp_first(model, m, pend_t + [t1], pend_h, mcache)
        yr = mtp_reference(model, m, pend_t + [t1], pend_h, rcache)
        rounds.check = max(rounds.check, mx.max(mx.abs(model.lm_head(m.shared_head_norm(y)).astype(mx.float32)
                                                       - model.lm_head(m.shared_head_norm(yr)).astype(mx.float32))).item())
        last = mcache[0].offset - 1
        drafts, dlog = [], []
        prev = m.shared_head_norm(y[:, -1:])
        for k in range(depth):
            if k > 0:
                prev = m.shared_head_norm(mtp_step(model, m, drafts[-1], prev, mcache, topk, last + k))
            lg = model.lm_head(prev)[0, -1]
            dlog.append(lg.astype(mx.float32).tolist())
            drafts.append(oracle[len(emitted) + 1 + k] if k < f else argmax(lg))
        rw.on = True
        try:
            hv = model.model(mx.array([t1] + drafts)[None], cache)
        finally:
            rw.on = False
        tt = [argmax(r) for r in model.lm_head(hv)[0]]
        a = 0
        while a < depth and drafts[a] == tt[a]:
            a += 1
        trim(cache, depth - a)
        out.append({"t1": t1, "forced": f, "drafts": drafts, "draft_logits": dlog, "accepted": a, "next": tt[a]})
        emitted += [t1] + drafts[:a]
        pend_h, pend_t, t1 = hv[:, :a + 1], drafts[:a], tt[a]
    return out, emitted


rounds.check = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--converter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=53)
    a = ap.parse_args()
    mx.set_default_device(mx.cpu)
    ref = load_reference(a.reference)
    rw = rowwise(ref)
    model = build(ref, a.seed)
    m, kv_b = build_mtp(ref, a.seed + 1)
    snap, pack = os.path.join(a.out, "snapshot"), os.path.join(a.out, "pack")
    os.makedirs(snap, exist_ok=True)
    mx.save_safetensors(os.path.join(snap, "model.safetensors"), dict(tree_flatten(model.parameters())),
                        metadata={"format": "mlx"})
    with open(os.path.join(snap, "config.json"), "w") as f:
        json.dump({**CONFIG, **MTP_CONFIG, "quantization": QUANT}, f, indent=1)
    subprocess.run([sys.executable, a.converter, "--src", snap, "--dst", pack, "--verify", "all", "--source-repo",
                    "synthetic"], check=True)
    os.makedirs(os.path.join(pack, "mtp"), exist_ok=True)
    mx.save_safetensors(os.path.join(pack, "mtp", "mtp-residents.safetensors"), mtp_tensors(m, kv_b),
                        metadata={"format": "mlx"})
    cases = []
    bad = 0
    for name, prompt in PROMPTS.items():
        want = serial(model, prompt, N_TOKENS + 6)
        for forced in [False, True]:
            for depth in range(1, 6):
                rs, emitted = rounds(model, m, rw, prompt, depth, N_TOKENS, want, forced)
                same = emitted[:N_TOKENS] == want[:N_TOKENS]
                bad += not same
                print("%s %s depth %d: %d rounds, %d drafts accepted, the rounds' tokens %s the serial decode's" % (
                    name, "forced" if forced else "mtp", depth, len(rs), sum(r["accepted"] for r in rs),
                    "equal" if same else "DIFFER from"))
                cases.append({"name": name, "schedule": "forced" if forced else "mtp", "prompt": prompt,
                              "depth": depth, "serial": want[:N_TOKENS], "rounds": rs})
    print("the first step's logits against the reference's decoder layer over the same pairs: max |delta| %.5f"
          % rounds.check)
    if bad:
        sys.exit("%d cases' rounds differ from the serial decode" % bad)
    with open(os.path.join(pack, "mtp-goldens.json"), "w") as f:
        json.dump({"tokens": N_TOKENS, "cases": cases}, f)
    print("wrote %s" % os.path.join(pack, "mtp-goldens.json"))


if __name__ == "__main__":
    main()
