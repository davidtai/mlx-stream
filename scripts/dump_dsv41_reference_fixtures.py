#!/usr/bin/env python3
"""Reproduce consumer-only fixtures from pinned official source, never model packs.

Engram executes the official constructor and forward on CPU. Stored-weight
values use Torch casts and the official FP4 level table, not plugin output.
See README's independent-reference section for the pinned environment.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer

REVISION = "2cba9e42aa026125f3ed06c6d98c1db82f7ca027"
HASHES = {
    "inference/engram.py": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    "inference/kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
    "config.json": "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
    "tokenizer.json": "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
}


class TokenizerAdapter:
    def __init__(self, path):
        self.backend_tokenizer = Tokenizer.from_file(str(path))

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


def engram_fixture(reference):
    spec = importlib.util.spec_from_file_location("official_engram", reference / "inference/engram.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = json.loads((reference / "config.json").read_text())["text_config"]
    args = SimpleNamespace(**config, engram_pad_id=config["engram_pad_token_id"], max_batch_size=1, max_seq_len=1024)
    tokenizer = TokenizerAdapter(reference / "tokenizer.json")
    state = module.NgramHashState(args, module.EngramLayout.from_args(args), tokenizer)
    inputs = [
        ([0, 42, 1, 2, 129264, 123, 500, 129279], [True, True, True, True, False, True, True, True]),
        ([0, 42, 1, 123, 500, 1, 1, 2, 129279], [True] * 9),
        (list(range(256)) + [129264, 129279, 1, 0], [True] * 260),
        ([42] * 17, [False, False, True, True, True, True, False, True, False, False, True, True, True, True, True, True, False]),
        ([], []),
    ]
    cases = []
    for ids, mask in inputs:
        rows = state(torch.tensor([ids], dtype=torch.int64), 0, torch.tensor([mask], dtype=torch.bool)).tolist()[0]
        cases.append(dict(input_ids=ids, token_mask=mask, expected_rows=rows))
    used = sorted({args.engram_pad_id, 0, 1, 42} | {i for ids, _ in inputs for i in ids})
    return dict(
        provenance=dict(revision=REVISION, engram_sha256=HASHES["inference/engram.py"],
                        config_sha256=HASHES["config.json"], tokenizer_sha256=HASHES["tokenizer.json"],
                        evidence="Official NgramHashState.forward on CPU; sparse token-map entries from the official tokenizer normalization; no full-model parity"),
        pad_id=args.engram_pad_id, compressed_vocab_size=args.engram_compressed_vocab_size,
        vocab_size=len(tokenizer), token_map=[dict(id=i, compressed=int(state.token_map[i])) for i in used],
        multipliers=state.multipliers.tolist(), primes=state.primes.reshape(2, 24).tolist(),
        offsets=state.offsets.tolist(), cases=cases)


def bf16_bits(x):
    return x.to(torch.bfloat16).contiguous().view(torch.uint16).flatten().tolist()


def stored_case(fmt, n, k):
    t = torch.arange(n * k).reshape(n, k)
    if fmt == "fp8":
        codes = ((t * 13 + 1) % 127).to(torch.uint8)
        codes |= ((t // 3) % 2).to(torch.uint8) * 128
        weights = codes.view(torch.float8_e4m3fn).float()
        scale_codes = (torch.arange(((n + 31) // 32) * (k // 32)).reshape((n + 31) // 32, k // 32) % 5 + 121).to(torch.uint8)
        scales = scale_codes.view(torch.float8_e8m0fnu).float()
        expanded = scales.repeat_interleave(32, 0)[:n].repeat_interleave(32, 1)
    else:
        levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
        nibbles = ((t * 7 + 1) % 16).to(torch.uint8)
        weights = levels[(nibbles & 7).long()] * torch.where((nibbles & 8) != 0, -1.0, 1.0)
        codes = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
        scale_codes = (torch.arange(n * (k // 32)).reshape(n, k // 32) % 7 + 121).to(torch.uint8)
        expanded = scale_codes.view(torch.float8_e8m0fnu).float().repeat_interleave(32, 1)
    return dict(format=fmt, n=n, k=k, weight=codes.flatten().tolist(),
                scales=scale_codes.flatten().tolist(), dequant=bf16_bits(weights * expanded))


def engram_rows_case():
    n, k = 2 * 2 * 24, 256
    t = torch.arange(n * k).reshape(n, k)
    codes = ((t * 13 + 1) % 127).to(torch.uint8)
    codes |= ((t // 3) % 2).to(torch.uint8) * 128
    weights = codes.view(torch.float8_e4m3fn).float()
    scale_codes = (torch.arange(n * (k // 32)).reshape(n, k // 32) % 7 + 121).to(torch.uint8)
    scales = scale_codes.view(torch.float8_e8m0fnu).float().repeat_interleave(32, 1)
    return dict(format="fp8", n=n, k=k, weight=codes.flatten().tolist(),
                scales=scale_codes.flatten().tolist(), dequant=bf16_bits(weights * scales),
                engram_shape=[2, 2, 24])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("src/fixtures"))
    args = parser.parse_args()
    for name, expected in HASHES.items():
        if hashlib.sha256((args.reference / name).read_bytes()).hexdigest() != expected:
            parser.error(f"{name}: does not match pinned official source")
    if torch.__version__ != "2.14.1":
        parser.error("reproduction requires torch==2.14.1")
    torch.set_num_threads(1)
    quant = dict(
        provenance=dict(revision=REVISION, kernel_sha256=HASHES["inference/kernel.py"],
                        torch_version=torch.__version__,
                        evidence="CPU Torch E4M3/E8M0/BF16 casts and FP4 level transcription; stored finite weights only, no activation quantization or GEMM parity"),
        cases=[stored_case(fmt, n, k) for fmt in ("fp8", "fp4") for n, k in ((7, 64), (35, 96))] + [engram_rows_case()])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, fixture in (("dsv41_official_engram.json", engram_fixture(args.reference)), ("dsv41_official_dequant.json", quant)):
        (args.output / name).write_text(json.dumps(fixture, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
