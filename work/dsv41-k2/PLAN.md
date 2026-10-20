# K=2 EXL3 experts for DeepSeek-V4.1 (layers 18-22 of the 1017 EAQ 2.9 bpw bank)

Stage 1: scripts/dsv41_exl3_k2_gen.py writes the K=2 twins of the live K=3 texts (8 decode GEMVs, 3 m128 prefill GEMMs,
header_dig_mul1h_k2, 2 decmat golden texts) by exact-count substitution + derived arithmetic, and proves itself by
regenerating the K=3 header tail and the K=3 lane plan from the same formulas.
Stage 2: registry (kernels.zig bank_ks/Kernel/Header), manifest.json K2 entries + golden planes + pin, bank.zig dsv41.k,
quant.zig entry sets keyed by code tile width, selfcheck tables.
Stage 3: bill (alt transient scratch), tests (planes, peek fixture, trace), hermetic + GPU + real bank on ai-server.
Stage 4: serve the 1017 pack, rebenchmark (ladder, headline, HumanEval), update the HF card.
