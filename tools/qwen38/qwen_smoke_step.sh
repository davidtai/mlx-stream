#!/bin/bash
# One guarded window: boot BIN on MODEL (port 8097, the standard serve flags, MTP lane), wait for ready, send one short GREEDY
# chat request and print the completion plus the server's engagement lines (exl3 trunk registry, exl3 engine, ngram table).
# The load test of a new pack composition before any standard cell. Env: BIN (abs path), MODEL (pack dir), PROMPT (optional),
# MAX_TOKENS (default 96), ENV_EXTRA (space-separated assignments for the server). Output: reports/qwen-smoke-<date>-<TAG>/.
set -u
BIN="${BIN:?}"; M="${MODEL:?}"; PROMPT="${PROMPT:-Write a Python function that returns the n-th Fibonacci number, then explain its complexity in two sentences.}"
MAXTOK="${MAX_TOKENS:-96}"; EX="${ENV_EXTRA:-}"
OUT=/Users/davidtai/projects/OpenSourceWTF/reports/qwen-smoke-$(date +%Y%m%d-%H%M)${TAG:+-$TAG}; mkdir -p "$OUT"
PORT=8097; export PATH=/opt/homebrew/bin:$PATH
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/step.log" >&2; }
SPID=""; stop_server() { [ -n "$SPID" ] && kill "$SPID" 2>/dev/null; for i in 1 2 3 4 5 6; do sleep 2; kill -0 "$SPID" 2>/dev/null || break; done; kill -9 "$SPID" 2>/dev/null; SPID=""; }
trap "stop_server; pkill -9 -f 'mlx-serve --serve.*--port $PORT' 2>/dev/null; true" EXIT
log "SMOKE_START bin=$(shasum -a 256 $BIN | cut -c1-12) model=$M env='$EX'"
nice -n 19 python3 -I /Users/davidtai/projects/OpenSourceWTF/reports/find_cached.py --reclaim "$M" > "$OUT/reclaim.txt" 2>&1
env MLX_SERVE_GPU_CEILING_MB=114688 MLX_SERVE_NOCACHE_LOAD=1 MLX_SERVE_NGRAM_WARM=0 $EX nice -n 5 "$BIN" --serve --model "$M" --host 127.0.0.1 --port $PORT --ctx-size 32768 --max-concurrent 1 --mtp-max-depth 3 --mtp-typical 0.2 > "$OUT/server.log" 2>&1 &
SPID=$!
for i in $(seq 1 240); do sleep 5; kill -0 $SPID 2>/dev/null || { log "SERVER_DIED after $((i*5))s"; grep -E "error|Error|MISSING|refus|panic|exl3|ngram" "$OUT/server.log" | tail -20 | cut -c1-220 | tee -a "$OUT/step.log"; exit 3; }; curl -s -m 5 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q '"ready"' && break; done
curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q '"ready"' || { log "NOT_READY"; tail -20 "$OUT/server.log" | cut -c1-220 | tee -a "$OUT/step.log"; exit 3; }
log "READY after $((i*5))s"
grep -E "\[load\] exl3 trunk|\[exl3\]|ngram table|ngram_table|trellis|\[qwen4\]" "$OUT/server.log" | head -12 | cut -c1-200 | tee -a "$OUT/step.log"
python3 -I - "$PORT" "$OUT" "$PROMPT" "$MAXTOK" <<'PY' | tee -a "$OUT/step.log"
import json, sys, time, urllib.request
port, out, prompt, maxtok = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
body = {"model": "default_model", "messages": [{"role": "user", "content": prompt}], "max_tokens": maxtok, "temperature": 0.0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=600)); dt = time.time() - t0
m = r["choices"][0]["message"]; text = (m.get("reasoning_content") or "") + (m.get("content") or "")
json.dump(r, open(f"{out}/response.json", "w"), indent=1)
print(f"SMOKE_RESULT wall={dt:.1f}s usage={r.get('usage')} finish={r['choices'][0].get('finish_reason')}")
print("SMOKE_TEXT_BEGIN"); print(text[:1200]); print("SMOKE_TEXT_END")
PY
grep -E "<- [0-9]+\+[0-9]+ tokens|spec-stats\] mode" "$OUT/server.log" | tail -2 | cut -c1-200 | tee -a "$OUT/step.log"
log "SMOKE_END out=$OUT"
