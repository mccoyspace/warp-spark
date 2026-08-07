#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
# pipeline.sh — download -> convert -> verify -> first K3 run, unattended.
#
# Every stage is resumable and refuses to start on a failed predecessor, so
# this can be killed and restarted at any point. It writes a running report
# to $RUN/pipeline.log and a final summary to $RUN/REPORT.md, where
# $RUN defaults to $OUT.runs — beside the container, not inside it, so
# the container holds only what the engine loads.
#
#   tools/pipeline.sh
#   SRC=... OUT=... JOBS=3 tools/pipeline.sh
#   RECLAIM=on tools/pipeline.sh          # delete staging as it is consumed
#
# Stages
#   1 download   loops tools/fetch_weights.sh until all shards verify
#   2 probe      tools/convert.py on the first MoE layer, plus the trunk
#   3 verify     that layer round-trips against the source weights
#   4 convert    the rest, --skip-trunk (stage 2 built it), resumable
#   5 run        the C engine generates from a prompt
#   6 oracle     PyTorch reference on the same prompt, logits diffed
#
# Stage 6 is the one that matters: it is the first end-to-end check that
# K3's latent MoE, Attention Residuals, SiTU and full-rank gate are wired
# up correctly, not merely implemented correctly in isolation.
#
# Why the round-trip sits at 3 rather than after the whole conversion: it
# is the only check that reads the source weights, and RECLAIM=on deletes
# each shard as convert.py finishes with it (peak staging becomes the
# container plus the shards still owed, instead of 1.42 TB alongside a
# 982 GiB container). A verification that runs after that has nothing left
# to compare against. So the recipe is proven on one layer while the
# checkpoint is still whole, and only then is the destructive path allowed.
# It is also the better order regardless of RECLAIM: a quantizer bug now
# costs one layer of conversion instead of a night of it.
#
# RECLAIM defaults to off. It is not reversible — a reclaimed shard has to
# be downloaded again — so it stays something the caller asks for.

set -uo pipefail
cd "$(dirname "$0")/.."

SRC="${SRC:-/Volumes/WasteDisk/k3}"
OUT="${OUT:-$HOME/models/k3.waste}"
JOBS="${JOBS:-3}"
RECLAIM="${RECLAIM:-off}"            # off | dry | on — see the note above
MIN_FREE_GB="${MIN_FREE_GB:-1100}"   # K3's container; a smaller model wants less
PROMPT="${PROMPT:-The capital of France is}"
NTOK="${NTOK:-24}"
RUN="${RUN_DIR:-$OUT.runs}"          # reports and logs, never inside $OUT
LOG="$RUN/pipeline.log"
UV="uv run --quiet --with torch --with fla-core --with einops --no-project python"

mkdir -p "$OUT" "$RUN"
say() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }
die() { say "FAILED at $*"; echo "$*" > "$RUN/.failed"; exit 1; }

rm -f "$RUN/.failed"
say "=== pipeline start ==="
say "src $SRC -> out $OUT, $JOBS conversion processes"

# ---- 1. download ---------------------------------------------------------
NEED=$(python3 -c "
import json;print(len(set(json.load(open('$SRC/model.safetensors.index.json'))['weight_map'].values())))" 2>/dev/null || echo 0)
[ "$NEED" -gt 0 ] || die "download (no index at $SRC)"

have() { wc -l < "$SRC/.download-state" 2>/dev/null | tr -d ' '; }
say "stage 1: download — $(have)/$NEED shards complete"
tries=0
while [ "$(have)" -lt "$NEED" ]; do
    tries=$((tries + 1))
    [ "$tries" -gt 200 ] && die "download (gave up after $tries passes)"
    ./tools/fetch_weights.sh --dest "$SRC" >/dev/null 2>&1
    say "  pass $tries: $(have)/$NEED"
done
say "stage 1 done: all $NEED shards verified"

# ---- 2. probe: one layer, while the checkpoint is still whole ------------
FREE=$(( $(df -kP "$(dirname "$OUT")" | awk 'NR==2 {print $4}') / 1048576 ))
say "stage 2: probe — ${FREE} GB free on the target volume"
[ "$FREE" -lt "$MIN_FREE_GB" ] && die "convert (need ${MIN_FREE_GB} GB, ${FREE} GB free)"

# The first MoE layer. K3 nests the text model, and the dense prefix has no
# experts to round-trip, so neither is a detail this can guess at.
PROBE=$(python3 -c "
import json
c = json.load(open('$SRC/config.json'))
c = c.get('text_config', c)
print(c.get('first_k_dense_replace', 0))" 2>/dev/null || echo "")
[ -n "$PROBE" ] || die "probe (no config.json at $SRC)"

probe_done() {
    python3 -c "
import json, sys
try:
    m = json.load(open('$OUT/manifest.json'))
except Exception:
    sys.exit(1)
sys.exit(0 if '$PROBE' in m.get('layers', {}) and m.get('trunk') else 1)" 2>/dev/null
}

if probe_done; then
    say "stage 2 skipped: layer $PROBE and the trunk are already published"
else
    T0=$(date +%s)
    $UV tools/convert.py --src "$SRC" --out "$OUT" --jobs 1 --layers "$PROBE" \
        >>"$LOG" 2>&1 || die "probe"
    say "stage 2 done in $(( ($(date +%s) - T0) / 60 )) min: layer $PROBE + trunk"
fi

# ---- 3. verify the container --------------------------------------------
# Skipped once it has passed, because it is the stage that cannot be redone:
# stage 4 may have reclaimed the weights it compares against.
if grep -q "^PASS" "$RUN/verify.txt" 2>/dev/null; then
    say "stage 3 skipped: round-trip already passed ($RUN/verify.txt)"
else
    if [ -s "$SRC/.reclaimed" ]; then
        die "verify (staging was reclaimed and $RUN/verify.txt holds no PASS \
— re-download $SRC, or point RUN_DIR at the run that proved this container)"
    fi
    say "stage 3: container round-trip on layer $PROBE"
    $UV tools/verify_container.py --container "$OUT" --src "$SRC" --experts 1 \
        > "$RUN/verify.txt" 2>&1
    grep -q "^PASS" "$RUN/verify.txt" || die "verify (see $RUN/verify.txt)"
    say "stage 3 done: round-trip PASS"
fi
ERR=$(grep -oE "rel err +[0-9.]+%" "$RUN/verify.txt" | head -1)

# ---- 4. convert the rest -------------------------------------------------
# --skip-trunk because stage 2 built it: a K3 trunk takes hours, and doing it
# twice per pipeline run was the cost of moving the probe in front.
say "stage 4: convert (reclaim=$RECLAIM)"
T0=$(date +%s)
$UV tools/convert.py --src "$SRC" --out "$OUT" --jobs "$JOBS" --skip-trunk \
    --reclaim "$RECLAIM" >>"$LOG" 2>&1 || die "convert"
say "stage 4 done in $(( ($(date +%s) - T0) / 60 )) min, $(du -sh "$OUT" | cut -f1)"
RECLAIMED=$(grep -E "^reclaim: .* (reclaimed|reclaimable) from " "$LOG" \
    | tail -1 | sed 's/^reclaim: //')
[ -n "$RECLAIMED" ] && say "  staging: $RECLAIMED"

# ---- 5. the engine actually runs -----------------------------------------
say "stage 5: engine"
make -s >>"$LOG" 2>&1 || die "build"
./waste run "$OUT" "$PROMPT" -n "$NTOK" --budget 46G > "$RUN/generated.txt" 2>&1 \
    || die "engine run (see $RUN/generated.txt)"
say "stage 5 done: $(head -c 200 "$RUN/generated.txt")"

# ---- 6. against the oracle ----------------------------------------------
say "stage 6: PyTorch oracle (slow — 93 layers in Python)"
IDS=$(./test_tokenizer "$OUT" "$PROMPT" | head -1 | cut -d' ' -f2- | tr ' ' ',')
[ -n "$IDS" ] || die "tokenize"
say "  prompt ids: $IDS"

./test_forward "$OUT" "$IDS" "$RUN/logits_c.bin" 0 >>"$LOG" 2>&1 || die "engine logits"
$UV tools/kimi_ref.py --container "$OUT" --tokens 0 --prompt-ids "$IDS" \
    --dump "$RUN/logits_ref.bin" >>"$LOG" 2>&1 || die "oracle"

python3 - "$RUN/logits_c.bin" "$RUN/logits_ref.bin" > "$RUN/diff.txt" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
md = max(abs(x - y) for x, y in zip(a, b))
na = sum(x * x for x in b) ** 0.5
rel = (sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5) / (na or 1)
ta = sorted(range(len(a)), key=lambda i: -a[i])[:10]
tb = sorted(range(len(b)), key=lambda i: -b[i])[:10]
print(f"max|diff| {md:.3e}   rel {rel:.3e}")
print(f"argmax  engine {ta[0]}  oracle {tb[0]}  {'MATCH' if ta[0]==tb[0] else 'MISMATCH'}")
print(f"top-10  {'identical' if ta==tb else 'differ'}")
print("ORACLE OK" if md < 1e-2 and ta[0] == tb[0] else "ORACLE MISMATCH")
PY
cat "$RUN/diff.txt" | tee -a "$LOG"
grep -q "^ORACLE OK" "$RUN/diff.txt" || die "oracle diff (see $RUN/diff.txt)"

# ---- report --------------------------------------------------------------
{
    echo "# K3 pipeline report"
    echo
    echo "Finished $(date '+%Y-%m-%d %H:%M')."
    echo
    echo '## Container'
    echo
    echo "- source: \`$SRC\` ($(du -sh "$SRC" | cut -f1) left on disk)"
    echo "- container: \`$OUT\` ($(du -sh "$OUT" | cut -f1))"
    echo "- round-trip error: $ERR (layer $PROBE, before the bulk convert)"
    if [ "$RECLAIM" != "off" ]; then
        # Say what was given back *and* that it is gone: a source that is
        # 40 GB in this report and 1.42 TB in the last one is not a mystery
        # anyone should have to solve from the sizes alone.
        echo "- staging reclaimed (RECLAIM=$RECLAIM): ${RECLAIMED:-none}"
        echo "  Re-verifying this container against its source needs the"
        echo "  checkpoint downloaded again."
    fi
    echo
    echo '## Generated'
    echo
    echo '```'
    cat "$RUN/generated.txt"
    echo '```'
    echo
    echo '## Against the PyTorch oracle'
    echo
    echo '```'
    cat "$RUN/diff.txt"
    echo '```'
} > "$RUN/REPORT.md"

say "=== pipeline complete — see $RUN/REPORT.md ==="
