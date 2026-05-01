#!/usr/bin/env bash
# Pull KV-cache Phase 1 results from Kaggle and display the table.
# Usage: ./scripts/pull_kv_results.sh

set -e
export KAGGLE_API_TOKEN=KGAT_a133fa1ec8f74503863c38bdf46578c6

KERNEL="chakradharvijayarao/credence-phase-1-kv-cache-eql-experiment"
OUTDIR="evals/kv_cache"

echo "==> Checking kernel status..."
STATUS=$(kaggle kernels status "$KERNEL" 2>&1)
echo "    $STATUS"

if echo "$STATUS" | grep -qi "RUNNING"; then
  echo "Kernel still running — try again later."
  exit 1
fi

if echo "$STATUS" | grep -qi "ERROR\|FAIL"; then
  echo "Kernel errored — check the Kaggle UI for logs."
  exit 2
fi

echo "==> Pulling output to $OUTDIR/ ..."
kaggle kernels output "$KERNEL" -p "$OUTDIR/"

echo "==> Files downloaded:"
ls -lh "$OUTDIR/"

RESULTS_FILE="$OUTDIR/kv_cache_results.json"
if [ ! -f "$RESULTS_FILE" ]; then
  echo "ERROR: $RESULTS_FILE not found in output."
  exit 3
fi

echo ""
echo "==> Results table:"
python3 - "$RESULTS_FILE" <<'EOF'
import json, sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

model  = data.get("model", "?")
date   = data.get("date", "?")
print(f"Model: {model}  |  Date: {date}")
print()

methods = data.get("methods", [])
results = data.get("results", {})

# Header
hdr = f"{'Method':<18} {'EQLR':>6} {'FCR':>6} {'Ghost-FCR':>10} {'Sem':>6} {'Lat(s)':>8}"
print(hdr)
print("-" * len(hdr))

for m in methods:
    agg = results.get(m, {}).get("aggregate", {})
    lat = results.get(m, {}).get("latency_s", 0)
    eqlr   = agg.get("eqlr_token", -1)
    fcr    = agg.get("fcr", -1)
    ghost  = agg.get("ghost_fcr", -1)
    sem    = agg.get("eqlr_semantic", -1)
    ghost_str = f"{ghost:.3f}" if ghost >= 0 else "  n/a"
    print(f"{m:<18} {eqlr:>6.3f} {fcr:>6.3f} {ghost_str:>10} {sem:>6.3f} {lat:>8.1f}")

print()
print("EQLR  = fraction of scenarios where qualifier was LOST (lower is better)")
print("FCR   = False Certainty Rate (lower is better)")
print("Ghost = FCR on ghost scenarios (no surface hedging, highest-risk category)")
print("Sem   = semantic qualifier overlap score (higher is better)")
EOF

echo ""
echo "==> Copying results to canonical path..."
cp "$RESULTS_FILE" "evals/kv_cache/kv_results.json"
echo "    Saved to evals/kv_cache/kv_results.json"
