"""
Run all figure generation scripts and report what was produced.
Usage: cd paper/figures && python3 generate_all.py
"""

import importlib.util
import os
import sys
from pathlib import Path

SCRIPTS = [
    "fig0_pipeline_diagram.py",
    "fig1_multimodel_eqlr.py",
    "fig2_compressor_comparison.py",
    "fig3_ghost_bothrate.py",
    "fig4_e8_recall.py",
    "fig5_latency.py",
]

here = Path(__file__).parent
os.chdir(here)   # save outputs alongside scripts

ok, failed = [], []
for script in SCRIPTS:
    try:
        spec = importlib.util.spec_from_file_location("_fig", here / script)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ok.append(script)
    except Exception as e:
        failed.append((script, e))

print(f"\n{'─'*50}")
print(f"Generated {len(ok)}/{len(SCRIPTS)} figures in {here}")
for s in ok:
    print(f"  ✓  {s}")
if failed:
    print("Failures:")
    for s, e in failed:
        print(f"  ✗  {s}: {e}")
sys.exit(0 if not failed else 1)
