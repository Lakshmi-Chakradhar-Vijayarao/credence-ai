"""
Terminal demo for GIF recording — Credence gate blocking scenario.
Run with: python3 demo/gate_demo.py
Record with: asciinema rec demo/gate_demo.cast --command "python3 demo/gate_demo.py" --overwrite
Convert:    agg demo/gate_demo.cast demo/gate_demo.gif --theme monokai
"""
import time, sys

G  = "\033[32m"   # green
Y  = "\033[33m"   # yellow
R  = "\033[31m"   # red
C  = "\033[36m"   # cyan
W  = "\033[37m"   # white dim
B  = "\033[1m"    # bold
DIM= "\033[2m"    # dim
RST= "\033[0m"    # reset

def p(text="", delay=0.04):
    print(text)
    time.sleep(delay)

def typewrite(text, color="", delay=0.03):
    print(color, end="", flush=True)
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print(RST)
    time.sleep(0.1)

# ── scene 1: developer says something uncertain ──────────────────────────────
time.sleep(0.4)
p(f"{DIM}$ claude  {RST}", 0.3)
p()
typewrite("You: The auth token probably expires in 3600s — I haven't confirmed with the vendor yet.", C, 0.025)
time.sleep(0.3)
p()
typewrite("Claude: Got it. I'll flag that as unverified.", G, 0.02)
p(f"{DIM}         [credence] registered → conf=0.28  zone=LOW{RST}", 0.4)

# ── scene 2: work continues ───────────────────────────────────────────────────
p()
p(f"{DIM}... 8 turns of normal work ...{RST}", 0.8)
p()

# ── scene 3: claude tries to write the code ──────────────────────────────────
typewrite("You: Implement the token refresh logic.", C, 0.025)
p()
typewrite("Claude: Writing token_refresh.py ...", G, 0.02)
time.sleep(0.5)

# ── scene 4: GATE FIRES ───────────────────────────────────────────────────────
p()
p(f"{R}{B}╔══════════════════════════════════════════════════════╗{RST}")
p(f"{R}{B}║  CREDENCE GATE — TOOL BLOCKED                        ║{RST}")
p(f"{R}{B}╚══════════════════════════════════════════════════════╝{RST}")
p()
p(f"  {W}Tool:    {RST}Edit  →  token_refresh.py")
p(f"  {Y}⚠ [LOW, conf=0.21] auth token expiry ~3600s — unconfirmed{RST}")
p(f"  {DIM}   Overlap: token, expiry, auth{RST}")
p()
p(f"  {W}Resolve:{RST}  credence_verify(<id>, confirmed_value=3600)")
p()
time.sleep(1.8)

# ── scene 5: developer verifies ───────────────────────────────────────────────
typewrite("You: Confirmed with vendor — token expires in 3600s.", C, 0.025)
p(f"{DIM}     [credence] verified → conf=1.00  zone=HIGH{RST}", 0.5)
p()

# ── scene 6: code writes ──────────────────────────────────────────────────────
typewrite("Claude: Writing token_refresh.py ...", G, 0.02)
time.sleep(0.3)
p(f"{G}  TOKEN_EXPIRY = 3600   # verified ✓{RST}", 0.2)
p(f"{G}  ✓ Edit complete — no unverified constraints.{RST}", 0.6)
p()
