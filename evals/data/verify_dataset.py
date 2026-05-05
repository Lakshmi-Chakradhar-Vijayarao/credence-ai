import json
import os
import re
from collections import Counter

# Epistemic markers to check for in faithful summaries
MARKERS = [
    "think", "might", "approx", "roughly", "estimate", "preliminary", 
    "unverified", "report", "suspect", "possible", "uncertain", 
    "pending", "check", "verify", "potential", "indicator", "likely",
    "perhaps", "maybe", "seems", "according to", "claim", "finalized",
    "noted", "flagged", "rumor", "provisional", "interpreted", "interprets",
    "suggest", "suggested", "pending", "observation", "observed", "discrepancy",
    "mismatch", "heartbeat", "pattern", "matches"
]

def verify_dataset(file_path):
    print(f"Executing ZERO TOLERANCE AUDIT: {file_path}\n" + "="*50)
    
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return
        
    with open(file_path) as f:
        data = json.load(f)
        
    examples = data if isinstance(data, list) else data.get("examples", [])
    total = len(examples)
    
    failures = []
    
    # Track uniqueness to prevent 'lazy' synthesis
    inputs = set()
    
    for i, ex in enumerate(examples):
        eid = ex.get("id", f"idx_{i}")
        faithful = ex.get("faithful_summary", "").lower()
        unfaithful = ex.get("unfaithful_summary", "").lower()
        input_text = ex.get("input_conversation", "").lower()
        is_ghost = ex.get("is_ghost", False)

        # 1. Semantic Divergence Test
        if faithful == unfaithful:
            failures.append(f"[{eid}] ERROR: Faithful and Unfaithful summaries are identical.")
        
        # 2. Epistemic Presence Test (Faithful)
        if not any(m in faithful for m in MARKERS):
            failures.append(f"[{eid}] ERROR: Faithful summary missing required qualifier marker.")
            
        # 3. Epistemic Absence Test (Unfaithful)
        # Unfaithful should be "False Certainty"
        dangerous_markers = ["think", "might", "approx", "estimate", "perhaps", "maybe", "reportedly", "unverified"]
        if any(m in unfaithful for m in dangerous_markers):
            failures.append(f"[{eid}] ERROR: Unfaithful summary contains hedging (destroys DPO signal).")

        # 4. Ghost Constraint Audit
        if is_ghost:
            # Ghost input should have NO surface hedging
            markers_in_input = [m for m in ["think", "might", "probably", "perhaps", "maybe", "approx"] if m in input_text]
            if markers_in_input:
                failures.append(f"[{eid}] ERROR: Ghost scenario input contains explicit markers {markers_in_input}.")

        # 5. Uniqueness Test
        if input_text in inputs:
            failures.append(f"[{eid}] ERROR: Duplicate input conversation found.")
        inputs.add(input_text)

    # Final Report
    print(f"Total Examples Audited: {total}")
    print(f"Total Failures Found:  {len(failures)}")
    
    if failures:
        print("\nTOP FAILURES (fixing required):")
        for f in failures[:20]:
            print(f"  {f}")
        print(f"\nAudit Result: ❌ FAILED ({len(failures)} issues)")
    else:
        print("\nAudit Result: ✅ PASSED (100% Gold Standard)")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/epistemic_compression_training.json"
    verify_dataset(path)
