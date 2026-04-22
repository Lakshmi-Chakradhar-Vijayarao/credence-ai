# CAMS — Submission Summary

**CAMS (Confidence-Adaptive Memory System)** is a cognitive governor for Claude Opus 4.7: a system that uses the model's own linguistic output as a real-time signal to control what it remembers — and specifically what it must not compress.

The project descends from two research strands — FAIL-CHAIN (how errors propagate in multi-step LLM pipelines) and Fisher J-signal (a hidden-state reliability indicator validated on transformer internals). Because Opus 4.7 exposes no hidden states, CAMS implements a language-level proxy: five linguistic factors that correlate with the resolved/uncertain distinction, validated against the Φ(√J̄/2) theoretical ceiling (AUARC 0.324, 49.0% of ceiling at the API surface).

Six guard rails prevent unsafe compression: attention sink protection (first 2 turns always preserved), Type Prior (code and error traces capped at MEDIUM zone regardless of tone), novelty guard (domain pivots detected via content-word ratio), faithfulness probe (refuses to compress old segments containing uncertainty markers — prevents Haiku from silently stripping "I'm not certain" into apparent fact), semantic entropy proxy (MEDIUM-zone responses with multi-answer markers downgraded to PRESERVE), and drift detection (3 consecutive LOW turns locks PRESERVE).

The benchmarks are honest: on a 30-question diverse QA benchmark, CAMS achieves 29.4% token reduction and 24.2% cost reduction vs. the same-prompt baseline, with +0.002 ROUGE-L delta (not statistically significant at n=30, bootstrap CI [-0.131, +0.137]). The system prompt contributes +0.078 ROUGE-L independently.

The measurable value is demonstrated in two targeted experiments. E6 (Negative Needle): when uncertain constraints are planted at turns 3-4 and 8 filler turns build compression pressure, CAMS achieves 100% correction recall vs. 0% for naive sliding window — the faithfulness probe refuses to compress segments containing uncertainty markers, preventing the model from confabulating confident wrong values. E7 (Multi-Hop Chain): CAMS preserves all 3 hops of a Falcon→Nexus CVE→Python upgrade chain through Haiku compression, matching baseline (3/3), while naive window drops to 1/3 hops.

A random-J ablation in E4 separates causality: CAMS 0.875 mean recall vs. naive window 0.812 and random-J 0.812 over a 20-turn session — CAMS > random_j confirms J-routing contributes beyond mere compression schedule. E2 found that long code blocks naturally stay in MEDIUM due to the brevity factor, so the Type Prior's value manifests on short code snippets rather than long function bodies.

Every file was written during the hackathon using Claude Code.
