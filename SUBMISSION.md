# CAMS — Submission Summary

**CAMS (Confidence-Adaptive Memory System)** is a cognitive governor for Claude Opus 4.7: a system that uses the model's own linguistic output as a real-time signal to control both what it remembers and how hard it thinks.

The project descends from two research strands — FAIL-CHAIN (how errors propagate in multi-step LLM pipelines) and Fisher J-signal (a hidden-state reliability indicator validated on transformer internals). Because Opus 4.7 exposes no hidden states, CAMS implements a language-level proxy: five linguistic factors measuring whether Claude's response is in a resolved or uncertain epistemic configuration. The same J-signal that governs memory (compress / trim / preserve) continuously scales the thinking budget for the next turn — uncertain turns get more compute, confident turns get less.

Four guard rails prevent unsafe compression: attention sink protection (first 2 turns are never compressed), Type Prior (error traces and code blocks get a J-ceiling so they can't be compressed regardless of tone), novelty guard (topic shifts force preserve), and a 3-turn drift detector for sustained instability.

Benchmark result on 30 QA pairs using real Opus 4.7 API calls: −26.5% tokens, −24.6% cost, +63% ROUGE-L quality vs. baseline. AUARC of 0.285 validates the J-proxy as a calibrated uncertainty signal, not a style detector.

Every file was written during the hackathon using Claude Code.
