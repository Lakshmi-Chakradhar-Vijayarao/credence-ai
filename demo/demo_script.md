# CAMS Live Demo Script — 14-Turn Session
# Verified: COMPRESS fires at T14 with 2,159 tokens saved. Recall test passes.
# Run: streamlit run demo/app.py  — use your API key in the sidebar

---

## WHAT THIS DEMO PROVES

1. CAMS preserves unique context across 13 turns of unrelated conversation
2. On a confident recall answer (J=0.750), COMPRESS fires and saves 2,159 tokens
3. A naive sliding window dropping oldest turns at T11 would have lost the T1 fact

---

## THE PLANTED FACT (Turn 1)
Unique fictional system config — not in Claude's training data.

**Type this first:**
> I'm designing a distributed cache called 'Heliograph' with a worker pool of 7 threads
> and a write-back buffer of 3,841 bytes. Can you explain how write-back caching differs
> from write-through caching?

---

## HISTORY-BUILDING TURNS (Type these in order, turns 2–13)

**T02:** Explain how TCP's three-way handshake works and what problem SYN flooding exploits.

**T03:** How does TLS 1.3 differ from TLS 1.2 in terms of the handshake steps and forward secrecy?

**T04:** What exactly is the difference between symmetric and asymmetric encryption, and when is each used?

**T05:** Explain how DNS resolution works from the moment a browser types a URL to getting an IP address.

**T06:** How does a CPU cache hierarchy (L1, L2, L3) work and what is cache coherence?

**T07:** What is consistent hashing and why is it preferred over modulo hashing in distributed systems?

**T08:** Explain how HTTPS certificate validation works — what does the browser actually verify?

**T09:** What is the exact sequence of steps in the OAuth 2.0 authorization code flow?

**T10:** What are the specific HTTP status code ranges — 1xx, 2xx, 3xx, 4xx, 5xx — and what does each represent?

**T11:** How does a relational database engine execute a JOIN operation internally?

**T12:** What is the difference between optimistic and pessimistic locking in databases?

**T13:** Explain how garbage collection works in the JVM — what are the generational heap regions?

---

## THE RECALL TEST + COMPRESS EVENT (Turn 14)

**Type this:**
> What was the name of the distributed cache system I described at the very start of our
> conversation, how many worker threads did it have, and what was the exact write-back
> buffer size in bytes?

**Expected on screen simultaneously:**
- J-gauge: 0.750 (HIGH zone, green)
- Decision Log: COMPRESS ← 2,159 tokens saved
- Claude's answer: "Heliograph, 7 worker threads, 3,841 bytes"

---

## RECORDING CHECKLIST

□ API key entered in Streamlit sidebar (CAMS Signal Panel visible on right)
□ Screen recording started before Turn 1
□ CAMS Signal Panel (J-gauge + Decision Log) always visible
□ Zoom in on Decision Log when COMPRESS fires at T14
□ Pause on Claude's correct recall answer
□ Session stats visible: tokens used + tokens saved updating live

---

## THE BASELINE FAILURE LINE (say in voiceover)

"A naive sliding window that drops turns older than 10 would have discarded Turn 1
at Turn 11 — losing 'Heliograph', 7 threads, and 3,841 bytes before the recall test.
CAMS preserved it because no turn in this conversation signaled enough confidence
to compress until Turn 14."

