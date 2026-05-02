# Removed Dead Code

Code removed from live files during open-source cleanup (2026-05-02).
All blocks are recoverable from git history or this file.

---

## `credence/context_manager.py` — Novelty Guard

### `_check_novelty()` and `_update_content_vocab()` methods

Removed from between `_extract_content_words()` and `_has_multi_answer()`.

**Why removed**: Novelty guard measured 79-87% false-positive rate on stable-domain
technical sessions. `_check_novelty()` was already returning `False` unconditionally
(disabled). `_update_content_vocab()` had no callers after its call site was removed.
The `_content_vocab` and `_recent_vocab_window` instance variables were also removed
from `__init__()`, `reset()`, `save()`, and `load()`.

**Note**: `_extract_content_words()` is **kept** — still used by `_detect_contradiction()`
and `_summary_faithful()`.

```python
def _check_novelty(self, text: str) -> bool:
    """
    Novelty guard — DISABLED after empirical measurement showed 79-87% FP rate.

    Technical writing introduces new vocabulary every sentence within the same domain
    ("vacuum", "partitioning", "B-tree" are all "new" in a PostgreSQL session).
    A vocabulary-distance signal cannot distinguish same-domain progression from
    a real domain pivot without semantic embeddings — which are not available here.

    The cases this guard was intended to protect are already covered by:
      - Faithfulness probe: detects uncertainty in compressible segments → PRESERVE
      - Selective J-compression: LOW/MEDIUM-J turns always kept verbatim
      - Regime detection: low J-variance → PRESERVE mode (no compression)

    Kept as a stub so the call site and decision log field remain intact.
    Returns False always. Re-enable with a proper embedding-based implementation.
    """
    return False

def _update_content_vocab(self, text: str):
    words = self._extract_content_words(text)
    self._content_vocab.update(words)
    # Maintain sliding window of last 3 turns
    self._recent_vocab_window.append(words)
    if len(self._recent_vocab_window) > 3:
        self._recent_vocab_window.pop(0)
```

### Instance variables removed from `__init__()` and `reset()`

```python
self._content_vocab: set[str] = set()
self._recent_vocab_window: list[set[str]] = []
```

### Constants removed from module-level

```python
_NOVELTY_THRESHOLD = 0.75
_NOVELTY_MIN_ENTITIES = 5
_NOVELTY_MIN_VOCAB = 10
```

### `save()` / `load()` entries removed

```python
# In save():
"content_vocab": list(self._content_vocab),

# In load():
self._content_vocab = set(state.get("content_vocab", []))
```

### Call site removed from `chat()`

```python
novelty_override = self._check_novelty(text)
# ...
"novelty_override": novelty_override,
# ...
self._update_content_vocab(text)
```

### `_apply_credence()` signature and guard removed

```python
# Signature was:
def _apply_credence(self, cr, novelty_override):
# Changed to:
def _apply_credence(self, cr):

# Guard block removed:
if novelty_override:
    return "PRESERVE", 0
```

---

## `credence/__init__.py` — Archived Module Exports

Removed these exports (the underlying modules were moved to `_archive/api_dependent/`):

```python
from .confidence_proxy import CredenceProxy, CredenceResult
from .epistemic_manifest import EpistemicManifest

# Also removed from __all__:
"EpistemicManifest",
"CredenceProxy", "CredenceResult",
```

**Note**: `confidence_proxy.py` was restored to `credence/` as an unexported internal
module because `ContextManager` (a valid power-user export) depends on it. It is not
re-exported in `__all__`.

---

## Files fully archived (see `_archive/ARCHIVE_LOG.md` for rationale)

| File | Archived to |
|------|------------|
| `credence/confidence_proxy.py` | `_archive/api_dependent/confidence_proxy.py` |
| `credence/epistemic_manifest.py` | `_archive/api_dependent/epistemic_manifest.py` |
| `credence/behavioral_signal.py` | `_archive/api_dependent/behavioral_signal.py` |
| `credence/claim_extractor.py` | `_archive/api_dependent/claim_extractor.py` |
| `credence/dpo_proxy.py` | `_archive/api_dependent/dpo_proxy.py` |
