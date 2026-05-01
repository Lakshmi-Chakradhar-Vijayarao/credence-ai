"""
evals/kv_cache/experiment_runner.py
====================================
KV-cache EQL experiment runner.

Designed to run on Kaggle T4 GPU with HuggingFace Transformers.
All HuggingFace and KV-eviction imports are lazy — the script runs
in --dry-run mode without any GPU or ML libraries installed.

Usage
-----
    # Validate structure, show table header, print 3 sample scenarios:
    python -m evals.kv_cache.experiment_runner --dry-run

    # Run one condition (10 scenarios):
    python -m evals.kv_cache.experiment_runner --method h2o --budget 0.70 --n 10

    # Run one condition (all 102 scenarios):
    python -m evals.kv_cache.experiment_runner --method h2o --budget 0.70 --all

    # Run all 7 conditions and save results:
    python -m evals.kv_cache.experiment_runner --all-conditions --out results/kv_cache_results.json

Models / libraries required for full run
-----------------------------------------
    pip install transformers torch accelerate
    # H2O:
    pip install h2o-llm   # or copy h2o_llm.py from the H2O repo
    # SnapKV:
    pip install snapkv    # or apply patch from github.com/FasterDecoding/SnapKV
    # StreamingLLM:
    pip install streaming-llm  # or apply patch from github.com/mit-han-lab/streaming-llm
"""

from __future__ import annotations

import abc
import argparse
import json
import os
import sys
import time
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evals.kv_cache.metrics import score_scenario, aggregate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL     = "meta-llama/Meta-Llama-3-8B-Instruct"
_DEFAULT_DEVICE    = "cuda"          # falls back to cpu in dry-run
_MAX_NEW_TOKENS    = 200
_CONTEXT_TARGET_TOKENS = 8000        # target conversation length

_CONDITIONS = [
    ("baseline",      None,          1.00),
    ("h2o_70",        "h2o",         0.70),
    ("h2o_50",        "h2o",         0.50),
    ("snapkv_70",     "snapkv",      0.70),
    ("snapkv_50",     "snapkv",      0.50),
    ("streaming_70",  "streaming",   0.70),
    ("streaming_50",  "streaming",   0.50),
]

# Generic filler Q&A pool — neutral technical content for context padding.
# Each pair adds ~100-150 tokens. 6 turns ≈ 600-900 tokens of padding.
FILLER_POOL = [
    ("What is the difference between TCP and UDP?",
     "TCP is connection-oriented with guaranteed delivery; UDP is connectionless and faster "
     "but provides no delivery guarantee. Use TCP for reliability (HTTP, databases), UDP for "
     "latency-sensitive streaming (video, DNS)."),
    ("What is a load balancer?",
     "A load balancer distributes incoming network traffic across multiple servers to ensure "
     "no single server is overwhelmed. It improves availability and scalability by routing "
     "requests based on health checks and balancing algorithms like round-robin or least-connections."),
    ("Explain HTTP caching headers.",
     "Cache-Control specifies directives for caching mechanisms. max-age defines how long a "
     "resource is fresh. ETag provides a validator for conditional requests. Last-Modified "
     "is a timestamp validator. Vary tells caches to consider request headers when caching."),
    ("What is a database index?",
     "An index is a data structure that improves query performance by allowing the database "
     "engine to find rows without scanning the entire table. Common types are B-tree for "
     "range queries and hash for equality lookups. Indexes speed reads but slow writes."),
    ("What is JWT?",
     "JSON Web Token is a compact, self-contained way to transmit information as a JSON "
     "object that can be verified and trusted because it is digitally signed. JWTs consist "
     "of three parts: header, payload, and signature, joined by dots."),
    ("What is eventual consistency?",
     "Eventual consistency is a consistency model where updates propagate to all replicas "
     "asynchronously. Given enough time with no new updates, all replicas will converge to "
     "the same value. Used in distributed systems like DynamoDB and Cassandra."),
    ("What is a CDN?",
     "A Content Delivery Network is a geographically distributed network of servers that "
     "caches and delivers content from locations close to users. It reduces latency, improves "
     "availability, and offloads traffic from origin servers."),
    ("What is the difference between REST and GraphQL?",
     "REST uses fixed endpoints that return predefined data shapes; GraphQL uses a single "
     "endpoint where clients specify exactly what data they need. GraphQL reduces over-fetching "
     "but adds query complexity. REST is simpler for caching."),
    ("What is database connection pooling?",
     "Connection pooling maintains a pool of database connections that can be reused across "
     "requests. This avoids the overhead of creating a new connection per query. Tools like "
     "PgBouncer or HikariCP manage connection pool sizing and idle timeouts."),
    ("What is a deadlock?",
     "A deadlock occurs when two or more transactions wait indefinitely for each other to "
     "release locks. Detection strategies include timeout-based rollback or cycle detection "
     "in the wait-for graph. Prevention strategies include lock ordering and optimistic locking."),
    ("What is exponential backoff?",
     "Exponential backoff is a retry strategy where the wait time between retries grows "
     "exponentially (e.g., 1s, 2s, 4s, 8s). It reduces thundering-herd pressure on "
     "overloaded services. Adding jitter randomises the wait to prevent synchronised retries."),
    ("What is container orchestration?",
     "Container orchestration automates deployment, scaling, and management of containerised "
     "applications. Kubernetes is the de facto standard, providing declarative configuration, "
     "service discovery, rolling updates, and self-healing via pod restarts."),
    ("What is idempotency in APIs?",
     "An idempotent operation produces the same result when applied multiple times as when "
     "applied once. GET and DELETE are inherently idempotent. POST typically isn't, but "
     "idempotency keys (unique per request) allow safe retries by deduplicating on the server."),
    ("What is the CAP theorem?",
     "CAP states that a distributed system can guarantee at most two of: Consistency "
     "(all nodes see the same data), Availability (every request gets a response), and "
     "Partition Tolerance (system operates despite network partitions). In practice, "
     "partition tolerance is required, so the tradeoff is between C and A."),
    ("What is a circuit breaker pattern?",
     "A circuit breaker monitors failures in external service calls and trips open after a "
     "threshold, preventing further calls for a cooldown period. States: Closed (normal), "
     "Open (failing, reject calls), Half-Open (test if service recovered). Prevents cascading "
     "failures in distributed systems."),
]


# ---------------------------------------------------------------------------
# Abstract base class for KV eviction methods
# ---------------------------------------------------------------------------

class KVEvictionMethod(abc.ABC):
    """
    Abstract base for KV-cache eviction methods.

    Subclasses implement generate() which returns the model's text output
    and optionally the attention matrix for QAR computation.
    """

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        kv_budget: float = 1.0,
    ) -> tuple[str, Optional[Any]]:
        """
        Generate a response with KV eviction applied.

        Parameters
        ----------
        prompt : str
            Full conversation text formatted as a HuggingFace chat template.
        kv_budget : float
            Fraction of KV pairs to retain (1.0 = no eviction).

        Returns
        -------
        (text, attention_matrix) : tuple
            text             : the generated response string
            attention_matrix : np.ndarray (n_layers, n_heads, seq_len, seq_len)
                               or None if attention extraction is not available
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Method", "").lower()


# ---------------------------------------------------------------------------
# Baseline (no eviction) — HuggingFace standard generate
# ---------------------------------------------------------------------------

class BaselineMethod(KVEvictionMethod):
    """
    No KV eviction. Standard HuggingFace generate() with output_attentions=True.

    This is the floor condition — any EQLR increase above this baseline in
    other conditions is attributable to KV eviction.
    """

    def __init__(self, model_id: str = _DEFAULT_MODEL, device: str = _DEFAULT_DEVICE):
        self._model_id = model_id
        self._device   = device
        self._model    = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy-load model and tokenizer on first use."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "HuggingFace Transformers and PyTorch are required for full runs.\n"
                "Install with: pip install transformers torch accelerate\n"
                f"Original error: {e}"
            ) from e

        print(f"Loading {self._model_id} on {self._device} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self._device == "cuda" else self._device,
        )
        self._model.eval()
        print("Model loaded.")

    def generate(
        self,
        prompt: str,
        kv_budget: float = 1.0,
    ) -> tuple[str, Optional[Any]]:
        self._load()
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs.sequences[0, input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Extract attention matrices from the final generation step if available
        attention_matrix = self._extract_attention(outputs)
        return text.strip(), attention_matrix

    @staticmethod
    def _extract_attention(outputs: Any) -> Optional[Any]:
        """
        Extract a (n_layers, n_heads, seq_len, seq_len) attention matrix
        from HuggingFace GenerateOutput.

        HuggingFace returns attentions as a tuple of tuples:
            outputs.attentions[step][layer]  shape: (batch, heads, 1, seq_len)

        We take the first generation step (full prompt attention), stack
        layers, and squeeze the batch dimension.

        Returns None if attentions are not available.
        """
        try:
            import torch
            import numpy as np
            if not hasattr(outputs, "attentions") or not outputs.attentions:
                return None
            # attentions: tuple[step] of tuple[layer] of Tensor(1, heads, 1, seq_len)
            step0 = outputs.attentions[0]
            # Stack into (n_layers, heads, 1, seq_len), squeeze to (n_layers, heads, seq_len)
            stacked = torch.stack([layer[0].squeeze(2) for layer in step0], dim=0)
            # stacked: (n_layers, heads, seq_len)
            # Expand to (n_layers, heads, seq_len, seq_len) by broadcasting
            # (we treat it as attention FROM each position equally — approximation)
            # For QAR we actually want column-wise aggregation across all positions
            attn_np = stacked.cpu().float().numpy()
            n_layers, n_heads, seq_len = attn_np.shape
            # Broadcast: (n_layers, n_heads, 1, seq_len) repeated along query axis
            full_attn = attn_np[:, :, None, :].repeat(seq_len, axis=2)
            return full_attn  # (n_layers, n_heads, seq_len, seq_len)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# H2O Method
# ---------------------------------------------------------------------------

class H2OMethod(KVEvictionMethod):
    """
    H2O (Heavy Hitter Oracle) KV eviction.

    Reference: Zhang et al., NeurIPS 2023.
    Mechanism: retains heavy-hitter tokens (high cumulative attention) plus
    a recency window. Controlled by heavy_ratio = budget * 0.8 (80% heavy
    hitters, 20% recency).

    Requires: pip install h2o-llm
    Or apply the H2O KV cache manager patch from:
        https://github.com/FMInference/H2O
    """

    def __init__(self, model_id: str = _DEFAULT_MODEL, device: str = _DEFAULT_DEVICE):
        self._model_id = model_id
        self._device   = device
        self._model    = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            # H2O monkey-patches the model's attention modules
            try:
                from h2o_llm import H2OLlamaAttention, H2OLlamaForCausalLM
                _h2o_available = True
            except ImportError:
                _h2o_available = False

            if not _h2o_available:
                raise ImportError(
                    "H2O library not found. Install with: pip install h2o-llm\n"
                    "Or apply the H2O patch from: https://github.com/FMInference/H2O\n"
                    "For a manual patch, add the H2O KV cache manager to the model's "
                    "attention forward() method before running this condition."
                )
        except ImportError as e:
            raise ImportError(str(e)) from e

        print(f"Loading {self._model_id} with H2O eviction ...")
        from h2o_llm import H2OLlamaForCausalLM
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = H2OLlamaForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self._device == "cuda" else self._device,
        )
        self._model.eval()
        print("H2O model loaded.")

    def generate(
        self,
        prompt: str,
        kv_budget: float = 0.70,
    ) -> tuple[str, Optional[Any]]:
        self._load()
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        seq_len = inputs["input_ids"].shape[1]
        max_kv_size = max(32, int(seq_len * kv_budget))
        heavy_ratio = 0.80  # 80% heavy hitters, 20% recency within the budget

        # H2O models accept kv_cache_budget or heavy_budget parameters
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                heavy_ratio=heavy_ratio,
                recent_ratio=1.0 - heavy_ratio,
                kv_cache_budget=max_kv_size,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs.sequences[0, input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        attention_matrix = BaselineMethod._extract_attention(outputs)
        return text.strip(), attention_matrix


# ---------------------------------------------------------------------------
# SnapKV Method
# ---------------------------------------------------------------------------

class SnapKVMethod(KVEvictionMethod):
    """
    SnapKV KV eviction.

    Reference: Li et al., 2024.
    Mechanism: pools attention from the instruction window to decide which
    earlier KV pairs to retain. The observation window is the last
    window_size=32 tokens of the prompt.

    Requires: pip install snapkv
    Or apply the SnapKV patch from:
        https://github.com/FasterDecoding/SnapKV
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        device: str = _DEFAULT_DEVICE,
        window_size: int = 32,
    ):
        self._model_id   = model_id
        self._device     = device
        self._window_size = window_size
        self._model      = None
        self._tokenizer  = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            try:
                from snapkv import enable_snapkv
                _snapkv_available = True
            except ImportError:
                _snapkv_available = False

            if not _snapkv_available:
                raise ImportError(
                    "SnapKV library not found. Install with: pip install snapkv\n"
                    "Or apply the SnapKV patch from: https://github.com/FasterDecoding/SnapKV\n"
                    "Patch applies to LlamaAttention.forward() to intercept the KV cache "
                    "and apply observation-window-based selection."
                )
        except ImportError as e:
            raise ImportError(str(e)) from e

        print(f"Loading {self._model_id} with SnapKV eviction ...")
        from snapkv import enable_snapkv
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self._device == "cuda" else self._device,
        )
        # SnapKV patches the attention modules in-place
        enable_snapkv(base_model, window_size=self._window_size)
        self._model = base_model
        self._model.eval()
        print("SnapKV model loaded.")

    def generate(
        self,
        prompt: str,
        kv_budget: float = 0.70,
    ) -> tuple[str, Optional[Any]]:
        self._load()
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        seq_len = inputs["input_ids"].shape[1]
        max_capacity = max(32, int(seq_len * kv_budget))

        # SnapKV models accept max_capacity_prompt controlling how many KV pairs to keep
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                max_capacity_prompt=max_capacity,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs.sequences[0, input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        attention_matrix = BaselineMethod._extract_attention(outputs)
        return text.strip(), attention_matrix


# ---------------------------------------------------------------------------
# StreamingLLM Method
# ---------------------------------------------------------------------------

class StreamingLLMMethod(KVEvictionMethod):
    """
    StreamingLLM KV eviction.

    Reference: Xiao et al., ICLR 2024.
    Mechanism: retains the first sink_size tokens (attention sinks) plus a
    rolling recency window. Evicts all other tokens.

    Requires: pip install streaming-llm
    Or apply the StreamingLLM patch from:
        https://github.com/mit-han-lab/streaming-llm
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        device: str = _DEFAULT_DEVICE,
        sink_size: int = 4,
    ):
        self._model_id  = model_id
        self._device    = device
        self._sink_size = sink_size
        self._model     = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            try:
                from streaming_llm.enable_streaming_llm import enable_streaming_llm
                _streaming_available = True
            except ImportError:
                _streaming_available = False

            if not _streaming_available:
                raise ImportError(
                    "streaming-llm library not found. Install with: pip install streaming-llm\n"
                    "Or apply the StreamingLLM patch from: "
                    "https://github.com/mit-han-lab/streaming-llm\n"
                    "The patch modifies attention_forward() to implement the attention sink + "
                    "recency window eviction strategy."
                )
        except ImportError as e:
            raise ImportError(str(e)) from e

        print(f"Loading {self._model_id} with StreamingLLM eviction ...")
        from streaming_llm.enable_streaming_llm import enable_streaming_llm
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self._device == "cuda" else self._device,
        )
        # StreamingLLM patches attention in-place
        enable_streaming_llm(base_model, start_size=self._sink_size, recent_size=256)
        self._model = base_model
        self._model.eval()
        print("StreamingLLM model loaded.")

    def generate(
        self,
        prompt: str,
        kv_budget: float = 0.70,
    ) -> tuple[str, Optional[Any]]:
        self._load()
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        seq_len = inputs["input_ids"].shape[1]
        recent_size = max(16, int(seq_len * kv_budget) - self._sink_size)

        # StreamingLLM models accept recent_size to configure the window
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                recent_size=recent_size,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs.sequences[0, input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        attention_matrix = BaselineMethod._extract_attention(outputs)
        return text.strip(), attention_matrix


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def _get_method(method_name: Optional[str], budget: float) -> KVEvictionMethod:
    """Instantiate the right method class."""
    if method_name is None or method_name == "baseline":
        return BaselineMethod()
    name = method_name.lower()
    if name == "h2o":
        return H2OMethod()
    if name == "snapkv":
        return SnapKVMethod()
    if name in ("streaming", "streamingllm", "streaming_llm"):
        return StreamingLLMMethod()
    raise ValueError(f"Unknown method: {method_name!r}. Choose from: baseline, h2o, snapkv, streaming")


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_eql_bench() -> list[dict]:
    """Load EQL-Bench v1 scenarios."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "eql_bench", "eql_bench_v1.json",
    )
    if not os.path.exists(path):
        print(f"Warning: EQL-Bench not found at {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("scenarios", [])


def _load_ghost_scenarios() -> list[dict]:
    """Load ghost constraint scenarios."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "eql_bench", "ghost_scenarios.json",
    )
    if not os.path.exists(path):
        print(f"Warning: ghost scenarios not found at {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    # Convert to EQL-Bench-compatible format
    converted = []
    for item in data:
        converted.append({
            "scenario_id":       item.get("id", ""),
            "domain":            item.get("domain", ""),
            "qualifier_type":    item.get("qualifier_type", ""),
            "uncertain_statement": item.get("text", ""),
            "value_fragments":   [item.get("value_fragment", "")],
            "qualifier_fragments": [item.get("qualifier_fragment", "")],
            "reference_answer":  item.get("text", "") + " (unverified — " + item.get("source_note", "") + ")",
            "notes":             item.get("source_note", ""),
            "_is_ghost":         True,
        })
    return converted


# ---------------------------------------------------------------------------
# Context embedding
# ---------------------------------------------------------------------------

def embed_in_context(scenario: dict, n_filler_turns: int = 6) -> str:
    """
    Embed a scenario's uncertain claim in an 8K-token multi-turn conversation.

    Structure:
        Turn 1: Context establishment (user)
        Turn 2: Acknowledgement (assistant)
        Turn 3: Uncertain claim from scenario (user)
        Turn 4: Acknowledgement (assistant)
        Turns 5-10: Neutral filler Q&A (n_filler_turns=6)
        Turn 11: Callback question

    Parameters
    ----------
    scenario : dict
        A scenario dict with uncertain_statement, domain, value_fragments.
    n_filler_turns : int
        Number of filler turns to inject (default 6 — each ≈100-150 tokens).

    Returns
    -------
    str
        Formatted conversation prompt (LLaMA-3-Instruct chat template style).
    """
    domain = scenario.get("domain", "software")
    claim  = scenario.get("uncertain_statement", scenario.get("text", ""))

    # Build callback question from value fragments
    val_frags = scenario.get("value_fragments", [])
    if val_frags:
        callback_q = (
            f"Based on what we discussed, what is the {val_frags[0]!r} "
            "figure we should use in our implementation? Please be precise "
            "about any uncertainty you flagged earlier."
        )
    else:
        callback_q = (
            "Based on what we discussed, what constraint or limit should "
            "we implement? Please be precise about any uncertainty you flagged earlier."
        )

    # Format filler turns (cycle through FILLER_POOL)
    filler_text = ""
    for i in range(n_filler_turns):
        fq, fa = FILLER_POOL[i % len(FILLER_POOL)]
        filler_text += f"<|start_header_id|>user<|end_header_id|>\n\n{fq}<|eot_id|>"
        filler_text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{fa}<|eot_id|>"

    # Assemble full conversation in LLaMA-3-Instruct format
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a precise technical assistant helping plan a {domain} integration. "
        "When asked to recall earlier facts, state them exactly as established, "
        "including any uncertainty flags.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"We're starting a new {domain} project.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"Understood. Let's document the requirements as we go for the {domain} project.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{claim}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "Noted. I've recorded that constraint with any uncertainty flags as stated.<|eot_id|>"
        + filler_text
        + "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{callback_q}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt


# ---------------------------------------------------------------------------
# Dry-run stub
# ---------------------------------------------------------------------------

class _DryRunMethod(KVEvictionMethod):
    """
    Dry-run method — returns a fake response without loading any models.
    Used with --dry-run to validate structure without GPU.
    """

    def __init__(self, name_str: str = "dry_run", inject_qualifier: bool = False):
        self._name_str = name_str
        self._inject_qualifier = inject_qualifier

    @property
    def name(self) -> str:
        return self._name_str

    def generate(
        self,
        prompt: str,
        kv_budget: float = 1.0,
    ) -> tuple[str, Optional[Any]]:
        # Extract the claim from the prompt to synthesise a plausible response
        if "50 req/min" in prompt.lower():
            if self._inject_qualifier:
                return "I believe the rate limit is approximately 50 req/min, but this needs verification.", None
            else:
                return "The rate limit is 50 req/min.", None
        if "3600" in prompt:
            if self._inject_qualifier:
                return "Tokens appear to expire after roughly 3600 seconds, though this is unverified.", None
            else:
                return "Token expiry is 3600 seconds.", None
        if self._inject_qualifier:
            return "Based on what was discussed, the value is approximately as stated — but flagged as uncertain.", None
        return "The value is as previously specified.", None


# ---------------------------------------------------------------------------
# Condition runner
# ---------------------------------------------------------------------------

def run_condition(
    method: KVEvictionMethod,
    budget: float,
    scenarios: list[dict],
    condition_id: str = "",
    verbose: bool = False,
) -> list[dict]:
    """
    Run all scenarios under one condition and return per-scenario score dicts.

    Parameters
    ----------
    method : KVEvictionMethod
        Eviction method to use.
    budget : float
        KV budget fraction (1.0 = no eviction).
    scenarios : list[dict]
        Scenario dicts from EQL-Bench or ghost_scenarios.
    condition_id : str
        Label for this condition (e.g., "h2o_70").
    verbose : bool
        Print progress.

    Returns
    -------
    list[dict]
        Per-scenario score dicts (output of score_scenario).
    """
    results = []
    n = len(scenarios)
    start_time = time.time()

    for i, scenario in enumerate(scenarios):
        prompt   = embed_in_context(scenario)
        t0       = time.time()
        answer, attn = method.generate(prompt, kv_budget=budget)
        elapsed  = time.time() - t0

        score = score_scenario(scenario, answer, attention_matrix=attn)
        score["condition"]  = condition_id or method.name
        score["kv_budget"]  = budget
        score["elapsed_s"]  = round(elapsed, 2)

        results.append(score)

        if verbose:
            status = "LOST" if score["eqlr_token"] else "OK"
            fcr_str = " FCR!" if score["fcr"] else ""
            print(
                f"  [{i+1:3d}/{n}] {scenario.get('scenario_id',''):<12} "
                f"EQLR={status}{fcr_str}  QAR={score['qar']:6.4f}  ({elapsed:.1f}s)"
            )

    elapsed_total = time.time() - start_time
    if verbose:
        print(f"  Condition {condition_id}: {n} scenarios in {elapsed_total:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def run_all_conditions(
    scenarios: list[dict],
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run all 7 conditions (baseline + 2 budgets × 3 methods).

    Parameters
    ----------
    scenarios : list[dict]
        All 102 scenarios (EQL-Bench + ghost).
    dry_run : bool
        If True, use _DryRunMethod instead of loading real models.
    verbose : bool
        Print per-scenario progress.

    Returns
    -------
    dict
        Keys are condition IDs; values are aggregate dicts from metrics.aggregate().
        Also includes "metadata" key.
    """
    output = {
        "metadata": {
            "model":      _DEFAULT_MODEL if not dry_run else "dry_run",
            "n_scenarios": len(scenarios),
            "conditions": [c[0] for c in _CONDITIONS],
            "dry_run":    dry_run,
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "conditions": {},
    }

    for condition_id, method_name, budget in _CONDITIONS:
        print(f"\nRunning condition: {condition_id} (method={method_name}, budget={budget})")

        if dry_run:
            inject = method_name is None  # baseline injects qualifier in dry run
            method = _DryRunMethod(condition_id, inject_qualifier=inject)
        else:
            try:
                method = _get_method(method_name, budget)
            except ImportError as e:
                print(f"  Skipped ({e})")
                output["conditions"][condition_id] = {"error": str(e)}
                continue

        raw_scores = run_condition(method, budget, scenarios, condition_id=condition_id, verbose=verbose)
        summary    = aggregate(raw_scores)
        summary["scenarios"] = raw_scores
        output["conditions"][condition_id] = summary

        print(
            f"  EQLR={summary['eqlr_token']:.3f}  "
            f"FCR={summary['fcr']:.3f}  "
            f"GhostFCR={summary.get('ghost_fcr', -1.0):.3f}  "
            f"QAR={summary.get('mean_qar', -1.0):.4f}"
        )

    return output


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def make_comparison_table(results: dict) -> str:
    """
    Format a comparison table of all conditions.

    Parameters
    ----------
    results : dict
        Output from run_all_conditions().

    Returns
    -------
    str
        Formatted ASCII table.
    """
    header = (
        f"{'Method':<14} {'Budget':>7}  "
        f"{'EQLR-Token':>12}  {'FCR':>8}  {'Ghost-FCR':>10}  {'Mean-QAR':>10}"
    )
    sep = "-" * len(header)
    rows = [header, sep]

    for condition_id, method_name, budget in _CONDITIONS:
        cdata = results.get("conditions", {}).get(condition_id, {})
        if "error" in cdata:
            rows.append(f"{condition_id:<14} {'N/A':>7}  (skipped: {cdata['error'][:40]})")
            continue

        budget_str = f"{budget*100:.0f}%" if budget < 1.0 else "100%"
        method_label = method_name.upper() if method_name else "baseline"

        eqlr     = cdata.get("eqlr_token", -1.0)
        eqlr_ci  = cdata.get("eqlr_token_ci", [-1.0, -1.0])
        fcr_v    = cdata.get("fcr", -1.0)
        fcr_ci   = cdata.get("fcr_ci", [-1.0, -1.0])
        g_fcr    = cdata.get("ghost_fcr", -1.0)
        g_ci     = cdata.get("ghost_fcr_ci", [-1.0, -1.0])
        qar_v    = cdata.get("mean_qar", -1.0)

        eqlr_str  = f"{eqlr:.3f} [{eqlr_ci[0]:.3f},{eqlr_ci[1]:.3f}]" if eqlr >= 0 else "  N/A"
        fcr_str   = f"{fcr_v:.3f}" if fcr_v >= 0 else "  N/A"
        g_fcr_str = f"{g_fcr:.3f}" if g_fcr >= 0 else "  N/A"
        qar_str   = f"{qar_v:.4f}" if qar_v >= 0 else "  N/A"

        rows.append(
            f"{method_label:<14} {budget_str:>7}  "
            f"{eqlr_str:>22}  {fcr_str:>8}  {g_fcr_str:>10}  {qar_str:>10}"
        )

    rows.append(sep)
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KV-cache EQL experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--method", choices=["baseline", "h2o", "snapkv", "streaming"],
                        help="Single method to run")
    parser.add_argument("--budget", type=float, default=0.70,
                        help="KV budget fraction (0.0–1.0, default 0.70)")
    parser.add_argument("--n",   type=int, default=None,
                        help="Limit to first N scenarios")
    parser.add_argument("--all", action="store_true",
                        help="Run all scenarios for the given method+budget")
    parser.add_argument("--all-conditions", action="store_true",
                        help="Run all 7 conditions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate structure, show 3 samples, no model loading")
    parser.add_argument("--out", default="results/kv_cache_results.json",
                        help="Output path for results JSON")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print per-scenario progress")
    args = parser.parse_args()

    # Load datasets
    eql_bench_scenarios = _load_eql_bench()
    ghost_scenarios     = _load_ghost_scenarios()
    all_scenarios       = eql_bench_scenarios + ghost_scenarios

    if args.n is not None:
        all_scenarios = all_scenarios[: args.n]

    print(f"Loaded {len(eql_bench_scenarios)} EQL-Bench + {len(ghost_scenarios)} ghost = "
          f"{len(all_scenarios)} total scenarios")

    # --- Dry run mode ---
    if args.dry_run:
        print("\n=== DRY RUN MODE (no models loaded) ===\n")
        print(f"Conditions that would be run:")
        for cid, mname, bgt in _CONDITIONS:
            print(f"  {cid:<16}  method={mname or 'baseline'!r:<12}  budget={bgt:.0%}")

        print(f"\nSample scenarios (first 3):")
        for s in all_scenarios[:3]:
            sid   = s.get("scenario_id", s.get("id", "?"))
            dom   = s.get("domain", "?")
            qual  = s.get("qualifier_type", "?")
            claim = (s.get("uncertain_statement") or s.get("text", ""))[:80]
            print(f"  {sid:<16} domain={dom:<12} type={qual:<20}")
            print(f"    claim: {claim!r}")
            print(f"    value_fragments:     {s.get('value_fragments', [s.get('value_fragment','?')])}")
            print(f"    qualifier_fragments: {s.get('qualifier_fragments', [s.get('qualifier_fragment','?')])}")
            print()

        print("\nRunning dry-run scoring on first 3 scenarios ...")
        dry_method = _DryRunMethod("dry_run", inject_qualifier=False)
        dry_scores = run_condition(dry_method, 1.0, all_scenarios[:3], condition_id="dry_run")
        table_data = {
            "conditions": {"dry_run": aggregate(dry_scores)}
        }
        table_data["conditions"]["dry_run"]["scenarios"] = dry_scores

        print("\nMetric preview (3 scenarios, no model):")
        print(f"  EQLR-Token: {table_data['conditions']['dry_run']['eqlr_token']:.3f}")
        print(f"  FCR:        {table_data['conditions']['dry_run']['fcr']:.3f}")
        print(f"  Mean-QAR:   {table_data['conditions']['dry_run']['mean_qar']:.4f}")

        # Print table header
        fake_results = {"conditions": {}}
        for cid, mn, bg in _CONDITIONS:
            fake_results["conditions"][cid] = {
                "eqlr_token": 0.0, "eqlr_token_ci": [0.0, 0.0],
                "fcr": 0.0, "fcr_ci": [0.0, 0.0],
                "ghost_fcr": 0.0, "ghost_fcr_ci": [0.0, 0.0],
                "mean_qar": -1.0,
            }
        print("\nComparison table template:")
        print(make_comparison_table(fake_results))
        print("\nDry run complete. No models were loaded.")
        return

    # --- Single condition ---
    if args.method or args.all:
        method_name = args.method or "baseline"
        budget      = args.budget
        condition_id = f"{method_name}_{int(budget*100)}" if budget < 1.0 else "baseline"

        try:
            method = _get_method(method_name, budget)
        except ImportError as e:
            print(f"Error loading method: {e}")
            sys.exit(1)

        print(f"\nRunning single condition: {condition_id} on {len(all_scenarios)} scenarios")
        raw_scores = run_condition(method, budget, all_scenarios, condition_id=condition_id, verbose=args.verbose)
        summary    = aggregate(raw_scores)
        print(f"\nResults for {condition_id}:")
        print(f"  n          = {summary['n']}")
        print(f"  EQLR-Token = {summary['eqlr_token']:.3f}  95%CI={summary['eqlr_token_ci']}")
        print(f"  FCR        = {summary['fcr']:.3f}  95%CI={summary['fcr_ci']}")
        print(f"  Ghost-FCR  = {summary.get('ghost_fcr', -1.0):.3f}  95%CI={summary.get('ghost_fcr_ci', [-1,-1])}")
        print(f"  Mean-QAR   = {summary.get('mean_qar', -1.0):.4f}")
        return

    # --- All conditions ---
    if args.all_conditions:
        print(f"\nRunning all {len(_CONDITIONS)} conditions on {len(all_scenarios)} scenarios")
        results = run_all_conditions(all_scenarios, dry_run=False, verbose=args.verbose)

        out_path = args.out
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w") as f:
            # Serialise numpy booleans etc. safely
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

        print("\nComparison table:")
        print(make_comparison_table(results))
        return

    # Default: print help
    parser.print_help()


if __name__ == "__main__":
    main()
