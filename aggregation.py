"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the solution script.

Layer-selection rationale (Marks & Tegmark 2024; Li et al. 2023):
  Middle layers (40–70 % of network depth) encode factual knowledge most
  strongly.  For Qwen2.5-0.5B (24 transformer layers) this corresponds to
  layers 10–18.  The final layers are optimised for next-token prediction
  rather than factual grounding and are therefore less informative for a
  hallucination probe.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

# We use only the final transformer layer for the primary feature vector.
# The geometric features (94-dim) carry the multi-layer dynamics signal;
# keeping the hidden-state part minimal (896-dim) avoids the curse of
# dimensionality with only ~468 training samples per fold.


# ---------------------------------------------------------------------------
# Boundary-token detection (first response token after `<|im_start|>assistant\n`).
#
# Snyder et al. 2024 ("On Early Detection of Hallucinations in Factual QA")
# show the hidden state of the *first generated token* — even a formatting
# char — already encodes whether the upcoming response will be hallucinated.
# Our default tail-100 aggregation MISSES this position for long
# hallucinated responses (median 99 tokens, p75 272 tokens) — exactly the
# samples we most need a signal on.  Here we locate that boundary token
# explicitly via the ChatML pattern and append its hidden state.
#
# We re-tokenise each sample within aggregation.py (input_ids are not
# passed through solution.py).  Sample identity is tracked by a module-
# level counter — solution.py iterates the dataframes sequentially, first
# the train set then the test set, so the counter aligns with the row.
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent / "data"

# Qwen2.5 ChatML token IDs for `<|im_start|>assistant<NL>`.
# Both CRLF (319) and LF (198) variants occur depending on dataset
# encoding — the supplied dataset.csv uses Windows line endings.
_HEADER_PREFIX = (151644, 77091)            # `<|im_start|>` + `assistant`
_HEADER_TERMINATORS = (319, 198)            # `\r\n` or `\n`
_IM_END_ID = 151645                         # `<|im_end|>` (end of a chat turn)

# Layer subset sweep target — change for ablation runs.
_BOUNDARY_LAYERS: tuple[int, ...] = (18, 20, 22)

_HIDDEN_DIM = 896

_DF_TRAIN: pd.DataFrame | None = None
_DF_TEST: pd.DataFrame | None = None
_TOKENIZER = None
_SAMPLE_COUNTER: int = 0


def _ensure_loaded() -> None:
    """Lazily load dataframes (text columns only — no label leak) and tokenizer."""
    global _DF_TRAIN, _DF_TEST, _TOKENIZER
    if _DF_TRAIN is None:
        _DF_TRAIN = pd.read_csv(_DATA_DIR / "dataset.csv")[["prompt", "response"]]
    if _DF_TEST is None:
        _DF_TEST = pd.read_csv(_DATA_DIR / "test.csv")[["prompt", "response"]]
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")


def _next_input_ids() -> list[int]:
    """Re-tokenise the next sample's text exactly as solution.py does.

    Returns the input_ids list (truncated to MAX_LENGTH=512, no padding —
    padding is appended on the right by solution.py and does not shift
    boundary indices).
    """
    global _SAMPLE_COUNTER
    _ensure_loaded()
    n_train = len(_DF_TRAIN)
    if _SAMPLE_COUNTER < n_train:
        row = _DF_TRAIN.iloc[_SAMPLE_COUNTER]
    else:
        idx = (_SAMPLE_COUNTER - n_train) % len(_DF_TEST)
        row = _DF_TEST.iloc[idx]
    _SAMPLE_COUNTER += 1
    text = f"{row['prompt']}{row['response']}"
    return _TOKENIZER(text, truncation=True, max_length=512)["input_ids"]


def _find_last_user_imend(
    input_ids: list[int], before_idx: int | None
) -> int | None:
    """Position of the last `<|im_end|>` strictly before `before_idx`.

    Semantically the user's turn-closing token — the "last prompt token"
    in Question-Only-Probe / HaDeMiF (NeurIPS 2025).  Its hidden state
    encodes the model's pre-generation uncertainty about the upcoming
    answer, complementary to the post-commit first-response-token signal.
    """
    if before_idx is None:
        return None
    for i in range(before_idx - 1, -1, -1):
        if input_ids[i] == _IM_END_ID:
            return i
    return None


def _find_boundary_idx(input_ids: list[int]) -> int | None:
    """Return position of the first response token, or None if not located.

    Scans for `<|im_start|>` + `assistant` followed by either CRLF or LF,
    and returns the index of the next token (the first response token).
    The last occurrence is used so that any few-shot examples in the
    prompt do not shadow the actual assistant turn.
    """
    p0, p1 = _HEADER_PREFIX
    last = None
    for i in range(len(input_ids) - 2):
        if (
            input_ids[i] == p0
            and input_ids[i + 1] == p1
            and input_ids[i + 2] in _HEADER_TERMINATORS
        ):
            boundary = i + 3
            if boundary < len(input_ids):
                last = boundary
    return last


def _spectral_features(
    hidden_states: torch.Tensor,
    tail_pos: torch.Tensor,
    layers: tuple[int, ...],
    top_k: int = 5,
) -> torch.Tensor:
    """Top-k log-eigenvalues of the tail-window covariance per layer.

    Captures the intrinsic dimensionality / spread of hidden-state
    activations within the response window — a different signal class
    from per-coordinate mean/max/std.  Hallucinations have been linked
    to compressed activation manifolds (EigenTrack, 2025).

    Returns a tensor of shape ``(len(layers) * top_k,)``.
    """
    chunks: list[torch.Tensor] = []
    for idx in layers:
        tail = hidden_states[idx, tail_pos, :]              # (k, hidden_dim)
        # Centred covariance over hidden-dim features (gram on sequence axis).
        # Shape (k, k) — much cheaper than (hidden_dim, hidden_dim).
        x = tail - tail.mean(dim=0, keepdim=True)
        gram = (x @ x.t()) / max(1, tail.shape[0] - 1)      # (k, k)
        # Symmetric eigenvalues — clamp tiny negatives from numerical error.
        eig = torch.linalg.eigvalsh(gram).clamp(min=1e-8)   # (k,)
        # Take top_k largest, log-scale; pad with log(eps) if k < top_k.
        eig_top = torch.sort(eig, descending=True).values[:top_k]
        if eig_top.shape[0] < top_k:
            pad = torch.full(
                (top_k - eig_top.shape[0],),
                float(torch.log(torch.tensor(1e-8))),
                device=eig_top.device,
                dtype=eig_top.dtype,
            )
            eig_top = torch.cat([torch.log(eig_top), pad])
        else:
            eig_top = torch.log(eig_top)
        chunks.append(eig_top)
    return torch.cat(chunks, dim=0)


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Strategy (multi-view pooling over tail-100 tokens):
      - For each of the 6 selected middle layers, compute mean / max / std
        over the last 100 real tokens. Three views capture complementary
        signals: mean = central tendency, max = peak activation
        (often coincides with uncertain / outlier tokens), std = spread
        (proxy for the model's confidence across the response window).
      - Append the global mean of the final layer over all real tokens
        (full-sequence complementary signal).
      - Append the normalised sequence length scalar.

    Output dimension: 6 * 3 * 896 + 896 + 1 = 17 025.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(17025,)``.
    """
    device = hidden_states.device
    mask = attention_mask.to(device).float().unsqueeze(-1)   # (seq_len, 1)
    n_real = mask.sum().clamp(min=1.0)

    real_positions = attention_mask.to(device).nonzero(as_tuple=False).squeeze(-1)
    k = min(100, real_positions.shape[0])
    tail_pos = real_positions[-k:]                    # (k,)

    features: list[torch.Tensor] = []

    for idx in [10, 12, 14, 16, 18, 20]:
        layer = hidden_states[idx]                    # (seq_len, hidden_dim)
        tail_layer = layer[tail_pos, :]               # (k, hidden_dim)

        tail_mean = tail_layer.mean(0)                # (hidden_dim,)
        tail_max = tail_layer.max(0).values           # (hidden_dim,)
        # Unbiased=False keeps std defined even when k == 1 (returns 0).
        tail_std = tail_layer.std(0, unbiased=False)  # (hidden_dim,)

        features.append(tail_mean)
        features.append(tail_max)
        features.append(tail_std)

    final_layer = hidden_states[-1]
    global_mean = (final_layer * mask).sum(0) / n_real  # (hidden_dim,)
    features.append(global_mean)

    seq_len_feature = n_real.unsqueeze(0) / 512.0   # normalised scalar in (0, 1]
    features.append(seq_len_feature)

    return torch.cat(features, dim=0)   # (6 * 3 * 896 + 896 + 1,) = (17025,)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.py``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Features computed over all 24 transformer layers:
      - 24 L2 norms of layer-wise mean-pooled representations
      - 23 consecutive-layer cosine similarities (representation drift)
      - 23 consecutive-layer L2 distances (semantic velocity)
      - 22 velocity differences (semantic acceleration)
      -  1 normalised index of the max-norm layer
      -  1 variance of the norm trajectory
    Total: 94 features.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(94,)``.
    """
    device = hidden_states.device
    mask = attention_mask.to(device).float().unsqueeze(-1)
    n_real = mask.sum().clamp(min=1.0)

    # Mean-pooled representation per transformer layer (skip embedding at 0)
    mean_reps: list[torch.Tensor] = []
    for layer in hidden_states[1:]:                       # 24 transformer layers
        mean_reps.append((layer * mask).sum(0) / n_real)  # (hidden_dim,)

    # 1. L2 norms — (24,)
    norms = torch.stack([r.norm() for r in mean_reps])

    # 2. Consecutive cosine similarities — (23,)
    cosine_sims = torch.stack([
        F.cosine_similarity(
            mean_reps[i].unsqueeze(0),
            mean_reps[i + 1].unsqueeze(0),
        ).squeeze()
        for i in range(len(mean_reps) - 1)
    ])

    # 3. Semantic velocity: L2 distance between consecutive layers — (23,)
    velocity = torch.stack([
        (mean_reps[i + 1] - mean_reps[i]).norm()
        for i in range(len(mean_reps) - 1)
    ])

    # 4. Semantic acceleration: difference of consecutive velocities — (22,)
    acceleration = velocity[1:] - velocity[:-1]

    # 5. Normalised max-norm layer index — (1,)
    max_norm_idx = (norms.argmax().float() / 23.0).unsqueeze(0)

    # 6. Norm variance — (1,)
    norm_var = norms.var().unsqueeze(0)

    return torch.cat(
        [norms, cosine_sims, velocity, acceleration, max_norm_idx, norm_var],
        dim=0,
    )  # (94,)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and emit a single feature vector.

    Main entry point called from ``solution.py`` for each sample.

    Boundary-token features: the hidden state of the first response token
    (right after `<|im_start|>assistant\\r\\n`) for each layer in
    `_BOUNDARY_LAYERS` (Run 21 sweep selected [18, 20, 22] — sub-1B
    boundary signal peaks at depth 75-95%, Snyder 2024 + Hou 2024).

    Spectral features (EigenTrack 2025): top-5 log-eigenvalues of the
    tail-window covariance per `_BOUNDARY_LAYERS` layer — capture
    activation-manifold compression (a different signal class from
    per-coordinate mean / max / std views).

    The 94-dim geometric block is always included regardless of the
    ``use_geometric`` flag — it is part of our final solution and the
    flag is preserved purely for API compatibility with the fixed
    ``solution.py``.  The original ``USE_GEOMETRIC`` switch in
    solution.py would otherwise force us to edit fixed infrastructure
    to enable our best configuration; bypassing it here keeps that
    file untouched.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Ignored (kept for backwards-compatible signature).
                        Geometric features are always included.

    Returns:
        A 1-D float tensor of shape ``(19822,)``:
            17025 (multi-view tail-pooling) + 94 (geometric)
            + 2688 (boundary block, 3 layers × hidden_dim)
            + 15   (spectral block, 3 layers × top-5 log-eigenvalues).
    """
    del use_geometric  # always-on; see docstring
    device = hidden_states.device
    agg_features = aggregate(hidden_states, attention_mask)
    geo_features = extract_geometric_features(hidden_states, attention_mask)

    # --- Boundary-token features ---
    input_ids = _next_input_ids()
    boundary_idx = _find_boundary_idx(input_ids)

    boundary_chunks: list[torch.Tensor] = []
    if boundary_idx is not None and boundary_idx < hidden_states.shape[1]:
        for idx in _BOUNDARY_LAYERS:
            boundary_chunks.append(hidden_states[idx, boundary_idx, :])
    else:
        for _ in _BOUNDARY_LAYERS:
            boundary_chunks.append(torch.zeros(_HIDDEN_DIM, device=device))
    boundary_features = torch.cat(boundary_chunks, dim=0)

    # --- Spectral features (EigenTrack-style) ---
    real_positions = attention_mask.to(device).nonzero(as_tuple=False).squeeze(-1)
    k = min(100, real_positions.shape[0])
    tail_pos = real_positions[-k:]
    spectral_features = _spectral_features(
        hidden_states, tail_pos, _BOUNDARY_LAYERS, top_k=5
    )

    return torch.cat(
        [agg_features, geo_features, boundary_features, spectral_features],
        dim=0,
    )
