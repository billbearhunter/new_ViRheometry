"""
surrogate/gating.py
====================
GMM-based expert selection (gating network) for MoE inference.

Supports three gating modes:
  - phi:          18D observation-space GMM (original)
  - input:        5D parameter-space GMM
  - hierarchical: two-stage (geometry group -> per-group phi-space)

Previously in Optimization/libs/moe_core.py (get_adaptive_weights,
hierarchical_get_weights).
"""

import numpy as np
from .features import build_phi, build_input_features


def get_adaptive_weights(
    gate_dict, phi, strategy="threshold",
    threshold=0.01, topk_hard=None,
    confidence_threshold=0.7, max_experts=5,
):
    """Compute expert IDs and weights from the GMM gate.

    Strategies
    ----------
    topk       : top-k experts with probability-weighted weight
    threshold  : experts whose GMM probability >= threshold
    adaptive   : high-confidence -> 1 expert; medium -> threshold; low -> top-8
    all        : all experts weighted by GMM probability
    """
    gmm    = gate_dict["gmm"]
    scaler = gate_dict.get("scaler")
    phi_in = phi.reshape(1, -1)
    if scaler is not None:
        phi_in = scaler.transform(phi_in)
    probs = gmm.predict_proba(phi_in)[0]

    expert_indices, weights = _apply_strategy(
        probs, strategy, threshold, topk_hard,
        confidence_threshold, max_experts,
    )

    expert_ids = [int(idx) + 1 for idx in expert_indices]
    return expert_ids, np.array(weights, dtype=float)


def hierarchical_get_weights(
    gate_dict, y_obs, W: float, H: float,
    strategy="threshold", threshold=0.01, max_experts=5,
    confidence_threshold=0.7, topk_hard=None,
):
    """Two-stage gating: geometry group -> per-group phi-space expert selection.

    Returns (expert_ids, weights) in the same format as get_adaptive_weights.
    """
    geo_gmm     = gate_dict["geo_gmm"]
    geo_scaler  = gate_dict["geo_scaler"]
    phi_gmms    = gate_dict["phi_gmms"]
    phi_scalers = gate_dict["phi_scalers"]
    k_phi       = gate_dict["k_phi"]  # list (per-group) or int (uniform)

    # Compute per-group offsets
    if isinstance(k_phi, list):
        k_phi_offsets = gate_dict.get("k_phi_offsets")
        if k_phi_offsets is None:
            k_phi_offsets = [0]
            for kp in k_phi[:-1]:
                k_phi_offsets.append(k_phi_offsets[-1] + kp)
    else:
        # Legacy: uniform K_phi for all groups
        k_phi_offsets = None

    # Stage 1: geometry group
    geo_feat = np.array([[W, H]], dtype=np.float64)
    geo_group = int(geo_gmm.predict(geo_scaler.transform(geo_feat))[0])

    # Stage 2: phi-space within this geometry group
    phi = build_phi(y_obs, W, H)
    phi_scaled = phi_scalers[geo_group].transform(phi.reshape(1, -1))
    probs = phi_gmms[geo_group].predict_proba(phi_scaled)[0]

    expert_indices, weights = _apply_strategy(
        probs, strategy, threshold, topk_hard,
        confidence_threshold, max_experts,
    )

    # Convert local indices to global expert IDs (1-indexed)
    if k_phi_offsets is not None:
        offset = k_phi_offsets[geo_group]
    else:
        offset = geo_group * k_phi
    expert_ids = [offset + int(idx) + 1 for idx in expert_indices]
    return expert_ids, np.array(weights, dtype=float)


def _apply_strategy(probs, strategy, threshold, topk_hard,
                    confidence_threshold, max_experts):
    """Shared expert selection logic for all gating modes."""
    if strategy == "all":
        expert_indices = np.arange(len(probs))
        weights = probs / np.sum(probs)

    elif strategy == "topk":
        k = topk_hard if topk_hard is not None else 2
        expert_indices = np.argsort(-probs)[:k]
        top_probs = probs[expert_indices]
        weights = top_probs / np.sum(top_probs)

    elif strategy == "threshold":
        mask = probs >= threshold
        if not np.any(mask):
            mask[np.argmax(probs)] = True
        expert_indices  = np.where(mask)[0]
        filtered_probs  = probs[mask]
        weights         = filtered_probs / np.sum(filtered_probs)
        if len(expert_indices) > max_experts:
            top_idx        = np.argsort(-filtered_probs)[:max_experts]
            expert_indices = expert_indices[top_idx]
            filtered_probs = filtered_probs[top_idx]
            weights        = filtered_probs / np.sum(filtered_probs)

    elif strategy == "adaptive":
        max_prob = np.max(probs)
        if max_prob > confidence_threshold:
            expert_indices = [np.argmax(probs)]
            weights        = [1.0]
        elif max_prob > 0.3:
            mask = probs >= 0.05
            if not np.any(mask):
                mask[np.argmax(probs)] = True
            expert_indices  = np.where(mask)[0]
            filtered_probs  = probs[mask]
            weights         = filtered_probs / np.sum(filtered_probs)
            if len(expert_indices) > max_experts:
                top_idx        = np.argsort(-filtered_probs)[:max_experts]
                expert_indices = expert_indices[top_idx]
                filtered_probs = filtered_probs[top_idx]
                weights        = filtered_probs / np.sum(filtered_probs)
        else:
            expert_indices = np.argsort(-probs)[:8]
            filtered_probs = probs[expert_indices]
            weights        = filtered_probs / np.sum(filtered_probs)

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return expert_indices, weights
