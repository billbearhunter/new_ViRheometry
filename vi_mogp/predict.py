"""Inference wrapper for trained HVIMoGP_rBCM checkpoints.

Loads a model.pt saved by train.py and exposes a physical-units predict
interface matching the existing downstream tools (diagnose_model,
inverse solvers, etc).

Interface:
    predictor = HVIMoGPrBCMPredictor.load("path/to/model.pt")
    y_phys = predictor.predict(
        X_raw=(B, 5),           # [n, eta, sigma_y, W, H] physical
        W=None, H=None,         # optional overrides
        y_obs=None,             # (B, 8) observed y — used for level-2 routing
        return_weights=False,   # if True, also return (expert_ids, weights)
    )

No softmax gate, no ELBO — just:
  Level 1:  (W, H) -> argmax geo group
  Level 2:  phi(y_obs, W, H) -> top-K phi experts within geo
  Combine:  rBCM precision-weighted mean/variance
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import gpytorch

from vi_mogp import config as HC
from vi_mogp.model import HVIMoGP_rBCM
from vi_mogp.data import InputScaler, OutputScaler
from surrogate.features import build_phi


class HVIMoGPrBCMPredictor:
    def __init__(self, model: HVIMoGP_rBCM, xs: InputScaler, ys: OutputScaler,
                 device=HC.DEVICE, dtype=HC.DTYPE):
        self.model = model.eval()
        self.xs, self.ys = xs, ys
        self.device, self.dtype = device, dtype
        # Lazily computed per-expert / per-baseline parameter-space boxes
        # in PHYSICAL units (n, eta, sigma_y). See `expert_boxes_physical`.
        self._expert_boxes: Optional[list] = None
        self._baseline_boxes: Optional[list] = None

    # ══════════════════════════════════════════════════════════════════
    # Factory
    # ══════════════════════════════════════════════════════════════════
    @classmethod
    def load(cls, ckpt_path: str | Path) -> "HVIMoGPrBCMPredictor":
        model = HVIMoGP_rBCM.load(ckpt_path, map_location=HC.DEVICE)
        extra = getattr(model, "_ckpt_extra", {})
        xs = InputScaler.from_dict(extra["xs"])
        ys = OutputScaler.from_dict(extra["ys"])
        return cls(model, xs, ys)

    # ══════════════════════════════════════════════════════════════════
    # Phi feature builder (row-wise)
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _build_phi_batch(y_obs: Optional[np.ndarray],
                         W: np.ndarray, H: np.ndarray, B: int) -> np.ndarray:
        """Build (B, D_PHI) phi features. If y_obs is None, uses zeros
        (dataset-mean after centering) which falls back to the phi-GMM's
        prior weights in routing."""
        if y_obs is None:
            phi = np.stack([build_phi(np.zeros(8, dtype=np.float64),
                                      W[i], H[i])[0]
                            for i in range(B)], axis=0)
        else:
            y_arr = np.asarray(y_obs, dtype=np.float64)
            if y_arr.ndim == 1:
                phi = np.stack([build_phi(y_arr, W[i], H[i])[0]
                                for i in range(B)], axis=0)
            else:
                phi = np.stack([build_phi(y_arr[i], W[i], H[i])[0]
                                for i in range(B)], axis=0)
        return phi

    # ══════════════════════════════════════════════════════════════════
    # Parameter-space boxes (for CMA bound tightening)
    # ══════════════════════════════════════════════════════════════════
    def _boxes_from_modulelist(self, modules) -> list:
        """Inverse-transform stored scaled training inputs → physical
        (n, eta, sigma_y) min/max per expert/baseline. Returns list with
        None for empty clusters."""
        boxes = []
        for mod in modules:
            X_s = getattr(mod, "_X_train", None)
            if X_s is None or X_s.numel() == 0:
                boxes.append(None)
                continue
            X_phys = self.xs.inverse_transform(
                X_s.detach().cpu().numpy().astype(np.float64)
            )
            boxes.append({
                "n":       (float(X_phys[:, 0].min()), float(X_phys[:, 0].max())),
                "eta":     (float(X_phys[:, 1].min()), float(X_phys[:, 1].max())),
                "sigma_y": (float(X_phys[:, 2].min()), float(X_phys[:, 2].max())),
            })
        return boxes

    def expert_boxes_physical(self) -> list:
        """Per-expert (n, eta, sigma_y) training-data boxes, physical units.

        Lazily computed once, cached. `None` for empty experts.
        """
        if self._expert_boxes is None:
            self._expert_boxes = self._boxes_from_modulelist(self.model.experts)
        return self._expert_boxes

    def baseline_boxes_physical(self) -> list:
        if self._baseline_boxes is None:
            self._baseline_boxes = self._boxes_from_modulelist(self.model.baselines)
        return self._baseline_boxes

    @staticmethod
    def _pad_interval(lo: float, hi: float, frac: float,
                      log: bool = False) -> Tuple[float, float]:
        if hi <= lo:
            return lo, hi
        if log and lo > 0 and hi > 0:
            log_span = math.log(hi) - math.log(lo)
            return (math.exp(math.log(lo) - log_span * frac),
                    math.exp(math.log(hi) + log_span * frac))
        span = hi - lo
        return lo - span * frac, hi + span * frac

    def tightened_bounds(self,
                         expert_ids,
                         geo_id: Optional[int],
                         base_bounds: dict,
                         include_baseline: bool = True,
                         pad_frac: float = 0.02) -> dict:
        """Hard union of routed experts' (+ geo-baseline) parameter boxes,
        intersected with `base_bounds`.

        This is the "change 2 hard union" tightening for CMA-ES: for each
        of n / eta / sigma_y we take min-of-lows and max-of-highs across all
        routed experts (slightly padded, then clipped to base_bounds).
        Degenerate results (lo >= hi for any dim) fall back to base_bounds.

        Parameters
        ----------
        expert_ids : Iterable[int]
            Expert indices (global, 0..n_experts-1). Negative/invalid ids
            are silently skipped. Pass the full top-K set you will route
            through at predict time.
        geo_id : Optional[int]
            Geo cluster index for the fallback baseline. If None or out of
            range, the baseline box is not unioned in.
        base_bounds : dict
            {"n":(lo,hi), "eta":(lo,hi), "sigma_y":(lo,hi)} — e.g. the
            project-wide PARAM_BOUNDS. The tightened box is intersected
            with this.
        include_baseline : bool
            If True (default) and `geo_id` is valid, also union in the
            baseline's box — since GRBCM always blends the baseline in.
        pad_frac : float
            Relative pad applied outward before intersection (log-scale
            for eta / sigma_y; linear for n). 0.02 = 2 %.
        """
        expert_boxes = self.expert_boxes_physical()
        active = []
        for eid in expert_ids:
            try:
                eid_i = int(eid)
            except Exception:
                continue
            if 0 <= eid_i < len(expert_boxes) and expert_boxes[eid_i] is not None:
                active.append(expert_boxes[eid_i])
        if include_baseline and geo_id is not None:
            base_boxes = self.baseline_boxes_physical()
            if 0 <= int(geo_id) < len(base_boxes) and base_boxes[int(geo_id)] is not None:
                active.append(base_boxes[int(geo_id)])
        if not active:
            return base_bounds

        lo_n = min(b["n"][0]       for b in active)
        hi_n = max(b["n"][1]       for b in active)
        lo_e = min(b["eta"][0]     for b in active)
        hi_e = max(b["eta"][1]     for b in active)
        lo_s = min(b["sigma_y"][0] for b in active)
        hi_s = max(b["sigma_y"][1] for b in active)

        lo_n, hi_n = self._pad_interval(lo_n, hi_n, pad_frac, log=False)
        lo_e, hi_e = self._pad_interval(lo_e, hi_e, pad_frac, log=True)
        lo_s, hi_s = self._pad_interval(lo_s, hi_s, pad_frac, log=True)

        tight = {
            "n":       (max(lo_n, base_bounds["n"][0]),
                        min(hi_n, base_bounds["n"][1])),
            "eta":     (max(lo_e, base_bounds["eta"][0]),
                        min(hi_e, base_bounds["eta"][1])),
            "sigma_y": (max(lo_s, base_bounds["sigma_y"][0]),
                        min(hi_s, base_bounds["sigma_y"][1])),
        }
        for k, (lo, hi) in tight.items():
            if hi <= lo:
                return base_bounds
        return tight

    def nearest_neighbor_theta(self,
                                expert_id: int,
                                y_obs: np.ndarray,
                                ) -> Optional[np.ndarray]:
        """Find the training point in `expert_id` whose output is closest
        to `y_obs` (in physical units, after monotonicity enforcement) and
        return its (n, eta, sigma_y) — physical — as a length-3 array.

        Returns None if the expert is empty or has no training data.

        Speed optimisation C (inverse warm-start): CMA/grad solvers can
        use this as x0 instead of the bound-midpoint. The nearest-neighbor
        is a crude but data-honest guess — it can't be outside the
        training distribution and is typically within 10–30% of truth,
        letting CMA converge in far fewer iters.
        """
        exp = self.model.experts[int(expert_id)]
        X_s = getattr(exp, "_X_train", None)
        Y_s = getattr(exp, "_Y_train", None)
        if X_s is None or Y_s is None or X_s.numel() == 0:
            return None
        # Physical space conversion (train inputs/outputs are stored in
        # centered/scaled space inside the GP model).
        X_phys = self.xs.inverse_transform(
            X_s.detach().cpu().numpy().astype(np.float64)
        )
        Y_phys = self.ys.inverse_transform(
            Y_s.detach().cpu().numpy().astype(np.float64)
        )
        # Apply the same monotonicity that predict() enforces on outputs
        # so NN distance is measured in the same space as the CMA loss.
        Y_phys = np.maximum.accumulate(Y_phys, axis=-1)
        y_ref = np.maximum.accumulate(np.asarray(y_obs, dtype=np.float64))
        # Normalised L2 in output-space (scale-invariant across samples).
        norm = max(float(np.mean(y_ref ** 2)), 1e-12)
        d2 = np.mean((Y_phys - y_ref[None, :]) ** 2, axis=1) / norm
        i_star = int(np.argmin(d2))
        return X_phys[i_star, :3].astype(np.float64)

    def route_for(self,
                  W: float, H: float,
                  y_obs: Optional[np.ndarray],
                  top_k_phi: int = HC.INFER_TOP_K_PHI,
                  ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Convenience: return (geo_id, expert_ids, weights) for a single
        (W, H, y_obs). Used by inverse solvers to pre-compute the routed
        expert set so they can tighten CMA bounds, reuse the Cholesky
        cache, etc.
        """
        W_arr = np.array([float(W)], dtype=np.float64)
        H_arr = np.array([float(H)], dtype=np.float64)
        phi_arr = self._build_phi_batch(y_obs, W_arr, H_arr, 1)
        geo_ids, eids, wts = self.model.route(
            W_arr, H_arr, phi=phi_arr, top_k_phi=top_k_phi,
        )
        return int(geo_ids[0]), eids[0], wts[0]

    # ══════════════════════════════════════════════════════════════════
    # Predict with a PRE-ROUTED expert set (optimisation A for inverse).
    #
    # Inverse loops (CMA / grad) keep (W, H, y_obs) fixed — so the routed
    # experts don't change across iters. This version skips build_phi +
    # gate-softmax inside the loop (saves ~50 ms / iter and all Python-
    # side overhead) and feeds pre-computed (expert_ids, geo_id) straight
    # to model.predict_grbcm.
    # ══════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def predict_fixed_route(self,
                            X_raw: np.ndarray,           # (B, 5) physical
                            expert_ids_row: np.ndarray,  # (K,) int, fixed
                            geo_id: int,
                            clear_cache: bool = False,
                            use_baseline: bool = True,
                            phi_weights_row: Optional[np.ndarray] = None,  # (K,)
                            ) -> np.ndarray:
        """Batched predict where every row in X_raw shares the same
        routed expert set (eids_row) and geo_id.

        Parameters
        ----------
        use_baseline : bool
            If True (default), use GRBCM aggregation with the geo-baseline
            tie-break term. If False, fall back to rBCM aggregation which
            uses only the local experts' posterior variances — removes the
            baseline-smoothing bias at the cost of less robust extrapolation.
            For top-1 routing, rBCM reduces to the single expert's mean.
        phi_weights_row : Optional[np.ndarray] of shape (K,)
            BGM posterior probabilities p(k|φ,g) for each routed expert.
            If provided, multiplies into the rBCM/GRBCM β_k weights so the
            gate's confidence in each expert is honoured (Step-A fix,
            2026-04-24). If None, falls back to unweighted rBCM/GRBCM.
            Inverse loops should pass `route_for(...)[2]` here.

        Returns monotonised y_phys of shape (B, 8).
        """
        X_arr = np.atleast_2d(np.asarray(X_raw, dtype=np.float64))
        B = X_arr.shape[0]
        K = len(expert_ids_row)
        X_s = self.xs.transform(X_arr)
        X_t = torch.tensor(X_s, dtype=self.dtype, device=self.device)

        eids_B = np.broadcast_to(
            np.asarray(expert_ids_row, dtype=np.int64).reshape(1, K),
            (B, K),
        ).copy()
        gids_B = np.full((B,), int(geo_id), dtype=np.int64)

        # Broadcast phi_weights_row -> (B, K) if provided.
        phi_w_B: Optional[np.ndarray] = None
        if phi_weights_row is not None:
            phi_w_row = np.asarray(phi_weights_row, dtype=np.float64).reshape(K)
            phi_w_B = np.broadcast_to(phi_w_row.reshape(1, K), (B, K)).copy()

        with gpytorch.settings.fast_pred_var():
            if use_baseline:
                mu_c, _var_c = self.model.predict_grbcm(
                    X_t, eids_B, gids_B, clear_cache=clear_cache,
                    phi_weights=phi_w_B,
                )
            else:
                mu_c, _var_c = self.model.predict_rbcm(
                    X_t, eids_B, clear_cache=clear_cache,
                    phi_weights=phi_w_B,
                )
        y_phys = self.ys.inverse_transform(mu_c.cpu().numpy())
        y_phys = np.maximum.accumulate(y_phys, axis=-1)
        return y_phys

    # ══════════════════════════════════════════════════════════════════
    # Predict (physical units in/out)
    # ══════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def predict(self,
                X_raw: np.ndarray | torch.Tensor,
                W: float | np.ndarray | None = None,
                H: float | np.ndarray | None = None,
                y_obs: np.ndarray | None = None,
                top_k_phi: int = HC.INFER_TOP_K_PHI,
                use_baseline: bool = True,
                return_weights: bool = False,
                return_variance: bool = False,
                clear_cache: bool = True,
                use_phi_weights: bool = True,
                ) -> np.ndarray | Tuple[np.ndarray, ...]:
        X_arr = np.atleast_2d(np.asarray(X_raw, dtype=np.float64))
        B = X_arr.shape[0]
        if W is None: W = X_arr[:, 3]
        if H is None: H = X_arr[:, 4]
        W_arr = np.broadcast_to(np.asarray(W, dtype=np.float64), (B,)).copy()
        H_arr = np.broadcast_to(np.asarray(H, dtype=np.float64), (B,)).copy()

        phi_arr = self._build_phi_batch(y_obs, W_arr, H_arr, B)
        X_s = self.xs.transform(X_arr)
        X_t = torch.tensor(X_s, dtype=self.dtype, device=self.device)

        with gpytorch.settings.fast_pred_var():
            mu_c, var_c, eids, wts = self.model.predict(
                X_t, W_arr, H_arr, phi=phi_arr, top_k_phi=top_k_phi,
                use_baseline=use_baseline, clear_cache=clear_cache,
                use_phi_weights=use_phi_weights,
            )
        y_phys = self.ys.inverse_transform(mu_c.cpu().numpy())
        # Change 1: monotonicity along the 8 flow-distance columns, applied
        # in PHYSICAL space (the only space where cummax is semantically
        # sound — see HVIMoGP_rBCM.predict_grbcm's comment for the
        # centered-space pitfall). Training data is 100 % monotonic;
        # GRBCM aggregation can produce ~5 % mild violations that this
        # cheap cummax tidies up to strictly non-decreasing output.
        y_phys = np.maximum.accumulate(y_phys, axis=-1)

        if return_weights and return_variance:
            return y_phys, eids, wts, var_c.cpu().numpy()
        if return_weights:
            return y_phys, eids, wts
        if return_variance:
            return y_phys, var_c.cpu().numpy()
        return y_phys
