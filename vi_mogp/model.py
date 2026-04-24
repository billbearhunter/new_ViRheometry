"""Hierarchical MoGP with rBCM aggregation (post-refactor 2026-04-19).

Architecture
------------
  * 540 independent exact-GP experts, one per (g, k) cluster
    - cluster membership is the hard argmax of sklearn BGM fits
      (12 geo × variable K_phi from BIC scan)
    - each expert is trained in isolation via ExactMarginalLogLikelihood
      (standard type-II MLE, no joint ELBO, no gate KL, no hyperprior)
  * Two-level hard routing at test time:
      Level 1: (W, H) -> geo group g  via geo GMM argmax
      Level 2: phi(y_obs, W, H) -> top-K phi experts within g
    (with y_obs == None, level 2 degrades to phi-GMM prior weights)
  * rBCM aggregation over the selected experts (Deisenroth & Ng 2015):
        β_k(x*) = 0.5 * (log σ²_prior,k - log σ²_k(x*))
        σ²_rBCM^{-1}(x*) = Σ_k β_k σ²_k(x*)^{-1}
                          + (1 - Σ_k β_k) σ²_prior^{-1}
        μ_rBCM(x*)       = σ²_rBCM Σ_k β_k σ²_k(x*)^{-1} μ_k(x*)

Why this replaces HVI-MoGP
--------------------------
  * No softmax gate over 540 experts -> cannot collapse
  * No Python for-loop through all experts during training -> each expert
    trains on its own cluster's data only; trivially parallelisable
  * Precision-weighted aggregation mitigates the "top-1 routing is wrong"
    tail pathology of hard-routed baselines

Checkpoint layout
-----------------
A saved model.pt contains:
  - state_dict              : kernel/noise hyperparams of every expert
  - train_X_list            : list of N_clusters tensors (the per-expert X)
  - train_Y_list            : list of N_clusters tensors (the per-expert Y)
  - cluster_sizes           : list[int]
  - geo_of_expert           : list[int]  mapping expert idx -> geo group
  - K_phi_per_geo, K_geo, kernel
  - xs, ys                  : scaler dicts
  - geo_gmm, geo_scaler, phi_gmms, phi_scalers  : hard-routing GMMs
    (stored as pickled sklearn objects — include with torch.save)
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from vi_mogp import config as HC
from vi_mogp.experts import ExactExpert


class HVIMoGP_rBCM(nn.Module):
    """540 independent exact-GP experts + hard 2-level router + rBCM combine."""

    # ══════════════════════════════════════════════════════════════════════
    # Construction
    # ══════════════════════════════════════════════════════════════════════
    def __init__(self,
                 experts: nn.ModuleList,
                 K_phi_per_geo: List[int],
                 kernel_name: str = HC.KERNEL):
        super().__init__()
        self.experts = experts
        self.K_phi_per_geo = list(K_phi_per_geo)
        self.K_geo = len(K_phi_per_geo)
        self.N_EXPERTS = sum(K_phi_per_geo)
        self.kernel_name = kernel_name
        assert len(experts) == self.N_EXPERTS

        # expert index -> geo group id, and (g, local_k) -> expert index
        geo_of, offset_of_geo = [], [0]
        for g, kg in enumerate(K_phi_per_geo):
            geo_of.extend([g] * kg)
            offset_of_geo.append(offset_of_geo[-1] + kg)
        self.register_buffer("geo_of_expert",
                             torch.tensor(geo_of, dtype=torch.long))
        self._offset_of_geo = offset_of_geo  # CPU list

        # GRBCM baselines: 1 exact-GP expert per geo group, trained on a
        # stratified subsample of that geo's data. Populated by
        # fit_baselines(). Empty ModuleList = rBCM fallback (no baselines).
        self.baselines = nn.ModuleList()

        # Null phi-experts: expert indices with too-few pts (N < MIN_CLUSTER_SIZE).
        # These are BUILT (so .experts stays len==N_EXPERTS and routing/offsets
        # stay aligned) but NOT trained, and at predict time they contribute
        # zero weight so the per-geo baseline handles their rows entirely.
        # Populated by from_cluster_assignments().
        self.null_experts: set = set()

        # Per-cluster actual training size after MIN/MAX clamping (len = N_EXPERTS).
        # Useful for downstream analysis + saved in checkpoint.
        self.cluster_sizes: list = [0] * self.N_EXPERTS

        # GMM routers — attached via attach_router() after construction so
        # the model itself stays torch-serialisable.
        self.geo_gmm = None
        self.geo_scaler = None
        self.phi_gmms: List = []
        self.phi_scalers: List = []

    # ══════════════════════════════════════════════════════════════════════
    # Factory: build from hard cluster assignments
    # ══════════════════════════════════════════════════════════════════════
    @classmethod
    def from_cluster_assignments(cls,
                                 X: torch.Tensor,          # (N, D_X) scaled
                                 Y: torch.Tensor,          # (N, D_Y) centred
                                 geo_ids: np.ndarray,      # (N,) in [0, K_geo)
                                 phi_ids: np.ndarray,      # (N,) local id
                                 K_phi_per_geo: List[int],
                                 kernel_name: str = HC.KERNEL,
                                 seed: int = HC.SEED,
                                 ) -> "HVIMoGP_rBCM":
        """Build one ExactExpert per (g, k) cluster from hard assignments.

        Three size regimes:
          * N < MIN_CLUSTER_SIZE : NULL expert — build a placeholder (borrowing
            geo-level data just so ExactGP construction doesn't choke), mark
            it in `null_experts`, skip training, and route around it at
            predict time so the per-geo baseline gets full weight for its rows.
          * MIN_CLUSTER_SIZE ≤ N ≤ EXPERT_MAX_SIZE : use all cluster rows.
          * N > EXPERT_MAX_SIZE  : stratified-random subsample down to
            EXPERT_MAX_SIZE (bounds Cholesky cost; only affects the few
            largest clusters).
        """
        K_geo = len(K_phi_per_geo)
        experts = nn.ModuleList()
        null_experts: set = set()
        actual_sizes: list = []
        capped_experts: set = set()
        rng = np.random.default_rng(seed)

        cap = getattr(HC, "EXPERT_MAX_SIZE", 0) or 0

        eid = 0
        for g in range(K_geo):
            for k in range(K_phi_per_geo[g]):
                mask = (geo_ids == g) & (phi_ids == k)
                n_k = int(mask.sum())
                if n_k < HC.MIN_CLUSTER_SIZE:
                    # NULL expert: build placeholder from any nearby geo rows
                    # (we fall back to the whole geo, or — if even that is
                    # empty — to whatever points exist, to keep the ExactGP
                    # constructor happy).  This placeholder will never be
                    # trained or used at predict time.
                    geo_mask = (geo_ids == g)
                    if int(geo_mask.sum()) >= HC.MIN_CLUSTER_SIZE:
                        idx = np.where(geo_mask)[0]
                    elif n_k > 0:
                        idx = np.where(mask)[0]
                    else:
                        # extreme edge: empty geo (shouldn't happen) — fall
                        # back to random N rows from the full dataset
                        idx = rng.choice(X.shape[0],
                                         size=min(HC.MIN_CLUSTER_SIZE,
                                                  X.shape[0]),
                                         replace=False)
                    # tiny placeholder — clamp to avoid wasting memory
                    take = min(len(idx), HC.MIN_CLUSTER_SIZE)
                    if len(idx) > take:
                        idx = rng.choice(idx, size=take, replace=False)
                    experts.append(ExactExpert(X[idx], Y[idx],
                                               kernel_name=kernel_name))
                    null_experts.add(eid)
                    actual_sizes.append(n_k)  # the REAL cluster size, not the placeholder
                else:
                    idx = np.where(mask)[0]
                    if cap and n_k > cap:
                        idx = rng.choice(idx, size=cap, replace=False)
                        capped_experts.add(eid)
                    experts.append(ExactExpert(X[idx], Y[idx],
                                               kernel_name=kernel_name))
                    actual_sizes.append(int(len(idx)))
                eid += 1

        inst = cls(experts, K_phi_per_geo, kernel_name=kernel_name)
        inst.null_experts = null_experts
        inst.cluster_sizes = actual_sizes
        inst._capped_experts = capped_experts
        return inst

    def attach_router(self, geo_gmm, geo_scaler,
                      phi_gmms: List, phi_scalers: List):
        """Attach sklearn GMMs for the 2-level hard router."""
        assert len(phi_gmms) == self.K_geo
        self.geo_gmm = geo_gmm
        self.geo_scaler = geo_scaler
        self.phi_gmms = list(phi_gmms)
        self.phi_scalers = list(phi_scalers)

    # ══════════════════════════════════════════════════════════════════════
    # Training — fit each expert independently
    # ══════════════════════════════════════════════════════════════════════
    def fit_all(self, n_iters: int = HC.EXPERT_N_ITERS,
                lr: float = HC.EXPERT_LR,
                verbose_every: int = HC.EXPERT_LOG_EVERY,
                experts_dir: Optional[Path] = None,
                ) -> List[float]:
        """Train every phi expert independently (sequential loop).

        If `experts_dir` is given, after training each expert its state is
        dumped to `experts_dir/expert_{EID:04d}.pt` (state_dict + train data
        + kernel info) before moving on. On restart, existing expert files
        are skipped so training resumes near where it crashed — this is the
        same per-expert checkpointing pattern the old MoE workspace used.
        """
        losses = []
        import time
        t0 = time.time()
        n_null = len(self.null_experts)
        if n_null:
            print(f"  skipping {n_null} null phi-experts (N<{HC.MIN_CLUSTER_SIZE}); "
                  f"their rows will be served by the per-geo baseline.",
                  flush=True)

        n_resumed = 0
        if experts_dir is not None:
            experts_dir = Path(experts_dir)
            experts_dir.mkdir(parents=True, exist_ok=True)

        for j, expert in enumerate(self.experts):
            if j in self.null_experts:
                losses.append(float("nan"))
                continue

            ckpt = (experts_dir / f"expert_{j:04d}.pt") if experts_dir else None

            # Resume: if a checkpoint exists, load it and skip training.
            if ckpt is not None and ckpt.exists():
                try:
                    blob = torch.load(ckpt, map_location=HC.DEVICE,
                                      weights_only=False)
                    expert.set_train_data(blob["X_train"], blob["Y_train"])
                    expert.load_state_dict(blob["state_dict"], strict=False)
                    losses.append(float(blob.get("loss", float("nan"))))
                    n_resumed += 1
                    continue
                except Exception as e:
                    print(f"  [WARN] resume failed for expert {j}: "
                          f"{type(e).__name__}: {e}  — refitting",
                          flush=True)

            loss = expert.fit(n_iters=n_iters, lr=lr, verbose=False)
            losses.append(loss)

            # Per-expert save (crash-resistant)
            if ckpt is not None:
                try:
                    torch.save(dict(
                        state_dict=expert.state_dict(),
                        X_train=expert._X_train.detach().cpu(),
                        Y_train=expert._Y_train.detach().cpu(),
                        kernel_name=expert.kernel_name,
                        loss=float(loss),
                        expert_id=int(j),
                    ), ckpt)
                except Exception as e:
                    print(f"  [WARN] save failed for expert {j}: "
                          f"{type(e).__name__}: {e}", flush=True)

            if (j + 1) % verbose_every == 0 or j == len(self.experts) - 1:
                elapsed = time.time() - t0
                n_trained_so_far = (j + 1) - n_null - n_resumed
                rate = n_trained_so_far / max(elapsed, 1e-3)
                n_left = len(self.experts) - j - 1
                eta = n_left / max(rate, 1e-3)
                print(f"  [{j+1:4d}/{len(self.experts)}]  "
                      f"last neg_mll={loss:.4f}  "
                      f"{rate:.2f} experts/s  ETA {eta:.0f}s  "
                      f"(resumed={n_resumed})",
                      flush=True)

        if n_resumed:
            print(f"  resumed {n_resumed} experts from disk (no retraining)",
                  flush=True)
        return losses

    # ══════════════════════════════════════════════════════════════════════
    # GRBCM baselines — one exact-GP per geo, trained on stratified subsample
    # ══════════════════════════════════════════════════════════════════════
    def fit_baselines(self,
                      X: torch.Tensor, Y: torch.Tensor,
                      geo_ids: np.ndarray,
                      subsample_n: int = HC.BASELINE_SUBSAMPLE,
                      n_iters: int = HC.BASELINE_N_ITERS,
                      lr: float = HC.BASELINE_LR,
                      phi_ids: Optional[np.ndarray] = None,
                      seed: int = HC.SEED,
                      baselines_dir: Optional[Path] = None,
                      ) -> List[float]:
        """Fit K_geo baseline experts, each on a stratified subsample of
        its geo group's data.

        If `phi_ids` is given, the subsample is stratified across phi
        clusters within the geo (proportional allocation) — gives the
        baseline a balanced view of cluster boundaries. Otherwise pure
        random sample within geo.
        """
        rng = np.random.default_rng(seed)
        self.baselines = nn.ModuleList()
        losses: List[float] = []
        import time
        t0 = time.time()

        if baselines_dir is not None:
            baselines_dir = Path(baselines_dir)
            baselines_dir.mkdir(parents=True, exist_ok=True)

        for g in range(self.K_geo):
            geo_mask = (geo_ids == g)
            idx_in_geo = np.where(geo_mask)[0]
            N_g = len(idx_in_geo)
            take_n = min(subsample_n, N_g)

            if phi_ids is not None and N_g > 0:
                # Stratified across phi clusters: proportional allocation,
                # with at least 1 sample per non-empty cluster so every
                # boundary has some representation.
                sub_idx = []
                phi_in_geo = phi_ids[idx_in_geo]
                uniq = np.unique(phi_in_geo)
                per_k_quota = max(1, take_n // max(len(uniq), 1))
                remaining = take_n
                for k in uniq:
                    idx_k = idx_in_geo[phi_in_geo == k]
                    take_k = min(per_k_quota, len(idx_k), remaining)
                    if take_k > 0:
                        choice = rng.choice(idx_k, size=take_k, replace=False)
                        sub_idx.append(choice)
                        remaining -= take_k
                # top up remainder with uniform random across geo
                if remaining > 0:
                    already = np.concatenate(sub_idx) if sub_idx \
                              else np.array([], dtype=np.int64)
                    leftover = np.setdiff1d(idx_in_geo, already,
                                            assume_unique=False)
                    if len(leftover) > 0:
                        extra = rng.choice(leftover,
                                           size=min(remaining, len(leftover)),
                                           replace=False)
                        sub_idx.append(extra)
                sub_idx = np.concatenate(sub_idx)
            else:
                sub_idx = rng.choice(idx_in_geo, size=take_n, replace=False)

            X_g = X[torch.as_tensor(sub_idx, dtype=torch.long)]
            Y_g = Y[torch.as_tensor(sub_idx, dtype=torch.long)]
            baseline = ExactExpert(X_g, Y_g, kernel_name=self.kernel_name)

            ckpt = (baselines_dir / f"baseline_{g:02d}.pt") \
                   if baselines_dir else None
            resumed = False
            if ckpt is not None and ckpt.exists():
                try:
                    blob = torch.load(ckpt, map_location=HC.DEVICE,
                                      weights_only=False)
                    baseline.set_train_data(blob["X_train"], blob["Y_train"])
                    baseline.load_state_dict(blob["state_dict"], strict=False)
                    loss = float(blob.get("loss", float("nan")))
                    resumed = True
                except Exception as e:
                    print(f"  [WARN] resume failed for baseline {g}: "
                          f"{type(e).__name__}: {e}  — refitting",
                          flush=True)

            if not resumed:
                loss = baseline.fit(n_iters=n_iters, lr=lr, verbose=False)
                if ckpt is not None:
                    try:
                        torch.save(dict(
                            state_dict=baseline.state_dict(),
                            X_train=baseline._X_train.detach().cpu(),
                            Y_train=baseline._Y_train.detach().cpu(),
                            kernel_name=baseline.kernel_name,
                            loss=float(loss),
                            geo_id=int(g),
                        ), ckpt)
                    except Exception as e:
                        print(f"  [WARN] save failed for baseline {g}: "
                              f"{type(e).__name__}: {e}", flush=True)

            self.baselines.append(baseline)
            losses.append(loss)
            elapsed = time.time() - t0
            rate = (g + 1) / max(elapsed, 1e-3)
            tag = " [resumed]" if resumed else ""
            print(f"  [geo {g+1:2d}/{self.K_geo}]  N_sub={len(sub_idx):4d}/{N_g:6d}  "
                  f"neg_mll={loss:.3f}  {rate:.2f} geos/s{tag}",
                  flush=True)
        return losses

    # ══════════════════════════════════════════════════════════════════════
    # Routing
    # ══════════════════════════════════════════════════════════════════════
    def route(self,
              W: np.ndarray, H: np.ndarray,
              phi: Optional[np.ndarray] = None,
              top_k_phi: int = HC.INFER_TOP_K_PHI,
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Hard route each row to a geo group, then top-K phi experts.

        Returns:
            geo_ids   : (B,)           argmax geo group
            expert_ids: (B, top_k_phi) global expert indices
            weights   : (B, top_k_phi) phi-GMM posterior probs (sum<=1)
        """
        assert self.geo_gmm is not None, "call attach_router() first"
        B = len(W)
        wh = np.stack([W, H], axis=-1).astype(np.float64)
        geo_ids = self.geo_gmm.predict(self.geo_scaler.transform(wh))  # (B,)

        expert_ids = np.full((B, top_k_phi), -1, dtype=np.int64)
        weights    = np.zeros((B, top_k_phi), dtype=np.float64)

        for g in range(self.K_geo):
            mask = (geo_ids == g)
            if not mask.any():
                continue
            offset = self._offset_of_geo[g]
            K_in_g = self.K_phi_per_geo[g]
            k_top = min(top_k_phi, K_in_g)

            if phi is None:
                # use phi GMM prior weights — same for every row in this geo
                priors = self.phi_gmms[g].weights_  # (K_in_g,)
                order = np.argsort(-priors)[:k_top]
                expert_ids[mask, :k_top] = offset + order[None, :]
                weights[mask, :k_top] = priors[order][None, :]
            else:
                phi_g = phi[mask]
                sc = self.phi_scalers[g]
                probs = self.phi_gmms[g].predict_proba(sc.transform(phi_g))
                # top-K per row
                top_idx = np.argsort(-probs, axis=1)[:, :k_top]         # (n,k)
                top_p   = np.take_along_axis(probs, top_idx, axis=1)
                expert_ids[mask, :k_top] = offset + top_idx
                weights[mask, :k_top]    = top_p
        return geo_ids, expert_ids, weights

    # ══════════════════════════════════════════════════════════════════════
    # Prediction — rBCM aggregation
    # ══════════════════════════════════════════════════════════════════════
    # NOTE: No @torch.no_grad() — Change 4 gradient inverse reuses this.
    # All production callers (HVIMoGPrBCMPredictor.predict) wrap with their
    # own @torch.no_grad context.
    def predict_rbcm(self,
                     X: torch.Tensor,         # (B, D_X) scaled
                     expert_ids: np.ndarray,  # (B, K) int64, -1 for pad
                     clear_cache: bool = True,
                     phi_weights: Optional[np.ndarray] = None,  # (B, K) BGM posterior
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """rBCM-aggregate predictions over the per-row selected experts.

        For efficiency, bucket rows by expert: for each unique expert j we
        push exactly the rows that selected j once through expert j's GP,
        then scatter (mean, var) back. This is O(N_unique_experts) forward
        passes instead of O(B*K).

        Step-A fix (2026-04-24): if `phi_weights` is provided, multiply
        the BGM posterior p(k|φ,g) into β_k before aggregation. The
        original rBCM (Deisenroth & Ng 2015) weights only by GP info
        gain (½ log σ²_prior/σ²_k), which over-trusts low-gate-prob
        experts that happen to be uncertain — a misroute pathology
        observed on the 5 real materials. With phi_weights provided:
            β_k_used = β_k * p(k|φ,g)
        When phi_weights=None (default), preserves backward-compat
        behavior of the unweighted rBCM.

        Returns (μ, σ²) each of shape (B, D_Y).
        """
        B, K = expert_ids.shape
        D_Y = self.experts[0].D_Y
        device = X.device
        dtype  = X.dtype

        # Per (row, slot) storage of expert GP output
        mu_slot  = torch.full((B, K, D_Y), float("nan"),
                              dtype=dtype, device=device)
        var_slot = torch.full((B, K, D_Y), float("nan"),
                              dtype=dtype, device=device)
        # Per-expert prior variance (constant in x), needed for rBCM's β_k
        prior_var = torch.full((B, K, D_Y), float("nan"),
                               dtype=dtype, device=device)

        # Bucket rows by (expert_id, slot_index) — actually simpler to
        # bucket by expert_id and track which (row, slot) wanted it.
        unique_experts = np.unique(expert_ids)
        unique_experts = unique_experts[unique_experts >= 0]
        # Null-expert mask applied to slot validity (zero out contribution)
        null_slot = np.zeros_like(expert_ids, dtype=bool)
        for j in unique_experts:
            j = int(j)
            if j in self.null_experts:
                null_slot |= (expert_ids == j)
                continue
            where = np.where(expert_ids == j)  # (rows_idx, slot_idx)
            rows = where[0]
            slots = where[1]
            if len(rows) == 0:
                continue
            # Predict expert j on these rows' X
            X_sub = X[torch.as_tensor(rows, device=device, dtype=torch.long)]
            mu_j, var_j = self.experts[j].predict(X_sub)  # (n, D_Y)
            prior_j = self.experts[j].prior_variance()     # (D_Y,)
            mu_slot [rows, slots] = mu_j
            var_slot[rows, slots] = var_j
            prior_var[rows, slots] = prior_j[None, :].expand_as(mu_j)
            # Free gpytorch's per-expert prediction_strategy cache so GPU
            # memory doesn't grow O(#experts × N_train²) across batches.
            # Callers in tight loops (e.g. CMA-ES inverse) where the same
            # experts are reused can pass clear_cache=False to keep the
            # Cholesky cached between calls (40× speedup).
            if clear_cache:
                self.experts[j].clear_prediction_cache()

        # rBCM aggregation over slots
        valid_np = (expert_ids >= 0) & (~null_slot)
        valid = torch.from_numpy(valid_np).to(device)                  # (B,K)
        valid_f = valid.to(dtype).unsqueeze(-1)                       # (B,K,1)
        # Replace NaN placeholders in null/unvisited slots with a finite
        # dummy so subsequent log/clamp ops don't propagate NaN (valid_f
        # zeroes their contribution anyway).
        var_slot  = torch.nan_to_num(var_slot,  nan=1.0)
        mu_slot   = torch.nan_to_num(mu_slot,   nan=0.0)
        prior_var = torch.nan_to_num(prior_var, nan=1.0)
        # β_k = 0.5 * (log σ²_prior - log σ²_k)   clipped at floor
        eps = 1e-10
        log_prior = torch.log(prior_var.clamp_min(eps))
        log_post  = torch.log(var_slot.clamp_min(eps))
        beta = 0.5 * (log_prior - log_post)                           # (B,K,D)
        beta = beta.clamp_min(HC.RBCM_BETA_FLOOR)
        # mask out invalid slots
        beta = beta * valid_f
        # Step-A fix: BGM posterior weighting (if provided).
        if phi_weights is not None:
            w = torch.as_tensor(np.asarray(phi_weights, dtype=np.float64),
                                dtype=dtype, device=device)            # (B,K)
            # Defensive: clip negatives that can come from numeric BGM
            # roundoff, then broadcast over the D_Y dim.
            w = w.clamp_min(0.0).unsqueeze(-1)                         # (B,K,1)
            beta = beta * w
        inv_var_k = 1.0 / var_slot.clamp_min(eps)
        inv_var_k = inv_var_k * valid_f
        # shared prior var for the (1 - Σβ) term: mean over valid slots
        sum_valid = valid_f.sum(dim=1).clamp_min(1.0)                 # (B,1)
        prior_shared = (prior_var * valid_f).sum(dim=1) / sum_valid    # (B,D)
        inv_prior_shared = 1.0 / prior_shared.clamp_min(eps)           # (B,D)

        beta_sum = beta.sum(dim=1)                                     # (B,D)
        precision = (beta * inv_var_k).sum(dim=1) \
                    + (1.0 - beta_sum) * inv_prior_shared              # (B,D)
        precision = precision.clamp_min(eps)
        var_rbcm = 1.0 / precision                                     # (B,D)
        mu_rbcm = var_rbcm * (beta * inv_var_k * mu_slot).sum(dim=1)   # (B,D)
        # NOTE: Change-1 monotonicity enforcement now lives in PHYSICAL
        # space (at the HVIMoGPrBCMPredictor level + gradient-inverse loop),
        # NOT here. See detailed comment in predict_grbcm below.
        return mu_rbcm, var_rbcm

    # ══════════════════════════════════════════════════════════════════════
    # GRBCM aggregation (preferred when baselines are trained)
    # ══════════════════════════════════════════════════════════════════════
    # NOTE: No @torch.no_grad() — see predict_rbcm note.
    def predict_grbcm(self,
                      X: torch.Tensor,         # (B, D_X) scaled
                      expert_ids: np.ndarray,  # (B, K) int64, -1 for pad
                      geo_ids: np.ndarray,     # (B,) int64, geo of each row
                      clear_cache: bool = True,
                      phi_weights: Optional[np.ndarray] = None,  # (B, K) BGM posterior
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GRBCM-aggregate predictions (Liu et al. 2018).

        For each test row:
          - baseline expert = self.baselines[geo_id]  (trained on geo
            stratified subsample; provides σ_b², μ_b as reference)
          - local experts   = top-K phi experts within geo

        Formula:
            β_k = 0.5 * (log σ_b² - log σ_k²)            (clipped ≥ 0)
            σ_GRBCM^{-2} = Σ_k β_k σ_k^{-2}
                           + (1 - Σ_k β_k) σ_b^{-2}
            μ_GRBCM      = σ_GRBCM^2 [Σ_k β_k σ_k^{-2} μ_k
                                       + (1 - Σ_k β_k) σ_b^{-2} μ_b]

        Returns (μ, σ²) each of shape (B, D_Y).
        """
        assert len(self.baselines) == self.K_geo, \
            "baselines not trained; call fit_baselines() first or use predict_rbcm"
        B, K = expert_ids.shape
        D_Y = self.experts[0].D_Y
        device = X.device
        dtype  = X.dtype
        eps = 1e-10

        # ─── 1. Local experts' (μ_k, σ²_k) per (row, slot) ───────────────────
        mu_slot  = torch.full((B, K, D_Y), float("nan"),
                              dtype=dtype, device=device)
        var_slot = torch.full((B, K, D_Y), float("nan"),
                              dtype=dtype, device=device)
        unique_experts = np.unique(expert_ids)
        unique_experts = unique_experts[unique_experts >= 0]
        null_slot = np.zeros_like(expert_ids, dtype=bool)
        for j in unique_experts:
            j = int(j)
            if j in self.null_experts:
                null_slot |= (expert_ids == j)
                continue
            where = np.where(expert_ids == j)
            rows, slots = where[0], where[1]
            X_sub = X[torch.as_tensor(rows, device=device, dtype=torch.long)]
            mu_j, var_j = self.experts[j].predict(X_sub)
            mu_slot[rows, slots]  = mu_j
            var_slot[rows, slots] = var_j
            if clear_cache:
                self.experts[j].clear_prediction_cache()

        # ─── 2. Baseline (μ_b, σ²_b) bucketed by geo ─────────────────────────
        mu_b  = torch.full((B, D_Y), float("nan"), dtype=dtype, device=device)
        var_b = torch.full((B, D_Y), float("nan"), dtype=dtype, device=device)
        for g in range(self.K_geo):
            mask = (geo_ids == g)
            if not mask.any():
                continue
            rows = np.where(mask)[0]
            X_g = X[torch.as_tensor(rows, device=device, dtype=torch.long)]
            mu_g, var_g = self.baselines[g].predict(X_g)
            mu_b[rows]  = mu_g
            var_b[rows] = var_g
            if clear_cache:
                self.baselines[g].clear_prediction_cache()

        # ─── 3. GRBCM aggregation ────────────────────────────────────────────
        valid_np = (expert_ids >= 0) & (~null_slot)
        valid = torch.from_numpy(valid_np).to(device)                     # (B,K)
        valid_f = valid.to(dtype).unsqueeze(-1)                           # (B,K,1)

        # Null/unvisited slots → finite dummies (valid_f masks them to zero).
        var_slot = torch.nan_to_num(var_slot, nan=1.0)
        mu_slot  = torch.nan_to_num(mu_slot,  nan=0.0)

        # β_k = 0.5 * (log σ_b² - log σ_k²)
        log_var_b = torch.log(var_b.clamp_min(eps)).unsqueeze(1)          # (B,1,D)
        log_var_k = torch.log(var_slot.clamp_min(eps))                    # (B,K,D)
        beta = 0.5 * (log_var_b - log_var_k)
        beta = beta.clamp_min(HC.RBCM_BETA_FLOOR) * valid_f               # (B,K,D)
        # Step-A fix: BGM posterior weighting (if provided). See predict_rbcm
        # docstring for rationale.
        if phi_weights is not None:
            w = torch.as_tensor(np.asarray(phi_weights, dtype=np.float64),
                                dtype=dtype, device=device)
            w = w.clamp_min(0.0).unsqueeze(-1)                            # (B,K,1)
            beta = beta * w

        inv_var_k = (1.0 / var_slot.clamp_min(eps)) * valid_f             # (B,K,D)
        inv_var_b = 1.0 / var_b.clamp_min(eps)                            # (B,D)

        beta_sum = beta.sum(dim=1)                                        # (B,D)
        # For stability: clamp (1 - Σβ) so the baseline always contributes
        # non-negatively (the theoretical range allows it to go slightly
        # negative but that's a numerical artefact for us).
        w_baseline = (1.0 - beta_sum).clamp_min(eps)                      # (B,D)

        precision = (beta * inv_var_k).sum(dim=1) + w_baseline * inv_var_b
        precision = precision.clamp_min(eps)
        var_out = 1.0 / precision
        mu_out  = var_out * ((beta * inv_var_k * mu_slot).sum(dim=1)
                             + w_baseline * inv_var_b * mu_b)
        # NOTE: Change-1 monotonicity (cummax along the 8 columns) is
        # deliberately NOT applied here. The OutputScaler subtracts a
        # per-column dataset mean, so in this centered space a slow-flow
        # sample (raw below the dataset mean) has centered values that
        # are monotonically DECREASING even though physical flow is
        # monotonically increasing. cummax on a decreasing centered row
        # collapses every column to the first, producing catastrophic
        # NMSE (e.g. sample 909 would drop flow-curve accuracy by 5
        # orders of magnitude). Monotonicity is safely applied in
        # PHYSICAL space — see HVIMoGPrBCMPredictor.predict (after
        # inverse_transform) and invert_rbcm_grad (where it is added to
        # y_pred_phys before loss).
        return mu_out, var_out

    # ══════════════════════════════════════════════════════════════════════
    # Full predict (scaled-inputs interface — outer code handles scaling)
    # ══════════════════════════════════════════════════════════════════════
    # NOTE: No @torch.no_grad() — see predict_rbcm note.
    def predict(self,
                X: torch.Tensor,                  # (B, D_X) scaled
                W: np.ndarray, H: np.ndarray,     # (B,) raw
                phi: Optional[np.ndarray] = None, # (B, D_PHI) raw
                top_k_phi: int = HC.INFER_TOP_K_PHI,
                use_baseline: bool = True,
                clear_cache: bool = True,
                use_phi_weights: bool = True,
                ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """One-shot predict: route -> GRBCM (or rBCM fallback) aggregate.

        If baselines are trained (len(self.baselines) == K_geo) AND
        use_baseline=True, uses GRBCM. Setting use_baseline=False forces
        rBCM (phi-experts only, no baseline mixing) — useful to diagnose
        baseline drag on predictions.

        Returns (μ, σ², expert_ids, weights).
        """
        geo_ids, expert_ids, weights = self.route(W, H, phi,
                                                  top_k_phi=top_k_phi)
        # Step-A fix: pass BGM posterior into rBCM/GRBCM unless caller
        # explicitly opts out (legacy diagnostics may want unweighted).
        phi_w = weights if use_phi_weights else None
        if use_baseline and len(self.baselines) == self.K_geo:
            mu, var = self.predict_grbcm(X, expert_ids, geo_ids,
                                         clear_cache=clear_cache,
                                         phi_weights=phi_w)
        else:
            mu, var = self.predict_rbcm(X, expert_ids,
                                        clear_cache=clear_cache,
                                        phi_weights=phi_w)
        return mu, var, expert_ids, weights

    # ══════════════════════════════════════════════════════════════════════
    # Persistence
    # ══════════════════════════════════════════════════════════════════════
    def save(self, path: str | Path, extra: Optional[dict] = None):
        """Save complete model state (phi experts + baselines + GMMs)."""
        train_X_list = [exp._X_train.detach().cpu() for exp in self.experts]
        train_Y_list = [exp._Y_train.detach().cpu() for exp in self.experts]
        baseline_X = [b._X_train.detach().cpu() for b in self.baselines]
        baseline_Y = [b._Y_train.detach().cpu() for b in self.baselines]

        # Change 3: persist per-expert / per-baseline poly-residual corrections
        # if they've been fitted. None entries for unfitted experts.
        def _poly_tuple(mod):
            if (getattr(mod, "poly_powers", None) is None
                    or getattr(mod, "poly_coef", None) is None):
                return None
            return (mod.poly_powers.detach().cpu(),
                    mod.poly_coef.detach().cpu())
        expert_poly   = [_poly_tuple(exp) for exp in self.experts]
        baseline_poly = [_poly_tuple(b)   for b   in self.baselines]

        blob = dict(
            state_dict=self.state_dict(),
            train_X_list=train_X_list,
            train_Y_list=train_Y_list,
            baseline_X_list=baseline_X,
            baseline_Y_list=baseline_Y,
            expert_poly=expert_poly,
            baseline_poly=baseline_poly,
            cluster_sizes=[int(x.shape[0]) for x in train_X_list],
            baseline_sizes=[int(x.shape[0]) for x in baseline_X],
            null_experts=sorted(int(x) for x in self.null_experts),
            cluster_sizes_true=list(self.cluster_sizes),
            capped_experts=sorted(int(x) for x in
                                  getattr(self, "_capped_experts", set())),
            geo_of_expert=self.geo_of_expert.cpu().tolist(),
            K_phi_per_geo=self.K_phi_per_geo,
            K_geo=self.K_geo,
            kernel=self.kernel_name,
            # Per-expert kernel names: enables mixed-kernel checkpoints where
            # a hotfix has retrained a subset with a different kernel.
            # load() will prefer this list over the single `kernel` field.
            per_expert_kernel_names=[exp.kernel_name for exp in self.experts],
            per_baseline_kernel_names=[b.kernel_name for b in self.baselines],
            geo_gmm=self.geo_gmm,
            geo_scaler=self.geo_scaler,
            phi_gmms=self.phi_gmms,
            phi_scalers=self.phi_scalers,
        )
        if extra:
            blob.update(extra)
        torch.save(blob, path)

    @classmethod
    def load(cls, path: str | Path,
             map_location=HC.DEVICE) -> "HVIMoGP_rBCM":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        K_phi_per_geo = blob["K_phi_per_geo"]
        kernel = blob.get("kernel", HC.KERNEL)
        train_X = blob["train_X_list"]
        train_Y = blob["train_Y_list"]

        # Per-expert kernel names (mixed-kernel checkpoints from hotfix).
        # Fallback to single `kernel` for legacy checkpoints.
        per_expert_kernels = blob.get("per_expert_kernel_names", None)

        # Rebuild phi experts
        experts = nn.ModuleList()
        for j in range(sum(K_phi_per_geo)):
            kn = per_expert_kernels[j] if per_expert_kernels is not None else kernel
            experts.append(ExactExpert(train_X[j], train_Y[j],
                                       kernel_name=kn))
        model = cls(experts, K_phi_per_geo, kernel_name=kernel)

        # Restore null-expert set + true cluster sizes (may be absent in
        # pre-null-routing checkpoints).
        model.null_experts = set(blob.get("null_experts", []))
        model.cluster_sizes = list(blob.get("cluster_sizes_true",
                                            blob.get("cluster_sizes",
                                                     [0] * sum(K_phi_per_geo))))
        model._capped_experts = set(blob.get("capped_experts", []))

        # Rebuild baselines (may be absent in pre-GRBCM checkpoints)
        baseline_X = blob.get("baseline_X_list", [])
        baseline_Y = blob.get("baseline_Y_list", [])
        per_baseline_kernels = blob.get("per_baseline_kernel_names", None)
        for g, (X_b, Y_b) in enumerate(zip(baseline_X, baseline_Y)):
            kn = per_baseline_kernels[g] if per_baseline_kernels is not None else kernel
            model.baselines.append(ExactExpert(X_b, Y_b, kernel_name=kn))

        # Load kernel/noise hyperparams
        missing, unexpected = model.load_state_dict(blob["state_dict"],
                                                    strict=False)
        bad_missing = [k for k in missing if "_X_train" not in k
                       and "_Y_train" not in k]
        if bad_missing:
            print(f"[HVIMoGP_rBCM.load] WARN missing: {bad_missing[:3]}")
        if unexpected:
            print(f"[HVIMoGP_rBCM.load] WARN unexpected: {unexpected[:3]}")

        # Attach GMMs
        model.attach_router(blob["geo_gmm"], blob["geo_scaler"],
                            blob["phi_gmms"], blob["phi_scalers"])

        # Re-attach train data (state_dict load doesn't restore
        # gpytorch's train_inputs because they're not buffers)
        for j, exp in enumerate(model.experts):
            exp.set_train_data(train_X[j], train_Y[j])
        for g, b in enumerate(model.baselines):
            b.set_train_data(baseline_X[g], baseline_Y[g])

        # Change 3: restore poly-residual corrections if present. Stored
        # outside state_dict so missing-in-checkpoint just means "no
        # correction fitted yet" and loads cleanly on legacy models.
        expert_poly   = blob.get("expert_poly",   None)
        baseline_poly = blob.get("baseline_poly", None)
        if expert_poly is not None:
            for j, tup in enumerate(expert_poly):
                if tup is None:
                    continue
                pp, pc = tup
                model.experts[j].poly_powers = pp.to(HC.DEVICE)
                model.experts[j].poly_coef   = pc.to(HC.DEVICE).to(HC.DTYPE)
        if baseline_poly is not None:
            for g, tup in enumerate(baseline_poly):
                if tup is None:
                    continue
                pp, pc = tup
                model.baselines[g].poly_powers = pp.to(HC.DEVICE)
                model.baselines[g].poly_coef   = pc.to(HC.DEVICE).to(HC.DTYPE)

        # Stash aux info the caller might want
        model._ckpt_extra = {k: v for k, v in blob.items()
                             if k not in {"state_dict", "train_X_list",
                                          "train_Y_list", "baseline_X_list",
                                          "baseline_Y_list", "cluster_sizes",
                                          "baseline_sizes", "geo_of_expert",
                                          "K_phi_per_geo", "K_geo", "kernel",
                                          "geo_gmm", "geo_scaler",
                                          "phi_gmms", "phi_scalers",
                                          "expert_poly", "baseline_poly"}}
        return model
