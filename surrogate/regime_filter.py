"""
surrogate/regime_filter.py
==========================
Stage 0: Physics-based flow regime pre-classification.

Separates inertia-dominated ("splashing") samples from viscoplastic samples
using the dam-break Reynolds number:

    Re_dam = rho * H^(3/2) * g^(1/2) / eta

where H is the initial column height, eta is the consistency index, rho is
density, and g is gravitational acceleration.

Samples with Re_dam > Re_c are classified as inertia-dominated and excluded
from the viscoplastic MoE surrogate.

References
----------
  - Ancey & Cochard (2009), "The dam-break problem for Herschel-Bulkley
    viscoplastic fluids down steep flumes", J. Non-Newtonian Fluid Mech.
  - Balmforth et al. (2014), "Yielding to Stress: Recent Developments in
    Viscoplastic Fluid Mechanics", Annu. Rev. Fluid Mech.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Physical constants (CGS)
RHO = 1.0       # g/cm^3 (fluid density, ~water)
G   = 981.0     # cm/s^2 (gravitational acceleration)

# Default threshold (order-of-magnitude argument: Re >> 1 => inertia-dominated)
DEFAULT_RE_C = 100.0


def compute_re_dam(eta, H, rho=RHO, g=G):
    """Compute dam-break Reynolds number.

    Re_dam = rho * H^(3/2) * g^(1/2) / eta

    Parameters
    ----------
    eta : array-like — consistency index (CGS: g/(cm·s))
    H   : array-like — initial column height (cm)
    rho : float — density (g/cm^3)
    g   : float — gravity (cm/s^2)

    Returns
    -------
    Re : ndarray — Reynolds number (same shape as eta)
    """
    eta = np.asarray(eta, dtype=np.float64)
    H   = np.asarray(H,   dtype=np.float64)
    return rho * H**1.5 * np.sqrt(g) / np.maximum(eta, 1e-12)


def compute_bi(sigma_y, eta, H, g=G):
    """Compute Bingham number.

    Bi = sigma_y / (eta * sqrt(g / H))

    Parameters
    ----------
    sigma_y : array-like — yield stress (CGS: g/(cm·s^2) = dyn/cm^2)
    eta     : array-like — consistency index
    H       : array-like — column height (cm)

    Returns
    -------
    Bi : ndarray
    """
    sigma_y = np.asarray(sigma_y, dtype=np.float64)
    eta     = np.asarray(eta,     dtype=np.float64)
    H       = np.asarray(H,       dtype=np.float64)
    return sigma_y / (np.maximum(eta, 1e-12) * np.sqrt(g / np.maximum(H, 1e-6)))


def filter_splashing(df, re_c=DEFAULT_RE_C):
    """Split a DataFrame into viscoplastic and splashing regimes.

    Parameters
    ----------
    df   : DataFrame with columns 'eta' and 'height'
    re_c : float — Reynolds number threshold

    Returns
    -------
    df_viscous, df_splash : DataFrames
    """
    Re = compute_re_dam(df["eta"].values, df["height"].values)
    mask_viscous = Re <= re_c

    df_viscous = df[mask_viscous].copy()
    df_splash  = df[~mask_viscous].copy()

    n_total = len(df)
    n_splash = len(df_splash)
    log.info(f"Stage 0 regime filter (Re_c={re_c:.0f}):")
    log.info(f"  Viscoplastic: {len(df_viscous):,} ({len(df_viscous)/n_total*100:.1f}%)")
    log.info(f"  Splashing:    {n_splash:,} ({n_splash/n_total*100:.1f}%)")

    return df_viscous, df_splash
