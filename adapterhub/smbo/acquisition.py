"""
Acquisition function for the NASPEFT SMBO search (paper eq. 7 / Algorithm 1).

    alpha_t(a) = lambda(t) * [ p_tilde(a) - kappa * sigma_tilde(a) ]
               + (1 - lambda(t)) * c_tilde(a)

Asymmetric uncertainty (paper §"Acquisition with asymmetric uncertainty"):
  - Performance perf(a) needs fine-tuning -> surrogate gives predictive mean
    p_hat(a) and tree-variance std sigma_p(a). This axis carries uncertainty;
    we use the Lower Confidence Bound  p_tilde - kappa * sigma_tilde.
  - Parameter % c(a) is computed EXACTLY from the configuration in microseconds
    (param_oracle) and carries NO uncertainty -> deterministic term, NOT a
    predictive distribution (unlike AutoPEFT's EHVI which models both axes).

Randomized scalarization: lambda(t) ~ Uniform(0,1) drawn once per iteration so
a single run sweeps the bi-objective Pareto front instead of collapsing to one
fixed weighting.

Minimization convention: alpha is MINIMIZED (select lowest-alpha top-B).
The surrogate is trained on raw task metric. Sign handling:
  - GLUE  (accuracy / MCC / Spearman): higher is better. The performance term
    must be "lower = better" for the minimization in eq. 7, so we NEGATE the
    normalized performance prediction.
  - LLaMA (perplexity): lower is better already -> no negation.
This mirrors how Problem._evaluate_single bakes in (-1)**maximization for the
exact same reason, and keeps paper eq. 7 valid as written under the §3.1
"minimization convention".

Normalization (paper: "normalised analogously over D ... comparable scales"):
  - c_tilde = c / c_max in [0, 1], c_max = the search-space maximum param%.
  - p_hat, sigma_p are min-max normalized over the surrogate set D so the
    performance term also lies on a [0, 1] scale, making the LCB and the
    parameter term directly comparable for any lambda. (Pure z-scoring left
    performance on a ~+/-2 scale while c_tilde stayed in [0, ~0.5], so the
    performance term dominated and the lambda-sweep collapsed to a corner.)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class DNormStats:
    """Min-max normalization statistics for the performance axis, over D.

    Eq. 7 requires p_tilde and c_tilde on COMPARABLE scales. c_tilde is the
    parameter percentage normalised by the search-space maximum, so it lies in
    [0, 1]. We therefore map the performance prediction to [0, 1] as well, using
    the observed min/max of the surrogate's current observation set D ("the
    predictive mean and standard deviation are normalised analogously over D so
    that the two terms are on comparable scales"). Standard deviation is scaled
    by the same span so the LCB width stays on the [0,1] performance scale.
    """
    perf_min: float
    perf_span: float  # max - min over D (>= tiny epsilon)

    @classmethod
    def from_targets(cls, y_surrogate: np.ndarray) -> "DNormStats":
        y = np.asarray(y_surrogate, dtype=float).reshape(-1)
        if y.size == 0:
            return cls(perf_min=0.0, perf_span=1.0)
        lo = float(np.min(y))
        hi = float(np.max(y))
        span = hi - lo
        if span < 1e-12:
            span = 1.0
        return cls(perf_min=lo, perf_span=span)


def acquisition_scores(
    pred_perf: np.ndarray,
    pred_std: np.ndarray,
    param_pct: np.ndarray,
    lambda_t: float,
    kappa: float,
    d_stats: DNormStats,
    c_max: float,
    glue_higher_is_better: bool,
) -> np.ndarray:
    """Compute alpha_t(a) for a candidate pool (eq. 7). Lower alpha = better.

    Args:
        pred_perf : surrogate predictive mean of the RAW task metric, shape (N,).
        pred_std  : surrogate tree-variance std, shape (N,).
        param_pct : EXACT param% per candidate (from param_oracle), shape (N,).
        lambda_t  : weight in [0,1], drawn ~Uniform(0,1) once per iteration.
        kappa     : exploration constant (LCB width).
        d_stats   : standardization stats over the surrogate set D.
        c_max     : search-space maximum param% (normalizes c_tilde to [0,1]).
        glue_higher_is_better :
            True  for GLUE accuracy/MCC/Spearman  -> negate normalized perf so
                  "lower = better" for the minimization convention.
            False for LLaMA perplexity            -> already lower-is-better.

    Returns:
        alpha : np.ndarray shape (N,). Select the B candidates with LOWEST alpha.
    """
    pred_perf = np.asarray(pred_perf, dtype=float).reshape(-1)
    pred_std = np.asarray(pred_std, dtype=float).reshape(-1)
    param_pct = np.asarray(param_pct, dtype=float).reshape(-1)

    # --- normalize performance over D to [0,1] (comparable to c_tilde) ---
    p_tilde = (pred_perf - d_stats.perf_min) / d_stats.perf_span
    s_tilde = pred_std / d_stats.perf_span  # std scaled by the same span

    # --- minimization convention ---
    # GLUE: higher raw metric is better -> negate so larger metric => smaller
    # (better) performance term. LLaMA perplexity: lower already better.
    if glue_higher_is_better:
        p_tilde = -p_tilde

    # Lower Confidence Bound on the (sign-corrected) performance.
    # Under minimization, the optimistic/explorative bound subtracts kappa*std.
    lcb = p_tilde - kappa * s_tilde

    # --- exact, uncertainty-free parameter term in [0,1] ---
    if c_max <= 0:
        c_tilde = np.zeros_like(param_pct)
    else:
        c_tilde = np.clip(param_pct / c_max, 0.0, 1.0)

    alpha = lambda_t * lcb + (1.0 - lambda_t) * c_tilde
    return alpha


def select_topB_lowest(alpha: np.ndarray, B: int) -> np.ndarray:
    """Indices of the B candidates with the LOWEST alpha (eq. 7 selection)."""
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    B = min(B, alpha.shape[0])
    # argsort ascending; take first B (lowest alpha).
    return np.argsort(alpha, kind="stable")[:B]


def draw_lambda(rng: Optional[np.random.Generator] = None) -> float:
    """Draw lambda(t) ~ Uniform(0,1) for one iteration."""
    if rng is None:
        return float(np.random.uniform(0.0, 1.0))
    return float(rng.uniform(0.0, 1.0))