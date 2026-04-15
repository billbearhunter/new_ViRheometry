"""
surrogate/models.py
====================
GP model definitions shared by training (DataPipeline) and inference (Optimization).

Previously duplicated in:
  - DataPipeline/moe_utils.py       (SingleOutputExactGP, SingleOutputSVGP)
  - Optimization/libs/moe_core.py   (_OfflineExactGP, _OfflineSVGPModel)

Now there is exactly ONE definition of each model class.
"""

import torch
import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


# ── Device / dtype ─────────────────────────────────────────────────────────────
DTYPE  = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── ExactGP ────────────────────────────────────────────────────────────────────

class SingleOutputExactGP(gpytorch.models.ExactGP):
    """Exact GP for small clusters (N <= EXACT_THRESHOLD).

    Kernel: ScaleKernel(Matern(2.5) + Linear)
    """
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
            + gpytorch.kernels.LinearKernel()
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ── SVGP ───────────────────────────────────────────────────────────────────────

class SingleOutputSVGP(gpytorch.models.ApproximateGP):
    """Sparse Variational GP for large clusters (N > EXACT_THRESHOLD).

    Kernel: ScaleKernel(Matern(2.5) + Linear)
    """
    def __init__(self, inducing_points):
        q = inducing_points.size(0)
        vd = CholeskyVariationalDistribution(q)
        vs = VariationalStrategy(self, inducing_points, vd,
                                 learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
            + gpytorch.kernels.LinearKernel()
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )
