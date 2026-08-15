"""recon360: optimise gaussian POSITIONS in a contracted radial domain.

The gaussians stay world-space gaussians. Only the *parameter* changes: the
optimiser holds `u`, and `x = to_world(u)` is what the rasterizer receives, so
nothing propagates a covariance through a nonlinear map and there is no "a
gaussian is no longer a gaussian" approximation anywhere. Autograd supplies
`dL/du = J^T dL/dx` for free -- no CUDA, no kernel.

Why bother, given `reports/contracted_position_report.md` shows Adam already
divides out the far field's 1843x smaller gradients? Two things Adam does NOT
fix, both measured there:

  * the world-space step is a single global scalar (`means_lr * scene_scale`),
    so the ANGULAR step a camera sees falls as 1/r. Over a whole 30k run a
    gaussian at 12R can correct ~3.5 px of angular error; one at 0.5R gets 81.
  * the far field's RADIUS is not chosen by the loss at all. The coherent radial
    budget over a run is 0.6% of a far gaussian's own radius, so what actually
    puts gaussians out there is MCMC's noise injection. Contraction is what
    gives the photometric loss a vote on radius.

Conventions that matter:

  * `u` is in WORLD UNITS and the map is the IDENTITY inside `radius`. So for
    everything within the camera cloud, `u == x`, the gradient is untouched, and
    `means_lr * scene_scale` keeps its exact meaning -- including the value the
    trainer hands to MCMC as its noise scale. Only r > radius behaves
    differently. A dimensionless `u` would have been equivalent up to a constant
    but would have silently rescaled the MCMC noise by `1/scene_scale`.
  * `scales` are NOT contracted. `exp(s)` is already a multiplicative
    parameterisation whose step is distance-independent; there is no
    conditioning problem on that side to fix.

Two maps, both C^1 at the boundary. With `ry = r / radius`:

    log     rho = radius * (1 + ln ry)        J_rad = ry      J_tan = ry/(1+ln ry)
    mip360  rho = radius * (2 - 1/ry)         J_rad = ry^2    J_tan = ry/(2-1/ry)

`mip360` is Mip-NeRF 360's contraction. Its bounded range exists because NeRF
feeds the contracted coordinate to a hash grid, which needs a bounded input;
here nothing consumes `u` except Adam, so the bound buys nothing and costs a
clamp -- `|u - center| = 2*radius` is infinity, and one step past it gives a
negative radius. `log` has no singularity, needs no clamp, and at ry=12 gives
12x/3.4x against mip360's 144x/6.3x, which is the difference between handing
the loss a vote on radius and letting it move a far gaussian ten radii per run.
`log` is the intended default; `mip360` is here to be measured against it.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

MODES = ("none", "log", "mip360")

# |u-center| <= (2 - _MIP_EPS) * radius, i.e. mip360 caps the scene at 1/_MIP_EPS
# radii. float32 resolves the radius there to ~6e-4 relative, which is plenty.
_MIP_EPS = 1e-3
# exp() guard for `log`. rho/radius = 1 + ln ry, so this caps ry at e^40 ~ 2e17;
# it exists only so a runaway step yields a large number instead of an inf.
_LOG_MAX = 41.0


class PositionContraction:
    """Radial reparameterisation of gaussian positions about a fixed centre."""

    def __init__(self, mode: str, center: Tensor, radius: float):
        if mode not in MODES:
            raise ValueError(f"position_contraction must be one of {MODES}, got {mode!r}")
        if mode != "none" and not (radius > 0):
            raise ValueError(f"contraction radius must be positive, got {radius}")
        self.mode = mode
        self.center = center.reshape(1, 3).float()
        self.radius = float(radius)

    @property
    def active(self) -> bool:
        return self.mode != "none"

    def to(self, device) -> "PositionContraction":
        self.center = self.center.to(device)
        return self

    # -- the map ------------------------------------------------------------
    def to_world(self, u: Tensor) -> Tensor:
        """u -> x. Differentiable; this is the only place the rasterizer's input
        is produced, so `J^T` reaches the parameter through autograd."""
        if not self.active:
            return u
        c = self.center.to(u.device, u.dtype)
        d = u - c
        rho = d.norm(dim=-1, keepdim=True).clamp_min(1e-20)
        t = rho / self.radius  # |u - center| in units of radius
        # Both branches must be finite everywhere: torch.where propagates NaN
        # from the unselected branch straight into the gradient.
        if self.mode == "log":
            outer_ry = torch.exp((t - 1.0).clamp(max=_LOG_MAX))
        else:
            outer_ry = 1.0 / (2.0 - t).clamp_min(_MIP_EPS)
        # The inner branch returns `u` itself rather than rebuilding it from
        # `c + d/rho * rho`, which is only identity up to rounding. Bit-exact
        # identity inside `radius` is the property the whole convention rests on.
        return torch.where(t <= 1.0, u, c + d * (outer_ry * self.radius / rho))

    def to_param(self, x: Tensor) -> Tensor:
        """x -> u. Used at init and whenever a world-space checkpoint is loaded."""
        if not self.active:
            return x
        c = self.center.to(x.device, x.dtype)
        d = x - c
        r = d.norm(dim=-1, keepdim=True).clamp_min(1e-20)
        ry = r / self.radius
        if self.mode == "log":
            outer_t = 1.0 + torch.log(ry.clamp_min(1e-20))
        else:
            outer_t = 2.0 - 1.0 / ry.clamp_min(1e-20)
        return torch.where(ry <= 1.0, x, c + d * (outer_t * self.radius / r))

    @torch.no_grad()
    def clamp_(self, u: Tensor) -> None:
        """Keep `u` inside the domain, in place. `log` has no finite boundary and
        is left alone; `mip360` does, and stepping past it gives a negative radius."""
        if self.mode != "mip360":
            return
        c = self.center.to(u.device, u.dtype)
        d = u - c
        rho = d.norm(dim=-1, keepdim=True)
        limit = (2.0 - _MIP_EPS) * self.radius
        over = rho > limit
        if bool(over.any()):
            u.copy_(torch.where(over, c + d * (limit / rho.clamp_min(1e-20)), u))

    # -- bookkeeping --------------------------------------------------------
    def state(self) -> dict:
        return {"mode": self.mode, "radius": self.radius,
                "center": self.center.detach().cpu().tolist()[0]}

    @staticmethod
    def from_state(state: Optional[dict]) -> "PositionContraction":
        if not state:
            return PositionContraction("none", torch.zeros(3), 1.0)
        return PositionContraction(state["mode"], torch.tensor(state["center"]),
                                   state["radius"])

    def __repr__(self) -> str:
        if not self.active:
            return "PositionContraction(none)"
        c = [round(v, 4) for v in self.center.flatten().tolist()]
        return (f"PositionContraction({self.mode}, center={c}, "
                f"radius={self.radius:.6f})")
