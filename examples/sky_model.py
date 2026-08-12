"""Global directional sky/background model for simple_trainer.

A tiny spherical-harmonics function of VIEW DIRECTION only, composited behind
the gaussians exactly where --random_bkgd composites:

    colors = colors + sky(dir) * (1 - alpha)

Direction-only is the load-bearing property: app_opt / bilagrid / ppisp all
measured NEGATIVE on held-out views in this repo because their correction lives
in per-training-image modules that a held-out view has no entry for. An
environment map is a global function of ray direction, so the identical
correction exists for every view, held-out or not.

Ray directions come from the same Kannala-Brandt model the rasterizer uses
(OPENCV_FISHEYE: r(theta) = f * (theta + k1 th^3 + k2 th^5 + k3 th^7 + k4 th^9)),
inverted per pixel by Newton, once per camera, cached. A pinhole path exists so
the model is also usable on the ERP-cube datasets.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


def _kb_theta_from_r(r_over_f: Tensor, k: Tensor, iters: int = 12) -> Tensor:
    """Invert the OPENCV_FISHEYE radial polynomial. r_over_f = theta_d.

    Pixels OUTSIDE the image circle (frame corners) have r/f beyond the
    polynomial's monotonic range; unclamped Newton diverges there and the NaNs
    poison every pixel through the background composite. Those pixels are
    black/masked anyway, so clamping theta to [0, pi] is purely protective."""
    th = r_over_f.clamp(max=math.pi)
    for _ in range(iters):
        th2 = th * th
        poly = th * (1 + th2 * (k[0] + th2 * (k[1] + th2 * (k[2] + th2 * k[3]))))
        dpoly = 1 + th2 * (3 * k[0] + th2 * (5 * k[1] + th2 * (7 * k[2] + th2 * 9 * k[3])))
        th = (th - (poly - r_over_f) / dpoly.clamp(min=1e-3)).clamp(0.0, math.pi)
    return torch.nan_to_num(th, nan=0.0)


@torch.no_grad()
def camera_ray_dirs(
    K: Tensor,
    width: int,
    height: int,
    camera_model: str = "pinhole",
    radial_coeffs: Optional[Tensor] = None,
    device: str = "cuda",
) -> Tensor:
    """[H, W, 3] unit ray directions in the CAMERA frame (x right, y down,
    z forward). K is [3, 3]; radial_coeffs is [4] (k1..k4) for fisheye."""
    K = K.to(device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    u = (xx - K[0, 2]) / K[0, 0]
    v = (yy - K[1, 2]) / K[1, 1]
    if camera_model == "fisheye":
        r = torch.sqrt(u * u + v * v).clamp(min=1e-9)
        if radial_coeffs is None:
            k = torch.zeros(4, device=device)
        else:
            k = radial_coeffs.to(device=device, dtype=torch.float32).reshape(-1)[:4]
        theta = _kb_theta_from_r(r, k)  # can exceed pi/2: z goes negative
        s = torch.sin(theta) / r
        dirs = torch.stack([u * s, v * s, torch.cos(theta)], dim=-1)
        # Normalise: the components above are only unit-length to ~2e-7, and
        # sh_basis divides by sqrt(1 - z^2), so a z a hair past 1 sends that to
        # zero and the high-order phi recurrence to inf * 0 = NaN. Three pixels
        # in 11M were enough to NaN every SH8 coefficient on the first step.
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    else:  # pinhole
        dirs = torch.stack([u, v, torch.ones_like(u)], dim=-1)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    return dirs


def sh_basis(dirs: Tensor, degree: int) -> Tensor:
    """Real orthonormal SH basis for unit dirs [..., 3] -> [..., (degree+1)^2].

    By recurrence, not a table: the table version stopped at degree 3 and
    silently returned 16 columns for ANY higher degree, which is how this repo
    recorded a bogus "SH3 = SH6, so capacity is not the limit" -- both arms were
    degree 3 and agreed to the last digit. Verified against orthonormality on
    uniform directions (max |Gram - I| < 0.02 through degree 8).

    Note phi comes from x/sin(theta), y/sin(theta): P_l^m already carries the
    sin^m(theta) factor, so feeding it x, y directly double-counts it.
    """
    import math as _math
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    s = (1.0 - z * z).clamp(min=0.0).sqrt()
    inv = 1.0 / s.clamp(min=1e-9)
    # cos/sin of the azimuth are bounded by construction; clamping makes the
    # degenerate pole (s == 0, where P_l^m is 0 anyway) harmless instead of inf.
    cph, sph = (x * inv).clamp(-1.0, 1.0), (y * inv).clamp(-1.0, 1.0)

    P = {(0, 0): torch.ones_like(z)}
    for l in range(1, degree + 1):
        P[(l, l)] = -(2 * l - 1) * s * P[(l - 1, l - 1)]
        if l - 1 >= 0:
            P[(l, l - 1)] = z * (2 * l - 1) * P[(l - 1, l - 1)]
        for m in range(l - 2, -1, -1):
            P[(l, m)] = ((2 * l - 1) * z * P[(l - 1, m)]
                         - (l - 1 + m) * P[(l - 2, m)]) / (l - m)

    cos_m = [torch.ones_like(x), cph]
    sin_m = [torch.zeros_like(x), sph]
    for m in range(2, degree + 1):
        cos_m.append(cph * cos_m[m - 1] - sph * sin_m[m - 1])
        sin_m.append(sph * cos_m[m - 1] + cph * sin_m[m - 1])

    out = []
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            K = _math.sqrt((2 * l + 1) / (4 * _math.pi)
                           * _math.factorial(l - am) / _math.factorial(l + am))
            base = K * P[(l, am)]
            if m == 0:
                out.append(base)
            elif m > 0:
                out.append(_math.sqrt(2.0) * base * cos_m[am])
            else:
                out.append(_math.sqrt(2.0) * base * sin_m[am])
    return torch.stack(out, -1)


class SkyModel(nn.Module):
    """sky(dir) = sigmoid(SH(dir) @ coeffs), one global set of coefficients."""

    def __init__(self, degree: int = 3, init_rgb: float = 0.8):
        super().__init__()
        self.degree = degree
        n = (degree + 1) ** 2
        coeffs = torch.zeros(n, 3)
        # sigmoid(dc * 0.282095) == init_rgb at start
        coeffs[0, :] = math.log(init_rgb / (1 - init_rgb)) / 0.282095
        self.coeffs = nn.Parameter(coeffs)

    def forward(self, dirs: Tensor) -> Tensor:
        """dirs [..., 3] (unit, world frame) -> rgb [..., 3] in [0, 1]."""
        return torch.sigmoid(sh_basis(dirs, self.degree) @ self.coeffs)


class SkyRayCache:
    """Per-camera [H, W, 3] camera-frame ray grids, built on first use."""

    def __init__(self, camera_model: str, device: str = "cuda"):
        self.camera_model = camera_model
        self.device = device
        self._cache: dict = {}

    def dirs_world(
        self,
        cam_key,
        K: Tensor,
        width: int,
        height: int,
        camtoworld: Tensor,
        radial_coeffs: Optional[Tensor] = None,
    ) -> Tensor:
        """[1, H, W, 3] unit world-frame ray dirs for a [1,...] batch."""
        key = (cam_key, width, height)
        if key not in self._cache:
            self._cache[key] = camera_ray_dirs(
                K.reshape(3, 3), width, height, self.camera_model,
                None if radial_coeffs is None else radial_coeffs.reshape(-1),
                device=self.device,
            )
        dirs = self._cache[key]
        R = camtoworld.reshape(4, 4)[:3, :3].to(dirs)
        return (dirs @ R.T)[None]
