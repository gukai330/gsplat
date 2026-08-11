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
    else:  # pinhole
        dirs = torch.stack([u, v, torch.ones_like(u)], dim=-1)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    return dirs


def sh_basis(dirs: Tensor, degree: int) -> Tensor:
    """Real SH basis values for unit dirs [..., 3] -> [..., (degree+1)^2].
    Hardcoded through degree 3; any fixed orthogonal basis works here because
    the coefficients are learned."""
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    out = [torch.full_like(x, 0.282095)]
    if degree >= 1:
        out += [-0.488603 * y, 0.488603 * z, -0.488603 * x]
    if degree >= 2:
        out += [
            1.092548 * x * y,
            -1.092548 * y * z,
            0.315392 * (3 * z * z - 1),
            -1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ]
    if degree >= 3:
        out += [
            -0.590044 * y * (3 * x * x - y * y),
            2.890611 * x * y * z,
            -0.457046 * y * (5 * z * z - 1),
            0.373176 * z * (5 * z * z - 3),
            -0.457046 * x * (5 * z * z - 1),
            1.445306 * z * (x * x - y * y),
            0.590044 * x * (x * x - 3 * y * y),
        ]
    return torch.stack(out, dim=-1)


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
