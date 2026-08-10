# SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for StopThePop-style per-pixel depth resorting on the eval3d forward.

The sorted kernel re-orders blending per pixel by the depth of maximum
response along each pixel's own ray,

    t* = d^T Sigma^-1 (mu - o) / (d^T Sigma^-1 d),

inside a sliding window, while the stock kernel blends in the per-tile global
order (one center-depth key per gaussian). The tests pin three behaviours:

1. When the global order is already per-pixel correct, the sorted kernel is
   bit-identical to the stock kernel (same blend sequence, same arithmetic).
2. When a large slanted gaussian's per-ray depth crosses a compact gaussian's
   (the popping scenario), the sorted kernel matches an analytic per-pixel
   reference that blends in the true t* order — and the stock kernel provably
   does not, so the test discriminates.
3. The path is render-only: inputs that require grad are rejected.
"""

import math

import torch
import torch.nn.functional as F

device = torch.device("cuda:0")


def _render(means, quats, scales, opacities, colors, viewmat, K, W, H, window):
    from gsplat.rendering import rasterization

    render, alphas, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmat[None],
        K[None],
        W,
        H,
        packed=False,
        with_ut=True,
        with_eval3d=True,
        global_z_order=False,
        near_plane=0.01,
        rasterize_mode="classic",
        camera_model="pinhole",
        # Pin the tile size for the window=0 arm too: the sorted path forces
        # 16, and per-tile candidate lists must be identical for the
        # bit-exactness assertion below.
        tile_size=16,
        per_pixel_sort_window=window,
    )
    return render[0], alphas[0]


def _rays(K, W, H):
    """World rays for an identity camera pose, pixel-center convention."""
    jj, ii = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32) + 0.5,
        torch.arange(H, device=device, dtype=torch.float32) + 0.5,
        indexing="xy",
    )
    pts = torch.stack([jj, ii, torch.ones_like(jj)], dim=-1)  # [H, W, 3]
    dirs = pts @ torch.linalg.inv(K).T
    return F.normalize(dirs, dim=-1)  # [H, W, 3]


def _per_ray_response(means, quats, scales, opacities, dirs):
    """alpha and t* per (gaussian, pixel) with the kernel's exact math."""
    # M = diag(1/s) R^T  (iscl_rot in the kernel)
    w, x, y, z = quats.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    M = torch.diag_embed(1.0 / scales) @ R.transpose(-1, -2)  # [N, 3, 3]

    H, W, _ = dirs.shape
    alphas = torch.zeros(len(means), H, W, device=device)
    tstars = torch.full((len(means), H, W), torch.inf, device=device)
    for g in range(len(means)):
        gro = (-means[g]) @ M[g].T  # camera at origin: o - mu = -mu
        grd_un = dirs @ M[g].T  # [H, W, 3]
        inv_len = 1.0 / grd_un.norm(dim=-1)
        grd = grd_un * inv_len[..., None]
        hit_t = -(grd * gro).sum(-1)
        gcrod = torch.cross(grd, gro.expand_as(grd), dim=-1)
        alpha = torch.clamp(
            opacities[g] * torch.exp(-0.5 * (gcrod * gcrod).sum(-1)), max=0.99
        )
        alpha = torch.where((hit_t >= 0) & (alpha >= 1.0 / 255.0), alpha, 0.0)
        alphas[g] = alpha
        tstars[g] = torch.where(alpha > 0, hit_t * inv_len, torch.inf)
    return alphas, tstars


def _make_camera(W=64, H=64, focal=80.0):
    K = torch.tensor(
        [[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]],
        device=device,
        dtype=torch.float32,
    )
    viewmat = torch.eye(4, device=device)
    return K, viewmat


def test_sorted_matches_stock_when_order_already_correct():
    """Well-separated isotropic gaussians on-axis: per-pixel order == global
    order, so the sorted kernel must reproduce the stock kernel bit-exactly."""
    torch.manual_seed(0)
    K, viewmat = _make_camera()
    N = 8
    means = torch.zeros(N, 3, device=device)
    means[:, 2] = torch.linspace(4.0, 18.0, N, device=device)
    means[:, 0] = 0.05 * torch.randn(N, device=device)
    means[:, 1] = 0.05 * torch.randn(N, device=device)
    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = 1.0
    scales = torch.full((N, 3), 0.6, device=device)
    opacities = torch.full((N,), 0.7, device=device)
    colors = torch.rand(N, 3, device=device)

    stock, stock_a = _render(means, quats, scales, opacities, colors, viewmat, K, 64, 64, 0)
    for window in (4, 8, 16, 24):
        srt, srt_a = _render(
            means, quats, scales, opacities, colors, viewmat, K, 64, 64, window
        )
        torch.testing.assert_close(srt, stock, atol=0.0, rtol=0.0)
        torch.testing.assert_close(srt_a, stock_a, atol=0.0, rtol=0.0)


def test_sorted_matches_per_pixel_reference_on_order_flip():
    """A large slanted gaussian crossing a compact one in depth: the true
    blending order flips across the image while the global order cannot."""
    K, viewmat = _make_camera()
    W = H = 64

    # B: compact-ish, isotropic, centered on axis at depth 5.
    # A: elongated along a slanted direction so its per-ray max-response depth
    #    runs from in front of B (left of image) to behind B (right), while its
    #    center Euclidean distance (the global sort key) stays > |B|'s.
    ang = math.radians(35.0)
    qa = torch.tensor(
        [math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0], device=device
    )  # rotation about +y: slants the x-axis in the x/z plane
    means = torch.tensor([[0.0, 0.0, 5.2], [0.0, 0.0, 5.0]], device=device)
    quats = torch.stack([qa, torch.tensor([1.0, 0, 0, 0], device=device)])
    scales = torch.tensor([[6.0, 1.5, 0.05], [0.8, 0.8, 0.8]], device=device)
    opacities = torch.tensor([0.8, 0.6], device=device)
    colors = torch.tensor([[1.0, 0.1, 0.0], [0.0, 0.2, 1.0]], device=device)

    dirs = _rays(K, W, H)
    alphas, tstars = _per_ray_response(means, quats, scales, opacities, dirs)

    # Sanity: the scenario must actually contain flipped pixels where both
    # gaussians contribute, or the test discriminates nothing.
    both = (alphas[0] > 0) & (alphas[1] > 0)
    a_first = both & (tstars[0] < tstars[1])
    b_first = both & (tstars[0] > tstars[1])
    assert a_first.sum() > 100 and b_first.sum() > 100, (
        f"degenerate scenario: {a_first.sum()} vs {b_first.sum()} flip pixels"
    )
    # The global (Euclidean center) order is B-first everywhere:
    assert means[0].norm() > means[1].norm()

    # Analytic per-pixel reference in the true t* order.
    aA, aB = alphas[0][..., None], alphas[1][..., None]
    cA, cB = colors[0], colors[1]
    ref_a_first = cA * aA + cB * aB * (1 - aA)
    ref_b_first = cB * aB + cA * aA * (1 - aB)
    ref = torch.where(a_first[..., None], ref_a_first, ref_b_first)
    ref_alpha = (1 - (1 - aA) * (1 - aB)).squeeze(-1)

    stock, _ = _render(means, quats, scales, opacities, colors, viewmat, K, W, H, 0)
    srt, srt_a = _render(means, quats, scales, opacities, colors, viewmat, K, W, H, 4)

    # Tolerance rationale: torch (exact fp32 exp/normalize) vs kernel
    # (__expf, rsqrtf, fast-math) drift is ~1e-5; a pixel sitting within fp
    # noise of the 1/255 alpha threshold can be included by one side only,
    # contributing up to ~4e-3. Both are far below the 0.4-scale order-swap
    # signal asserted underneath.
    torch.testing.assert_close(srt, ref, atol=2e-2, rtol=0.0)
    torch.testing.assert_close(srt_a.squeeze(-1), ref_alpha, atol=2e-2, rtol=0.0)

    # And the stock kernel must NOT match the reference on the flipped pixels
    # (otherwise this test lost its teeth).
    stock_err = (stock - ref).abs().max()
    assert stock_err > 0.1, f"stock order unexpectedly correct ({stock_err})"


def test_rejects_grad_and_bad_window():
    import pytest

    from gsplat.rendering import rasterization

    K, viewmat = _make_camera()
    means = torch.zeros(1, 3, device=device)
    means[0, 2] = 5.0
    means.requires_grad_(True)
    quats = torch.tensor([[1.0, 0, 0, 0]], device=device)
    scales = torch.full((1, 3), 0.5, device=device)
    opacities = torch.full((1,), 0.7, device=device)
    colors = torch.rand(1, 3, device=device)

    kwargs = dict(
        packed=False,
        with_ut=True,
        with_eval3d=True,
        global_z_order=False,
        camera_model="pinhole",
    )
    with pytest.raises(RuntimeError, match="no backward"):
        rasterization(
            means, quats, scales, opacities, colors, viewmat[None], K[None],
            64, 64, per_pixel_sort_window=16, **kwargs,
        )
    with pytest.raises(ValueError, match="per_pixel_sort_window"):
        rasterization(
            means.detach(), quats, scales, opacities, colors, viewmat[None],
            K[None], 64, 64, per_pixel_sort_window=5, **kwargs,
        )
    with pytest.raises(ValueError, match="with_eval3d"):
        rasterization(
            means.detach(), quats, scales, opacities, colors, viewmat[None],
            K[None], 64, 64, per_pixel_sort_window=16, packed=False,
        )


if __name__ == "__main__":
    test_sorted_matches_stock_when_order_already_correct()
    print("order-preserving bit-exactness: PASS")
    test_sorted_matches_per_pixel_reference_on_order_flip()
    print("order-flip vs analytic reference: PASS")
    try:
        test_rejects_grad_and_bad_window()
        print("guard rejections: PASS")
    except ModuleNotFoundError:
        print("guard rejections: SKIPPED (no pytest)")
    print("all per-pixel sort tests passed")
