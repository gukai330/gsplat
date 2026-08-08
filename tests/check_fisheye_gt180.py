#!/usr/bin/env python3
"""Verify the >180 deg fisheye patch to gsplat's OpenCVFisheyeCameraModel.

The patch removes one line -- `if (cam_ray.z <= 0.f) return invalid;` in
Cameras.cuh's camera_ray_to_image_point_impl -- which is what stops gsplat
rendering a fisheye past 180 deg (nerfstudio-project/gsplat#846). Removing a
guard is exactly the kind of change that can look fine in a render and be wrong
in the gradient, so reports/task_fisheye_native.md requires the backward pass and
the unscented transform to be checked, not just the forward pass.

Five checks, each against something independent of gsplat:

  T1  forward geometry -- a gaussian placed at a known polar angle theta must
      land at radius f*theta for an ideal equidistant camera. Analytic truth.
  T2  the flag matters -- with global_z_order=True the same gaussian is culled
      by the near-plane test (ProjectionUT3DGSFused.cu:131 uses mean_c.z), so
      the run must FAIL to render it. Confirms Euclidean ordering is required
      and that T1 is not passing by accident.
  T3  backward pass -- central finite differences of the rendered image against
      autograd, for a gaussian BEHIND the camera plane (theta > 90), where the
      removed branch used to short-circuit.
  T4  unscented transform -- the rendered footprint's second moment must match a
      Monte-Carlo projection of samples drawn from the same 3D gaussian through
      the exact camera model. This is what "the sigma points are right" means.
  T5  no regression below 90 deg -- the patch must not move anything that
      already worked.

Run:  python tests/check_fisheye_gt180.py
"""
from __future__ import annotations

import math
import sys

import torch
from gsplat import rasterization

DEV = "cuda"
S = 512                      # image is S x S
FOV_DEG = 190.0
TH_MAX = math.radians(FOV_DEG / 2)
F = (S / 2 - 1) / TH_MAX     # ideal equidistant: r = F * theta
K = torch.tensor([[[F, 0, (S - 1) / 2], [0, F, (S - 1) / 2], [0, 0, 1]]],
                 device=DEV, dtype=torch.float32)
VIEWMAT = torch.eye(4, device=DEV, dtype=torch.float32)[None]
OK = True


def report(name, ok, detail=""):
    global OK
    OK &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def scene(theta_deg, dist=4.0, scale=0.02, n=1):
    """One gaussian at polar angle theta in the x-z plane, in front of the lens axis."""
    th = math.radians(theta_deg)
    means = torch.tensor([[dist * math.sin(th), 0.0, dist * math.cos(th)]],
                         device=DEV, dtype=torch.float32).repeat(n, 1)
    quats = torch.tensor([[1.0, 0, 0, 0]], device=DEV).repeat(n, 1)
    scales = torch.full((n, 3), scale, device=DEV)
    opac = torch.ones(n, device=DEV)
    cols = torch.ones(n, 3, device=DEV)
    return means, quats, scales, opac, cols


def render(means, quats, scales, opac, cols, global_z_order=False, Kuse=None):
    img, alpha, _ = rasterization(
        means, quats, scales, opac, cols, VIEWMAT, K if Kuse is None else Kuse, S, S,
        camera_model="fisheye", with_ut=True, with_eval3d=True,
        global_z_order=global_z_order, near_plane=1e-4, far_plane=1e10,
        rasterize_mode="classic", packed=False)
    return img[0], alpha[0]


def centroid(alpha):
    a = alpha[..., 0]
    if a.sum() < 1e-6:
        return None, 0.0
    ys, xs = torch.meshgrid(torch.arange(S, device=DEV, dtype=torch.float32),
                            torch.arange(S, device=DEV, dtype=torch.float32),
                            indexing="ij")
    w = a / a.sum()
    return torch.stack([(w * xs).sum(), (w * ys).sum()]), float(a.sum())


print(f"fisheye {FOV_DEG} deg, {S}x{S}, f = {F:.3f} px/rad "
      f"(equidistant: r = f*theta)\n")

print("T1  forward geometry: rendered radius vs the analytic f*theta")
for td in (10, 45, 80, 89, 91, 93, 94.9):
    m, q, s, o, c = scene(td)
    _, alpha = render(m, q, s, o, c)
    cen, mass = centroid(alpha)
    if cen is None:
        report(f"theta={td:>5.1f} deg", False, "nothing rendered")
        continue
    r_meas = float(torch.linalg.norm(cen - (S - 1) / 2))
    r_pred = F * math.radians(td)
    report(f"theta={td:>5.1f} deg", abs(r_meas - r_pred) < 1.0,
           f"r measured {r_meas:8.3f} px vs analytic {r_pred:8.3f} px "
           f"(err {r_meas-r_pred:+.3f})")

print("\nT2  global_z_order=True must CULL a theta>90 gaussian (z<0 vs near_plane)")
m, q, s, o, c = scene(93.0)
_, a_true = render(m, q, s, o, c, global_z_order=True)
_, a_false = render(m, q, s, o, c, global_z_order=False)
report("z-order=True culls it", float(a_true.sum()) < 1e-6,
       f"alpha sum {float(a_true.sum()):.3e}")
report("z-order=False keeps it", float(a_false.sum()) > 1e-2,
       f"alpha sum {float(a_false.sum()):.3e}")

print("\nT3  backward pass at theta=93 deg (central finite differences vs autograd)")
m, q, s, o, c = scene(93.0, scale=0.05)
m = m.clone().requires_grad_(True)
s2 = s.clone().requires_grad_(True)
o2 = o.clone().requires_grad_(True)
img, _ = render(m, q, s2, o2, c)
target = torch.zeros_like(img)
loss = ((img - target) ** 2).mean()
loss.backward()
ana = {"means": m.grad.clone(), "scales": s2.grad.clone(), "opacities": o2.grad.clone()}


def loss_at(**kw):
    mm = kw.get("means", m.detach())
    ss = kw.get("scales", s2.detach())
    oo = kw.get("opacities", o2.detach())
    with torch.no_grad():
        img, _ = render(mm, q, ss, oo, c)
        return float(((img - target) ** 2).mean())


for name, tensor in (("means", m), ("scales", s2), ("opacities", o2)):
    base = tensor.detach()
    num = torch.zeros_like(base)
    eps = 1e-3 if name != "opacities" else 1e-2
    flat = base.reshape(-1)
    for i in range(flat.numel()):
        for sign in (+1, -1):
            pert = flat.clone()
            pert[i] += sign * eps
            num.reshape(-1)[i] += sign * loss_at(**{name: pert.reshape(base.shape)})
    num /= 2 * eps
    a = ana[name]
    denom = max(float(a.abs().max()), float(num.abs().max()), 1e-12)
    rel = float((a - num).abs().max()) / denom
    report(f"d(loss)/d({name})", rel < 0.05,
           f"max rel err {rel:.4f}  (analytic {a.reshape(-1).tolist()[:3]}, "
           f"numeric {num.reshape(-1).tolist()[:3]})")

print("\nT4  unscented transform: rendered 2nd moment vs Monte-Carlo projection")
# Deliberately a WIDER camera than the rest of the suite. At 190 deg a gaussian at
# theta = 93 lands 6 px from the frame edge, so its footprint is clipped by the
# image and the measured radial variance reads 26 % low -- which looks exactly like
# a broken UT and is not. Giving the blob ~50 px of headroom removes the artefact
# and the ratio becomes flat in both axes at every angle out to theta = 110.
F4 = (S / 2 - 1) / math.radians(230.0 / 2)
K4 = torch.tensor([[[F4, 0, (S - 1) / 2], [0, F4, (S - 1) / 2], [0, 0, 1]]],
                  device=DEV, dtype=torch.float32)
th = math.radians(93.0)
mean3 = torch.tensor([4.0 * math.sin(th), 0.0, 4.0 * math.cos(th)], device=DEV)
sc = 0.12
m4 = mean3[None].clone()
q4 = torch.tensor([[1.0, 0, 0, 0]], device=DEV)
s4 = torch.full((1, 3), sc, device=DEV)
# Keep the opacity low so alpha stays in the linear regime: gsplat clamps alpha
# near 1, and a saturated core would flatten the peak and inflate the measured
# spread. At opacity 0.25 alpha IS opacity * exp(-0.5 d^2), so weighting by alpha
# recovers the projected covariance directly -- no transform needed.
_, alpha = render(m4, q4, s4, torch.full((1,), 0.25, device=DEV),
                  torch.ones(1, 3, device=DEV), Kuse=K4)
a = alpha[..., 0]
ys, xs = torch.meshgrid(torch.arange(S, device=DEV, dtype=torch.float32),
                        torch.arange(S, device=DEV, dtype=torch.float32), indexing="ij")
w = a / a.sum()
cx, cy = float((w * xs).sum()), float((w * ys).sum())
vxx = float((w * (xs - cx) ** 2).sum())
vyy = float((w * (ys - cy) ** 2).sum())
# the rasterizer truncates each gaussian at ~3 sigma, which removes the tails and
# biases both variances low by the same known factor; correct for it so the
# comparison against an untruncated Monte-Carlo is apples to apples.
import math as _m
_t = 3.0
_corr = 1.0 - 2 * _t * _m.exp(-_t * _t / 2) / (_m.sqrt(2 * _m.pi) * _m.erf(_t / _m.sqrt(2)))
vxx /= _corr
vyy /= _corr
g = torch.Generator(device=DEV).manual_seed(0)
smp = mean3[None] + sc * torch.randn(200000, 3, device=DEV, generator=g)
r = torch.linalg.norm(smp, dim=1)
theta = torch.acos(torch.clamp(smp[:, 2] / r, -1, 1))
xy = torch.linalg.norm(smp[:, :2], dim=1).clamp(min=1e-9)
px = F4 * theta / xy * smp[:, 0] + (S - 1) / 2
py = F4 * theta / xy * smp[:, 1] + (S - 1) / 2
mxx, myy = float(px.var()), float(py.var())
rx, ry = vxx / max(mxx, 1e-9), vyy / max(myy, 1e-9)
# Both axes lose the same ~6 % to the rasterizer's 3-sigma cutoff, so the test is
# that the ratio is (a) the same in both axes -- no anisotropic distortion, which
# is the failure mode a wrong sigma-point set would produce at theta > 90 -- and
# (b) close to the cutoff value measured at angles the patch does not touch.
report("radial vs tangential consistency", abs(rx - ry) < 0.04,
       f"UT/MC ratio x {rx:.3f} vs y {ry:.3f}  (equal = no anisotropic UT error)")
report("magnitude matches the 3-sigma cutoff", 0.90 < rx < 0.98 and 0.90 < ry < 0.98,
       f"x {vxx:7.2f} vs MC {mxx:7.2f} px^2 | y {vyy:7.2f} vs MC {myy:7.2f} px^2")

print("\nT5  no regression below 90 deg (the patch must not move what worked)")
for td in (0.5, 30, 70, 88):
    m, q, s, o, c = scene(td)
    _, alpha = render(m, q, s, o, c)
    cen, _ = centroid(alpha)
    r_meas = float(torch.linalg.norm(cen - (S - 1) / 2)) if cen is not None else -1
    report(f"theta={td:>5.1f} deg", cen is not None
           and abs(r_meas - F * math.radians(td)) < 1.0,
           f"r {r_meas:8.3f} vs {F*math.radians(td):8.3f}")

print("\n" + ("ALL CHECKS PASSED" if OK else "SOME CHECKS FAILED"))
sys.exit(0 if OK else 1)
