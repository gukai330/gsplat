"""Per-eye radial gain g_e(theta): a global lens property, learned jointly.

The sky variation probe measured the biggest source of sky-pixel variance to
be CAMERA-FRAME radial (per-eye vignetting/flare: same instant, same world
direction, two eyes disagree by ~50 grey levels at p50, with strongly
different theta profiles). Left unmodelled, the optimizer expresses it the
only way it can -- floating pale "diffuser" gaussians in the 10-40 m band,
i.e. the welded fog: every pruning of that fog measured on this line grew
back within 6k steps, with or without sky pixels in the loss.

This is the legal home for that variance: ONE 1-D log-gain profile per
physical lens, shared by every frame, so unlike the thrice-refuted per-image
modules (pitfalls C3) it cannot absorb per-image appearance and applies
identically to held-out views.
"""
import torch


class FlareGain(torch.nn.Module):
    def __init__(self, n_cams: int, knots: int = 12):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.zeros(n_cams, knots))
        self._cache = {}

    def gain_image(self, cam_idx: int, K, width: int, height: int):
        key = (int(cam_idx), width, height)
        if key not in self._cache:
            dev = self.raw.device
            u = torch.arange(width, device=dev, dtype=torch.float32) - float(K[0, 2])
            v = torch.arange(height, device=dev, dtype=torch.float32) - float(K[1, 2])
            r = torch.sqrt(u[None, :] ** 2 + v[:, None] ** 2)
            theta = r / float(K[0, 0])              # ideal equidistant fisheye
            t = (theta / theta.max()).clamp(0, 1) * (self.raw.shape[1] - 1)
            i0 = t.floor().long().clamp(max=self.raw.shape[1] - 2)
            self._cache[key] = (i0, t - i0.float())
        i0, w = self._cache[key]
        # Gauge: pin g(theta=0)=0 so the curve can only express RELATIVE
        # radial variation. Without this the F1 arm degenerated to a flat
        # +11% global gain (a dynamic-range buffer for saturated sky, not
        # flare) and eval-without-the-module read 2 dB dark.
        prof = self.raw[int(cam_idx)]
        prof = prof - prof[0]
        g = prof[i0] * (1.0 - w) + prof[i0 + 1] * w
        return torch.exp(g)[None, ..., None]        # [1, H, W, 1]
