"""Veiling glare as scattering of the RENDER itself.

Physically, glare is incident radiance convolved with a wide-tailed PSF; the
fog measured on the sky-welding line is the optimizer hand-building exactly
that (floating diffusers that brighten views near bright content). This gives
it the legal version: per-eye learnable weights over three fixed-sigma
radial kernels, convolved with the (detached, linear-light) render at 1/8
resolution. The per-frame input is the render -- available for ANY view,
including novel ones -- and a convolution of the render structurally cannot
absorb scene content, which is what sank the free per-image appearance
modules (pitfalls C3).
"""
import torch
import torch.nn.functional as F


def _srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92,
                       ((x.clamp(min=0.04045) + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    return torch.where(x <= 0.0031308, x * 12.92,
                       1.055 * x.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)


class GlarePSF(torch.nn.Module):
    SIGMAS = (4.0, 16.0, 48.0)   # px at the 1/8-res working grid

    def __init__(self, n_cams: int):
        super().__init__()
        self.logw = torch.nn.Parameter(torch.full((n_cams, len(self.SIGMAS)), -5.0))
        self._k = None

    def _kernels(self, dev):
        if self._k is None:
            ks = []
            for s in self.SIGMAS:
                r = int(3 * s)
                x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
                g = torch.exp(-0.5 * (x / s) ** 2)
                ks.append((g / g.sum(), r))
            self._k = ks
        return self._k

    def apply(self, cam_idx: int, colors, down: int = 8):
        """colors [1,H,W,3] sRGB in [0,1] -> composited sRGB."""
        lin = _srgb_to_linear(colors)
        src = lin.detach().permute(0, 3, 1, 2)          # gradients reach only
        H, W = src.shape[-2:]                           # the kernel weights
        x = F.avg_pool2d(src, down)
        w = F.softplus(self.logw[int(cam_idx)])
        g = None
        for (k1d, r), wi in zip(self._kernels(src.device), w):
            kv = k1d.view(1, 1, -1, 1).expand(3, 1, -1, 1)
            kh = k1d.view(1, 1, 1, -1).expand(3, 1, 1, -1)
            y = F.conv2d(F.conv2d(x, kv, padding=(r, 0), groups=3),
                         kh, padding=(0, r), groups=3)
            g = wi * y if g is None else g + wi * y
        g = F.interpolate(g, size=(H, W), mode="bilinear",
                          align_corners=False).permute(0, 2, 3, 1)
        return _linear_to_srgb(lin + g)
