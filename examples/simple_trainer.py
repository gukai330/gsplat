# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from gsplat.color_correct import color_correct_affine, color_correct_quadratic
from contraction import PositionContraction
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from gsplat.losses import (
    depth_l1_loss,
    l1_loss,
    opacity_reg_loss,
    scale_reg_loss,
    ssim_loss,
    total_variation_loss,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    RigCameraOptModule,
    knn,
    rig_groups_from_names,
    rgb_to_sh,
    set_random_seed,
)

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization, RasterizeMode

try:
    from gsplat.scene import GaussianScene
    from gsplat.stage import Stage
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{e.name} is not installed. The example trainers require the "
        "scene/stage helper packages, which ship with gsplat. Install gsplat with:\n"
        "    python -m pip install -e ."
    ) from e
from gsplat.cuda._wrapper import CameraModel
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.ops import inject_noise_to_position
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap


def _srgb_to_linear(x: Tensor) -> Tensor:
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: Tensor) -> Tensor:
    # The clamp inside the power branch is load-bearing, not defensive. torch.where
    # evaluates both branches in backward, d/dx x**(1/2.4) is infinite at x = 0, and
    # 0 * inf = NaN -- so without it the loss goes NaN a few hundred steps in and
    # MCMC dies inside multinomial sampling on an all-NaN opacity vector.
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.0031308, x * 12.92,
                       1.055 * x.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


def apply_exposure_prior(colors: Tensor, exposure: Tensor, alpha: Tensor) -> Tensor:
    """Re-expose the render to each image's capture exposure, in LINEAR light.

    Two corrections over the first version of this, both measured in
    reports/camera_metadata_review.md:

    1. The gain goes on linear radiance, not on the gamma-encoded render. The
       renders and the JPEG targets live in sRGB, so a raw multiply there is off
       by roughly the 2.4 exponent.
    2. The coefficient `alpha` is not pinned to 1. alpha=1 is the textbook
       pixel = radiance * shutter model, which needs a transparent ISP. This
       footage is auto-exposed: shutter is ANTI-correlated with frame brightness
       (r = -0.676 over take2's 860 frames), so alpha=1 amplifies the photometric
       inconsistency 3.9x. Let the data set alpha -- if the shutter track carries
       nothing usable, alpha goes to 0 and this reduces to the plain path.
    """
    gain = torch.exp2(alpha * exposure).reshape(-1, 1, 1, 1)
    rgb = _linear_to_srgb(_srgb_to_linear(colors[..., :3]) * gain)
    if colors.shape[-1] == 3:
        return rgb
    return torch.cat([rgb, colors[..., 3:]], dim=-1)



@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # recon360: warm start. `ckpt` above is an EVAL-ONLY path (it loads splats then
    # calls eval and returns). These two continue TRAINING from a checkpoint instead.
    # init_ckpt loads the gaussians; init_step fast-forwards the LR schedulers so the
    # run picks up the schedule of a max_steps-long run at that point, rather than
    # restarting the decay. Without init_step a "continuation" would re-anneal the
    # position LR from its initial value, which is a different optimisation
    # trajectory, not a longer one.
    init_ckpt: Optional[str] = None
    init_step: int = 0
    # write optimizer + MCMC strategy state into the checkpoint. Costs roughly 3x
    # the file size (1.42 GB vs 0.47 at cap 2M) and MEASURES AS A WASH, so it is
    # off: restoring Adam's moments across a 30k -> 60k continuation moved take2 by
    # -0.004 dB / 0.0000 SSIM, an order of magnitude under the +/-0.04 dB noise
    # floor. (An earlier A/B said "harmful"; that run was void -- load_state_dict
    # also restored the saved LR, see the restore site.) There is no MCMC state to
    # lose either -- MCMCStrategy's whole state is a precomputed binomial table.
    # Turn it on to warm-start experiments that want the moments anyway.
    save_train_state: bool = False
    # set False to reproduce the stateless warm start, for the A/B
    restore_train_state: bool = True
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path: "interp", "ellipse", "spiral", or "raw" (use captured poses as-is)
    render_traj_path: str = "interp"

    # Dataset backend: "colmap" or "ncore"
    data_type: str = "colmap"
    # Path to the Mip-NeRF 360 dataset (colmap) or NCore v4 meta-JSON file (ncore)
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: CameraModel = "pinhole"
    # Load EXIF exposure metadata from images (if available)
    load_exposure: bool = True
    # Apply centred EXIF exposure to RGB on the plain reg0 rendering path.
    # OFF by default: the implemented gain is 2**EV, which assumes pixel value is
    # proportional to radiance*shutter. This footage is auto-exposed, so shutter is
    # ANTI-correlated with frame brightness (r=-0.676 on take2's 860 frames) and the
    # gain amplifies the photometric inconsistency 3.9x instead of removing it.
    # See reports/camera_metadata_review.md before turning this back on.
    apply_exposure_prior: bool = False
    # Coefficient on the centred EV. 1.0 is the naive radiance*shutter model and is
    # wrong for auto-exposed footage; the take2 least-squares fit is about -0.22 in
    # sRGB terms. Prefer --exposure_alpha_learnable and let training settle it.
    exposure_alpha: float = 1.0
    # Optimise exposure_alpha as a single global scalar (1 parameter, so it cannot
    # overfit, and unlike --app_opt it applies to held-out views too because the
    # shutter is recorded for every frame).
    exposure_alpha_learnable: bool = False
    exposure_alpha_lr: float = 1e-2
    # Backend to train on: "cuda" for standard multi-process training,
    # or "dgx" for torch-dgx single-process multi-GPU training.
    backend: str = "cuda"

    # --- NCore-specific options (only used when data_type="ncore") ---
    # Camera sensor IDs to load (auto-detected from sequence if empty)
    ncore_camera_ids: List[str] = field(default_factory=list)
    # Point cloud source IDs to load -- accepts lidar, radar, or native point cloud
    # source IDs (auto-detected from sequence if empty). Field name kept for backward compat.
    ncore_lidar_ids: List[str] = field(default_factory=list)
    # Temporal seek offset in seconds
    ncore_seek_offset_sec: Optional[float] = None
    # Clip duration in seconds (None = full sequence)
    ncore_duration_sec: Optional[float] = None
    # Maximum number of lidar init points
    ncore_max_lidar_points: int = 500_000
    # Generic-data key for lidar point RGB colors (fallback to gray if unavailable)
    ncore_lidar_color_generic_data_name: str = "rgb"
    # NCore component group names
    ncore_poses_component_group: str = "default"
    ncore_intrinsics_component_group: str = "default"
    ncore_masks_component_group: str = "default"

    # Port for the viewer server
    port: int = 8080

    # recon360: expose the RNG seed. It was hardcoded to 42, which makes it
    # impossible to repeat a config and measure the run-to-run noise floor -- and
    # without that floor a +-0.1 dB ablation delta cannot be called signal or noise.
    seed: int = 42

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Cast SH coefficients to fp16 before feeding the SH kernel.
    # Parameters and Adam state stay fp32.
    sh_fp16: bool = False
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # recon360: global directional sky/background model (sky_model.py). A tiny
    # SH-of-view-direction environment composited at the random_bkgd site:
    # colors + sky(dir) * (1 - alpha). Direction-only is what keeps it valid on
    # held-out views (app_opt/bilagrid/ppisp all measured negative there because
    # their correction is keyed to training images). With --sky_mask_dir set,
    # pixels labelled sky (white, eroded masks -- see fisheye_sky_mask.py) also
    # get a BCE pushing alpha toward 0, which is what actually transfers the sky
    # from gaussians to the background; MCMC then relocates the freed budget.
    sky_model: bool = False
    # SH degree of the sky model ((d+1)^2 coeffs x 3 channels)
    sky_sh_degree: int = 3
    # dir with <image_name>.png sky masks, WHITE = sky. None -> composite only,
    # no alpha pressure.
    sky_mask_dir: Optional[str] = None
    # weight of the alpha->0 BCE on sky pixels
    sky_alpha_lambda: float = 0.05
    # LR for the sky SH coefficients
    sky_lr: float = 1e-2
    # Apply the alpha BCE to every valid pixel instead of a sky mask. Only has an
    # effect when --sky_mask_dir is unset.
    sky_bce_everywhere: bool = False
    # recon360: replace the BCE -log(1-a) with plain a.mean() on sky pixels.
    # BCE's gradient is 1/(1-a): warm-starting a converged model (sky alpha
    # 1-eps) hands every gaussian grazing a sky pixel a transiently enormous
    # gradient and shreds the scene (measured: non-sky held-out 22.9 -> 12.1
    # after 6k continuation steps). L1's pressure is constant and consistent,
    # which under Adam is exactly what moves parameters -- fog with no
    # photometric defence still dies, structure defended by the photometric
    # term does not get the 1e6-scale kick.
    sky_alpha_l1: bool = False
    # recon360: learn a per-eye radial gain g_e(theta) applied to the rendered
    # image before the loss (see examples/flare_model.py for why).
    flare_gain: bool = False
    flare_knots: int = 12
    flare_lr: float = 3e-3
    # recon360: geometric alternative to the sky BCE. On sky-mask pixels,
    # penalise the rendered EXPECTED HIT DISTANCE falling below sky_far_min
    # (normalized units): sky content may exist, but only far away. Unlike
    # the BCE it neither kills the sky nor leaves it unsupervised -- the S1
    # mask-out arm measured that unsupervised sky rays are free real estate
    # and the fog grows WORSE. Gated on alpha>0.5 so empty sky is legal.
    # On the fisheye path (global_z_order False) the UT projection depth is
    # Euclidean distance, which is exactly the right quantity here.
    sky_far_lambda: float = 0.0
    sky_far_min: float = 8.5
    # Apply sky_far only before this step (-1 = whole run). The fog is built
    # while densification runs; past refine_stop there is nothing left to
    # steer and the hinge is pure cost.
    sky_far_stop_iter: int = -1
    # Render the ED channel and apply the depth auxiliaries (mono_depth,
    # sky_far) only every K steps. Under Adam what moves parameters is
    # gradient CONSISTENCY, not per-step magnitude (pitfalls: contraction
    # line, kappa), so K=4 keeps the steering and gives back most of the
    # ED-render and npy-IO overhead. depth_loss (sparse SfM) is unaffected.
    aux_depth_every: int = 1
    # recon360: dir from scripts/fisheye_erp_depth.py (disp/ + scatter_e*.npz).
    # Scatter pearson on DA360 disparities: sky arrives as a VALID ~0
    # disparity, so unlike MoGe's NaN the anchor also pushes sky rays far.
    mono_depth_erp: Optional[str] = None
    # recon360: veiling-glare PSF (see examples/glare_model.py).
    glare_psf: bool = False
    glare_lr: float = 1e-2
    # Seed the point cloud with N extra points on a distant sphere, coloured by
    # the sky model in --sky_init. The alternative to a background function: give
    # the sky its own dedicated far gaussians from step 0, so MCMC never has to
    # manufacture a depth-scattered shell to explain sky pixels.
    sky_sphere_points: int = 0
    # radius of that sphere, in units of the camera-cloud radius
    sky_sphere_radius: float = 8.0
    # evaluate the SH background every Nth pixel and bilinearly upsample
    sky_eval_stride: int = 4
    # sky_init.pt from scripts/fisheye_bg_sphere.py: SH fitted to the capture's own
    # rotation-aligned per-direction median. The SH endpoint does not depend on this
    # (48 smooth parameters converge either way, measured), but the GAUSSIANS' does:
    # with a correct background from step 0 the BCE has something to hand the sky to
    # immediately, so the shell may never be built rather than being built and pushed
    # off. MCMC relocation history is path-dependent.
    sky_init: Optional[str] = None

    # recon360: optimise POSITIONS in a contracted radial domain -- the optimiser
    # holds `u`, the rasterizer gets `x = uncontract(u)`. "none" is a true no-op.
    # Motivation and the measurements it rests on: examples/contraction.py and
    # reports/contracted_position_report.md. Note the far field's problem is NOT
    # gradient conditioning (Adam divides that out); it is that the angular step
    # falls as 1/r and the far radius is set by MCMC noise rather than the loss.
    position_contraction: Literal["none", "log", "mip360"] = "none"
    # Where the contraction starts, in world units of the NORMALIZED frame.
    # Defaults to parser.scene_scale, the camera-cloud radius -- which is also the
    # constant means_lr is multiplied by, so ry = 1 is at once the edge of the
    # walked region, the contraction boundary, and the unit of the position LR.
    contraction_radius: Optional[float] = None

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # recon360: adaptive regularizer guard. On the ERP-cube route the MCMC
    # opacity/scale penalty can take the scene fully transparent at a high cap
    # (or with a weak data term -- exposure flicker, soft poses) and PSNR lands
    # at 8-13 dB. Watch median sigmoid(opacity) and back the penalty off before
    # that happens instead of making the user guess --reg per dataset.
    # NOT because "the penalty grows with the gaussian count": opacity_reg_loss
    # is sigmoid(opacities).mean(), a mean, which does not scale with N. The
    # cap-dependence is measured; the mechanism is not established. The guard
    # also does NOT rescue a collapse (10.78 dB vs 18.64 for plain reg 0) --
    # by the time opacity has fallen the damage is done. On the fisheye-native
    # route reg 0.01 does not collapse at all and is the default there
    # (2026-08-15, reports/reg_default_report.md).
    adaptive_reg: bool = False
    # Back off when the median opacity drops below this (healthy runs measured
    # 0.02-0.09; fully collapsed runs measured 1.6e-6).
    adaptive_reg_median: float = 5e-3
    # How often to check
    adaptive_reg_every: int = 200
    # Multiply both regularizers by this on each trigger
    adaptive_reg_factor: float = 0.5
    # Stop backing off here (0.0 = allow disabling the regularizer entirely)
    adaptive_reg_min: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # recon360: constrain pose_opt to one 6-DoF delta per cubemap rig (ERP frame)
    # instead of one per image, so a frame's faces cannot drift apart.
    pose_opt_rig: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Post-processing method for appearance correction (experimental)
    post_processing: Optional[Literal["bilateral_grid", "ppisp"]] = None
    # Use fused implementation for bilateral grid (only applies when post_processing="bilateral_grid")
    bilateral_grid_fused: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    # Enable PPISP controller
    ppisp_use_controller: bool = True
    # Use controller distillation in PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_distillation: bool = True
    # Controller activation ratio for PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_activation_num_steps: int = 25_000
    # Color correction method for cc_* metrics (only applies when post_processing is set)
    color_correct_method: Literal["affine", "quadratic"] = "affine"
    # Compute color-corrected metrics (cc_psnr, cc_ssim, cc_lpips) during evaluation
    use_color_correction_metric: bool = False

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # recon360: DENSE monocular depth prior (scripts/mono_depth.py writes
    # <data_dir>/mono_depth/<image>.npy). Note --depth_loss above supervises with
    # SfM *sparse* point depths -- a few hundred pixels per image -- which is the
    # same weak data term we are trying to prop up. This one covers every pixel.
    mono_depth: bool = False
    # Weight for the monocular depth loss
    mono_depth_lambda: float = 0.05
    # "pearson": 1 - Pearson correlation of disparities, invariant to any residual
    #   per-image scale+shift left over from the SfM alignment.
    # "l1": L1 straight on the aligned depth -- a stronger constraint,
    #   but it trusts the per-image affine fit.
    mono_depth_mode: Literal["pearson", "l1"] = "pearson"
    # Stop applying the prior after this step (-1 = never). Late in training the
    # photometric term should own the fine detail.
    mono_depth_stop_iter: int = -1

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False
    # recon360: for a fisheye wider than 180 deg, z-depth is the wrong quantity to
    # both cull and sort by -- half the field of view has z < 0, so near/far culling
    # (ProjectionUT3DGSFused.cu:131) deletes it and the sort key changes sign across
    # the equator. gsplat already supports Euclidean distance instead; it just was
    # not exposed here. Set False together with --camera_model fisheye --with_ut
    # whenever the FOV exceeds 180 deg.
    global_z_order: bool = True
    # recon360: keep the images as captured and let the rasterizer model the lens.
    # The Parser otherwise remaps OPENCV_FISHEYE to a perspective image, which cannot
    # represent a >180 deg fisheye at all. With this off, the COLMAP distortion
    # coefficients are forwarded to gsplat as `radial_coeffs` instead -- for
    # OPENCV_FISHEYE they are the same Kannala-Brandt k1..k4 the CUDA camera model
    # expects, so nothing is converted or approximated. Needs --with_ut, because the
    # non-UT projection ignores radial_coeffs entirely (Utils.cuh fisheye_proj).
    undistort: bool = True

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            if strategy.noise_injection_stop_iter >= 0:
                strategy.noise_injection_stop_iter = int(
                    strategy.noise_injection_stop_iter * factor
                )
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    contraction: Optional[PositionContraction] = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm" or init_type == "lidar":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm, random, or lidar")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    # recon360: store the contracted coordinate. AFTER knn, which needs real
    # world distances to size the initial gaussians. `u` is in world units and
    # the map is the identity inside the contraction radius, so `means_lr *
    # scene_scale` keeps its exact meaning -- including the value the trainer
    # later hands to MCMC as its noise scale.
    if contraction is not None and contraction.active:
        points = contraction.to_param(points)

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    # recon360 fix: `fused` is a torch.optim.Adam argument. SelectiveAdam takes only
    # (params, eps, betas) and SparseAdam has no fused path either, so passing it
    # unconditionally makes --visible_adam and --sparse_grad die at startup with
    # "TypeError: SelectiveAdam.__init__() got an unexpected keyword argument 'fused'".
    extra_opt_kwargs = {"fused": True} if optimizer_class is torch.optim.Adam else {}
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            **extra_opt_kwargs,
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(cfg.seed + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"{cfg.backend}:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # recon360: record of every adaptive_reg back-off, dumped next to the stats
        self.adaptive_reg_events = []

        # Load data: Training data should contain initial points and colors.
        if cfg.data_type == "ncore":
            from datasets.ncore import NCoreDataset, NCoreParser

            self.parser = NCoreParser(
                meta_json_path=cfg.data_dir,
                factor=1.0 / cfg.data_factor if cfg.data_factor > 1 else 1.0,
                test_every=cfg.test_every,
                camera_ids=cfg.ncore_camera_ids or None,
                lidar_ids=cfg.ncore_lidar_ids or None,
                seek_offset_sec=cfg.ncore_seek_offset_sec,
                duration_sec=cfg.ncore_duration_sec,
                max_lidar_points=cfg.ncore_max_lidar_points,
                lidar_color_generic_data_name=cfg.ncore_lidar_color_generic_data_name,
                poses_component_group=cfg.ncore_poses_component_group,
                intrinsics_component_group=cfg.ncore_intrinsics_component_group,
                masks_component_group=cfg.ncore_masks_component_group,
                normalize_world_space=cfg.normalize_world_space,
            )
            self.trainset = NCoreDataset(self.parser, split="train")
            self.valset = NCoreDataset(self.parser, split="val")
            self.ncore_camera_data = [
                self.parser.camera_render_data[cam_id]
                for cam_id in self.parser.camera_ids
            ]
            if (
                any(d.camera_model == "ftheta" for d in self.ncore_camera_data)
                and not cfg.with_eval3d
            ):
                print(
                    "[NCore] Warning: FTheta cameras detected; pass --with-eval3d True for correct results."
                )
        else:
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
                undistort=cfg.undistort,
                load_exposure=cfg.load_exposure,
            )
            self.trainset = Dataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
                load_mono_depth=cfg.mono_depth,
                sky_mask_dir=cfg.sky_mask_dir,
                mono_depth_erp=cfg.mono_depth_erp,
            )
            self.valset = Dataset(self.parser, split="val")
        if cfg.sky_sphere_points > 0:
            import numpy as _np
            _cams = _np.asarray(self.parser.camtoworlds, float)[:, :3, 3]
            _ctr = _cams.mean(0)
            _rad = float(_np.linalg.norm(_cams - _ctr, axis=1).max()) * cfg.sky_sphere_radius
            _n = cfg.sky_sphere_points
            _i = _np.arange(_n) + 0.5                      # Fibonacci sphere
            _phi = _np.arccos(1 - 2 * _i / _n)
            _th = _np.pi * (1 + 5 ** 0.5) * _i
            _d = _np.stack([_np.cos(_th) * _np.sin(_phi), _np.sin(_th) * _np.sin(_phi),
                            _np.cos(_phi)], 1)
            if cfg.sky_init is not None:
                from sky_model import SkyModel as _SM
                _si = torch.load(cfg.sky_init, map_location="cpu", weights_only=False)
                _m = _SM(int(_si["sh_degree"]))
                _m.coeffs.data = _si["coeffs"]
                with torch.no_grad():
                    _rgb = _m(torch.from_numpy(_d).float()).numpy() * 255.0
            else:
                _rgb = _np.full((_n, 3), 200.0)
            self.parser.points = _np.concatenate([self.parser.points, _ctr + _rad * _d])
            self.parser.points_rgb = _np.concatenate(
                [self.parser.points_rgb, _rgb]).astype(self.parser.points_rgb.dtype)
            print(f"[recon360] seeded {_n} sky-sphere points at {_rad:.2f} "
                  f"({cfg.sky_sphere_radius:g}x the camera radius)")

        self.radial_coeffs_table = None
        if not cfg.undistort:
            import numpy as _np
            _tbl = torch.zeros(self.parser.num_cameras, 4)
            for _cid, _idx in self.parser.camera_id_to_idx.items():
                _p = _np.asarray(self.parser.params_dict[_cid], dtype=_np.float32)
                _tbl[_idx, : min(4, len(_p))] = torch.from_numpy(_p[:4])
            self.radial_coeffs_table = _tbl.to(self.device)
            print(f"[recon360] undistort=False: forwarding radial_coeffs for "
                  f"{self.parser.num_cameras} camera(s) -> {_tbl.tolist()}")
            if not cfg.with_ut:
                print("[recon360] !! --with_ut is OFF, so the non-UT projection will "
                      "IGNORE these coefficients (Utils.cuh fisheye_proj is pure "
                      "equidistant). Add --with_ut --with_eval3d.")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # recon360: the position parameterisation. Built before the splats,
        # because create_splats_with_optimizers stores `u` rather than `x`. The
        # centre is the camera centroid: a radial contraction has to have one,
        # and that is the only centre the rest of this route already agrees on
        # (it is what the probes in reports/contracted_position_report.md bin
        # about). The radius is parser.scene_scale -- the RAW camera-cloud
        # radius, not self.scene_scale, which carries an extra 1.1 * global_scale.
        self.contraction = PositionContraction(
            cfg.position_contraction,
            torch.as_tensor(self.parser.camtoworlds[:, :3, 3]).float().mean(0),
            float(cfg.contraction_radius if cfg.contraction_radius is not None
                  else self.parser.scene_scale),
        ).to(self.device)
        if self.contraction.active:
            print(f"[recon360] {self.contraction}", flush=True)

        # recon360: MCMC adds its position noise as `Sigma @ xi` in WORLD space,
        # in place on params["means"] -- which under contraction is `u`. Feeding a
        # world covariance to a contracted coordinate would fling the far field by
        # a factor of J. So the strategy's own injection is switched off and the
        # trainer does the round trip: uncontract, perturb, contract back. That is
        # exact, not the first-order `J^-1 Sigma J^-T`, and it keeps the existing
        # fused CUDA kernel. relocate/sample_add need nothing: they copy the
        # position verbatim, and a verbatim copy is the same point either way --
        # which is also why contraction cannot touch MCMC's inward budget drift.
        self._mcmc_noise_stop = None
        if self.contraction.active and isinstance(cfg.strategy, MCMCStrategy):
            self._mcmc_noise_stop = cfg.strategy.noise_injection_stop_iter
            cfg.strategy.noise_injection_stop_iter = 0  # off, inside the strategy
        if self.contraction.active and cfg.sparse_grad:
            raise ValueError(
                "--sparse_grad with --position_contraction is untested: the "
                "position gradient now reaches the parameter through the "
                "contraction map, and the dense->sparse conversion in the train "
                "loop indexes the parameter's own .grad. Verify it before using "
                "the two together.")

        if self.parser.num_cameras > 1 and cfg.batch_size != 1:
            raise ValueError(
                f"When using multiple cameras ({self.parser.num_cameras} found), batch_size must be 1, "
                f"but got batch_size={cfg.batch_size}."
            )
        if cfg.post_processing == "ppisp" and cfg.batch_size != 1:
            raise ValueError(
                f"PPISP post-processing requires batch_size=1, got batch_size={cfg.batch_size}"
            )
        if cfg.post_processing is not None and world_size > 1:
            raise ValueError(
                f"Post-processing ({cfg.post_processing}) requires single-GPU training, "
                f"but world_size={world_size}."
            )

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            contraction=self.contraction,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        self.scene = GaussianScene.from_splats(self.splats, id="scene")
        self.splats = self.scene.splats

        # recon360: global directional sky/background model
        self.sky_module = None
        self.sky_optimizers = []
        if cfg.sky_model:
            from sky_model import SkyModel, SkyRayCache

            self.sky_module = SkyModel(cfg.sky_sh_degree).to(self.device)
            if cfg.sky_init is not None:
                _si = torch.load(cfg.sky_init, map_location=self.device,
                                 weights_only=False)
                if int(_si["sh_degree"]) != cfg.sky_sh_degree:
                    raise ValueError(
                        f"--sky_init was fitted at SH degree {_si['sh_degree']} but "
                        f"--sky_sh_degree is {cfg.sky_sh_degree}")
                self.sky_module.coeffs.data = _si["coeffs"].to(self.device)
                print(f"[recon360] sky initialised from {cfg.sky_init}")
            self.sky_optimizers = [
                torch.optim.Adam(self.sky_module.parameters(), lr=cfg.sky_lr)
            ]
            self.sky_rays = SkyRayCache(cfg.camera_model, self.device)
            print(f"[recon360] sky model on: SH degree {cfg.sky_sh_degree}, "
                  f"masks {cfg.sky_mask_dir or '(none: composite only)'}, "
                  f"alpha lambda {cfg.sky_alpha_lambda}")
        self.flare_module = None
        self.flare_optimizers = []
        if cfg.flare_gain:
            from flare_model import FlareGain

            self.flare_module = FlareGain(
                len(self.parser.Ks_dict), cfg.flare_knots).to(self.device)
            self.flare_optimizers = [
                torch.optim.Adam(self.flare_module.parameters(), lr=cfg.flare_lr)
            ]
            print(f"[recon360] flare gain on: {len(self.parser.Ks_dict)} cams x "
                  f"{cfg.flare_knots} knots, lr {cfg.flare_lr}")

        self.glare_module = None
        self.glare_optimizers = []
        if cfg.glare_psf:
            from glare_model import GlarePSF

            self.glare_module = GlarePSF(len(self.parser.Ks_dict)).to(self.device)
            self.glare_optimizers = [
                torch.optim.Adam(self.glare_module.parameters(), lr=cfg.glare_lr)
            ]
            print(f"[recon360] glare PSF on: {len(self.parser.Ks_dict)} cams, "
                  f"sigmas {GlarePSF.SIGMAS}, lr {cfg.glare_lr}")

        self.stage = Stage()
        self.stage.add_scene(self.scene, self.rasterize_splats)
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            if cfg.pose_opt_rig:
                # image_id in a batch is the index into the TRAINSET, so the rig
                # grouping and the reference poses must be built over trainset
                # order too -- and from parser.camtoworlds, which is already in
                # the normalized world frame the trainer trains in.
                tr_idx = self.trainset.indices
                names = [self.parser.image_names[i] for i in tr_idx]
                group_ids, n_rigs, ref_local = rig_groups_from_names(names)
                ref_c2w = torch.from_numpy(
                    np.stack([self.parser.camtoworlds[tr_idx[j]] for j in ref_local.tolist()])
                ).float()
                self.pose_adjust = RigCameraOptModule(group_ids, ref_c2w).to(self.device)
                print(
                    f"[pose_opt_rig] {len(names)} train images -> {n_rigs} rigs "
                    f"({len(names)/max(n_rigs,1):.1f} faces/rig), "
                    f"{n_rigs*6} free pose parameters instead of {len(names)*6}"
                )
            else:
                self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.post_processing_module = None
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_module = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
        elif cfg.post_processing == "ppisp":
            ppisp_config = PPISPConfig(
                use_controller=cfg.ppisp_use_controller,
                controller_distillation=cfg.ppisp_controller_distillation,
                controller_activation_ratio=cfg.ppisp_controller_activation_num_steps
                / cfg.max_steps,
            )
            self.post_processing_module = PPISP(
                num_cameras=self.parser.num_cameras,
                num_frames=len(self.trainset),
                config=ppisp_config,
            ).to(self.device)

        # Global coefficient on the recorded EV. One scalar, shared by every image,
        # so it carries over to held-out views -- which is exactly what app_opt and
        # bilagrid cannot do (their correction lives in a per-training-image module).
        self.exposure_alpha = torch.tensor(
            float(cfg.exposure_alpha), device=self.device,
            requires_grad=cfg.exposure_alpha_learnable)
        self.exposure_alpha_optimizers = (
            [torch.optim.Adam([self.exposure_alpha], lr=cfg.exposure_alpha_lr)]
            if cfg.exposure_alpha_learnable else [])

        self.post_processing_optimizers = []
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_optimizers = [
                torch.optim.Adam(
                    self.post_processing_module.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]
        elif cfg.post_processing == "ppisp":
            self.post_processing_optimizers = (
                self.post_processing_module.create_optimizers()
            )

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        # Track if Gaussians are frozen (for controller distillation)
        self._gaussians_frozen = False

    def freeze_gaussians(self):
        """Freeze all Gaussian parameters for controller distillation.

        This prevents Gaussians from being updated by any loss (including regularization)
        while the controller learns to predict per-frame corrections.
        """
        if self._gaussians_frozen:
            return

        for name, param in self.splats.items():
            param.requires_grad = False

        self._gaussians_frozen = True
        print("[Distillation] Gaussian parameters frozen")

    @torch.no_grad()
    def _inject_noise_in_world(self, step: int, lr: float) -> None:
        """recon360: MCMC's position noise, applied in world space under a
        contracted parameterisation. See the note in __init__."""
        strat = self.cfg.strategy
        stop = self._mcmc_noise_stop
        if stop is not None and stop >= 0 and step >= stop:
            return
        world = self.contraction.to_world(self.splats["means"].detach()).contiguous()
        inject_noise_to_position(
            params={"means": world, "quats": self.splats["quats"],
                    "scales": self.splats["scales"],
                    "opacities": self.splats["opacities"]},
            optimizers={},
            state={},
            noise_scale=lr * strat.noise_lr,
            t=strat.noise_opacity_t,
            k=strat.noise_opacity_k,
        )
        self.splats["means"].data.copy_(self.contraction.to_param(world))

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[RasterizeMode] = None,
        camera_model: Optional[CameraModel] = None,
        frame_idcs: Optional[Tensor] = None,
        camera_idcs: Optional[Tensor] = None,
        exposure: Optional[Tensor] = None,
        splats: Optional[torch.nn.ParameterDict] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        splats = splats if splats is not None else self.splats
        # recon360: `means` is the PARAMETER. Under --position_contraction that is
        # a contracted coordinate, and everything below -- the rasterizer, and
        # app_opt's view directions -- needs world space. This is the single place
        # the map is applied in the forward pass, so autograd delivers J^T here.
        means = self.contraction.to_world(splats["means"])  # [N, 3]
        # quats = F.normalize(splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = splats["quats"]  # [N, 4]
        scales = torch.exp(splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            # Cast before the cat so both the cat and the SH kernel run on fp16.
            if self.cfg.sh_fp16:
                colors = torch.cat(
                    [splats["sh0"].half(), splats["shN"].half()], 1
                )  # [N, K, 3]
            else:
                colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        ftheta_coeffs = None
        radial_coeffs = None
        tangential_coeffs = None
        thin_prism_coeffs = None
        with_ut = self.cfg.with_ut

        if (radial_coeffs is None and getattr(self, "radial_coeffs_table", None) is not None
                and camera_idcs is not None):
            # shape must mirror viewmats' leading dims: this trainer passes
            # viewmats as [C, 4, 4] with no batch axis, so radial_coeffs is [C, 4].
            radial_coeffs = self.radial_coeffs_table[
                camera_idcs.reshape(-1).long()].reshape(-1, 4)
        if camera_idcs is not None and hasattr(self, "ncore_camera_data"):
            cam = self.ncore_camera_data[camera_idcs.item()]
            camera_model = cam.camera_model
            ftheta_coeffs = cam.ftheta_coeffs
            if cam.radial_coeffs is not None:
                radial_coeffs = (
                    torch.from_numpy(cam.radial_coeffs).to(means.device).unsqueeze(0)
                )
            if cam.tangential_coeffs is not None:
                tangential_coeffs = (
                    torch.from_numpy(cam.tangential_coeffs)
                    .to(means.device)
                    .unsqueeze(0)
                )
            if cam.thin_prism_coeffs is not None:
                thin_prism_coeffs = (
                    torch.from_numpy(cam.thin_prism_coeffs)
                    .to(means.device)
                    .unsqueeze(0)
                )

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv_ex(camtoworlds).inverse,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=camera_model,
            with_ut=with_ut,
            with_eval3d=self.cfg.with_eval3d,
            global_z_order=self.cfg.global_z_order,
            ftheta_coeffs=ftheta_coeffs,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            **kwargs,
        )
        if (exposure is not None and self.cfg.apply_exposure_prior
                and self.cfg.post_processing is None):
            # The splats represent reference-exposure radiance; compare each
            # image against a render scaled to that capture exposure.
            render_colors = apply_exposure_prior(
                render_colors, exposure, self.exposure_alpha)
        if masks is not None:
            render_colors[~masks] = 0

        if self.cfg.post_processing is not None:
            # Create pixel coordinates [H, W, 2] with +0.5 center offset
            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(height, device=self.device) + 0.5,
                torch.arange(width, device=self.device) + 0.5,
                indexing="ij",
            )
            pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1)  # [H, W, 2]

            # Split RGB from extra channels (e.g. depth) for post-processing
            rgb = render_colors[..., :3]
            extra = render_colors[..., 3:] if render_colors.shape[-1] > 3 else None

            if self.cfg.post_processing == "bilateral_grid":
                if frame_idcs is not None:
                    grid_xy = (
                        pixel_coords / torch.tensor([width, height], device=self.device)
                    ).unsqueeze(0)
                    rgb = slice(
                        self.post_processing_module,
                        grid_xy.expand(rgb.shape[0], -1, -1, -1),
                        rgb,
                        frame_idcs.unsqueeze(-1),
                    )["rgb"]
            elif self.cfg.post_processing == "ppisp":
                camera_idx = camera_idcs.item() if camera_idcs is not None else None
                frame_idx = frame_idcs.item() if frame_idcs is not None else None
                rgb = self.post_processing_module(
                    rgb=rgb,
                    pixel_coords=pixel_coords,
                    resolution=(width, height),
                    camera_idx=camera_idx,
                    frame_idx=frame_idx,
                    exposure_prior=exposure,
                )

            render_colors = (
                torch.cat([rgb, extra], dim=-1) if extra is not None else rgb
            )

        return render_colors, render_alphas, info

    def sky_background(self, camera_idx, Ks, camtoworlds, width, height):
        """[1, H, W, 3] sky RGB for this view, from the same KB unprojection
        the rasterizer's fisheye path models (radial_coeffs included)."""
        idx = int(camera_idx.reshape(-1)[0])
        rc = None
        if getattr(self, "radial_coeffs_table", None) is not None:
            rc = self.radial_coeffs_table[idx]
        # Evaluate on a coarse grid and upsample. A degree-d SH is band-limited
        # far below pixel resolution, but sh_basis materialises (d+1)^2 tensors
        # the size of the image: at degree 8 that is 81 x 14.7 MB per step, ~2.5
        # GB of intermediates, which is enough to wedge two concurrent trainings
        # on a 10 GB card (measured -- both froze at ~700 steps).
        step = max(1, int(self.cfg.sky_eval_stride))
        dirs = self.sky_rays.dirs_world(
            idx, Ks[0], width, height, camtoworlds[0], rc)
        if step == 1:
            return self.sky_module(dirs)
        coarse = self.sky_module(dirs[:, ::step, ::step])
        return torch.nn.functional.interpolate(
            coarse.permute(0, 3, 1, 2), size=(height, width),
            mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

    def mono_depth_loss(self, depths, gt, alphas, masks):
        """recon360: dense monocular depth prior against the rendered depth.

        depths [B,H,W,1] rendered expected depth, gt [B,H,W] prior (NaN = invalid),
        alphas [B,H,W,1], masks [B,H,W] bool or None.

        Pixels are only supervised where the render is actually opaque -- an
        under-populated region renders alpha~0 with a meaningless expected depth,
        and pulling that toward the prior would fight densification rather than help it.
        """
        cfg = self.cfg
        d = depths[..., 0]
        valid = torch.isfinite(gt) & (gt > 0) & (alphas[..., 0] > 0.5) & (d > 0)
        if masks is not None:
            valid = valid & masks
        n = int(valid.sum())
        if n < 1024:
            return depths.new_zeros(())

        if cfg.mono_depth_mode == "pearson":
            # Correlate in disparity space: bounded, invariant to any residual
            # per-image scale+shift, and it weights near geometry rather than
            # letting one distant surface dominate the statistic.
            x = (1.0 / d.clamp_min(1e-6))[valid]
            y = (1.0 / gt.clamp_min(1e-6))[valid]
            x = x - x.mean()
            y = y - y.mean()
            return 1.0 - (x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-8)
        # "l1": trusts the per-image affine alignment; normalized by scene scale
        # so the weight means the same thing across captures.
        return (d[valid] - gt[valid]).abs().mean() / self.scene_scale

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = cfg.init_step

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        # Post-processing module has a learning rate schedule
        if cfg.post_processing == "bilateral_grid":
            # Linear warmup + exponential decay
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.post_processing_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.post_processing_optimizers[0],
                            gamma=0.01 ** (1.0 / max_steps),
                        ),
                    ]
                )
            )
        elif cfg.post_processing == "ppisp":
            ppisp_schedulers = self.post_processing_module.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=max_steps,
            )
            schedulers.extend(ppisp_schedulers)

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        if init_step > 0:
            for _ in range(init_step):
                for scheduler in schedulers:
                    scheduler.step()
            print(f"[warm start] schedulers advanced to step {init_step}; "
                  f"means lr {schedulers[0].get_last_lr()[0]:.3e}", flush=True)

        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # Freeze Gaussians when PPISP controller distillation starts
            if (
                cfg.post_processing == "ppisp"
                and cfg.ppisp_use_controller
                and cfg.ppisp_controller_distillation
                and step >= cfg.ppisp_controller_activation_num_steps
            ):
                self.freeze_gaussians()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            exposure = (
                data["exposure"].to(device) if "exposure" in data else None
            )  # [B,]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]
            mono_depth_gt = (
                data["mono_depth"].to(device) if "mono_depth" in data else None
            )  # [1, H, W], NaN where the prior is invalid

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.stage.render(
                self.scene.id,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if (cfg.depth_loss or (
                    (cfg.mono_depth or cfg.sky_far_lambda > 0
                     or cfg.mono_depth_erp is not None)
                    and step % cfg.aux_depth_every == 0)) else "RGB",
                masks=masks,
                frame_idcs=image_ids,
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            if self.sky_module is not None:
                sky_rgb = self.sky_background(
                    data["camera_idx"], Ks, camtoworlds, width, height)
                colors = colors + sky_rgb * (1.0 - alphas)

            if self.flare_module is not None:
                colors = colors * self.flare_module.gain_image(
                    int(data["camera_idx"].reshape(-1)[0]), Ks[0], width, height)

            if self.glare_module is not None:
                colors = self.glare_module.apply(
                    int(data["camera_idx"].reshape(-1)[0]), colors)

            # While Gaussians are frozen for PPISP controller distillation the render
            # output has requires_grad=False, so densification bookkeeping (e.g.
            # DefaultStrategy's retain_grad) is both invalid and unnecessary.
            if not self._gaussians_frozen:
                self.cfg.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            # loss
            if masks is not None:
                # Exclude masked pixels (e.g. ego vehicle) from L1.
                # For SSIM (patch-based), zero out both sides at masked locations
                # so masked patches don't pull colors toward an arbitrary value.
                l1loss = l1_loss(colors[masks], pixels[masks]).mean()
                colors_ssim = colors * masks[..., None]
                pixels_ssim = pixels * masks[..., None]
            else:
                l1loss = l1_loss(colors, pixels).mean()
                colors_ssim = colors
                pixels_ssim = pixels
            ssimloss = ssim_loss(
                colors_ssim.permute(0, 3, 1, 2), pixels_ssim.permute(0, 3, 1, 2)
            )
            loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)
            skyloss = None
            if (self.sky_module is not None and "sky_mask" not in data
                    and cfg.sky_alpha_lambda > 0 and cfg.sky_bce_everywhere):
                # No sky mask at all: push alpha toward 0 on EVERY valid pixel and
                # let the photometric term hold up whatever is really there. The
                # mask machinery is what produced the protection rings welded to
                # the cables, so this asks whether the data alone can do the job.
                a = alphas[..., 0]
                if masks is not None:
                    a = a[masks]
                if cfg.sky_alpha_l1:
                    skyloss = a.mean()
                else:
                    skyloss = -torch.log1p(-a.clamp(max=1.0 - 1e-6)).mean()
                loss = loss + cfg.sky_alpha_lambda * skyloss
            elif self.sky_module is not None and "sky_mask" in data:
                # BCE pushing alpha -> 0 where the (eroded) sky mask says sky.
                # Spatially targeted, unlike the global opacity_reg mean that
                # collapses scenes here -- but watch ckpt_health's opacity
                # median on any new capture anyway.
                skym = data["sky_mask"].to(device)  # [1, H, W]
                if masks is not None:
                    skym = skym & masks
                if skym.any():
                    a = alphas[..., 0][skym].clamp(max=1.0 - 1e-6)
                    if cfg.sky_alpha_l1:
                        skyloss = a.mean()
                    else:
                        skyloss = -torch.log1p(-a).mean()
                    loss = loss + cfg.sky_alpha_lambda * skyloss
            if (cfg.sky_far_lambda > 0 and "sky_mask" in data
                    and depths is not None
                    and (cfg.sky_far_stop_iter < 0
                         or step < cfg.sky_far_stop_iter)):
                skym_f = data["sky_mask"].to(device)
                if masks is not None:
                    skym_f = skym_f & masks
                if mono_depth_gt is not None:
                    # MoGe's validity mask is a per-pixel second opinion on
                    # "really sky": it returns geometry for pale poles the
                    # segmenter swallows (no finite point exists for true sky).
                    # Without this gate the hinge dims the near mass on
                    # mislabelled pole pixels -- measured as semi-transparent
                    # poles in the viewer, the same wound the BCE used to
                    # inflict through the same mask errors.
                    skym_f = skym_f & ~torch.isfinite(mono_depth_gt)
                near = skym_f & (alphas[..., 0] > 0.5)
                if near.any():
                    d = depths[..., 0][near]
                    skyfar = torch.relu(1.0 - d / cfg.sky_far_min).mean()
                    loss = loss + cfg.sky_far_lambda * skyfar
            if (cfg.mono_depth_erp is not None and depths is not None
                    and "mono_pts" in data
                    and (cfg.mono_depth_stop_iter < 0
                         or step < cfg.mono_depth_stop_iter)):
                pts = data["mono_pts"].to(device)          # [1, M, 2] pixels
                gtd = data["mono_gt_disp"].to(device)      # [1, M]
                grid = torch.stack(
                    [pts[..., 0] / (width - 1) * 2 - 1,
                     pts[..., 1] / (height - 1) * 2 - 1], -1)[:, None]
                pd = torch.nn.functional.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True)[:, 0, 0]
                pa = torch.nn.functional.grid_sample(
                    alphas.permute(0, 3, 1, 2), grid, align_corners=True)[:, 0, 0]
                vm = torch.isfinite(gtd) & (pd > 1e-6) & (pa > 0.5)
                if vm.sum() >= 512:
                    pr = 1.0 / pd[vm]
                    gt = gtd[vm]
                    pr = pr - pr.mean()
                    gt = gt - gt.mean()
                    dn = pr.norm() * gt.norm()
                    if dn > 0:
                        loss = loss + cfg.mono_depth_lambda * (
                            1.0 - (pr * gt).sum() / dn)

            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                depthloss = depth_l1_loss(
                    depths, depths_gt, scene_scale=self.scene_scale
                )
                loss += depthloss * cfg.depth_lambda
            if (
                cfg.mono_depth
                and mono_depth_gt is not None
                and depths is not None
                and (cfg.mono_depth_stop_iter < 0 or step < cfg.mono_depth_stop_iter)
            ):
                monodepthloss = self.mono_depth_loss(
                    depths, mono_depth_gt, alphas, masks
                )
                loss = loss + cfg.mono_depth_lambda * monodepthloss
            if cfg.post_processing == "bilateral_grid":
                post_processing_reg_loss = 10 * total_variation_loss(
                    self.post_processing_module.grids
                )
                loss += post_processing_reg_loss
            elif cfg.post_processing == "ppisp":
                post_processing_reg_loss = (
                    self.post_processing_module.get_regularization_loss()
                )
                loss += post_processing_reg_loss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * opacity_reg_loss(self.splats["opacities"])
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * scale_reg_loss(self.splats["scales"])

            loss.backward()

            # recon360: adaptive regularizer guard. One-directional -- we only ever
            # back the penalty off, never restore it, so the run cannot oscillate.
            if (
                cfg.adaptive_reg
                and step > 0
                and step % cfg.adaptive_reg_every == 0
                and (cfg.opacity_reg > 0.0 or cfg.scale_reg > 0.0)
            ):
                with torch.no_grad():
                    opa_med = torch.sigmoid(self.splats["opacities"]).median().item()
                if world_rank == 0 and self.writer is not None:
                    self.writer.add_scalar("train/opacity_median", opa_med, step)
                if opa_med < cfg.adaptive_reg_median:
                    old_o, old_s = cfg.opacity_reg, cfg.scale_reg
                    cfg.opacity_reg = max(
                        cfg.adaptive_reg_min, cfg.opacity_reg * cfg.adaptive_reg_factor
                    )
                    cfg.scale_reg = max(
                        cfg.adaptive_reg_min, cfg.scale_reg * cfg.adaptive_reg_factor
                    )
                    print(
                        f"\n[adaptive_reg] step {step}: median opacity {opa_med:.2e} < "
                        f"{cfg.adaptive_reg_median:.1e} (collapse risk) -- backing reg off "
                        f"{old_o:.4g}/{old_s:.4g} -> {cfg.opacity_reg:.4g}/{cfg.scale_reg:.4g}",
                        flush=True,
                    )
                    self.adaptive_reg_events.append(
                        {
                            "step": step,
                            "opa_median": opa_med,
                            "opacity_reg": cfg.opacity_reg,
                            "scale_reg": cfg.scale_reg,
                        }
                    )

            desc = f"loss={loss.item():.3f}| sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                if skyloss is not None:
                    self.writer.add_scalar("train/skyloss", skyloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.post_processing is not None:
                    self.writer.add_scalar(
                        "train/post_processing_reg_loss",
                        post_processing_reg_loss.item(),
                        step,
                    )
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    # allocated is what the tensors need; reserved is what the
                    # caching allocator holds on the device and is the number that
                    # decides whether a second job fits. They differ by ~2x here.
                    "mem_reserved": torch.cuda.max_memory_reserved() / 1024**3,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {
                    "step": step,
                    "scene_id": self.scene.id,
                    "splats": self.splats.state_dict(),
                }
                if self.contraction.active:
                    # recon360: a checkpoint always stores WORLD means. That keeps
                    # every downstream reader working unchanged (gsplat_data's
                    # load_splats, export_ply, ckpt_health, tsdf_from_splats,
                    # viewer_qt) and stops a .pt existing whose coordinate
                    # convention is only recoverable from the run's flags.
                    # --init-ckpt re-contracts on load. Saved Adam moments, if
                    # any, are u-space and stay consistent because the means they
                    # are restored alongside get contracted back.
                    data["splats"] = dict(data["splats"])
                    data["splats"]["means"] = self.contraction.to_world(
                        self.splats["means"].detach())
                    data["position_contraction"] = self.contraction.state()
                if cfg.save_train_state:
                    # recon360: everything a TRUE continuation needs. Without it,
                    # --init-ckpt restarts Adam's moments and the MCMC strategy's
                    # own bookkeeping from scratch: measured, the first 2k-step
                    # window after a warm start reads 0.0659 against the 0.0645 the
                    # run ended on, and it takes ~8k steps to recover. Tensors are
                    # moved to CPU so the file loads on any device.
                    data["optimizers"] = {
                        k: o.state_dict() for k, o in self.optimizers.items()}
                    st = self.strategy_state
                    data["strategy_state"] = {
                        k: (v.detach().cpu() if torch.is_tensor(v) else v)
                        for k, v in st.items()} if isinstance(st, dict) else None
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                if self.post_processing_module is not None:
                    data["post_processing"] = self.post_processing_module.state_dict()
                if self.sky_module is not None:
                    # eval_masked composites the same background, so the
                    # checkpoint carries the model, not a matching flag.
                    data["sky_module"] = self.sky_module.state_dict()
                    data["sky_sh_degree"] = cfg.sky_sh_degree
                if cfg.apply_exposure_prior:
                    # eval_masked has to re-expose with the SAME coefficient, so the
                    # checkpoint carries it rather than relying on a matching flag.
                    data["exposure_alpha"] = float(self.exposure_alpha.detach())
                    print(f"[exposure] alpha = {float(self.exposure_alpha.detach()):+.4f}"
                          f" ({'learned' if cfg.exposure_alpha_learnable else 'fixed'})")
                if cfg.adaptive_reg:
                    # recon360: keep the back-off history with the checkpoint, so a
                    # run's effective regularizer is recoverable after the fact.
                    data["adaptive_reg_events"] = self.adaptive_reg_events
                    data["adaptive_reg_final"] = {
                        "opacity_reg": cfg.opacity_reg,
                        "scale_reg": cfg.scale_reg,
                    }
                    with open(f"{self.stats_dir}/adaptive_reg.json", "w") as f:
                        json.dump(
                            {
                                "events": self.adaptive_reg_events,
                                "final_opacity_reg": cfg.opacity_reg,
                                "final_scale_reg": cfg.scale_reg,
                            },
                            f,
                            indent=1,
                        )
                if self.glare_module is not None:
                    torch.save({"logw": self.glare_module.logw.detach().cpu()},
                               f"{cfg.result_dir}/glare.pt")
                if self.flare_module is not None:
                    # recon360: the learned per-eye radial gain is an analysis
                    # product in its own right (compare against the variation
                    # probe's measured profiles) -- keep it with the run.
                    torch.save({"raw": self.flare_module.raw.detach().cpu()},
                               f"{cfg.result_dir}/flare.pt")
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                means = self.contraction.to_world(self.splats["means"].detach())
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # recon360: mip360's domain has a finite boundary -- |u - c| = 2R is
            # infinity, and one step past it gives a negative radius. No-op for
            # `log`, which has no boundary, and for `none`.
            self.contraction.clamp_(self.splats["means"].data)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.post_processing_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.exposure_alpha_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.sky_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.flare_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.glare_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer.
            # Skip structural updates while Gaussians are frozen for PPISP controller distillation.
            if self._gaussians_frozen:
                pass
            elif isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                    scene=self.scene,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                    scene=self.scene,
                )
                if self.contraction.active:
                    self._inject_noise_in_world(
                        step, lr=schedulers[0].get_last_lr()[0])
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            if masks is not None:
                # zero both sides at masked pixels so metrics ignore them
                pixels[~masks] = 0.0
            height, width = pixels.shape[1:3]

            # Exposure metadata is available for any image with EXIF data (train or val)
            exposure = data["exposure"].to(device) if "exposure" in data else None

            torch.cuda.synchronize()
            tic = time.time()
            colors, alphas, _ = self.stage.render(
                self.scene.id,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                frame_idcs=None,  # For novel views, pass None (no per-frame parameters available)
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )  # [1, H, W, 3]
            if self.sky_module is not None:
                with torch.no_grad():
                    sky_rgb = self.sky_background(
                        data["camera_idx"], Ks, camtoworlds, width, height)
                    colors = colors + sky_rgb * (1.0 - alphas)
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                # Compute color-corrected metrics for fair comparison across methods
                if cfg.use_color_correction_metric:
                    if cfg.color_correct_method == "affine":
                        cc_colors = color_correct_affine(colors, pixels)
                    else:
                        cc_colors = color_correct_quadratic(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.use_color_correction_metric:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "raw":
            # Use captured poses as-is
            camtoworlds_all = camtoworlds_all[:, :3, :]  # [N, 3, 4]
        elif cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.stage.render(
                self.scene.id,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def export_ppisp_reports(self) -> None:
        """Export PPISP visualization reports (PDF) and parameter JSON."""
        if self.cfg.post_processing != "ppisp":
            return
        print("Exporting PPISP reports...")

        # Compute frames per camera from training dataset
        num_cameras = self.parser.num_cameras
        frames_per_camera = [0] * num_cameras
        for idx in self.trainset.indices:
            cam_idx = self.parser.camera_indices[idx]
            frames_per_camera[cam_idx] += 1

        # Generate camera names from COLMAP camera IDs
        # camera_id_to_idx maps COLMAP ID -> 0-based index
        idx_to_camera_id = {v: k for k, v in self.parser.camera_id_to_idx.items()}
        camera_names = [f"camera_{idx_to_camera_id[i]}" for i in range(num_cameras)]

        # Export reports
        output_dir = Path(self.cfg.result_dir) / "ppisp_reports"
        pdf_paths = export_ppisp_report(
            self.post_processing_module,
            frames_per_camera,
            output_dir,
            camera_names=camera_names,
        )
        print(f"PPISP reports saved to {output_dir}")
        for path in pdf_paths:
            print(f"  - {path.name}")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.stage.render(
            self.scene.id,
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    # Import post-processing modules based on configuration
    # These imports must be here (not in __main__) for distributed workers
    if cfg.post_processing == "bilateral_grid":
        global BilateralGrid, slice
        if cfg.bilateral_grid_fused:
            from fused_bilagrid import (
                BilateralGrid,
                slice,
            )
        else:
            from lib_bilagrid import (
                BilateralGrid,
                slice,
            )
    elif cfg.post_processing == "ppisp":
        global PPISP, PPISPConfig, export_ppisp_report
        from ppisp import PPISP, PPISPConfig
        from ppisp.report import export_ppisp_report

    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        # recon360: checkpoints always store WORLD means; the parameter is `u`.
        # No-op unless --position_contraction is on.
        runner.splats["means"].data = runner.contraction.to_param(
            runner.splats["means"].data)
        runner.scene = GaussianScene.from_splats(runner.splats, id="scene")
        runner.splats = runner.scene.splats
        runner.stage = Stage()
        runner.stage.add_scene(runner.scene, runner.rasterize_splats)
        if runner.post_processing_module is not None:
            pp_state = ckpts[0].get("post_processing")
            if pp_state is not None:
                runner.post_processing_module.load_state_dict(pp_state)
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        if cfg.init_ckpt is not None:
            ck = torch.load(cfg.init_ckpt, map_location=runner.device, weights_only=False)
            for k in runner.splats.keys():
                runner.splats[k].data = ck["splats"][k].to(runner.device)
            # recon360: as above -- the file is world, the parameter is `u`.
            runner.splats["means"].data = runner.contraction.to_param(
                runner.splats["means"].data)
            if runner.sky_module is not None and "sky_module" in ck:
                runner.sky_module.load_state_dict(ck["sky_module"])
            # The scene's component bookkeeping (GaussianScene.component_index)
            # was sized at SfM init; the loaded checkpoint replaces the params
            # with a different N. MCMC relocation then indexes component_index
            # with indices up to the new N -> device-side assert. Latent until a
            # continuation opens the refine window past init_step (the default
            # refine_stop_iter=25000 kept relocation off in every earlier warm
            # start). Rebuild scene + stage exactly as the eval-only path does;
            # from_splats reuses the ParameterDict, so optimizer references to
            # the Parameters stay valid.
            runner.scene = GaussianScene.from_splats(runner.splats, id="scene")
            runner.splats = runner.scene.splats
            runner.stage = Stage()
            runner.stage.add_scene(runner.scene, runner.rasterize_splats)
            n_opt = 0
            if cfg.restore_train_state and ck.get("optimizers"):
                for k, st in ck["optimizers"].items():
                    if k in runner.optimizers:
                        opt = runner.optimizers[k]
                        # load_state_dict ALSO restores param_groups, i.e. the LR
                        # the saved run had decayed to (0.01x initial at the end of
                        # its schedule). train() then builds ExponentialLR from
                        # those groups and fast-forwards it, decaying a SECOND
                        # time: measured 3.490e-07 against the stateless arm's
                        # 3.489e-05 -- exactly 100x, which silently turned the
                        # "does Adam state help" A/B into a "100x lower LR" A/B.
                        # Only the moments belong to the checkpoint; the LR belongs
                        # to the new schedule.
                        lrs = [g["lr"] for g in opt.param_groups]
                        opt.load_state_dict(st)
                        for g, lr in zip(opt.param_groups, lrs):
                            g["lr"] = lr
                        n_opt += 1
                if ck.get("strategy_state") and isinstance(runner.strategy_state, dict):
                    for k, v in ck["strategy_state"].items():
                        runner.strategy_state[k] = (
                            v.to(runner.device) if torch.is_tensor(v) else v)
            print(f"[warm start] loaded {len(runner.splats['means']):,} gaussians from "
                  f"{cfg.init_ckpt} (saved at step {ck.get('step')}); "
                  f"restored {n_opt} optimizer states"
                  f"{' + strategy state' if n_opt and ck.get('strategy_state') else ''}",
                  flush=True)
        runner.train()
        runner.export_ppisp_reports()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    if cfg.backend == "dgx":
        import torch_dgx  # noqa: F401
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut and cfg.with_eval3d:
        print(
            "[Trainer] Note: with_ut=True + with_eval3d=True (full 3DGUT mode). "
            "DefaultStrategy is incompatible with eval3d; use MCMCStrategy (the `mcmc` subcommand)."
        )

    cli(main, cfg, verbose=True)
