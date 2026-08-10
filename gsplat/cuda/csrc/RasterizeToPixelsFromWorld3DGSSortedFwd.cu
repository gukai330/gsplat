/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// StopThePop-style per-pixel depth resorting for the world-space (eval3d)
// 3DGS forward rasterizer (https://arxiv.org/abs/2402.00525).
//
// The stock rasterizer blends in the global tile-sorted order: one 64-bit
// (image|tile|depth) key per (gaussian, tile), so all pixels of a 16x16 tile
// share one order, keyed by a single per-gaussian depth. With Euclidean
// sorting (global_z_order=False) that key equals the depth of maximum
// response along the ray through the gaussian's own center — the correct
// coarse level of the StopThePop hierarchy — but within a tile the true
// per-ray order of large gaussians varies per pixel, which is what pops
// when the camera moves.
//
// This kernel keeps the tile-level coarse order and adds the fine level:
// each pixel holds a small sorted window (k-buffer) of pending gaussians
// keyed by the depth of maximum response along ITS OWN ray,
//   t* = dot(grd, -gro) / |iscl_rot * ray_d|
//      = d^T Sigma^-1 (mu - o) / (d^T Sigma^-1 d),
// and blends the nearest pending entry only when the window overflows (or
// at tile end). Local order errors up to WINDOW positions in the incoming
// stream are thereby corrected exactly.
//
// Deliberately NOT ported from StopThePop: its tile/culling optimizations
// (opacity-aware tile masks, load balancing) are built on the 2D projected
// approximation. The UT projection refuses exactly that approximation
// (see ProjectionUT3DGSFused.cu: culling on the 2D footprint deletes
// gaussians whose 3D response along the ray is real), so only the sorting
// part applies here. StopThePop reports ~4% overhead for sorting alone.
//
// Forward/render only: blending order is part of the autograd contract, so
// this kernel is reachable exclusively through the fwd-only no-grad path
// (Rasterization.cpp rejects per_pixel_sort_window > 0 with grads). It is a
// separate translation unit with a restricted instantiation set
// (CDIM {1,2,3,4} x window {4,8,16,24} x hit-distance, tile 16 only) so the
// stock kernels stay byte-identical and compile time stays sane.

#include "Config.h"

#if GSPLAT_BUILD_3DGUT

#    include <ATen/Dispatch.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAStream.h>
#    include <cassert>
#    include <cuda/std/optional>

#    include "Common.h"
#    include "ExternalDistortion.cuh"
#    include "RasterizeToPixelsFromWorld3DGS.h"
#    include "RasterizeToPixelsFromWorld3DGS.cuh"
#    include "Cameras.cuh"
#    include "Lidars.cuh"
#    include "TorchUtils.h"
#    include "Utils.cuh"
#    include "Dispatch.h"

namespace gsplat
{
namespace
{
using SortedFwdChannels = dispatch::IntParam<1, 2, 3, 4>;
using SortedFwdWindows  = dispatch::IntParam<4, 8, 16, 24>;

constexpr float WINDOW_EMPTY_DEPTH = 1e30f; // ascending-order sentinel for unused slots

// One pending (not yet blended) gaussian contribution for one pixel.
// Kept as separate register arrays in the kernel (SoA) so the unrolled
// compare-exchange chains stay in registers; a struct array would tempt
// the compiler into local memory.
} // namespace

template<uint32_t CDIM, uint32_t TILE_SIZE, uint32_t CTA_SIZE, uint32_t WINDOW, bool UseHitDistance>
__global__ void __launch_bounds__(CTA_SIZE) rasterize_to_pixels_from_world_3dgs_sorted_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const vec3 *__restrict__ means,        // [B, N, 3]
    const vec4 *__restrict__ quats,        // [B, N, 4]
    const vec3 *__restrict__ scales,       // [B, N, 3]
    const float *__restrict__ colors,      // [B, C, N, CDIM]
    const float *__restrict__ opacities,   // [B, C, N]
    const float *__restrict__ backgrounds, // [B, C, CDIM]
    const bool *__restrict__ masks,        // [B, C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    // camera model
    const float *__restrict__ viewmats0, // [B, C, 4, 4]
    const float *__restrict__ viewmats1, // [B, C, 4, 4] optional for rolling shutter
    const float *__restrict__ Ks,        // [B, C, 3, 3]
    const CameraModelType camera_model_type,
    // unscented transform
    const UnscentedTransformParameters ut_params,
    const ShutterType rs_type,
    const float *__restrict__ rays,              // [B, C, H, W, 6]
    const float *__restrict__ radial_coeffs,     // [B, C, 6] or [B, C, 4] optional
    const float *__restrict__ tangential_coeffs, // [B, C, 2] optional
    const float *__restrict__ thin_prism_coeffs, // [B, C, 4] optional
    const FThetaCameraDistortionDeviceParams ftheta_device_coeffs,
    const cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> lidar_device_coeffs,
    const cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> external_distortion_device_params,
    // intersections
    const int32_t *__restrict__ isect_offsets, // [B, C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,   // [n_isects]
    float *__restrict__ render_colors,         // [B, C, image_height, image_width, CDIM]
    float *__restrict__ render_alphas          // [B, C, image_height, image_width, 1]
)
{
    // One thread per pixel; the window arrays live in registers and their
    // compare-exchange chains rely on full unrolling with static indices.
    static_assert(TILE_SIZE * TILE_SIZE == CTA_SIZE, "sorted fwd kernel requires one thread per pixel");
    constexpr uint32_t FETCH_SIZE = CTA_SIZE;
    constexpr uint32_t TILE_MASK  = TILE_SIZE - 1;
    constexpr uint32_t TILE_SHIFT = __builtin_ctz(TILE_SIZE);

    const int32_t iid          = blockIdx.x;
    const uint32_t grid_width  = gridDim.z;
    const uint32_t grid_height = gridDim.y;

    const uint32_t tile_x = blockIdx.z;
    const uint32_t tile_y = blockIdx.y;
    const int32_t tile_id = blockIdx.y * grid_width + blockIdx.z;

    bool masked_tile = false;
    if(masks != nullptr)
    {
        masks       += iid * grid_height * grid_width;
        masked_tile  = !masks[tile_id];
    }

    const uint32_t tid      = threadIdx.x;
    const uint32_t thread_x = tid & TILE_MASK;
    const uint32_t thread_y = tid >> TILE_SHIFT;

    // Offset pointers to current image
    isect_offsets += iid * grid_height * grid_width;
    render_colors += iid * image_height * image_width * CDIM;
    render_alphas += iid * image_height * image_width;
    if(backgrounds != nullptr)
    {
        backgrounds += iid * CDIM;
    }

    PixelCoords pc = compute_pixel_coords(
        camera_model_type,
        tile_id,
        tile_y,
        tile_x,
        TILE_SIZE,
        thread_y,
        thread_x,
        tid,
        image_width,
        image_height,
        lidar_device_coeffs
    );
    bool done = !pc.inside;

    // Masked tiles write deterministic empty outputs (the safe default of the
    // stock forward); the unsafe expert mode is not supported here.
    if(masked_tile)
    {
        if(pc.inside)
        {
            render_alphas[pc.pix_id] = 0.0f;
#    pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                render_colors[pc.pix_id * CDIM + k] = backgrounds == nullptr ? 0.0f : backgrounds[k];
            }
        }
        return;
    }

    if(rays != nullptr)
    {
        rays += iid * image_height * image_width * 6;
    }

    auto rs_params
        = RollingShutterParameters(viewmats0 + iid * 16, viewmats1 == nullptr ? nullptr : viewmats1 + iid * 16);

    vec3 ray_o = {};
    vec3 ray_d = {};
    if(!done)
    {
        WorldRay ray = compute_world_ray<float>(
            iid,
            pc.col,
            pc.row,
            pc.pix_id,
            /*inside=*/true,
            rs_params,
            rays,
            Ks,
            image_width,
            image_height,
            camera_model_type,
            rs_type,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            ftheta_device_coeffs,
            lidar_device_coeffs,
            external_distortion_device_params
        );
        if(!ray.valid_flag)
        {
            done = true;
        }
        else
        {
            ray_o = ray.ray_org;
            ray_d = ray.ray_dir;
        }
    }

    // Gaussian range for this tile
    const int32_t range_start = isect_offsets[tile_id];
    const int32_t range_end   = (iid == (int32_t)gridDim.x - 1) && (tile_id == (int32_t)(grid_width * grid_height) - 1)
                                  ? n_isects
                                  : isect_offsets[tile_id + 1];
    const uint32_t num_batches = (range_end - range_start + FETCH_SIZE - 1) / FETCH_SIZE;

    // Shared memory: FETCH_SIZE gaussians per cooperative fetch round.
    // No normals segment: normals are rejected on the sorted path.
    extern __shared__ int s[];
    int32_t *id_batch       = (int32_t *)s;                                             // [FETCH_SIZE]
    vec4 *xyz_opacity_batch = reinterpret_cast<vec4 *>(&id_batch[FETCH_SIZE]);          // [FETCH_SIZE]
    mat3 *iscl_rot_batch    = reinterpret_cast<mat3 *>(&xyz_opacity_batch[FETCH_SIZE]); // [FETCH_SIZE]
    vec3 *scale_batch       = reinterpret_cast<vec3 *>(&iscl_rot_batch[FETCH_SIZE]);    // [FETCH_SIZE]

    // Per-pixel accumulation state
    float T                = 1.0f;
    float pix_out[CDIM]    = {0.f};

    // Per-pixel sorted window (ascending in depth). SoA register arrays; all
    // indexing below is static (fully unrolled) so these stay in registers.
    float win_d[WINDOW];
    float win_a[WINDOW];
    int32_t win_id[WINDOW];
    float win_h[UseHitDistance ? WINDOW : 1];
#    pragma unroll
    for(uint32_t i = 0; i < WINDOW; ++i)
    {
        win_d[i]  = WINDOW_EMPTY_DEPTH;
        win_a[i]  = 0.f;
        win_id[i] = -1;
        if constexpr(UseHitDistance)
        {
            win_h[i] = 0.f;
        }
    }
    int32_t win_count = 0;

    // Blend one popped contribution front-to-back. Same saturation policy as
    // the stock forward (KeepPreSaturationT): crossing the transmittance
    // threshold drops the contribution, keeps the pre-saturation T and marks
    // the pixel done.
    auto blend_one = [&](float alpha, int32_t isect_id, float hit_distance)
    {
        if(done)
        {
            return;
        }
        const float next_T = T * (1.0f - alpha);
        if(next_T <= TRANSMITTANCE_THRESHOLD)
        {
            done = true;
            return;
        }
        const float vis    = alpha * T;
        const float *c_ptr = colors + static_cast<int64_t>(isect_id) * CDIM;
        if constexpr(UseHitDistance)
        {
#    pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                const float value  = (k == CDIM - 1) ? hit_distance : c_ptr[k];
                pix_out[k]        += value * vis;
            }
        }
        else
        {
#    pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                pix_out[k] += c_ptr[k] * vis;
            }
        }
        T = next_T;
    };

    // Pop the nearest pending entry and shift the window down one slot.
    auto pop_front = [&]()
    {
        blend_one(win_a[0], win_id[0], UseHitDistance ? win_h[0] : 0.f);
#    pragma unroll
        for(uint32_t i = 0; i + 1 < WINDOW; ++i)
        {
            win_d[i]  = win_d[i + 1];
            win_a[i]  = win_a[i + 1];
            win_id[i] = win_id[i + 1];
            if constexpr(UseHitDistance)
            {
                win_h[i] = win_h[i + 1];
            }
        }
        win_d[WINDOW - 1]  = WINDOW_EMPTY_DEPTH;
        win_a[WINDOW - 1]  = 0.f;
        win_id[WINDOW - 1] = -1;
        --win_count;
    };

#    pragma unroll 1
    for(uint32_t b = 0; b < num_batches; ++b)
    {
        const uint32_t batch_start = range_start + FETCH_SIZE * b;
        cooperative_load_fetch_round<FETCH_SIZE, CTA_SIZE, /*ReturnNormals=*/false, float>(
            tid,
            id_batch,
            xyz_opacity_batch,
            iscl_rot_batch,
            scale_batch,
            /*normal_batch=*/nullptr,
            batch_start,
            range_end,
            flatten_ids,
            means,
            quats,
            scales,
            opacities,
            C,
            N
        );
        cta_sync<CTA_SIZE>();

        const uint32_t batch_size = min(FETCH_SIZE, (uint32_t)range_end - batch_start);
#    pragma unroll 1
        for(uint32_t t = 0; t < batch_size; ++t)
        {
            if(done)
            {
                break;
            }
            const vec4 xyz_opac = xyz_opacity_batch[t];
            const float opac    = xyz_opac[3];
            const vec3 xyz      = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
            const mat3 iscl_rot = iscl_rot_batch[t];

            // Same expressions as the stock blend (safe_normalize included) so
            // that whenever the resorted order coincides with the stock order
            // the output is bit-identical — the same contraction-stability
            // property the fwd/bwd replay pair relies on.
            const vec3 gro    = iscl_rot * (ray_o - xyz);
            const vec3 grd_un = iscl_rot * ray_d;
            const vec3 grd    = safe_normalize(grd_un);
            // hit_t < 0: closest approach is behind the camera origin — skip.
            const float hit_t = -glm::dot(grd, gro);
            if(hit_t < 0.f)
            {
                continue;
            }
            const vec3 gcrod     = glm::cross(grd, gro);
            const float grayDist = glm::dot(gcrod, gcrod);
            const float power    = -0.5f * grayDist;
            const float max_response = __expf(power);
            const float alpha        = min(MAX_ALPHA, opac * max_response);
            if(alpha < ALPHA_THRESHOLD)
            {
                continue;
            }

            // Depth of maximum response along THIS pixel's ray, in units of
            // the world-space ray parameter — the per-ray sort key. hit_t is
            // measured along the normalized transformed ray, so divide by
            // |iscl_rot * ray_d| (same explicit-intrinsic reduction as
            // safe_normalize; the compiler CSEs the repeated rsqrt).
            float grd_len2 = __fmul_rn(grd_un.x, grd_un.x);
            grd_len2       = __fmaf_rn(grd_un.y, grd_un.y, grd_len2);
            grd_len2       = __fmaf_rn(grd_un.z, grd_un.z, grd_len2);
            if(grd_len2 <= 0.f)
            {
                continue;
            }
            const float depth = __fmul_rn(hit_t, rsqrtf(grd_len2));

            float hit_distance = 0.0f;
            if constexpr(UseHitDistance)
            {
                const vec3 scale = scale_batch[t];
                const vec3 grds  = scale * (grd * hit_t);
                hit_distance     = glm::length(grds);
            }
            const int32_t isect_id = id_batch[t];

            if(win_count < (int32_t)WINDOW)
            {
                // Insert keeping ascending order: compare-exchange chain from
                // the front; the ejected element is an empty-slot sentinel.
                float cd    = depth;
                float ca    = alpha;
                int32_t cid = isect_id;
                float ch    = hit_distance;
#    pragma unroll
                for(uint32_t i = 0; i < WINDOW; ++i)
                {
                    if(cd < win_d[i])
                    {
                        float td  = win_d[i];
                        float ta  = win_a[i];
                        int32_t ti = win_id[i];
                        win_d[i]  = cd;
                        win_a[i]  = ca;
                        win_id[i] = cid;
                        cd        = td;
                        ca        = ta;
                        cid       = ti;
                        if constexpr(UseHitDistance)
                        {
                            float th = win_h[i];
                            win_h[i] = ch;
                            ch       = th;
                        }
                    }
                }
                ++win_count;
            }
            else if(depth < win_d[0])
            {
                // Nearer than everything pending: blending it immediately IS
                // the sorted order (window contents all lie behind it).
                blend_one(alpha, isect_id, hit_distance);
            }
            else
            {
                // Window full: emit the nearest pending entry, then shift-left
                // and insert the newcomer in one pass.
                blend_one(win_a[0], win_id[0], UseHitDistance ? win_h[0] : 0.f);
                float cd    = depth;
                float ca    = alpha;
                int32_t cid = isect_id;
                float ch    = hit_distance;
#    pragma unroll
                for(uint32_t i = 0; i + 1 < WINDOW; ++i)
                {
                    const float nd   = win_d[i + 1];
                    const float na   = win_a[i + 1];
                    const int32_t ni = win_id[i + 1];
                    float nh         = 0.f;
                    if constexpr(UseHitDistance)
                    {
                        nh = win_h[i + 1];
                    }
                    if(nd <= cd)
                    {
                        win_d[i]  = nd;
                        win_a[i]  = na;
                        win_id[i] = ni;
                        if constexpr(UseHitDistance)
                        {
                            win_h[i] = nh;
                        }
                    }
                    else
                    {
                        win_d[i]  = cd;
                        win_a[i]  = ca;
                        win_id[i] = cid;
                        if constexpr(UseHitDistance)
                        {
                            win_h[i] = ch;
                        }
                        cd  = nd;
                        ca  = na;
                        cid = ni;
                        ch  = nh;
                    }
                }
                win_d[WINDOW - 1]  = cd;
                win_a[WINDOW - 1]  = ca;
                win_id[WINDOW - 1] = cid;
                if constexpr(UseHitDistance)
                {
                    win_h[WINDOW - 1] = ch;
                }
            }
        }

        // CTA-wide early stop once every pixel is saturated (or inactive).
        // The sync doubles as the inter-batch barrier protecting the shared
        // fetch buffers from being refilled while stragglers still read them.
        if(cta_sync_count<CTA_SIZE>(done) >= (int32_t)CTA_SIZE)
        {
            break;
        }
    }

    // Flush the pending window front-to-back (already ascending).
#    pragma unroll 1
    while(win_count > 0 && !done)
    {
        pop_front();
    }

    if(pc.inside)
    {
        render_alphas[pc.pix_id] = 1.0f - T;
#    pragma unroll
        for(uint32_t k = 0; k < CDIM; ++k)
        {
            render_colors[pc.pix_id * CDIM + k]
                = backgrounds == nullptr ? pix_out[k] : (pix_out[k] + T * backgrounds[k]);
        }
    }
}

void launch_rasterize_to_pixels_from_world_3dgs_sorted_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means,                     // [..., N, 3]
    const at::Tensor quats,                     // [..., N, 4]
    const at::Tensor scales,                    // [..., N, 3]
    const at::Tensor colors,                    // [..., C, N, channels]
    const at::Tensor opacities,                 // [..., C, N]
    const at::optional<at::Tensor> backgrounds, // [..., C, channels]
    const at::optional<at::Tensor> masks,       // [..., C, grid_h, grid_w]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const at::Tensor viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const CameraModelType camera_model,
    // unscented transform
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> rays,              // [...., C, H, W, 6]
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    // external distortion
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    // intersections
    const at::Tensor isect_offsets, // [..., C, grid_h, grid_w]
    const at::Tensor flatten_ids,   // [n_isects]
    const bool use_hit_distance,
    const int64_t per_pixel_sort_window,
    // outputs
    at::Tensor renders, // [..., C, image_height, image_width, channels]
    at::Tensor alphas   // [..., C, image_height, image_width]
)
{
    // Note: quats need to be normalized before passing in.
    bool packed = opacities.dim() == 1;
    TORCH_CHECK(!packed, "packed mode not supported for 3DGUT forward rasterization");

    const uint32_t N        = means.size(-2);
    const uint32_t B        = static_cast<uint32_t>(c10::multiply_integers(means.sizes().slice(0, means.dim() - 2)));
    const uint32_t C        = viewmats0.size(-3);
    const uint32_t I        = B * C;
    const uint32_t grid_h   = isect_offsets.size(-2);
    const uint32_t grid_w   = isect_offsets.size(-1);
    const uint32_t n_isects = flatten_ids.size(0);

    TORCH_CHECK(ut_params, "ut_params intrusive_ptr is null");
    FThetaCameraDistortionDeviceParams ftheta_device_coeffs(gsplat::checked_deref(ftheta_coeffs, "ftheta_coeffs"));
    cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> external_distortion_device_params
        = cuda::std::nullopt;
    if(external_distortion_params.has_value())
    {
        const auto &params = gsplat::checked_deref(external_distortion_params.value(), "external_distortion_params");
        CHECK_CONTIGUOUS(params.horizontal_poly);
        CHECK_CONTIGUOUS(params.vertical_poly);
        CHECK_CONTIGUOUS(params.horizontal_poly_inverse);
        CHECK_CONTIGUOUS(params.vertical_poly_inverse);
        external_distortion_device_params = extdist::BivariateWindshieldModelDeviceParams(params);
    }

    cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> lidar_device_coeffs
        = cuda::std::nullopt;
    if(lidar_coeffs.has_value())
    {
        TORCH_CHECK(
            camera_model == CameraModelType::LIDAR,
            "If lidar sensor coefficients are given, the camera model must be lidar"
        );
        lidar_device_coeffs = *lidar_coeffs.value();
    }
    else
    {
        TORCH_CHECK(
            camera_model != CameraModelType::LIDAR, "If the sensor isn't lidar, lidar coefficients must not be given"
        );
    }

    const int32_t channels = colors.size(-1);
    TORCH_CHECK_VALUE(
        SortedFwdChannels::contains(channels),
        "per_pixel_sort_window > 0 supports at most 4 feature channels per raster pass, got ",
        channels,
        ". Render RGB / RGB+depth, or lower channel_chunk to <= 4."
    );
    TORCH_CHECK_VALUE(
        tile_size == 16,
        "per_pixel_sort_window > 0 requires tile_size == 16 (one thread per pixel), got ",
        tile_size
    );
    TORCH_CHECK_VALUE(
        SortedFwdWindows::contains(static_cast<int>(per_pixel_sort_window)),
        "per_pixel_sort_window must be one of {4, 8, 16, 24}, got ",
        per_pixel_sort_window
    );

    auto launch_kernel = [&]<typename ChannelsT, typename WindowT, typename UseHitDistanceT>()
    {
        constexpr uint32_t CDIM       = ChannelsT::value;
        constexpr uint32_t TILE_SIZE  = 16;
        constexpr uint32_t CTA_SIZE   = TILE_SIZE * TILE_SIZE;
        constexpr uint32_t WINDOW     = WindowT::value;
        constexpr bool UseHitDistance = static_cast<bool>(UseHitDistanceT::value);

        const dim3 threads = {CTA_SIZE, 1, 1};
        const dim3 grid    = {I, grid_h, grid_w};
        // Shared memory: id_batch + xyz_opacity_batch + iscl_rot_batch + scale_batch
        const int64_t shmem_size = CTA_SIZE * (sizeof(int32_t) + sizeof(vec4) + sizeof(mat3) + sizeof(vec3));

        if(cudaFuncSetAttribute(
               rasterize_to_pixels_from_world_3dgs_sorted_fwd_kernel<CDIM, TILE_SIZE, CTA_SIZE, WINDOW, UseHitDistance>,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               shmem_size
           )
           != cudaSuccess)
        {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shmem_size, " bytes).");
        }

        rasterize_to_pixels_from_world_3dgs_sorted_fwd_kernel<CDIM, TILE_SIZE, CTA_SIZE, WINDOW, UseHitDistance>
            <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                C,
                N,
                n_isects,
                data_ptr_as<const vec3, float>(means),
                data_ptr_as<const vec4, float>(quats),
                data_ptr_as<const vec3, float>(scales),
                colors.const_data_ptr<float>(),
                opacities.const_data_ptr<float>(),
                data_ptr_or_null<const float>(backgrounds),
                data_ptr_or_null<const bool>(masks),
                image_width,
                image_height,
                viewmats0.const_data_ptr<float>(),
                data_ptr_or_null<const float>(viewmats1),
                Ks.const_data_ptr<float>(),
                camera_model,
                *ut_params,
                rs_type,
                data_ptr_or_null<const float>(rays),
                data_ptr_or_null<const float>(radial_coeffs),
                data_ptr_or_null<const float>(tangential_coeffs),
                data_ptr_or_null<const float>(thin_prism_coeffs),
                ftheta_device_coeffs,
                lidar_device_coeffs,
                external_distortion_device_params,
                isect_offsets.const_data_ptr<int32_t>(),
                flatten_ids.const_data_ptr<int32_t>(),
                renders.data_ptr<float>(),
                alphas.data_ptr<float>()
            );
    };
    const bool dispatched = dispatch::dispatch(
        SortedFwdChannels{channels},
        SortedFwdWindows{static_cast<int>(per_pixel_sort_window)},
        dispatch::IntParam<0, 1>{use_hit_distance ? 1 : 0},
        std::move(launch_kernel)
    );
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}
} // namespace gsplat

#endif
