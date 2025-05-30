# Panoramic Gaussian Splatting (yzslab/gsplat - Older Version)

This project extends the older `yzslab/gsplat` library to support panoramic (omni-directional) image rendering using 3D Gaussian Splatting.

## What's New

Panoramic rendering capabilities have been added to this version of the `gsplat` library. This allows for rendering 3D Gaussian Splatting scenes from panoramic camera perspectives.

## Implementation Details

The following key changes were made to integrate panoramic rendering into this older codebase:

- A `PANORAMA` camera model type was added to `yzslab/gsplat/gsplat/cuda/include/bindings.h`.
- Panoramic projection and its backward pass (`panorama_proj` and `panorama_proj_vjp`) were implemented in `yzslab/gsplat/gsplat/cuda/include/proj.cuh`, based on the logic from the `op43dgs` repository.
- The CUDA kernels for basic projection (`proj_fwd.cu`, `proj_bwd.cu`), fused projection (`fully_fused_projection_fwd.cu`, `fully_fused_projection_bwd.cu`), and packed fused projection (`fully_fused_projection_packed_fwd.cu`, `fully_fused_projection_packed_bwd.cu`) were modified to include a case for the `PANORAMA` camera model, calling the new panoramic projection functions.
- The Python wrapper file `yzslab/gsplat/gsplat/cuda/_wrapper.py` was updated to include "panorama" in the type hints for relevant projection functions.
- The C++ bindings in `yzslab/gsplat/gsplat/cuda/csrc/ext.cpp` were updated to expose the new `PANORAMA` camera model type.

## How to Use

To use the panoramic rendering feature, you will need to specify the `camera_model` parameter as `"panorama"` when calling the relevant projection functions in this version of the `gsplat` library.

For example, when using the `fully_fused_projection` function, you would pass `camera_model="panorama"`:

```python
import torch
from gsplat.cuda import fully_fused_projection # Assuming this is the correct import path for this version

# Assuming you have your means, quats, scales, viewmats, and Ks tensors ready
# ...

# Perform panoramic projection
# Note: The exact function signature might differ slightly in this older version.
# Refer to the source code or original documentation if needed.
radii, means2d, depths, conics, compensations = fully_fused_projection(
    means=means,
    quats=quats,
    scales=scales,
    viewmats=viewmats,
    Ks=Ks,
    width=image_width,
    height=image_height,
    camera_model="panorama", # Specify the panoramic camera model
    # Other parameters specific to this version...
)

# Continue with rasterization using the projected data
# ...
```

You would similarly specify `camera_model="panorama"` when using other projection functions that support the `CameraModelType` parameter in this version.

Please refer to the original documentation or source code of this specific `yzslab/gsplat` version for detailed usage of other functions and parameters.

## Building

After applying these changes, you will need to rebuild this specific `yzslab/gsplat` library to include the new CUDA code. Follow the standard build instructions provided with this version, which typically involve using CMake and Python's `setuptools`.
