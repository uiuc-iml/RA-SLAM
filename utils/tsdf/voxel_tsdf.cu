#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "utils/cuda/arithmetic.cuh"
#include "utils/cuda/errors.cuh"
#include "utils/tsdf/mcube_table.cuh"
#include "utils/tsdf/voxel_tsdf.cuh"
#include "utils/tsdf/label_color_palette.cuh"
#include "utils/time.hpp"

#define MAX_IMG_H 1920
#define MAX_IMG_W 1080
#define MAX_IMG_SIZE (MAX_IMG_H * MAX_IMG_W)

__global__ static void check_bound_kernel(const VoxelHashTable hash_table,
                                          const BoundingCube<short> volumn_grid,
                                          int* visible_mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const VoxelBlock& block = hash_table.GetBlock(idx);
  const Eigen::Matrix<short, 3, 1> voxel_grid = BlockToPoint(block.position);
  visible_mask[idx] =
      (block.idx >= 0 && voxel_grid[0] >= volumn_grid.xmin && voxel_grid[1] >= volumn_grid.ymin &&
       voxel_grid[2] >= volumn_grid.zmin && voxel_grid[0] + BLOCK_LEN - 1 <= volumn_grid.xmax &&
       voxel_grid[1] + BLOCK_LEN - 1 <= volumn_grid.ymax &&
       voxel_grid[2] + BLOCK_LEN - 1 <= volumn_grid.zmax);
}

__global__ static void check_valid_kernel(const VoxelHashTable hash_table, int* visible_mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const VoxelBlock& block = hash_table.GetBlock(idx);
  const Eigen::Matrix<short, 3, 1> pos_grid = BlockToPoint(block.position);
  visible_mask[idx] = (int)(block.idx >= 0);
}

__global__ static void download_tsdf_kernel(const VoxelHashTable hash_table,
                                            const VoxelBlock* blocks, const float voxel_size,
                                            VoxelSpatialTSDF* voxel_pos_tsdf) {
  const VoxelBlock& block = blocks[blockIdx.x];
  const Eigen::Matrix<short, 3, 1> offset_grid(threadIdx.x, threadIdx.y, threadIdx.z);
  const Eigen::Matrix<short, 3, 1> pos_grid = BlockToPoint(block.position) + offset_grid;
  const Eigen::Vector3f pos_world = pos_grid.cast<float>() * voxel_size;
  const int thread_idx = OffsetToIndex(offset_grid);

  const int idx = blockIdx.x * BLOCK_VOLUME + thread_idx;
  const VoxelTSDF& tsdf = hash_table.mem.GetVoxel<VoxelTSDF>(thread_idx, block);
  voxel_pos_tsdf[idx] = VoxelSpatialTSDF(pos_world, tsdf.tsdf);
}

__global__ static void download_semantic_kernel(const VoxelHashTable hash_table,
                                                const VoxelBlock* blocks, const float voxel_size,
                                                VoxelSpatialTSDFSEGM* voxel_pos_tsdf) {
  const VoxelBlock& block = blocks[blockIdx.x];
  const Eigen::Matrix<short, 3, 1> offset_grid(threadIdx.x, threadIdx.y, threadIdx.z);
  const Eigen::Matrix<short, 3, 1> pos_grid = BlockToPoint(block.position) + offset_grid;
  const Eigen::Vector3f pos_world = pos_grid.cast<float>() * voxel_size;
  const int thread_idx = OffsetToIndex(offset_grid);

  const int idx = blockIdx.x * BLOCK_VOLUME + thread_idx;
  const VoxelTSDF& tsdf = hash_table.mem.GetVoxel<VoxelTSDF>(thread_idx, block);
  const VoxelSEGM& voxel_segm = hash_table.mem.GetVoxel<VoxelSEGM>(thread_idx, block);
  // find max class
  int max_cls = voxel_segm.semantic_rep.get_max_class();
  voxel_pos_tsdf[idx] = VoxelSpatialTSDFSEGM(pos_world, tsdf.tsdf, max_cls);
}

__device__ static bool is_voxel_visible(const Eigen::Matrix<short, 3, 1>& pos_grid,
                                        const SE3<float>& cam_T_world,
                                        const CameraParams& cam_params, const float& voxel_size) {
  const Eigen::Vector3f pos_world = pos_grid.cast<float>() * voxel_size;
  const Eigen::Vector3f pos_cam = cam_T_world.Apply(pos_world);
  const Eigen::Vector3f pos_img_h = cam_params.intrinsics * pos_cam;
  const Eigen::Vector2f pos_img = pos_img_h.hnormalized();
  return (pos_img[0] >= 0 && pos_img[0] <= cam_params.img_w - 1 && pos_img[1] >= 0 &&
          pos_img[1] <= cam_params.img_h - 1 && pos_img_h[2] >= 0);
}

template <bool Full = true>
__device__ static bool is_block_visible(const Eigen::Matrix<short, 3, 1>& block_pos,
                                        const SE3<float>& cam_T_world,
                                        const CameraParams& cam_params, const float& voxel_size) {
  const Eigen::Matrix<short, 3, 1> pos_grid = BlockToPoint(block_pos);
  const short x = pos_grid[0], y = pos_grid[1], z = pos_grid[2];

  bool visible = Full;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    Eigen::Matrix<short, 3, 1> corner(x + ((i >> 0) & 1) * (BLOCK_LEN - 1),
                                      y + ((i >> 1) & 1) * (BLOCK_LEN - 1),
                                      z + ((i >> 2) & 1) * (BLOCK_LEN - 1));
    if (Full) {
      visible &= is_voxel_visible(corner, cam_T_world, cam_params, voxel_size);
    } else {
      visible |= is_voxel_visible(corner, cam_T_world, cam_params, voxel_size);
    }
  }

  return visible;
}

__global__ static void check_visibility_kernel(const VoxelHashTable hash_table,
                                               const float voxel_size, const float max_depth,
                                               const CameraParams cam_params,
                                               const SE3<float> cam_T_world, int* visible_mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const VoxelBlock& block = hash_table.GetBlock(idx);
  if (block.idx < 0) {
    visible_mask[idx] = 0;
    return;
  }
  visible_mask[idx] = is_block_visible<false>(block.position, cam_T_world, cam_params, voxel_size);
}

__global__ static void gather_visible_blocks_kernel(const VoxelHashTable hash_table,
                                                    const int* visible_mask,
                                                    const int* visible_indics, VoxelBlock* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (visible_mask[idx]) {
    output[visible_indics[idx] - 1] = hash_table.GetBlock(idx);
  }
}

__global__ static void block_allocate_kernel(VoxelHashTable hash_table, const float* img_depth,
                                             const CameraParams cam_params,
                                             const SE3<float> cam_T_world,
                                             const SE3<float> world_T_cam, const float voxel_size,
                                             const float max_depth, const float truncation,
                                             float* img_depth_to_range) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cam_params.img_w || y >= cam_params.img_h) {
    return;
  }
  const int idx = y * cam_params.img_w + x;
  const float depth = img_depth[idx];
  // calculate depth2range scale
  const Eigen::Vector3f pos_img_h(x, y, 1.);

  // homogeneous -> camera coordinate
  const Eigen::Vector3f pos_cam = cam_params.intrinsics_inv * pos_img_h;

  // range
  img_depth_to_range[idx] = pos_cam.norm();
  if (depth == 0 || depth > max_depth) {
    return;
  }

  // transform coordinate from image to world
  const Eigen::Vector3f pos_world = world_T_cam.Apply(pos_cam * depth);
  // calculate end coordinates of sample ray
  const Eigen::Vector3f ray_dir_cam = pos_cam / img_depth_to_range[idx];
  const Eigen::Quaternionf world_R_cam = world_T_cam.GetR();
  const Eigen::Vector3f ray_dir_world = world_R_cam * ray_dir_cam;
  const Eigen::Vector3f ray_start_world = pos_world - ray_dir_world * truncation;
  // put ray into voxel grid coordinate
  const Eigen::Vector3f ray_dir_grid = ray_dir_world / voxel_size;
  const Eigen::Vector3f ray_start_grid = ray_start_world / voxel_size;
  const Eigen::Vector3f ray_grid = 2 * truncation * ray_dir_grid;  // start -> end vector
  // DDA for finding ray / block intersection
  const int step_grid =
      ceilf(fmaxf(fmaxf(fabsf(ray_grid[0]), fabsf(ray_grid[1])), fabsf(ray_grid[2])) / BLOCK_LEN);
  const Eigen::Vector3f ray_step_grid = ray_grid / fmaxf((float)step_grid, 1);
  Eigen::Vector3f pos_grid = ray_start_grid;
  // allocate blocks along the ray
  for (int i = 0; i <= step_grid; ++i, pos_grid += ray_step_grid) {
    const Eigen::Matrix<short, 3, 1> pos_block =
        PointToBlock(pos_grid.unaryExpr([](const float x) { return roundf(x); }).cast<short>());
    if (is_block_visible(pos_block, cam_T_world, cam_params, voxel_size)) {
      hash_table.Allocate(pos_block);
    }
  }
}

__global__ static void tsdf_integrate_kernel(
    VoxelBlock* blocks, VoxelMemPool voxel_mem, const SE3<float> cam_T_world,
    const CameraParams cam_params, const int num_visible_blocks, const float max_depth,
    const float truncation, const float voxel_size, const uchar3* img_rgb, const float* img_depth,
    const float* prob_map, const float* img_depth_to_range, const int height, const int width) {
  if (blockIdx.x >= num_visible_blocks) {
    // If voxel block idx is not visible, return.
    return;
  }
  // voxel position w.r.t. its block (which contains 8 * 8 * 8 voxels)
  const Eigen::Matrix<short, 3, 1> pos_grid_rel(threadIdx.x, threadIdx.y, threadIdx.z);

  // voxel position in discrete world coordinate
  const Eigen::Matrix<short, 3, 1> pos_grid_abs =
      BlockToPoint(blocks[blockIdx.x].position) + pos_grid_rel;

  // voxel position in continuous world coordinate (in meter)
  const Eigen::Vector3f pos_world = pos_grid_abs.cast<float>() * voxel_size;

  // voxel position in camera coordinate
  const Eigen::Vector3f pos_cam = cam_T_world.Apply(pos_world);

  // voxel position in (img_i * depth, img_j * depth, depth)
  const Eigen::Vector3f pos_img_h = cam_params.intrinsics * pos_cam;

  // original (img_i * depth, img_j * depth, depth) -> (img_i, img_j)
  const Eigen::Vector2f pos_img = pos_img_h.hnormalized();

  // corresponding pixel index in camera width
  const int u = roundf(pos_img[0]);

  // corresponding pixel index in camera height
  const int v = roundf(pos_img[1]);

  const int class_offset = width * height;

  // update if visible
  if (u >= 0 && u < cam_params.img_w && v >= 0 && v < cam_params.img_h) {
    // image coordinate access
    const int img_idx = v * cam_params.img_w + u;

    // get depth
    const float depth = img_depth[img_idx];
    if (depth == 0 || depth > max_depth) {
      return;
    }

    // multiple voxels -> same
    const float sdf = img_depth_to_range[img_idx] * (depth - pos_img_h[2]);
    if (sdf > -truncation) {
      const float tsdf = fminf(1, sdf / truncation);
      const unsigned int idx = OffsetToIndex(pos_grid_rel);
      VoxelTSDF& voxel_tsdf = voxel_mem.GetVoxel<VoxelTSDF>(idx, blocks[blockIdx.x]);
      VoxelRGBW& voxel_rgbw = voxel_mem.GetVoxel<VoxelRGBW>(idx, blocks[blockIdx.x]);
      VoxelSEGM& voxel_segm = voxel_mem.GetVoxel<VoxelSEGM>(idx, blocks[blockIdx.x]);

      // weight running average
      // heuristic
      const float weight_new = (1 - depth / max_depth) * 4;  // TODO(alvin): better weighting here
      const float weight_old = voxel_rgbw.weight;
      const float weight_combined = weight_old + weight_new;

      // rgb running average
      const uchar3 rgb = img_rgb[img_idx];
      const Eigen::Vector3f rgb_old = voxel_rgbw.rgb.cast<float>();
      const Eigen::Vector3f rgb_new(rgb.x, rgb.y, rgb.z);
      const Eigen::Vector3f rgb_combined =
          (rgb_old * weight_old + rgb_new * weight_new) / weight_combined;
      voxel_tsdf.tsdf = (voxel_tsdf.tsdf * weight_old + tsdf * weight_new) / weight_combined;
      // TODO(roger): should be tuned/not hardcoded
      voxel_rgbw.weight = fminf(roundf(weight_combined), 40);
      voxel_rgbw.rgb =
          rgb_combined.unaryExpr([](const float x) { return roundf(x); }).cast<unsigned char>();
      // multi-class recursive Bayesian update
      voxel_segm.semantic_rep.update(prob_map, img_idx, class_offset);
    }
  }
}

__global__ static void space_carving_kernel(VoxelHashTable hash_table, const VoxelBlock* blocks,
                                            const int num_visible_blocks,
                                            const float min_tsdf_threshold) {
  if (blockIdx.x >= num_visible_blocks) {
    return;
  }

  __shared__ float tsdf_abs[BLOCK_VOLUME];
  // load shared buffer
  const int tx = threadIdx.x;
  const int tx2 = tx + BLOCK_VOLUME / 2;
  tsdf_abs[tx] = fabsf(hash_table.mem.GetVoxel<VoxelTSDF>(tx, blocks[blockIdx.x]).tsdf);
  tsdf_abs[tx2] = fabsf(hash_table.mem.GetVoxel<VoxelTSDF>(tx2, blocks[blockIdx.x]).tsdf);
// reduce min
#pragma unroll
  for (int stride = BLOCK_VOLUME / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (tx < stride) tsdf_abs[tx] = fminf(tsdf_abs[tx], tsdf_abs[tx + stride]);
  }
  // de-allocate block
  if (tx == 0 && tsdf_abs[0] >= min_tsdf_threshold) {
    hash_table.Delete(blocks[blockIdx.x].position);
  }
}

__global__ static void ray_cast_kernel(const VoxelHashTable hash_table,
                                       const CameraParams cam_params, const SE3<float> cam_T_world,
                                       const SE3<float> world_T_cam, const float step_size,
                                       const float max_depth, const float voxel_size,
                                       uchar4* img_tsdf_rgba, uchar4* img_tsdf_normal) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cam_params.img_w || y >= cam_params.img_h) {
    return;
  }
  const int idx = y * cam_params.img_w + x;
  const Eigen::Vector3f pos_img_h(x, y, 1);
  const Eigen::Vector3f pos_cam = cam_params.intrinsics_inv * pos_img_h;
  const Eigen::Vector3f ray_dir_cam = pos_cam.normalized();
  const Eigen::Quaternionf world_R_cam = world_T_cam.GetR();
  const Eigen::Vector3f ray_dir_world = world_R_cam * ray_dir_cam;

  // ray casting step vector

  Eigen::Vector3f ray_step_grid = ray_dir_world * step_size / voxel_size;
  const int max_step = ceil(max_depth / step_size);
  Eigen::Vector3f pos_grid = world_T_cam.GetT() / voxel_size;
  VoxelBlock cache;
  const auto roundf_func = [](const float x) { return roundf(x); };
  float tsdf_prev =
      hash_table.Retrieve<VoxelTSDF>(pos_grid.unaryExpr(roundf_func).cast<short>(), cache).tsdf;
  pos_grid += ray_step_grid;
  for (int i = 1; i < max_step; ++i) {
    const float tsdf_curr =
        hash_table.Retrieve<VoxelTSDF>(pos_grid.unaryExpr(roundf_func).cast<short>(), cache).tsdf;
    const unsigned char weight_curr =
        hash_table.Retrieve<VoxelRGBW>(pos_grid.unaryExpr(roundf_func).cast<short>(), cache).weight;

    // Skip voxels with insufficient observations.
    if (weight_curr < 10) {
      pos_grid += ray_step_grid;
      tsdf_prev = tsdf_curr;
      continue;
    }
    // ray hit front surface
    if (tsdf_prev > 0 && tsdf_curr <= 0 && tsdf_prev - tsdf_curr <= 2.0) {
      const Eigen::Vector3f pos1_grid = pos_grid - ray_step_grid;
      const Eigen::Vector3f pos2_grid = pos_grid;

      // When zero-cross happens, we use trilinear interpolation to get better TSDF estimate
      const auto accurate_tsdf_curr = hash_table.RetrieveTSDF(pos2_grid, cache);
      const auto accurate_tsdf_prev = hash_table.RetrieveTSDF(pos1_grid, cache);

      // computing zero-crossing voxel coordinates
      const Eigen::Vector3f pos_interp_grid =
          pos_grid + accurate_tsdf_curr / (accurate_tsdf_prev - accurate_tsdf_curr) * ray_step_grid;
      const Eigen::Matrix<short, 3, 1> final_grid =
          pos_interp_grid.unaryExpr(roundf_func).cast<short>();

      // Retrieve RGBW and SEGM voxel
      const VoxelRGBW voxel_rgbw = hash_table.Retrieve<VoxelRGBW>(final_grid, cache);
      const VoxelSEGM voxel_segm = hash_table.Retrieve<VoxelSEGM>(final_grid, cache);

      // calculate gradient for rendering diffusivity
      const Eigen::Matrix<short, 3, 1> x_pos(final_grid[0] + 1, final_grid[1], final_grid[2]);
      const Eigen::Matrix<short, 3, 1> x_neg(final_grid[0] - 1, final_grid[1], final_grid[2]);
      const Eigen::Matrix<short, 3, 1> y_pos(final_grid[0], final_grid[1] + 1, final_grid[2]);
      const Eigen::Matrix<short, 3, 1> y_neg(final_grid[0], final_grid[1] - 1, final_grid[2]);
      const Eigen::Matrix<short, 3, 1> z_pos(final_grid[0], final_grid[1], final_grid[2] + 1);
      const Eigen::Matrix<short, 3, 1> z_neg(final_grid[0], final_grid[1], final_grid[2] - 1);
      const Eigen::Vector3f norm_raw_grid(hash_table.Retrieve<VoxelTSDF>(x_pos, cache).tsdf -
                                              hash_table.Retrieve<VoxelTSDF>(x_neg, cache).tsdf,
                                          hash_table.Retrieve<VoxelTSDF>(y_pos, cache).tsdf -
                                              hash_table.Retrieve<VoxelTSDF>(y_neg, cache).tsdf,
                                          hash_table.Retrieve<VoxelTSDF>(z_pos, cache).tsdf -
                                              hash_table.Retrieve<VoxelTSDF>(z_neg, cache).tsdf);
      const float diffusivity = fmaxf(norm_raw_grid.dot(-ray_dir_world) / norm_raw_grid.norm(), 0);
      // find max class
      int max_cls = voxel_segm.semantic_rep.get_max_class();
      float alpha = 0;
      if (max_cls != 0) alpha = 0.5; // alpha for non-background classes
      img_tsdf_rgba[idx] =
          make_uchar4(alpha * label_color_palette[max_cls][0] + (1 - alpha) * voxel_rgbw.rgb[0],
                      alpha * label_color_palette[max_cls][1] + (1 - alpha) * voxel_rgbw.rgb[1],
                      alpha * label_color_palette[max_cls][2] + (1 - alpha) * voxel_rgbw.rgb[2], 255);
      img_tsdf_normal[idx] =
          make_uchar4(alpha * label_color_palette[max_cls][0] + (1 - alpha) * diffusivity * 255,
                      alpha * label_color_palette[max_cls][1] + (1 - alpha) * diffusivity * 255,
                      alpha * label_color_palette[max_cls][2] + (1 - alpha) * diffusivity * 255, 255);
      return;
    }
    tsdf_prev = tsdf_curr;

    if (tsdf_curr < 0.5) {
      // if we get close enough to a surface, use smaller step size for finer estimation
      ray_step_grid = (ray_dir_world * step_size / voxel_size) / 10;
    } else {
      // far away from surface. Use coarse step size to speed up
      ray_step_grid = ray_dir_world * step_size / voxel_size;
    }
    pos_grid += ray_step_grid;
  }

  // For loop terminates with no result. No surface intersection (zero-crossing) found
  img_tsdf_rgba[idx] = make_uchar4(0, 0, 0, 0);
  img_tsdf_normal[idx] = make_uchar4(0, 0, 0, 0);
}

TSDFGrid::TSDFGrid(float voxel_size, float truncation)
    : voxel_size_(voxel_size), truncation_(truncation) {
  // memory allocation
  CUDA_SAFE_CALL(cudaMalloc(&visible_mask_, sizeof(int) * NUM_ENTRY));
  CUDA_SAFE_CALL(cudaMalloc(&visible_indics_, sizeof(int) * NUM_ENTRY));
  CUDA_SAFE_CALL(cudaMalloc(&visible_indics_aux_, sizeof(int) * NUM_ENTRY / (2 * SCAN_BLOCK_SIZE)));
  CUDA_SAFE_CALL(
      cudaMalloc(&visible_blocks_,
                 sizeof(VoxelBlock) * NUM_ENTRY));  // TODO(roger): change to NUM_BLOCKS and test
  CUDA_SAFE_CALL(cudaMalloc(&img_rgb_, sizeof(uint3) * MAX_IMG_SIZE));
  CUDA_SAFE_CALL(cudaMalloc(&img_depth_, sizeof(float) * MAX_IMG_SIZE));
  CUDA_SAFE_CALL(cudaMalloc(&prob_map_, sizeof(float) * MAX_IMG_SIZE * NUM_CLASSES));
  CUDA_SAFE_CALL(cudaMalloc(&img_depth_to_range_, sizeof(float) * MAX_IMG_SIZE));
  CUDA_SAFE_CALL(cudaMalloc(&img_tsdf_rgba_, sizeof(uchar4) * MAX_IMG_SIZE));
  CUDA_SAFE_CALL(cudaMalloc(&img_tsdf_normal_, sizeof(uchar4) * MAX_IMG_SIZE));
  // stream init
  CUDA_SAFE_CALL(cudaStreamCreate(&stream_));
  CUDA_SAFE_CALL(cudaStreamCreate(&stream2_));
}

TSDFGrid::~TSDFGrid() {
  // release memory
  hash_table_.ReleaseMemory();
  CUDA_SAFE_CALL(cudaFree(visible_mask_));
  CUDA_SAFE_CALL(cudaFree(visible_indics_));
  CUDA_SAFE_CALL(cudaFree(visible_indics_aux_));
  CUDA_SAFE_CALL(cudaFree(visible_blocks_));
  CUDA_SAFE_CALL(cudaFree(img_rgb_));
  CUDA_SAFE_CALL(cudaFree(img_depth_));
  CUDA_SAFE_CALL(cudaFree(prob_map_));
  CUDA_SAFE_CALL(cudaFree(img_depth_to_range_));
  CUDA_SAFE_CALL(cudaFree(img_tsdf_rgba_));
  CUDA_SAFE_CALL(cudaFree(img_tsdf_normal_));
  // release cuda stream
  CUDA_SAFE_CALL(cudaStreamDestroy(stream_));
  CUDA_SAFE_CALL(cudaStreamDestroy(stream2_));
}

void TSDFGrid::Integrate(const cv::Mat& img_rgb, const cv::Mat& img_depth,
                         const torch::Tensor& prob_map, float max_depth,
                         const CameraIntrinsics<float>& intrinsics, const SE3<float>& cam_T_world) {
  // check rgb/depth type and shape
  assert(img_rgb.type() == CV_8UC3);
  assert(img_depth.type() == CV_32FC1);
  width_ = img_rgb.cols;
  height_ = img_rgb.rows;
  assert(img_rgb.cols == width_);
  assert(img_rgb.rows == height_);
  assert(img_rgb.cols == img_depth.cols);
  assert(img_rgb.rows == img_depth.rows);
  // check prob map type and shape
  assert(prob_map.dtype() == torch::kFloat32);
  assert(img_depth.cols == prob_map.sizes()[2]);
  assert(img_depth.rows == prob_map.sizes()[1]);
  assert(img_rgb.rows < MAX_IMG_H);
  assert(img_rgb.cols < MAX_IMG_W);
  assert(prob_map.sizes()[1] == height_);
  assert(prob_map.sizes()[2] == width_);
  assert(prob_map.sizes()[0] == NUM_CLASSES);

  const CameraParams cam_params(intrinsics, img_rgb.rows, img_rgb.cols);

  // data transfer
  CUDA_SAFE_CALL(cudaMemcpyAsync(prob_map_, prob_map.data_ptr(), sizeof(float) * width_ * height_ * NUM_CLASSES,
                                 cudaMemcpyDeviceToDevice, stream2_));
  CUDA_SAFE_CALL(cudaMemcpyAsync(img_rgb_, img_rgb.data, sizeof(uchar3) * img_rgb.total(),
                                 cudaMemcpyHostToDevice, stream_));
  CUDA_SAFE_CALL(cudaMemcpyAsync(img_depth_, img_depth.data, sizeof(float) * img_depth.total(),
                                 cudaMemcpyHostToDevice, stream_));
                                 // ht is 0
  // compute
  spdlog::debug("[TSDF] pre integrate: {} active blocks", hash_table_.NumActiveBlock());
  Allocate(img_rgb, img_depth, max_depth, cam_params, cam_T_world);
  const int num_visible_blocks = GatherVisible(max_depth, cam_params, cam_T_world);
  CUDA_SAFE_CALL(cudaStreamSynchronize(stream2_));  // synchronize ht / lt img copy
  // it takes ~2ms from first async memcpy to get here
  UpdateTSDF(num_visible_blocks, max_depth, cam_params, cam_T_world);

  // Deallocate empty voxels
  SpaceCarving(num_visible_blocks);
  CUDA_SAFE_CALL(cudaStreamSynchronize(stream_));
  spdlog::debug("[TSDF] post integrate: {} active blocks", hash_table_.NumActiveBlock());
}

void TSDFGrid::Allocate(const cv::Mat& img_rgb, const cv::Mat& img_depth, float max_depth,
                        const CameraParams& cam_params, const SE3<float>& cam_T_world) {
  const dim3 IMG_BLOCK_DIM(ceil((float)cam_params.img_w / 32), ceil((float)cam_params.img_h / 16));
  const dim3 IMG_THREAD_DIM(32, 16);
  block_allocate_kernel<<<IMG_BLOCK_DIM, IMG_THREAD_DIM, 0, stream_>>>(
      hash_table_, img_depth_, cam_params, cam_T_world, cam_T_world.Inverse(), voxel_size_,
      max_depth, truncation_, img_depth_to_range_);
  CUDA_STREAM_CHECK_ERROR(stream_);
  hash_table_.ResetLocks(stream_);
}

int TSDFGrid::GatherVisible(float max_depth, const CameraParams& cam_params,
                            const SE3<float>& cam_T_world) {
  constexpr int GATHER_BLOCK_DIM = NUM_ENTRY / BLOCK_VOLUME;
  // generate binary array of visibility
  check_visibility_kernel<<<GATHER_BLOCK_DIM, BLOCK_VOLUME, 0, stream_>>>(
      hash_table_, voxel_size_, max_depth, cam_params, cam_T_world, visible_mask_);
  CUDA_STREAM_CHECK_ERROR(stream_);

  return GatherBlock();
}

std::vector<VoxelSpatialTSDF> TSDFGrid::GatherValid() {
  spdlog::debug("[TSDF] {} active blocks before download", hash_table_.NumActiveBlock());

  constexpr int GATHER_BLOCK_DIM = NUM_ENTRY / BLOCK_VOLUME;

  check_valid_kernel<<<GATHER_BLOCK_DIM, BLOCK_VOLUME, 0, stream_>>>(hash_table_, visible_mask_);
  CUDA_STREAM_CHECK_ERROR(stream_);

  const int num_visible_blocks = GatherBlock();
  std::vector<VoxelSpatialTSDF> ret(num_visible_blocks * BLOCK_VOLUME);

  VoxelSpatialTSDF* voxel_pos_tsdf;
  CUDA_SAFE_CALL(
      cudaMalloc(&voxel_pos_tsdf, sizeof(VoxelSpatialTSDF) * num_visible_blocks * BLOCK_VOLUME));

  constexpr dim3 DOWNLOAD_THREAD_DIM(BLOCK_LEN, BLOCK_LEN, BLOCK_LEN);
  download_tsdf_kernel<<<num_visible_blocks, DOWNLOAD_THREAD_DIM, 0, stream_>>>(
      hash_table_, visible_blocks_, voxel_size_, voxel_pos_tsdf);
  CUDA_STREAM_CHECK_ERROR(stream_);

  CUDA_SAFE_CALL(cudaMemcpyAsync(ret.data(), voxel_pos_tsdf,
                                 sizeof(VoxelSpatialTSDF) * num_visible_blocks * BLOCK_VOLUME,
                                 cudaMemcpyDeviceToHost, stream_));
  CUDA_SAFE_CALL(cudaFree(voxel_pos_tsdf));

  return ret;
}

std::vector<VoxelSpatialTSDFSEGM> TSDFGrid::GatherValidSemantic() {
  spdlog::debug("[TSDF] {} active blocks before download", hash_table_.NumActiveBlock());

  constexpr int GATHER_BLOCK_DIM = NUM_ENTRY / BLOCK_VOLUME;

  check_valid_kernel<<<GATHER_BLOCK_DIM, BLOCK_VOLUME, 0, stream_>>>(hash_table_, visible_mask_);
  CUDA_STREAM_CHECK_ERROR(stream_);

  const int num_visible_blocks = GatherBlock();
  std::vector<VoxelSpatialTSDFSEGM> ret(num_visible_blocks * BLOCK_VOLUME);

  VoxelSpatialTSDFSEGM* voxel_pos_tsdf;
  CUDA_SAFE_CALL(cudaMalloc(&voxel_pos_tsdf,
                            sizeof(VoxelSpatialTSDFSEGM) * num_visible_blocks * BLOCK_VOLUME));

  constexpr dim3 DOWNLOAD_THREAD_DIM(BLOCK_LEN, BLOCK_LEN, BLOCK_LEN);
  download_semantic_kernel<<<num_visible_blocks, DOWNLOAD_THREAD_DIM, 0, stream_>>>(
      hash_table_, visible_blocks_, voxel_size_, voxel_pos_tsdf);
  CUDA_STREAM_CHECK_ERROR(stream_);

  CUDA_SAFE_CALL(cudaMemcpyAsync(ret.data(), voxel_pos_tsdf,
                                 sizeof(VoxelSpatialTSDFSEGM) * num_visible_blocks * BLOCK_VOLUME,
                                 cudaMemcpyDeviceToHost, stream_));
  CUDA_SAFE_CALL(cudaFree(voxel_pos_tsdf));

  return ret;
}

std::vector<VoxelSpatialTSDF> TSDFGrid::GatherVoxels(const BoundingCube<float>& volumn) {
  // convert bounds to grid coordinates
  const BoundingCube<short> volumn_grid = volumn.Scale<short>(1. / voxel_size_);

  constexpr int GATHER_BLOCK_DIM = NUM_ENTRY / BLOCK_VOLUME;
  check_bound_kernel<<<GATHER_BLOCK_DIM, BLOCK_VOLUME, 0, stream_>>>(hash_table_, volumn_grid,
                                                                     visible_mask_);
  CUDA_STREAM_CHECK_ERROR(stream_);

  const int num_visible_blocks = GatherBlock();
  std::vector<VoxelSpatialTSDF> ret(num_visible_blocks * BLOCK_VOLUME);

  VoxelSpatialTSDF* voxel_pos_tsdf;
  CUDA_SAFE_CALL(
      cudaMalloc(&voxel_pos_tsdf, sizeof(VoxelSpatialTSDF) * num_visible_blocks * BLOCK_VOLUME));

  constexpr dim3 DOWNLOAD_THREAD_DIM(BLOCK_LEN, BLOCK_LEN, BLOCK_LEN);
  download_tsdf_kernel<<<num_visible_blocks, DOWNLOAD_THREAD_DIM, 0, stream_>>>(
      hash_table_, visible_blocks_, voxel_size_, voxel_pos_tsdf);
  CUDA_STREAM_CHECK_ERROR(stream_);

  CUDA_SAFE_CALL(cudaMemcpyAsync(ret.data(), voxel_pos_tsdf,
                                 sizeof(VoxelSpatialTSDF) * num_visible_blocks * BLOCK_VOLUME,
                                 cudaMemcpyDeviceToHost, stream_));
  CUDA_SAFE_CALL(cudaFree(voxel_pos_tsdf));

  return ret;
}

__global__ static void marching_cube_kernel(const VoxelHashTable hash_table,
                                            const VoxelBlock* blocks, const float voxel_size,
                                            Eigen::Vector3f* vertices, float* vertices_prob_arr,
                                            Eigen::Vector3i* triangle_ids, int* vertex_mask,
                                            int* triangle_mask) {
  __shared__ float cube_tsdf[BLOCK_LEN * 2][BLOCK_LEN * 2][BLOCK_LEN * 2];
  __shared__ float cube_segm_prob[BLOCK_LEN * 2][BLOCK_LEN * 2][BLOCK_LEN * 2];

  // initalized as uint8_t array because shared memory buffer in CUDA
  // do not support object array instantiation
  __shared__ uint8_t cube_blocks_buff[8 * sizeof(VoxelBlock)];

  // reinterpret the above uint8_t array as VoxelBlock array of size [2][2][2]
  // cube_blocks in declared here as a reference to this array
  VoxelBlock(&cube_blocks)[2][2][2] = *reinterpret_cast<VoxelBlock(*)[2][2][2]>(cube_blocks_buff);

  const VoxelBlock& base_block = blocks[blockIdx.x];
  const Eigen::Matrix<short, 3, 1> offset_grid(threadIdx.x, threadIdx.y, threadIdx.z);
  const int offset_idx = OffsetToIndex(offset_grid);

  // load 2x2x2 block for boundary handling
  // TODO(alvin): to be optimized
  if (threadIdx.x <= 1 && threadIdx.y <= 1 && threadIdx.z <= 1) {
    hash_table.GetBlock(base_block.position + offset_grid,
                        &cube_blocks[threadIdx.z][threadIdx.y][threadIdx.x]);
  }
  __syncthreads();

// load voxel tsdf values
#pragma unroll
  for (int i = 0; i < 2; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        // test if this block has been allocated in the voxel hashmap
        if (cube_blocks[i][j][k].idx >= 0) {
          const VoxelTSDF& tsdf =
              hash_table.mem.GetVoxel<VoxelTSDF>(offset_idx, cube_blocks[i][j][k]);
          const int weight =
              hash_table.mem.GetVoxel<VoxelRGBW>(offset_idx, cube_blocks[i][j][k]).weight;
          // FIXME!!!
          // TODO(roger): maintain only the maximum class. Use TSDF to choose the nearest one as predicted class
          // const float segm_prob =
          //     __half2float(hash_table.mem.GetVoxel<VoxelSEGM>(offset_idx, cube_blocks[i][j][k]).prob_vec[0]);
          const float segm_prob = 0;
          if (weight > 10) {
            // if mutliple measurement, plug in tsdf
            cube_tsdf[i * BLOCK_LEN + threadIdx.z][j * BLOCK_LEN + threadIdx.y]
                     [k * BLOCK_LEN + threadIdx.x] = tsdf.tsdf;
            cube_segm_prob[i * BLOCK_LEN + threadIdx.z][j * BLOCK_LEN + threadIdx.y]
                          [k * BLOCK_LEN + threadIdx.x] = segm_prob;
          } else {
            cube_tsdf[i * BLOCK_LEN + threadIdx.z][j * BLOCK_LEN + threadIdx.y]
                     [k * BLOCK_LEN + threadIdx.x] = -10;
            cube_segm_prob[i * BLOCK_LEN + threadIdx.z][j * BLOCK_LEN + threadIdx.y]
                          [k * BLOCK_LEN + threadIdx.x] = 0;
          }
        } else {
          cube_tsdf[i * BLOCK_LEN + threadIdx.z][j * BLOCK_LEN + threadIdx.y]
                   [k * BLOCK_LEN + threadIdx.x] = -10;
          cube_segm_prob[i * BLOCK_LEN + threadIdx.z][j * BLOCK_LEN + threadIdx.y]
                        [k * BLOCK_LEN + threadIdx.x] = 0;
        }
      }
    }
  }
  __syncthreads();

  // local caching of the cube tsdf values + compute cube scenario
  float local_tsdf[NUM_VERTICES_PER_CUBE];
  // local 8-bit binary mask (lowest 8 bit)
  int cubeindex = 0;
#pragma unroll
  for (int i = 0; i < NUM_VERTICES_PER_CUBE; ++i) {
    local_tsdf[i] = cube_tsdf[threadIdx.z + offset_table[i][2]][threadIdx.y + offset_table[i][1]]
                             [threadIdx.x + offset_table[i][0]];
    // if tsdf < 0 for some i, the lowest i-th bit in the final cubeindex will be 1
    cubeindex |= ((local_tsdf[i] < 0) << i);
  }

// compute 3 vertices per cube
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int cube_offset_idx = offset_idx + i * BLOCK_VOLUME;
    if (cube_offset_idx < BLOCK_VERT_VOLUME) {
      const Eigen::Matrix<short, 3, 1> vertex_offset(
          cube_offset_idx % BLOCK_VERT_LEN, cube_offset_idx / BLOCK_VERT_LEN % BLOCK_VERT_LEN,
          cube_offset_idx / BLOCK_VERT_AREA);
      const Eigen::Vector3f v1 = (BlockToPoint(base_block.position) + vertex_offset).cast<float>();

      const float t1 = cube_tsdf[vertex_offset[2]][vertex_offset[1]][vertex_offset[0]];
      const float prob1 = cube_segm_prob[vertex_offset[2]][vertex_offset[1]][vertex_offset[0]];
      const int cube_idx = blockIdx.x * BLOCK_VERT_VOLUME + cube_offset_idx;

#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const int v2_idx = offset_dim_indics[j];
        const Eigen::Map<Eigen::Vector3i> v_offset(offset_table[v2_idx]);
        const float t2 = cube_tsdf[vertex_offset[2] + v_offset[2]][vertex_offset[1] + v_offset[1]]
                                  [vertex_offset[0] + v_offset[0]];
        const float prob2 =
            cube_segm_prob[vertex_offset[2] + v_offset[2]][vertex_offset[1] + v_offset[1]]
                          [vertex_offset[0] + v_offset[0]];
        vertices[cube_idx * 3 + j] = (v1 + (-t1) / (t2 - t1) * v_offset.cast<float>()) * voxel_size;

        // TODO(roger): is there a better way to compute vertex probabilities?
        vertices_prob_arr[cube_idx * 3 + j] = (prob1 + prob2) / 2;
        // not actually filling the array. This is placed here
        // as a speed up to initialize everything as zeros.
        vertex_mask[cube_idx * 3 + j] = 0;
      }
    }
  }

  __syncthreads();

  const int thread_idx = blockIdx.x * BLOCK_VOLUME + offset_idx;
// compute triangle indics
#pragma unroll
  for (int i = 0; i < MAX_TRIANGLES_PER_CUBE; ++i) {
    const int triangle_idx = thread_idx * MAX_TRIANGLES_PER_CUBE + i;
    if (tri_table[cubeindex][i * 3] != -1) {
      triangle_mask[triangle_idx] = 1;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        // get relative vertex index
        const int v1_idx = edge_table[tri_table[cubeindex][i * 3 + j]][0];
        const int v2_idx = edge_table[tri_table[cubeindex][i * 3 + j]][1];

        // check for accidental back faces due to constant initialization
        // First case: TSDF shouldn't be too close (potential numerical instability)
        // Second case: tsdf values are normalized (ranged in -1 to 1)
        if (fabsf(local_tsdf[v2_idx] - local_tsdf[v1_idx]) < 1e-3 ||
            fabsf(local_tsdf[v2_idx] - local_tsdf[v1_idx]) >= 2) {
          triangle_mask[triangle_idx] = 0;
          break;
        }

        // we use lower vertex idx here because
        // every voxel is in charge of positive x/y/z directions only
        const int lower_vertex_idx = edge_vertex_map[tri_table[cubeindex][i * 3 + j]][0];
        const int v_dim_offset = edge_vertex_map[tri_table[cubeindex][i * 3 + j]][1];
        const int cube_offset_idx =
            (offset_table[lower_vertex_idx][2] + threadIdx.z) * BLOCK_VERT_AREA +
            (offset_table[lower_vertex_idx][1] + threadIdx.y) * BLOCK_VERT_LEN +
            (offset_table[lower_vertex_idx][0] + threadIdx.x);
        const int cube_idx = blockIdx.x * BLOCK_VERT_VOLUME + cube_offset_idx;

        triangle_ids[triangle_idx][j] = cube_idx * 3 + v_dim_offset;
        vertex_mask[cube_idx * 3 + v_dim_offset] = 1;
      }

    } else {
      triangle_mask[triangle_idx] = 0;
    }
  }
}

template <typename T>
__global__ static void compactify_kernel(T* outputs, const T* inputs, const int* mask,
                                         const int* valid_map, const int length) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length && mask[idx]) {
    outputs[valid_map[idx] - 1] = inputs[idx];
  }
}

__global__ static void transform_triangle_id_kernel(int* triangle_ids, const int* vertex_valid_map,
                                                    const int num_triangle_ids) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_triangle_ids) {
    triangle_ids[idx] = vertex_valid_map[triangle_ids[idx]] - 1;
  }
}

void TSDFGrid::GatherValidMesh(std::vector<Eigen::Vector3f>* vertex_buffer,
                               std::vector<Eigen::Vector3i>* index_buffer,
                               std::vector<float>* vertex_prob_buffer) {
  constexpr int GATHER_BLOCK_DIM = NUM_ENTRY / BLOCK_VOLUME;

  // write binary valid mask to visible_mask_
  check_valid_kernel<<<GATHER_BLOCK_DIM, BLOCK_VOLUME, 0, stream_>>>(hash_table_, visible_mask_);
  CUDA_STREAM_CHECK_ERROR(stream_);

  // convert to contiguous block array
  const int num_visible_blocks = GatherBlock();

  // maximum possible number of vertices
  const int num_vertices = num_visible_blocks * BLOCK_VERT_VOLUME * 3;
  const int num_triangles = num_visible_blocks * BLOCK_VOLUME * MAX_TRIANGLES_PER_CUBE;
  int num_valid_triangles, num_valid_vertices;

  Eigen::Vector3f* vertices;
  Eigen::Vector3f* valid_vertices;
  Eigen::Vector3i* triangle_ids;
  Eigen::Vector3i* valid_triangle_ids;

  float* vertices_prob_arr;
  float* valid_vertices_prob_arr;

  // boolean arrays indicating whether a particular vertex or triangle is valid
  int* vertex_mask;
  int* triangle_mask;

  // map from sparse indices of triangles to contiguous triangles
  int* vertex_valid_map;
  int* triangle_valid_map;

  CUDA_SAFE_CALL(cudaMalloc(&vertices, sizeof(Eigen::Vector3f) * num_vertices));
  CUDA_SAFE_CALL(cudaMalloc(&vertices_prob_arr, sizeof(float) * num_vertices));
  CUDA_SAFE_CALL(cudaMalloc(&vertex_mask, sizeof(int) * num_vertices));
  CUDA_SAFE_CALL(cudaMalloc(&vertex_valid_map, sizeof(int) * num_vertices));
  CUDA_SAFE_CALL(cudaMalloc(&triangle_ids, sizeof(Eigen::Vector3i) * num_triangles));
  CUDA_SAFE_CALL(cudaMalloc(&triangle_mask, sizeof(int) * num_triangles));
  CUDA_SAFE_CALL(cudaMalloc(&triangle_valid_map, sizeof(int) * num_triangles));

  constexpr dim3 DOWNLOAD_THREAD_DIM(BLOCK_LEN, BLOCK_LEN, BLOCK_LEN);
  marching_cube_kernel<<<num_visible_blocks, DOWNLOAD_THREAD_DIM, 0, stream_>>>(
      hash_table_, visible_blocks_, voxel_size_, vertices, vertices_prob_arr, triangle_ids,
      vertex_mask, triangle_mask);

  prefix_sum<int>(triangle_mask, triangle_valid_map, nullptr, num_triangles, stream_);
  CUDA_SAFE_CALL(cudaMemcpyAsync(&num_valid_triangles, triangle_valid_map + num_triangles - 1,
                                 sizeof(int), cudaMemcpyDeviceToHost, stream_));

  prefix_sum<int>(vertex_mask, vertex_valid_map, nullptr, num_vertices, stream2_);
  CUDA_SAFE_CALL(cudaMemcpyAsync(&num_valid_vertices, vertex_valid_map + num_vertices - 1,
                                 sizeof(int), cudaMemcpyDeviceToHost, stream2_));

  CUDA_SAFE_CALL(cudaStreamSynchronize(stream_));
  CUDA_SAFE_CALL(cudaStreamSynchronize(stream2_));
  spdlog::debug("Gathered {} / {} vertices, {} / {} triangles", num_valid_vertices, num_vertices,
                num_valid_triangles, num_triangles);

  CUDA_SAFE_CALL(cudaMalloc(&valid_triangle_ids, sizeof(Eigen::Vector3i) * num_valid_triangles));
  CUDA_SAFE_CALL(cudaMalloc(&valid_vertices, sizeof(Eigen::Vector3f) * num_valid_vertices));
  CUDA_SAFE_CALL(cudaMalloc(&valid_vertices_prob_arr, sizeof(float) * num_valid_vertices));

  // Compactify sparse triangles to contiguous triangles
  compactify_kernel<<<ceil((float)num_triangles / 512), 512, 0, stream_>>>(
      valid_triangle_ids, triangle_ids, triangle_mask, triangle_valid_map, num_triangles);

  // Transform sparse vertices indices referenced by triangles
  // to contiguous triangles idx (which is to be obtained by next compactify call)
  transform_triangle_id_kernel<<<ceil((float)num_valid_triangles * 3 / 512), 512, 0, stream_>>>(
      (int*)valid_triangle_ids, vertex_valid_map, num_valid_triangles * 3);

  // Compactify sparse vertices to contiguous representation
  compactify_kernel<<<ceil((float)num_vertices / 512), 512, 0, stream2_>>>(
      valid_vertices, vertices, vertex_mask, vertex_valid_map, num_vertices);

  compactify_kernel<<<ceil((float)num_vertices / 512), 512, 0, stream2_>>>(
      valid_vertices_prob_arr, vertices_prob_arr, vertex_mask, vertex_valid_map, num_vertices);

  vertex_buffer->reserve(num_valid_vertices);
  vertex_prob_buffer->reserve(num_valid_vertices);
  index_buffer->reserve(num_valid_triangles);
  vertex_buffer->resize(num_valid_vertices);
  vertex_prob_buffer->resize(num_valid_vertices);
  index_buffer->resize(num_valid_triangles);

  CUDA_SAFE_CALL(cudaMemcpyAsync(index_buffer->data(), valid_triangle_ids,
                                 sizeof(Eigen::Vector3i) * num_valid_triangles,
                                 cudaMemcpyDeviceToHost, stream_));
  CUDA_SAFE_CALL(cudaMemcpyAsync(vertex_buffer->data(), valid_vertices,
                                 sizeof(Eigen::Vector3f) * num_valid_vertices,
                                 cudaMemcpyDeviceToHost, stream2_));
  CUDA_SAFE_CALL(cudaMemcpyAsync(vertex_prob_buffer->data(), valid_vertices_prob_arr,
                                 sizeof(float) * num_valid_vertices, cudaMemcpyDeviceToHost,
                                 stream2_));

  CUDA_SAFE_CALL(cudaStreamSynchronize(stream_));
  CUDA_SAFE_CALL(cudaStreamSynchronize(stream2_));

  CUDA_SAFE_CALL(cudaFree(vertices));
  CUDA_SAFE_CALL(cudaFree(vertices_prob_arr));
  CUDA_SAFE_CALL(cudaFree(triangle_ids));
  CUDA_SAFE_CALL(cudaFree(vertex_mask));
  CUDA_SAFE_CALL(cudaFree(vertex_valid_map));
  CUDA_SAFE_CALL(cudaFree(valid_vertices));
  CUDA_SAFE_CALL(cudaFree(valid_vertices_prob_arr));
  CUDA_SAFE_CALL(cudaFree(valid_triangle_ids));
  CUDA_SAFE_CALL(cudaFree(triangle_valid_map));
  CUDA_SAFE_CALL(cudaFree(triangle_mask));
}

int TSDFGrid::GatherBlock() {
  constexpr int GATHER_THREAD_DIM = 512;
  constexpr int GATHER_BLOCK_DIM = NUM_ENTRY / GATHER_THREAD_DIM;
  // parallel prefix sum scan
  // e.g., 1 0 1 0 0 1 0 -> 0 1 1 2 2 2 3
  // write to visible_indices
  prefix_sum<int>(visible_mask_, visible_indics_, visible_indics_aux_, NUM_ENTRY, stream_);

  // gather visible blocks into contiguous array
  gather_visible_blocks_kernel<<<GATHER_BLOCK_DIM, GATHER_THREAD_DIM, 0, stream_>>>(
      hash_table_, visible_mask_, visible_indics_, visible_blocks_);
  CUDA_STREAM_CHECK_ERROR(stream_);
  // get number of visible blocks from scanned index array
  int num_visible_blocks;

  CUDA_SAFE_CALL(cudaMemcpyAsync(&num_visible_blocks, visible_indics_ + NUM_ENTRY - 1, sizeof(int),
                                 cudaMemcpyDeviceToHost, stream_));
  CUDA_SAFE_CALL(cudaStreamSynchronize(stream_));
  spdlog::debug("[TSDF] Getting {} blocks", num_visible_blocks);
  return num_visible_blocks;
}

void TSDFGrid::UpdateTSDF(int num_visible_blocks, float max_depth, const CameraParams& cam_params,
                          const SE3<float>& cam_T_world) {
  const dim3 VOXEL_BLOCK_DIM(BLOCK_LEN, BLOCK_LEN, BLOCK_LEN);
  tsdf_integrate_kernel<<<num_visible_blocks, VOXEL_BLOCK_DIM, 0, stream_>>>(
      visible_blocks_, hash_table_.mem, cam_T_world, cam_params, num_visible_blocks, max_depth,
      truncation_, voxel_size_, img_rgb_, img_depth_, prob_map_, img_depth_to_range_, height_, width_);
  CUDA_STREAM_CHECK_ERROR(stream_);
}

void TSDFGrid::SpaceCarving(int num_visible_blocks) {
  space_carving_kernel<<<num_visible_blocks, BLOCK_VOLUME / 2, 0, stream_>>>(
      hash_table_, visible_blocks_, num_visible_blocks, .9);
  CUDA_STREAM_CHECK_ERROR(stream_);
  hash_table_.ResetLocks(stream_);
}

void TSDFGrid::RayCast(float max_depth, const CameraParams& virtual_cam,
                       const SE3<float>& cam_T_world, GLImage8UC4* tsdf_rgba,
                       GLImage8UC4* tsdf_normal) {
  const dim3 IMG_BLOCK_DIM(ceil((float)virtual_cam.img_w / 32),
                           ceil((float)virtual_cam.img_h / 16));
  const dim3 IMG_THREAD_DIM(32, 16);
  ray_cast_kernel<<<IMG_BLOCK_DIM, IMG_THREAD_DIM, 0, stream_>>>(
      hash_table_, virtual_cam, cam_T_world, cam_T_world.Inverse(), truncation_ / 2, max_depth,
      voxel_size_, img_tsdf_rgba_, img_tsdf_normal_);
  CUDA_STREAM_CHECK_ERROR(stream_);
  if (tsdf_rgba) {
    tsdf_rgba->LoadCuda(img_tsdf_rgba_, stream_);
  }
  if (tsdf_normal) {
    tsdf_normal->LoadCuda(img_tsdf_normal_, stream_);
  }
  CUDA_SAFE_CALL(cudaStreamSynchronize(stream_));
}
