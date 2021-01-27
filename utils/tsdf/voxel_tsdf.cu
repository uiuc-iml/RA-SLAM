#include <spdlog/spdlog.h>

#include "utils/cuda/arithmetic.cuh"
#include "utils/cuda/errors.cuh"
#include "utils/tsdf/voxel_tsdf.cuh"

#define MAX_IMG_H     1920
#define MAX_IMG_W     1080
#define MAX_IMG_SIZE  (MAX_IMG_H * MAX_IMG_W)

__global__ static void check_bound_kernel(const VoxelHashTable hash_table,
                                          const BoundingCube<short> volumn_grid,
                                          int *visible_mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const VoxelBlock &block = hash_table.GetBlock(idx);
  visible_mask[idx] = (block.idx >= 0 &&
      block.position.x >= volumn_grid.xmin &&
      block.position.y >= volumn_grid.ymin &&
      block.position.z >= volumn_grid.zmin &&
      block.position.x + BLOCK_LEN - 1 <= volumn_grid.xmax &&
      block.position.y + BLOCK_LEN - 1 <= volumn_grid.ymax &&
      block.position.z + BLOCK_LEN - 1 <= volumn_grid.zmax);
}

__global__ static void check_valid_kernel(const VoxelHashTable hash_table,
                                          int *visible_mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const VoxelBlock &block = hash_table.GetBlock(idx);
  const Vector3<short> pos_grid = block.position << BLOCK_LEN_BITS;
  visible_mask[idx] = (int)(block.idx >= 0);
}

__global__ static void download_tsdf_kernel(const VoxelHashTable hash_table,
                                            const VoxelBlock *blocks,
                                            const float voxel_size,
                                            VoxelSpatialTSDF *voxel_pos_tsdf) {
  const VoxelBlock &block = blocks[blockIdx.x];
  const Vector3<short> offset_grid(threadIdx.x, threadIdx.y, threadIdx.z);
  const Vector3<short> pos_grid = (block.position << BLOCK_LEN_BITS) + offset_grid;
  const Vector3<float> pos_world = pos_grid.cast<float>() * voxel_size;
  const int thread_idx = offset2index(offset_grid);

  const int idx = blockIdx.x * BLOCK_VOLUME + thread_idx;
  const VoxelTSDF &tsdf = hash_table.mem.GetVoxel<VoxelTSDF>(thread_idx, block);
  voxel_pos_tsdf[idx] = VoxelSpatialTSDF(pos_world, tsdf.tsdf);
}

__device__ static bool is_voxel_visible(const Vector3<short> &pos_grid,
                                        const SE3<float> &cam_P_world,
                                        const CameraParams &cam_params,
                                        const float &voxel_size) {
  const Vector3<float> pos_world = pos_grid.cast<float>() * voxel_size;
  const Vector3<float> pos_cam = cam_P_world.Apply(pos_world);
  const Vector3<float> pos_img_h = cam_params.intrinsics * pos_cam;
  const Vector3<float> pos_img = pos_img_h / pos_img_h.z;
  return (pos_img.x >= 0 && pos_img.x <= cam_params.img_w - 1 &&
          pos_img.y >= 0 && pos_img.y <= cam_params.img_h - 1 &&
          pos_img_h.z >= 0);
}

__device__ static bool is_full_block_visible(const Vector3<short> &block_pos,
                                             const SE3<float> &cam_P_world,
                                             const CameraParams &cam_params,
                                             const float &voxel_size) {
  const Vector3<short> pos_grid = block_pos << BLOCK_LEN_BITS;
  const short x = pos_grid.x, y = pos_grid.y, z = pos_grid.z, b = BLOCK_LEN - 1;
  return is_voxel_visible({ x, y, z }, cam_P_world, cam_params, voxel_size) &&
         is_voxel_visible({ x + b, y, z }, cam_P_world, cam_params, voxel_size) &&
         is_voxel_visible({ x, y + b, z }, cam_P_world, cam_params, voxel_size) &&
         is_voxel_visible({ x + b, y + b, z }, cam_P_world, cam_params, voxel_size) &&
         is_voxel_visible({ x, y, z + b }, cam_P_world, cam_params, voxel_size) &&
         is_voxel_visible({ x + b, y, z + b }, cam_P_world, cam_params, voxel_size) &&
         is_voxel_visible({ x, y + b, z + b }, cam_P_world, cam_params, voxel_size) &&
         is_voxel_visible({ x + b, y + b, z + b }, cam_P_world, cam_params, voxel_size);
}

__device__ static bool is_partial_block_visible(const Vector3<short> &block_pos,
                                                const SE3<float> &cam_P_world,
                                                const CameraParams &cam_params,
                                                const float &voxel_size) {
  const Vector3<short> pos_grid = block_pos << BLOCK_LEN_BITS;
  const short x = pos_grid.x, y = pos_grid.y, z = pos_grid.z, b = BLOCK_LEN - 1;
  return is_voxel_visible({ x, y, z }, cam_P_world, cam_params, voxel_size) ||
         is_voxel_visible({ x + b, y, z }, cam_P_world, cam_params, voxel_size) ||
         is_voxel_visible({ x, y + b, z }, cam_P_world, cam_params, voxel_size) ||
         is_voxel_visible({ x + b, y + b, z }, cam_P_world, cam_params, voxel_size) ||
         is_voxel_visible({ x, y, z + b }, cam_P_world, cam_params, voxel_size) ||
         is_voxel_visible({ x + b, y, z + b }, cam_P_world, cam_params, voxel_size) ||
         is_voxel_visible({ x, y + b, z + b }, cam_P_world, cam_params, voxel_size) ||
         is_voxel_visible({ x + b, y + b, z + b }, cam_P_world, cam_params, voxel_size);
}

__global__ static void check_visibility_kernel(const VoxelHashTable hash_table,
                                               const float voxel_size,
                                               const float max_depth,
                                               const CameraParams cam_params,
                                               const SE3<float> cam_P_world,
                                               int *visible_mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const VoxelBlock &block = hash_table.GetBlock(idx);
  if (block.idx < 0) {
    visible_mask[idx] = 0;
    return;
  }
  visible_mask[idx] = is_partial_block_visible(block.position, cam_P_world, cam_params, voxel_size);
}

__global__ static void gather_visible_blocks_kernel(const VoxelHashTable hash_table,
                                                    const int *visible_mask,
                                                    const int *visible_indics,
                                                    VoxelBlock *output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (visible_mask[idx]) {
    output[visible_indics[idx] - 1] = hash_table.GetBlock(idx);
  }
}

__global__ static void block_allocate_kernel(VoxelHashTable hash_table,
                                             const float *img_depth,
                                             const CameraParams cam_params,
                                             const SE3<float> cam_P_world,
                                             const SE3<float> world_P_cam,
                                             const float voxel_size,
                                             const float max_depth,
                                             const float truncation,
                                             float *img_depth_to_range) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cam_params.img_w || y >= cam_params.img_h) { return; }
  const int idx = y * cam_params.img_w + x;
  const float depth = img_depth[idx];
  // calculate depth2range scale
  const Vector3<float> pos_img_h(x, y, 1.);
  const Vector3<float> pos_cam = cam_params.intrinsics_inv * pos_img_h;
  img_depth_to_range[idx] = sqrtf(pos_cam.dot(pos_cam));
  if (depth == 0 || depth > max_depth) { return; }
  // transform coordinate from image to world
  const Vector3<float> pos_world = world_P_cam.Apply(pos_cam * depth);
  // calculate end coordinates of sample ray
  const Vector3<float> ray_dir_cam = pos_cam / img_depth_to_range[idx];
  const SO3<float> world_R_cam = world_P_cam.GetR();
  const Vector3<float> ray_dir_world = world_R_cam * ray_dir_cam;
  const Vector3<float> ray_start_world = pos_world - ray_dir_world * truncation;
  // put ray into voxel grid coordinate
  const Vector3<float> ray_dir_grid = ray_dir_world / voxel_size;
  const Vector3<float> ray_start_grid = ray_start_world / voxel_size;
  const Vector3<float> ray_grid = 2 * truncation * ray_dir_grid; // start -> end vector
  // DDA for finding ray / block intersection
  const int step_grid =
    ceilf(fmaxf(fmaxf(fabsf(ray_grid.x), fabsf(ray_grid.y)), fabsf(ray_grid.z)) / BLOCK_LEN);
  const Vector3<float> ray_step_grid = ray_grid / fmaxf((float)step_grid, 1);
  Vector3<float> pos_grid = ray_start_grid;
  // allocate blocks along the ray
  for (int i = 0; i <= step_grid; ++i, pos_grid += ray_step_grid) {
    const Vector3<short> pos_block = pos_grid.round<short>() >> BLOCK_LEN_BITS;
    if (is_full_block_visible(pos_block, cam_P_world, cam_params, voxel_size))
      hash_table.Allocate(pos_block);
  }
}

__global__ static void tsdf_integrate_kernel(VoxelBlock *blocks,
                                             VoxelMemPool voxel_mem,
                                             const SE3<float> cam_P_world,
                                             const CameraParams cam_params,
                                             const int num_visible_blocks,
                                             const float max_depth,
                                             const float truncation,
                                             const float voxel_size,
                                             const uchar3 *img_rgb,
                                             const float *img_depth,
                                             const float *img_ht,
                                             const float *img_lt,
                                             const float *img_depth_to_range) {
  if (blockIdx.x >= num_visible_blocks) {
    return;
  }
  const Vector3<short> pos_grid_rel(threadIdx.x, threadIdx.y, threadIdx.z);
  // transform to camera / image coordinates
  const Vector3<short> pos_grid_abs = (blocks[blockIdx.x].position << BLOCK_LEN_BITS)
                                      + pos_grid_rel;
  const Vector3<float> pos_world = pos_grid_abs.cast<float>() * voxel_size;
  const Vector3<float> pos_cam = cam_P_world.Apply(pos_world);
  const Vector3<float> pos_img_h = cam_params.intrinsics * pos_cam;
  const Vector3<float> pos_img = pos_img_h / pos_img_h.z;
  const int u = roundf(pos_img.x);
  const int v = roundf(pos_img.y);
  // update if visible
  if (u >= 0 && u < cam_params.img_w && v >= 0 && v < cam_params.img_h) {
    const int img_idx = v * cam_params.img_w + u;
    const float depth = img_depth[img_idx];
    if (depth == 0 || depth > max_depth) { return; }
    const float sdf = img_depth_to_range[img_idx] * (depth - pos_img_h.z);
    if (sdf > -truncation) {
      const float tsdf = fminf(1, sdf / truncation);
      const unsigned int idx = offset2index(pos_grid_rel);
      VoxelTSDF &voxel_tsdf = voxel_mem.GetVoxel<VoxelTSDF>(idx, blocks[blockIdx.x]);
      VoxelRGBW &voxel_rgbw = voxel_mem.GetVoxel<VoxelRGBW>(idx, blocks[blockIdx.x]);
      VoxelSEGM &voxel_segm = voxel_mem.GetVoxel<VoxelSEGM>(idx, blocks[blockIdx.x]);
      // weight running average
      const float weight_new = (1 - depth / max_depth) * 4; // TODO(alvin): better weighting here
      const float weight_old = voxel_rgbw.weight;
      const float weight_combined = weight_old + weight_new;
      // rgb running average
      const uchar3 rgb = img_rgb[img_idx];
      const Vector3<float> rgb_old = voxel_rgbw.rgb.cast<float>();
      const Vector3<float> rgb_new(rgb.x, rgb.y, rgb.z);
      const Vector3<float> rgb_combined =
        (rgb_old * weight_old + rgb_new * weight_new) / weight_combined;
      voxel_tsdf.tsdf = (voxel_tsdf.tsdf * weight_old + tsdf * weight_new) / weight_combined;
      voxel_rgbw.weight = fminf(roundf(weight_combined), 40);
      voxel_rgbw.rgb = rgb_combined.round<unsigned char>();
      // high touch / low touch
      const float positive = expf((weight_old * logf(voxel_segm.probability) +
                                   weight_new * logf(img_ht[img_idx])) / weight_combined);
      const float negative = expf((weight_old * logf(1 - voxel_segm.probability) +
                                   weight_new * logf(img_lt[img_idx])) / weight_combined);
      voxel_segm.probability = positive / (positive + negative);
    }
  }
}

__global__ static void space_carving_kernel(VoxelHashTable hash_table,
                                            const VoxelBlock *blocks,
                                            const int num_visible_blocks,
                                            const float min_tsdf_threshold) {
  if (blockIdx.x >= num_visible_blocks) {
    return;
  }

  __shared__ float tsdf_abs[BLOCK_VOLUME];
  // load shared buffer
  const int tx = threadIdx.x;
  const int tx2 = tx + BLOCK_VOLUME/2;
  tsdf_abs[tx] = fabsf(hash_table.mem.GetVoxel<VoxelTSDF>(tx, blocks[blockIdx.x]).tsdf);
  tsdf_abs[tx2] = fabsf(hash_table.mem.GetVoxel<VoxelTSDF>(tx2, blocks[blockIdx.x]).tsdf);
  // reduce min
  #pragma unroll
  for (int stride = BLOCK_VOLUME/2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (tx < stride)
      tsdf_abs[tx] = fminf(tsdf_abs[tx], tsdf_abs[tx + stride]);
  }
  // de-allocate block
  if (tx == 0 && tsdf_abs[0] >= min_tsdf_threshold) {
    hash_table.Delete(blocks[blockIdx.x].position);
  }
}

__global__ static void ray_cast_kernel(const VoxelHashTable hash_table,
                                       const CameraParams cam_params,
                                       const SE3<float> cam_P_world,
                                       const SE3<float> world_P_cam,
                                       const float step_size,
                                       const float max_depth,
                                       const float voxel_size,
                                       uchar4 *img_tsdf_rgba,
                                       uchar4 *img_tsdf_normal) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cam_params.img_w || y >= cam_params.img_h) {
    return;
  }
  const int idx = y * cam_params.img_w + x;
  const Vector3<float> pos_img_h(x, y, 1);
  const Vector3<float> pos_cam = cam_params.intrinsics_inv * pos_img_h;
  const Vector3<float> ray_dir_cam = pos_cam / sqrtf(pos_cam.dot(pos_cam));
  const SO3<float> world_R_cam = world_P_cam.GetR();
  const Vector3<float> ray_dir_world = world_R_cam * ray_dir_cam;
  const Vector3<float> ray_step_grid = ray_dir_world * step_size / voxel_size;
  const int max_step = ceil(max_depth / step_size);
  Vector3<float> pos_grid = world_P_cam.GetT() / voxel_size;
  VoxelBlock cache;
  float tsdf_prev = hash_table.Retrieve<VoxelTSDF>(
      pos_grid.round<short>(), cache).tsdf;
  pos_grid += ray_step_grid;
  for (int i = 1; i < max_step; ++i, pos_grid += ray_step_grid) {
    const float tsdf_curr = hash_table.Retrieve<VoxelTSDF>(pos_grid.round<short>(), cache).tsdf;
    // ray hit front surface
    if (tsdf_prev > 0 && tsdf_curr <= 0 && tsdf_prev - tsdf_curr <= 1.5) {
      Vector3<float> pos1_grid = pos_grid - ray_step_grid;
      Vector3<float> pos2_grid = pos_grid;
      Vector3<float> pos_mid_grid = (pos1_grid + pos2_grid) / 2.;
      // binary search refinement
      while((pos1_grid - pos2_grid).dot(pos1_grid - pos2_grid) > .1) {
        const float tsdf_mid = hash_table.Retrieve<VoxelTSDF>(
          pos_mid_grid.round<short>(), cache).tsdf;
        if (tsdf_mid < 0) {
          pos2_grid = pos_mid_grid;
        }
        else {
          pos1_grid = pos_mid_grid;
        }
        pos_mid_grid = (pos1_grid + pos2_grid) / 2.;
      }
      const Vector3<short> final_grid = pos_mid_grid.round<short>();
      const VoxelRGBW voxel_rgbw = hash_table.Retrieve<VoxelRGBW>(final_grid, cache);
      const VoxelSEGM voxel_segm = hash_table.Retrieve<VoxelSEGM>(final_grid, cache);
      // calculate gradient
      const Vector3<float> norm_raw_grid(
        hash_table.Retrieve<VoxelTSDF>(
          { final_grid.x + 1, final_grid.y, final_grid.z }, cache).tsdf -
        hash_table.Retrieve<VoxelTSDF>(
          { final_grid.x - 1, final_grid.y, final_grid.z }, cache).tsdf,
        hash_table.Retrieve<VoxelTSDF>(
          { final_grid.x, final_grid.y + 1, final_grid.z }, cache).tsdf -
        hash_table.Retrieve<VoxelTSDF>(
          { final_grid.x, final_grid.y - 1, final_grid.z }, cache).tsdf,
        hash_table.Retrieve<VoxelTSDF>(
          { final_grid.x, final_grid.y, final_grid.z + 1 }, cache).tsdf -
        hash_table.Retrieve<VoxelTSDF>(
          { final_grid.x, final_grid.y, final_grid.z - 1 }, cache).tsdf
      );
      const float diffusivity = fmaxf(norm_raw_grid.dot(-ray_dir_world) /
                                sqrtf(norm_raw_grid.dot(norm_raw_grid)), 0);
      const float alpha = fmaxf(voxel_segm.probability - 0.5, 0) / .5;
      img_tsdf_rgba[idx] = make_uchar4(alpha * 255 + (1 - alpha) * voxel_rgbw.rgb.x,
                                       (1 - alpha) * voxel_rgbw.rgb.y,
                                       (1 - alpha) * voxel_rgbw.rgb.z, 255);
      img_tsdf_normal[idx] = make_uchar4(alpha * 255 + (1 - alpha) * diffusivity * 255,
                                         (1 - alpha) * diffusivity * 255,
                                         (1 - alpha) * diffusivity * 255, 255);
      return;
    }
    tsdf_prev = tsdf_curr;
  }
  // no surface intersection found
  img_tsdf_rgba[idx] = make_uchar4(0, 0, 0, 0);
  img_tsdf_normal[idx] = make_uchar4(0, 0, 0, 0);
}

TSDFGrid::TSDFGrid(float voxel_size, float truncation)
  : voxel_size_(voxel_size), truncation_(truncation) {
  // memory allocation
  CUDA_SAFE_CALL(cudaMalloc(&visible_mask_, sizeof(int) * NUM_ENTRY));
  CUDA_SAFE_CALL(cudaMalloc(&visible_indics_, sizeof(int) * NUM_ENTRY));
  CUDA_SAFE_CALL(cudaMalloc(&visible_indics_aux_, sizeof(int) * NUM_ENTRY / (2 * SCAN_BLOCK_SIZE)));
  CUDA_SAFE_CALL(cudaMalloc(&visible_blocks_, sizeof(VoxelBlock) * NUM_ENTRY));
  CUDA_SAFE_CALL(cudaMalloc(&img_rgb_, sizeof(uint3) * MAX_IMG_SIZE));
  CUDA_SAFE_CALL(cudaMalloc(&img_depth_, sizeof(float) * MAX_IMG_SIZE));
  CUDA_SAFE_CALL(cudaMalloc(&img_ht_, sizeof(float) * MAX_IMG_SIZE));
  CUDA_SAFE_CALL(cudaMalloc(&img_lt_, sizeof(float) * MAX_IMG_SIZE));
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
  CUDA_SAFE_CALL(cudaFree(img_ht_));
  CUDA_SAFE_CALL(cudaFree(img_lt_));
  CUDA_SAFE_CALL(cudaFree(img_depth_to_range_));
  CUDA_SAFE_CALL(cudaFree(img_tsdf_rgba_));
  CUDA_SAFE_CALL(cudaFree(img_tsdf_normal_));
  // release cuda stream
  CUDA_SAFE_CALL(cudaStreamDestroy(stream_));
  CUDA_SAFE_CALL(cudaStreamDestroy(stream2_));
}

void TSDFGrid::Integrate(const cv::Mat &img_rgb, const cv::Mat &img_depth,
                         const cv::Mat &img_ht, const cv::Mat &img_lt,
                         float max_depth,
                         const CameraIntrinsics<float> &intrinsics,
                         const SE3<float> &cam_P_world) {
  assert(img_rgb.type() == CV_8UC3);
  assert(img_depth.type() == CV_32FC1);
  assert(img_rgb.cols == img_depth.cols);
  assert(img_rgb.rows == img_depth.rows);

  const CameraParams cam_params(intrinsics, img_rgb.rows, img_rgb.cols);

  // data transfer
  CUDA_SAFE_CALL(cudaMemcpyAsync(img_rgb_, img_rgb.data,
    sizeof(uchar3)*img_rgb.total(), cudaMemcpyHostToDevice, stream_));
  CUDA_SAFE_CALL(cudaMemcpyAsync(img_depth_, img_depth.data,
    sizeof(float)*img_depth.total(), cudaMemcpyHostToDevice, stream_));
  CUDA_SAFE_CALL(cudaMemcpyAsync(img_ht_, img_ht.data,
    sizeof(float)*img_depth.total(), cudaMemcpyHostToDevice, stream2_));
  CUDA_SAFE_CALL(cudaMemcpyAsync(img_lt_, img_lt.data,
    sizeof(float)*img_depth.total(), cudaMemcpyHostToDevice, stream2_));
  // compute
  spdlog::debug("[TSDF] pre integrate: {} active blocks", hash_table_.NumActiveBlock());
  Allocate(img_rgb, img_depth, max_depth, cam_params, cam_P_world);
  const int num_visible_blocks = GatherVisible(max_depth, cam_params, cam_P_world);
  CUDA_SAFE_CALL(cudaStreamSynchronize(stream2_)); // synchronize ht / lt img copy
  UpdateTSDF(num_visible_blocks, max_depth, cam_params, cam_P_world);
  SpaceCarving(num_visible_blocks);
  CUDA_SAFE_CALL(cudaStreamSynchronize(stream_));
  spdlog::debug("[TSDF] post integrate: {} active blocks", hash_table_.NumActiveBlock());
}

void TSDFGrid::Allocate(const cv::Mat &img_rgb, const cv::Mat &img_depth, float max_depth,
                        const CameraParams &cam_params, const SE3<float> &cam_P_world) {
  const dim3 IMG_BLOCK_DIM(ceil((float)cam_params.img_w/32), ceil((float)cam_params.img_h/16));
  const dim3 IMG_THREAD_DIM(32, 16);
  block_allocate_kernel<<<IMG_BLOCK_DIM, IMG_THREAD_DIM, 0, stream_>>>(
    hash_table_, img_depth_, cam_params, cam_P_world, cam_P_world.Inverse(),
    voxel_size_, max_depth, truncation_, img_depth_to_range_);
  CUDA_STREAM_CHECK_ERROR(stream_);
  hash_table_.ResetLocks(stream_);
}

int TSDFGrid::GatherVisible(float max_depth,
                            const CameraParams &cam_params, const SE3<float> &cam_P_world) {
  constexpr int GATHER_BLOCK_DIM = NUM_ENTRY / BLOCK_VOLUME;
  // generate binary array of visibility
  check_visibility_kernel<<<GATHER_BLOCK_DIM, BLOCK_VOLUME, 0, stream_>>>(
    hash_table_, voxel_size_, max_depth, cam_params,
    cam_P_world, visible_mask_);
  CUDA_STREAM_CHECK_ERROR(stream_);

  return GatherBlock();
}

std::vector<VoxelSpatialTSDF> TSDFGrid::GatherValid() {
  spdlog::debug("[TSDF] {} active blocks before download", hash_table_.NumActiveBlock());

  constexpr int GATHER_BLOCK_DIM = NUM_ENTRY / BLOCK_VOLUME;

  check_valid_kernel<<<GATHER_BLOCK_DIM, BLOCK_VOLUME, 0, stream_>>>(
    hash_table_, visible_mask_);
  CUDA_STREAM_CHECK_ERROR(stream_);

  const int num_visible_blocks = GatherBlock();
  std::vector<VoxelSpatialTSDF> ret(num_visible_blocks * BLOCK_VOLUME);

  VoxelSpatialTSDF *voxel_pos_tsdf;
  CUDA_SAFE_CALL(cudaMalloc(&voxel_pos_tsdf,
    sizeof(VoxelSpatialTSDF) * num_visible_blocks * BLOCK_VOLUME));

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

std::vector<VoxelSpatialTSDF> TSDFGrid::GatherVoxels(const BoundingCube<float> &volumn) {
  // convert bounds to grid coordinates
  const BoundingCube<short> volumn_grid = volumn.Scale<short>(1. / voxel_size_);

  constexpr int GATHER_BLOCK_DIM = NUM_ENTRY / BLOCK_VOLUME;
  check_bound_kernel<<<GATHER_BLOCK_DIM, BLOCK_VOLUME, 0, stream_>>>(
    hash_table_, volumn_grid, visible_mask_);
  CUDA_STREAM_CHECK_ERROR(stream_);

  const int num_visible_blocks = GatherBlock();
  std::vector<VoxelSpatialTSDF> ret(num_visible_blocks * BLOCK_VOLUME);

  VoxelSpatialTSDF *voxel_pos_tsdf;
  CUDA_SAFE_CALL(cudaMalloc(&voxel_pos_tsdf,
    sizeof(VoxelSpatialTSDF) * num_visible_blocks * BLOCK_VOLUME));

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

int TSDFGrid::GatherBlock() {
  constexpr int GATHER_THREAD_DIM = 512;
  constexpr int GATHER_BLOCK_DIM = NUM_ENTRY / GATHER_THREAD_DIM;
  // parallel prefix sum scan
  prefix_sum<int>(visible_mask_, visible_indics_, visible_indics_aux_, NUM_ENTRY, stream_);
  // gather visible blocks into contiguous array
  gather_visible_blocks_kernel<<<GATHER_BLOCK_DIM, GATHER_THREAD_DIM, 0, stream_>>>(
    hash_table_, visible_mask_, visible_indics_, visible_blocks_);
  CUDA_STREAM_CHECK_ERROR(stream_);
  // get number of visible blocks from scanned index array
  int num_visible_blocks;
  CUDA_SAFE_CALL(cudaMemcpyAsync(&num_visible_blocks, visible_indics_ + NUM_ENTRY - 1,
    sizeof(int), cudaMemcpyDeviceToHost, stream_));
  CUDA_SAFE_CALL(cudaStreamSynchronize(stream_));
  spdlog::debug("[TSDF] Getting {} blocks", num_visible_blocks);
  return num_visible_blocks;
}

void TSDFGrid::UpdateTSDF(int num_visible_blocks, float max_depth,
                          const CameraParams &cam_params, const SE3<float> &cam_P_world) {
  const dim3 VOXEL_BLOCK_DIM(BLOCK_LEN, BLOCK_LEN, BLOCK_LEN);
  tsdf_integrate_kernel<<<num_visible_blocks, VOXEL_BLOCK_DIM, 0, stream_>>>(
    visible_blocks_, hash_table_.mem, cam_P_world, cam_params, num_visible_blocks,
    max_depth, truncation_, voxel_size_,
    img_rgb_, img_depth_, img_ht_, img_lt_, img_depth_to_range_);
  CUDA_STREAM_CHECK_ERROR(stream_);
}

void TSDFGrid::SpaceCarving(int num_visible_blocks) {
  space_carving_kernel<<<num_visible_blocks, BLOCK_VOLUME/2, 0, stream_>>>(
    hash_table_, visible_blocks_, num_visible_blocks, .9);
  CUDA_STREAM_CHECK_ERROR(stream_);
  hash_table_.ResetLocks(stream_);
}

void TSDFGrid::RayCast(float max_depth,
                       const CameraParams &virtual_cam,
                       const SE3<float> &cam_P_world,
                       GLImage8UC4 *tsdf_rgba, GLImage8UC4 *tsdf_normal) {
  const dim3 IMG_BLOCK_DIM(ceil((float)virtual_cam.img_w/32),
                           ceil((float)virtual_cam.img_h/16));
  const dim3 IMG_THREAD_DIM(32, 16);
  ray_cast_kernel<<<IMG_BLOCK_DIM, IMG_THREAD_DIM, 0, stream_>>>(
    hash_table_, virtual_cam, cam_P_world, cam_P_world.Inverse(),
    truncation_/2, max_depth, voxel_size_, img_tsdf_rgba_, img_tsdf_normal_);
  CUDA_STREAM_CHECK_ERROR(stream_);
  if (tsdf_rgba) { tsdf_rgba->LoadCuda(img_tsdf_rgba_); }
  if (tsdf_normal) { tsdf_normal->LoadCuda(img_tsdf_normal_); }
}
