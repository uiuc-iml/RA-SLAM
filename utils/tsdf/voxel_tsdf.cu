#include "utils/cuda/arithmetic.cuh"
#include "utils/cuda/errors.cuh"
#include "utils/tsdf/voxel_tsdf.cuh"

#define MAX_IMG_H     1920
#define MAX_IMG_W     1080
#define MAX_IMG_SIZE  (MAX_IMG_H * MAX_IMG_W)

__global__ static void check_visibility_kernel(const VoxelHashTable hash_table, 
                                               const float voxel_size,
                                               const float max_depth,
                                               const CameraParams cam_params,
                                               const SE3<float> cam_P_world,
                                               int *visible_mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const VoxelBlock &block = hash_table.GetBlock(idx);
  if (!block.voxels) {
    visible_mask[idx] = 0;
    return;
  }
  const Vector3<float> pos_world = block.block_pos.cast<float>() * (voxel_size * BLOCK_LEN);
  const Vector3<float> pos_cam = cam_P_world.Apply(pos_world);
  const Vector3<float> pos_img_h = cam_params.intrinsics * pos_cam;
  const Vector3<float> pos_img = pos_img_h / pos_img_h.z;
  visible_mask[idx] = (pos_img.x >= 0 && pos_img.x < cam_params.img_w &&
                       pos_img.y >= 0 && pos_img.y < cam_params.img_h &&
                       pos_img_h.z >= 0 && pos_img_h.z < max_depth);
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

__global__ static void tsdf_integrate_kernel(VoxelBlock *blocks, 
                                             const SE3<float> cam_P_world,
                                             const CameraParams cam_params,
                                             const int num_visible_blocks,
                                             const float truncation,
                                             const float voxel_size,
                                             const uchar3 *img_rgb,
                                             const float *img_depth,
                                             const float *img_depth_to_range) {
  if (blockIdx.x >= num_visible_blocks) {
    return;
  }
  const Vector3<short> pos_grid_rel(threadIdx.x, threadIdx.y, threadIdx.z);
  // transform to camera / image coordinates
  const Vector3<short> pos_grid_abs = (blocks[blockIdx.x].block_pos << BLOCK_LEN_BITS) 
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
    const float sdf = img_depth_to_range[img_idx] * (img_depth[img_idx] - pos_img_h.z);
    if (sdf > -truncation) {
      const float tsdf = fminf(1, sdf / truncation);
      const unsigned int idx = offset2index(pos_grid_rel);
      Voxel &voxel = blocks[blockIdx.x].GetVoxelMutable(idx);
      // weight running average
      const float weight_new = 1; // TODO(alvin): add better weighting here
      const float weight_old = voxel.weight;
      const float weight_combined = weight_old + weight_new;
      // rgb running average
      const uchar3 rgb = img_rgb[img_idx];
      const Vector3<float> rgb_old = voxel.rgb.cast<float>();
      const Vector3<float> rgb_new(rgb.x, rgb.y, rgb.z);
      const Vector3<float> rgb_combined = 
        (rgb_old * weight_old + rgb_new * weight_new) / weight_combined;
      voxel.tsdf = (voxel.tsdf * weight_old + tsdf * weight_new) / weight_combined;
      voxel.weight = fminf(roundf(weight_combined), 200); // TODO(alvin): don't hardcode
      voxel.rgb = (rgb_combined + .5).cast<unsigned char>(); // rounding
    }
  }
}

__global__ static void block_allocate_kernel(VoxelHashTable hash_table,
                                             const float *img_depth, 
                                             const CameraParams cam_params,
                                             const SE3<float> world_P_cam,
                                             const float max_depth,
                                             const float truncation,
                                             float *img_depth_to_range) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cam_params.img_w || y >= cam_params.img_h) {
    return;
  }
  const int idx = y * cam_params.img_w + x;
  const float depth = img_depth[idx];
  if (depth > max_depth) {
    return;
  }
  // transform coordinate from image to world
  const Vector3<float> pos_img_h(x * depth, y * depth, depth);
  const Vector3<float> pos_cam = cam_params.intrinsics_inv * pos_img_h;
  const Vector3<float> pos_world = world_P_cam.Apply(pos_cam);
  // cache depth to range buffer
  const float range = sqrtf(pos_cam.dot(pos_cam));
  img_depth_to_range[idx] = range / depth;
  // calculate end coordinates of sample ray
  const Vector3<float> ray_dir_cam = pos_cam / range;
  const SO3<float> world_R_cam = world_P_cam.GetR();
  const Vector3<float> ray_dir_world = world_R_cam * ray_dir_cam;
  const Vector3<float> ray_start_world = pos_world - ray_dir_world * truncation;
  // put ray into voxel block coordinate
  const Vector3<float> ray_dir_block = ray_dir_world / BLOCK_LEN;
  const Vector3<float> ray_block = 2.f * truncation * ray_dir_world; // start -> end vector
  // DDA for finding ray / block intersection
  const int step = fmaxf(fmaxf(ray_block.x, ray_block.y), ray_block.z);
  const Vector3<float> ray_step_block = ray_block / (float)step;
  Vector3<float> pos_block = ray_start_world / BLOCK_LEN;
  // allocate blocks along the ray
  for (int i = 0; i <= step; ++i, pos_block += ray_step_block) {
    hash_table.Allocate((pos_block + .5).cast<short>());
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
  tsdf_abs[tx] = fabs(blocks[blockIdx.x].GetVoxel(tx).tsdf);
  tsdf_abs[tx + BLOCK_VOLUME/2] = fabs(blocks[blockIdx.x].GetVoxel(tx + BLOCK_VOLUME/2).tsdf);
  // reduce min
  for (int stride = BLOCK_VOLUME/2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (tx < stride) 
      tsdf_abs[tx] = fminf(tsdf_abs[tx], tsdf_abs[tx + stride]);
  }
  // de-allocate block
  if (tx == 0 && tsdf_abs[0] >= min_tsdf_threshold) {
    hash_table.Delete(blocks[blockIdx.x].block_pos);
  }
}

__global__ static void ray_cast_kernel(const VoxelHashTable hash_table,
                                       const CameraParams cam_params,
                                       const SE3<float> world_P_cam,
                                       const float truncation, 
                                       const float max_depth,
                                       const float voxel_size,
                                       float *img_gray) {
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
  const Vector3<float> ray_step_world = ray_dir_world * truncation / 2;
  const int max_step = ceil(max_depth / (truncation / 2));
  Vector3<float> pos_world = world_P_cam.GetT();
  VoxelBlock cache;
  Voxel voxel_prev = hash_table.Retrieve((pos_world / voxel_size + .5).cast<short>(), cache);
  pos_world += ray_step_world;
  for (int i = 1; i < max_step; ++i, pos_world += ray_step_world) {
    const Vector3<short> pos_grid = (pos_world / voxel_size + .5).cast<short>();
    const Voxel voxel_curr = hash_table.Retrieve(pos_grid, cache);
    // ray hit surface
    if (voxel_prev.tsdf > 0 && voxel_curr.tsdf <= 0) {
      Vector3<short> pos1_grid = ((pos_world - ray_step_world) / voxel_size + .5).cast<short>();
      Vector3<short> pos2_grid = pos_grid;
      Vector3<short> pos_mid_grid = (pos1_grid + pos2_grid) / 2;
      // binary search refinement
      while(pos1_grid != pos_mid_grid && pos2_grid != pos_mid_grid) {
        const Voxel voxel = hash_table.Retrieve(pos_mid_grid, cache);
        if (voxel.tsdf < 0) {
          pos2_grid = pos_mid_grid;
        }
        else {
          pos1_grid = pos_mid_grid;
        }
        pos_mid_grid = (pos1_grid + pos2_grid) / 2;
      }
      const Voxel voxel = hash_table.Retrieve(pos_mid_grid, cache);
      // calculate gradient
      const Vector3<float> norm_raw(
        hash_table.Retrieve({ pos2_grid.x + 1, pos2_grid.y, pos2_grid.z }, cache).tsdf - 
        hash_table.Retrieve({ pos2_grid.x - 1, pos2_grid.y, pos2_grid.z }, cache).tsdf,
        hash_table.Retrieve({ pos2_grid.x, pos2_grid.y + 1, pos2_grid.z }, cache).tsdf - 
        hash_table.Retrieve({ pos2_grid.x, pos2_grid.y - 1, pos2_grid.z }, cache).tsdf,
        hash_table.Retrieve({ pos2_grid.x, pos2_grid.y, pos2_grid.z + 1 }, cache).tsdf - 
        hash_table.Retrieve({ pos2_grid.x, pos2_grid.y, pos2_grid.z - 1 }, cache).tsdf
      );
      img_gray[idx] = -fminf(norm_raw.dot(ray_dir_world) / sqrtf(norm_raw.dot(norm_raw)), 0);
      return;
    }
    voxel_prev = voxel_curr;
  }
  img_gray[idx] = 0; // no surface intersection found
}

TSDFGrid::TSDFGrid(float voxel_size, float truncation, float max_depth) 
  : voxel_size_(voxel_size), truncation_(truncation), max_depth_(max_depth) {
  // memory allocation
  CUDA_SAFE_CALL(cudaMalloc(&visible_mask_, sizeof(int) * NUM_ENTRY));
  CUDA_SAFE_CALL(cudaMalloc(&visible_indics_, sizeof(int) * NUM_ENTRY));
  CUDA_SAFE_CALL(cudaMalloc(&visible_indics_aux_, sizeof(int) * SCAN_BLOCK_SIZE));
  CUDA_SAFE_CALL(cudaMalloc(&visible_blocks_, sizeof(VoxelBlock) * NUM_ENTRY));
  CUDA_SAFE_CALL(cudaMalloc(&img_rgb_, sizeof(uint3) * MAX_IMG_SIZE));
  CUDA_SAFE_CALL(cudaMalloc(&img_depth_, sizeof(float) * MAX_IMG_SIZE));
  CUDA_SAFE_CALL(cudaMalloc(&img_depth_to_range_, sizeof(float) * MAX_IMG_SIZE));
  CUDA_SAFE_CALL(cudaMalloc(&img_normal_, sizeof(float) * MAX_IMG_SIZE));
  // stream init
  CUDA_SAFE_CALL(cudaStreamCreate(&stream_));
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
  CUDA_SAFE_CALL(cudaFree(img_depth_to_range_));
  CUDA_SAFE_CALL(cudaFree(img_normal_));
  // release cuda stream
  CUDA_SAFE_CALL(cudaStreamDestroy(stream_));
}

void TSDFGrid::Integrate(const cv::Mat &img_rgb, const cv::Mat &img_depth, 
                         const CameraIntrinsics<float> &intrinsics, 
                         const SE3<float> &cam_P_world) {
  assert(img_rgb.type() == CV_8UC3);
  assert(img_depth.type() == CV_32FC1);
  assert(img_rgb.cols == img_depth.cols);
  assert(img_rgb.rows == img_depth.rows);

  const CameraParams cam_params(intrinsics, img_rgb.rows, img_rgb.cols);

  Allocate(img_rgb, img_depth, cam_params, cam_P_world);
  const int num_visible_blocks = GatherVisible(cam_params, cam_P_world);
  UpdateTSDF(num_visible_blocks, cam_params, cam_P_world);
  SpaceCarving(num_visible_blocks);
}

void TSDFGrid::Allocate(const cv::Mat &img_rgb, const cv::Mat &img_depth,
                        const CameraParams &cam_params, const SE3<float> &cam_P_world) {
  CUDA_SAFE_CALL(cudaMemcpyAsync(img_rgb_, img_rgb.data, 
    sizeof(char)*img_rgb.total(), cudaMemcpyHostToDevice, stream_));
  CUDA_SAFE_CALL(cudaMemcpyAsync(img_depth_, img_depth.data, 
    sizeof(float)*img_depth.total(), cudaMemcpyHostToDevice, stream_));
  const dim3 IMG_BLOCK_DIM(ceil((float)cam_params.img_w/32), ceil((float)cam_params.img_h/16));
  const dim3 IMG_THREAD_DIM(32, 16);
  block_allocate_kernel<<<IMG_BLOCK_DIM, IMG_THREAD_DIM, 0, stream_>>>(
    hash_table_, img_depth_, cam_params, cam_P_world.Inverse(), 
    max_depth_, truncation_, img_depth_to_range_);
  hash_table_.ResetLocks(stream_);
}

int TSDFGrid::GatherVisible(const CameraParams &cam_params, const SE3<float> &cam_P_world) {
  constexpr int GATHER_THREAD_DIM = 512;
  const int GATHER_BLOCK_DIM = ceil((float)NUM_ENTRY / GATHER_THREAD_DIM);
  // generate binary array of visibility
  check_visibility_kernel<<<GATHER_BLOCK_DIM, GATHER_THREAD_DIM, 0, stream_>>>(
    hash_table_, voxel_size_, max_depth_, cam_params,
    cam_P_world, visible_mask_);
  // parallel prefix sum scan
  prefix_sum<int>(visible_mask_, visible_indics_, visible_indics_aux_, NUM_ENTRY, stream_);
  // gather visible blocks into contiguous array
  gather_visible_blocks_kernel<<<GATHER_BLOCK_DIM, GATHER_THREAD_DIM, 0, stream_>>>(
    hash_table_, visible_mask_, visible_indics_, visible_blocks_);
  int num_visible_blocks;
  CUDA_SAFE_CALL(cudaMemcpyAsync(&num_visible_blocks, visible_indics_ + NUM_ENTRY - 1,
    sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaStreamSynchronize(stream_));
  return num_visible_blocks;
}

void TSDFGrid::UpdateTSDF(int num_visible_blocks, 
                          const CameraParams &cam_params, const SE3<float> &cam_P_world) {
  const dim3 VOXEL_BLOCK_DIM(BLOCK_LEN, BLOCK_LEN, BLOCK_LEN);
  tsdf_integrate_kernel<<<num_visible_blocks, VOXEL_BLOCK_DIM, 0, stream_>>>(
    visible_blocks_, cam_P_world, cam_params, num_visible_blocks, truncation_, voxel_size_,
    img_rgb_, img_depth_, img_depth_to_range_);
}

void TSDFGrid::SpaceCarving(int num_visible_blocks) {
  space_carving_kernel<<<num_visible_blocks, BLOCK_VOLUME/2, 0, stream_>>>(
    hash_table_, visible_blocks_, num_visible_blocks, .8); 
  hash_table_.ResetLocks(stream_);
}

void TSDFGrid::RayCast(cv::Mat *img, 
                       const CameraIntrinsics<float> &virtual_intrinsics,
                       const SE3<float> &cam_P_world) {
  assert(img->type() == CV_32FC1);

  const CameraParams cam_params(virtual_intrinsics, img->rows, img->cols);
  const dim3 IMG_BLOCK_DIM(ceil((float)cam_params.img_w/32), ceil((float)cam_params.img_h/16));
  const dim3 IMG_THREAD_DIM(32, 16);
  ray_cast_kernel<<<IMG_BLOCK_DIM, IMG_THREAD_DIM, 0, stream_>>>(
    hash_table_, cam_params, cam_P_world.Inverse(),
    truncation_, max_depth_, voxel_size_, img_normal_);
  CUDA_SAFE_CALL(cudaMemcpyAsync(img->data, img_normal_, 
    sizeof(float) * img->total(), cudaMemcpyDeviceToHost, stream_));
  CUDA_SAFE_CALL(cudaStreamSynchronize(stream_));
}
