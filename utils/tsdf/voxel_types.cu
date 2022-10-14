#include "utils/tsdf/voxel_types.cuh"

// __device__ __host__ MultiClsSemantics::MultiClsSemantics() { uniform_init(); }

// __device__ __host__ void MultiClsSemantics::uniform_init() {
//   max_cls = 0;
// }

// __device__ void MultiClsSemantics::update(const float* prob_map, const int img_idx,
//                                           const int class_offset) {
//   int cur_max_cls = 0;
//   float cur_max_prob = prob_map[img_idx];
//   #pragma unroll
//   for (int i = 0; i < NUM_CLASSES; ++i) {
//     float cur_prob = prob_map[img_idx + class_offset * i];
//     if (cur_prob > cur_max_prob) {
//       cur_max_cls = i;
//       cur_max_prob = cur_prob;
//     }
//   }
//   max_cls = cur_max_cls;
// }

// __device__ int MultiClsSemantics::get_max_class() const {
//   return max_cls;
// }

// __host__ __device__ void encoder(const float *prob_arr, float *latent) {
//   float linear1_weights[2][21] = {{-0.4452,  1.2788, -1.6547,  3.0422, -1.6771, -3.4774, -0.9816, -0.4947,
//           1.3808,  0.5453,  0.0251,  1.0826, -0.0458,  0.2517,  1.2300,  0.1493,
//           0.1871,  0.0395, -0.1356,  0.1410, -0.3524},
//         {-0.4132, -2.0518, -2.7553,  0.2146,  1.5178, -0.3728, -0.2114, -0.2159,
//           0.9625,  0.0715,  0.2339, -0.2266,  0.0805,  0.6653,  1.8413,  0.5473,
//           0.2610,  0.2050,  0.0998,  0.1401,  0.1974}};
//   #pragma unroll
//   for (int i = 0; i < 2; ++i) {
//     float ele = 0;
//     #pragma unroll
//     for (int j = 0; j < 21; ++j) {
//       ele += (prob_arr[j] * linear1_weights[i][j]);
//     }
//     latent[i] = ele;
//   }
// }

// __host__ __device__ void decoder(float *prob_arr, const float *latent) {
//   float inv_linear1_weights[21][2] = {{-0.3330, -0.4030},
//         { 1.3486, -1.7884},
//         {-1.7685, -2.3880},
//         { 1.8763,  0.3704},
//         {-1.6508,  1.8934},
//         {-2.4774, -0.4769},
//         {-1.1916, -0.0041},
//         {-0.5321, -0.4036},
//         { 0.9883, -0.1722},
//         { 0.8204, -0.1933},
//         {-0.0203,  0.0077},
//         { 0.8967, -0.6998},
//         { 0.0030, -0.5308},
//         { 0.0845,  0.4164},
//         { 1.4126,  2.3751},
//         { 0.3316,  0.2053},
//         { 0.8504,  0.9706},
//         { 0.0525,  0.1554},
//         { 0.1385,  0.2368},
//         { 0.3048,  0.3658},
//         {-0.3132,  0.1348}};
//   float softmax_sum = 0;
//   #pragma unroll
//   for (int i = 0; i < 21; ++i) {
//     float ele = 0;
//     #pragma unroll
//     for (int j = 0; j < 2; ++j) {
//       ele += (latent[j] * inv_linear1_weights[i][j]);
//     }
//     float new_ele = expf(ele);
//     softmax_sum += new_ele;
//     prob_arr[i] = new_ele;
//   }
//   // softmax
//   #pragma unroll
//   for (int i = 0; i < 21; ++i) {
//     prob_arr[i] = prob_arr[i] / softmax_sum;
//   }
// }

// __device__ __host__ MultiClsSemantics::MultiClsSemantics() { uniform_init(); }

// __device__ __host__ void MultiClsSemantics::uniform_init() {
//   float prob_arr[NUM_CLASSES];
//   #pragma unroll
//   for (int i =0; i < NUM_CLASSES; ++i) {
//     prob_arr[i] = 1. / NUM_CLASSES;
//   }
//   encoder(prob_arr, latent);
// }

// __device__ void MultiClsSemantics::update(const float* prob_map, const int img_idx,
//                                           const int class_offset) {
//   float norm_coeff = 0.0f;
//   float buffer[NUM_CLASSES];
//   decoder(buffer, latent);
// #pragma unroll
//   for (int i = 0; i < NUM_CLASSES; ++i) {
//     float new_prob = buffer[i] * prob_map[img_idx + class_offset * i];
//     buffer[i] = new_prob;
//     norm_coeff += new_prob;
//   }
// #pragma unroll
//   for (int i = 0; i < NUM_CLASSES; ++i) {
//     buffer[i] = buffer[i] / norm_coeff;
//   }
//   encoder(buffer, latent);
// }

// __device__ int MultiClsSemantics::get_max_class() const {
//   float buffer[NUM_CLASSES];
//   decoder(buffer, latent);
//   int max_cls = 0;
//   float max_prob = buffer[0];
// #pragma unroll
//   for (int i = 0; i < NUM_CLASSES; ++i) {
//     if (buffer[i] > max_prob) {
//       max_cls = i;
//       max_prob = buffer[i];
//     }
//   }
//   return max_cls;
// }

/* QUANTIZATION IMPL */
// #define QUAN_BITS 4

// __device__ __host__ MultiClsSemantics::MultiClsSemantics() { uniform_init(); }

// __device__ __host__ void MultiClsSemantics::uniform_init() {
//   #pragma unroll
//   for (int i =0; i < NUM_CLASSES; ++i) {
//     quantization[i] = 1;
//   }
// }

// __device__ void MultiClsSemantics::update(const float* prob_map, const int img_idx,
//                                           const int class_offset) {
//   float norm_coeff = 0.0f;
//   float buffer[NUM_CLASSES];
// #pragma unroll
//   for (int i = 0; i < NUM_CLASSES; ++i) {
//     float new_prob = (quantization[i] * 1.0f / (1 << QUAN_BITS)) * prob_map[img_idx + class_offset * i];
//     buffer[i] = new_prob;
//     norm_coeff += new_prob;
//   }
// #pragma unroll
//   for (int i = 0; i < NUM_CLASSES; ++i) {
//     quantization[i] = (buffer[i] / norm_coeff) * (1 << QUAN_BITS) + 1;
//   }
// }

// __device__ int MultiClsSemantics::get_max_class() const {
//   int max_cls = 0;
//   uint16_t max_prob = quantization[0];
// #pragma unroll
//   for (int i = 0; i < NUM_CLASSES; ++i) {
//     if (quantization[i] > max_prob) {
//       max_cls = i;
//       max_prob = quantization[i];
//     }
//   }
//   return max_cls;
// }

/* MAXCLASS IMPL */
__device__ __host__ MultiClsSemantics::MultiClsSemantics() { uniform_init(); }

__device__ __host__ void MultiClsSemantics::uniform_init() {
  max_cls = 0;
  observation_cnt = 1;
}

__device__ void MultiClsSemantics::update(const float* prob_map, const int img_idx,
                                          const int class_offset) {
  int cur_max_cls = 0;
  float cur_max_prob = prob_map[img_idx];
  #pragma unroll
  for (int i = 0; i < NUM_CLASSES; ++i) {
    float cur_prob = prob_map[img_idx + class_offset * i];
    if (cur_prob > cur_max_prob) {
      cur_max_cls = i;
      cur_max_prob = cur_prob;
    }
  }
  if (cur_max_cls == max_cls) {
    observation_cnt++;
  } else {
    observation_cnt--;
    if (observation_cnt <= 0) {
      max_cls = cur_max_cls;
      observation_cnt = 1;
    }
  }
}

__device__ int MultiClsSemantics::get_max_class() const {
  return max_cls;
}

/* FP16 IMPL */
// __device__ __host__ MultiClsSemantics::MultiClsSemantics() { uniform_init(); }

// __device__ __host__ void MultiClsSemantics::uniform_init() {
// #pragma unroll
//   for (int i = 0; i < NUM_CLASSES; ++i) {
//     prob_vec[i] = __float2half(1. / NUM_CLASSES);
//   }
// }

// __device__ void MultiClsSemantics::update(const float* prob_map, const int img_idx,
//                                           const int class_offset) {
//   __half norm_coeff = __float2half(0.0f);
// #pragma unroll
//   for (int i = 0; i < NUM_CLASSES; ++i) {
//     __half new_prob = __hmul(prob_vec[i], __float2half(prob_map[img_idx + class_offset * i]));
//     prob_vec[i] = new_prob;
//     norm_coeff = __hadd(norm_coeff, new_prob);
//   }
// #pragma unroll
//   for (int i = 0; i < NUM_CLASSES; ++i) {
//     prob_vec[i] = __hdiv(prob_vec[i], norm_coeff);
//   }
// }

// __device__ int MultiClsSemantics::get_max_class() const {
//   int max_cls = 0;
//   __half max_prob = prob_vec[0];
// #pragma unroll
//   for (int i = 0; i < NUM_CLASSES; ++i) {
//     if (__hgt(prob_vec[i], max_prob)) {
//       max_cls = i;
//       max_prob = prob_vec[i];
//     }
//   }
//   return max_cls;
// }

__device__ __host__ VoxelRGBW::VoxelRGBW() : rgb(0, 0, 0), weight(0) {}
__device__ __host__ VoxelRGBW::VoxelRGBW(const Eigen::Matrix<unsigned char, 3, 1>& rgb,
                                         unsigned char weight)
    : rgb(rgb), weight(weight) {}

__device__ __host__ VoxelTSDF::VoxelTSDF() : tsdf(-10) {}
__device__ __host__ VoxelTSDF::VoxelTSDF(float tsdf) : tsdf(tsdf) {}

__device__ __host__ VoxelSEGM::VoxelSEGM() { semantic_rep.uniform_init(); }

__device__ __host__ VoxelSpatialTSDF::VoxelSpatialTSDF(){};
__device__ __host__ VoxelSpatialTSDF::VoxelSpatialTSDF(const Eigen::Vector3f& position)
    : VoxelSpatialTSDF(position, 1.) {}
__device__ __host__ VoxelSpatialTSDF::VoxelSpatialTSDF(const Eigen::Vector3f& position, float tsdf)
    : position(position), tsdf(tsdf) {}

__device__ __host__ VoxelSpatialTSDFSEGM::VoxelSpatialTSDFSEGM(){};
__device__ __host__ VoxelSpatialTSDFSEGM::VoxelSpatialTSDFSEGM(const Eigen::Vector3f& position,
                                                               const float tsdf,
                                                               const int predicted_class_)
    : position(position), tsdf(tsdf), predicted_class(predicted_class_) {}
