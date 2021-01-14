#include "inference.h"

#include <iostream>

// gpumat to tensor
torch::Tensor mat_to_tensor(cv::Mat & my_mat) {
    // sizes: (1, C, H, W)
    //normalization
    my_mat.convertTo(my_mat, CV_32FC3, 1.0f / 255.0f);
    //opencv format H*W*C
    auto input_tensor = torch::from_blob(my_mat.data, {1, my_mat.rows, my_mat.cols, 3});
    //pytorch format N*C*H*W
    input_tensor = input_tensor.permute({0, 3, 1, 2});
    return input_tensor;
}

// (CPU) float32 tensor to float32 mat
cv::Mat tensor_to_mat(const torch::Tensor & my_tensor) {
    torch::Tensor temp_tensor;
    temp_tensor = temp_tensor.to(torch::kCPU);
    cv::Mat ret(352, 640, CV_32FC1);
    //copy the data from out_tensor to resultImg
    std::memcpy((void *) ret.data, temp_tensor.data_ptr(), sizeof(torch::kFloat32) * temp_tensor.numel());
    return ret;
}

inference_engine::inference_engine(const std::string & compiled_engine_path) {
    this->engine = torch::jit::load(compiled_engine_path);
    this->engine.to(torch::kCUDA);

    std::cout << "Model loaded." << std::endl;
}

std::vector<cv::Mat> inference_engine::infer_one(const cv::Mat & rgb_img) {
    cv::Mat downsized_rgb_img;
    std::vector<cv::Mat> ret;
    cv::resize(rgb_img, downsized_rgb_img, cv::Size(640, 352));

    torch::Tensor downsized_rgb_img_tensor = mat_to_tensor(downsized_rgb_img);
    torch::Tensor downsized_rgb_img_tensor_cuda = downsized_rgb_img_tensor.to(torch::kCUDA);

    this->input_buffer.clear();
    this->input_buffer.push_back(downsized_rgb_img_tensor_cuda);

    torch::Tensor ht_lt_prob_map = this->engine.forward(this->input_buffer).toTensor().squeeze().detach();
    
    ret.push_back(tensor_to_mat(ht_lt_prob_map[0]));
    ret.push_back(tensor_to_mat(ht_lt_prob_map[1]));

    return ret;
}