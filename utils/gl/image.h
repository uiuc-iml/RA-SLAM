#pragma once

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "utils/gl/shader.h"


class GLImage {
 public:
  GLImage();

  GLImage(int height, int width, void *data = nullptr);

  ~GLImage();

  void ReBindImage(int height, int width, void *data = nullptr);

  void Draw() const;

  void LoadCuda(const void *data, cudaStream_t stream = nullptr);

  int height, width;

 private:
  Shader shader_;
  unsigned int vbo_, vao_, ebo_, texture_;
  cudaGraphicsResource_t cuda_resrc_ = nullptr;
};
