#pragma once

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "utils/gl/shader.h"

class GLImageBase {
 public:
  GLImageBase(const char *fragment_shader);

  virtual ~GLImageBase();

  void BindImage(int height, int width, const void *data = nullptr);

  void Draw() const;

  void LoadCuda(const void *data, cudaStream_t stream = nullptr);

  int height, width;

 protected:
  virtual void GLTex2D(int height, int width, const void *data) const = 0;

  virtual size_t ElementSize() const = 0;

 private:
  Shader shader_;
  unsigned int vbo_, vao_, ebo_, texture_;
  cudaGraphicsResource_t cuda_resrc_ = nullptr;
};

class GLImage32FC1 : public GLImageBase {
 public:
  GLImage32FC1();

 protected:
  void GLTex2D(int height, int width, const void *data) const override final;

  size_t ElementSize() const override final;
};

class GLImage8UC4 : public GLImageBase {
 public:
  GLImage8UC4();

 protected:
  void GLTex2D(int height, int width, const void *data) const override final;

  size_t ElementSize() const override final;
};
