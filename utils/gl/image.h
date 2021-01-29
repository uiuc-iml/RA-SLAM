#pragma once

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "utils/gl/shader.h"

/**
 * @brief base class for image bridging between OpenGL and CUDA
 */
class GLImageBase {
 public:
  /**
   * @brief base GLImage constructor
   *
   * @param fragment_shader OpenGL fragment shader program for texture rendering
   */
  GLImageBase(const std::string& fragment_shader);

  /**
   * @brief virtual destructor for inheritance
   */
  virtual ~GLImageBase();

  /**
   * @brief bind GL texture with image stored with CUDA data pointer
   *
   * @param height  height of the image
   * @param width   width of the image
   * @param data    optional pointer to a CUDA image buffer
   */
  void BindImage(int height, int width, const void* data = nullptr);

  /**
   * @brief OpenGL draw call
   */
  void Draw() const;

  /**
   * @brief update GLImage with new CUDA pointer
   *
   * @param data    pointer to a CUDA image buffer
   * @param stream  optional CUDA stream
   */
  void LoadCuda(const void* data, cudaStream_t stream = nullptr);

  /**
   * @return image height
   */
  int GetHeight() { return height_; }

  /**
   * @return image width
   */
  int GetWidth() { return width_; }

 protected:
  /**
   * @brief bind data pointer with GL Texture2D
   *
   * @param height
   * @param width
   * @param data
   */
  virtual void GLTex2D(int height, int width, const void* data) const = 0;

  /**
   * @return size of a single pixel
   */
  virtual size_t ElementSize() const = 0;

 private:
  Shader shader_;
  unsigned int vbo_, vao_, ebo_, texture_;
  cudaGraphicsResource_t cuda_resrc_ = nullptr;
  int height_;
  int width_;
};

/**
 * @brief specialized GL Image with 32-bit float single channel pixels
 */
class GLImage32FC1 : public GLImageBase {
 public:
  GLImage32FC1();

 protected:
  void GLTex2D(int height, int width, const void* data) const override final;

  size_t ElementSize() const override final;
};

/**
 * @brief specialized GL Image with 8-bit unsigned int 4-channel pixels
 */
class GLImage8UC4 : public GLImageBase {
 public:
  GLImage8UC4();

 protected:
  void GLTex2D(int height, int width, const void* data) const override final;

  size_t ElementSize() const override final;
};
