#include "utils/gl/image.h"

#include <GL/glew.h>

#include <string>

#include "utils/cuda/errors.cuh"
#include "utils/gl/renderer_base.h"

const static float vertices[] = {
    1,  1,  0,  // top right
    1,  -1, 0,  // bottom right
    -1, -1, 0,  // bottom left
    -1, 1,  0,  // top left
};

const static unsigned int indices[] = {
    0, 1, 3,  // first triangle
    1, 2, 3,  // second triangle
};

const static char* vertex_shader = GLSL_VERSION
    "\n"
    R"END(
layout (location = 0) in vec3 pos;
out vec2 tex_coord;

void main() {
  gl_Position = vec4(pos, 1.0);
  tex_coord = vec2((pos.x + 1) / 2, 1 - (pos.y + 1) / 2);
}
)END";

const static char* frag_shader_c1 = GLSL_VERSION
    "\n"
    R"END(
out vec4 frag_color;
in vec2 tex_coord;
uniform sampler2D tex;

void main() {
  frag_color = vec4(vec3(texture(tex, tex_coord).r), 1);
}
)END";

const static char* frag_shader_c4 = GLSL_VERSION
    "\n"
    R"END(
out vec4 frag_color;
in vec2 tex_coord;
uniform sampler2D tex;

void main() {
  frag_color = texture(tex, tex_coord);
}
)END";

GLImageBase::GLImageBase(const std::string& fragment_shader)
    : height_(0), width_(0), shader_(vertex_shader, fragment_shader, false) {
  // vertices stuff
  glGenVertexArrays(1, &vao_);
  glGenBuffers(1, &vbo_);
  glGenBuffers(1, &ebo_);
  glBindVertexArray(vao_);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  // texture stuff
  glGenTextures(1, &texture_);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

GLImageBase::~GLImageBase() {
  glDeleteVertexArrays(1, &vao_);
  glDeleteBuffers(1, &vbo_);
  glDeleteBuffers(1, &ebo_);
  glDeleteTextures(1, &texture_);
  if (cuda_resrc_) {
    CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cuda_resrc_));
  }
}

void GLImageBase::BindImage(int height, int width, const void* data) {
  if (cuda_resrc_) {
    CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cuda_resrc_));
  }
  this->height_ = height;
  this->width_ = width;
  glBindTexture(GL_TEXTURE_2D, texture_);
  GLTex2D(height, width, data);
  CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&cuda_resrc_, texture_, GL_TEXTURE_2D,
                                             cudaGraphicsRegisterFlagsWriteDiscard));
}

void GLImageBase::Draw() const {
  shader_.Bind();
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glBindVertexArray(vao_);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void GLImageBase::LoadCuda(const void* data, cudaStream_t stream) {
  if (!cuda_resrc_) {
    return;
  }
  cudaArray_t array;
  CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cuda_resrc_, stream));
  CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array, cuda_resrc_, 0, 0));
  CUDA_SAFE_CALL(cudaMemcpy2DToArrayAsync(array, 0, 0, data, width_ * ElementSize(),
                                          width_ * ElementSize(), height_, cudaMemcpyDeviceToDevice,
                                          stream));
  CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &cuda_resrc_, stream));
}

GLImage32FC1::GLImage32FC1() : GLImageBase(frag_shader_c1) {}

void GLImage32FC1::GLTex2D(int height, int width, const void* data) const {
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, data);
}

size_t GLImage32FC1::ElementSize() const { return sizeof(float); }

GLImage8UC4::GLImage8UC4() : GLImageBase(frag_shader_c4) {}

void GLImage8UC4::GLTex2D(int height, int width, const void* data) const {
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
}

size_t GLImage8UC4::ElementSize() const { return sizeof(unsigned char) * 4; }
