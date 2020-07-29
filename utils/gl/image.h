#pragma once

#include "utils/gl/shader.h"

class GLImage {
 public:
  GLImage(int height, int width, void *data = nullptr);

  ~GLImage();

  void ReBindImage(int height, int width, void *data = nullptr);

  void Draw() const;

  int height, width;

 private:
  Shader shader_;
  unsigned int vbo_, vao_, ebo_, texture_;
};
