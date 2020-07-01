#pragma once

#include <imgui.h>

#include <GLFW/glfw3.h>

#include <string>

class RendererBase {
 public:
  RendererBase(const std::string &name, int width = 1280, int height = 720);
  virtual ~RendererBase();

  void Run();

  virtual void Render() = 0;

 private:
  static void GLFWErrorHandler(int error, const char *desc);

  GLFWwindow *window_; 
};
