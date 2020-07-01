#pragma once

#include <imgui.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <string>

class RendererBase {
 public:
  RendererBase(const std::string &name, int width = 1280, int height = 720);
  virtual ~RendererBase();

  void Run();

 protected:
  GLFWwindow *window_; 
  virtual void Render() = 0;

 private:
  static void GLFWErrorHandler(int error, const char *desc);
  static void GLErrorHandler(GLenum source, GLenum type, GLuint id, GLenum severity,
                             GLsizei length, const GLchar *msg, const void *args);
  static bool initialized_;
};
