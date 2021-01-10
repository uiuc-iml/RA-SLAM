#pragma once

#include <imgui.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <string>

// minimum OpenGL 4.3
#define GL_VERSION_MAJOR 4
#define GL_VERSION_MINOR 3
#define GLSL_VERSION "#version 430"

class RendererBase {
 public:
  RendererBase(const std::string &name, int width = 1280, int height = 720);
  virtual ~RendererBase();

  void Run();

 protected:
  GLFWwindow *window_;
  virtual void Render() = 0;
  virtual void DispatchInput();
  virtual void RenderExit();

 private:
  static void GLFWErrorHandler(int error, const char *desc);
  static void GLErrorHandler(GLenum source, GLenum type, GLuint id, GLenum severity,
                             GLsizei length, const GLchar *msg, const void *args);
  static bool initialized_;
};
