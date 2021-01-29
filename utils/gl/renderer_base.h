#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>

#include <string>

// minimum OpenGL 4.3
#define GL_VERSION_MAJOR 4
#define GL_VERSION_MINOR 3
#define GLSL_VERSION "#version 430"

/**
 * @brief base OpenGL renderer class
 */
class RendererBase {
 public:
  /**
   * @brief create rendering window
   *
   * @param name    name of the window
   * @param width   width of the window in [pixel]
   * @param height  height of the window in [pixel]
   */
  RendererBase(const std::string& name, int width = 1280, int height = 720);

  /**
   * @brief destroy the window
   */
  virtual ~RendererBase();

  /**
   * @brief launch the rendering window
   */
  void Run();

 protected:
  GLFWwindow* window_;

  /**
   * @brief render call that needs to be implemented by child classes
   */
  virtual void Render() = 0;

  /**
   * @brief dispatch input to the GUI before render
   */
  virtual void DispatchInput();

  /**
   * @brief optional exit cleanup before exiting rendering loop
   */
  virtual void RenderExit();

 private:
  static void GLFWErrorHandler(int error, const char* desc);
  static void GLErrorHandler(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
                             const GLchar* msg, const void* args);
  static bool initialized_;
};
