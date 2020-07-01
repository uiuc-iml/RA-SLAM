#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <spdlog/spdlog.h>

#include "renderer_base.h"

// minimum OpenGL 4.3
#define GL_VERSION_MAJOR 4
#define GL_VERSION_MINOR 3
#define GLSL_VERSION "#version 430"

RendererBase::RendererBase(const std::string &name, int width, int height) {
  // GLFW init
  glfwSetErrorCallback(&RendererBase::GLFWErrorHandler);
  if (!glfwInit()) {
    spdlog::error("cannot initialize glfw");
    exit(EXIT_FAILURE);
  }
  // GL version hints
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, GL_VERSION_MAJOR);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, GL_VERSION_MINOR);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  // window creation
  window_ = glfwCreateWindow(width, height, name.c_str(), NULL, NULL);
  if (window_ == NULL) {
    spdlog::error("cannot create glfw window");
    exit(EXIT_FAILURE);
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1); // Enable vsync
  // glew init
  if (glewInit() != GLEW_OK) {
    spdlog::error("cannot initialize glew");
    exit(EXIT_FAILURE);
  }
  // imgui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  ImGui_ImplOpenGL3_Init(GLSL_VERSION);
}

RendererBase::~RendererBase() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window_);
  glfwTerminate();
}

void RendererBase::Run() {
  while (!glfwWindowShouldClose(window_)) {
    // new frame + dispatch events
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    // custom rendering logic to be overloaded
    Render();
    // render imgui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    // double buffer swap
    glfwSwapBuffers(window_);
  }
}

void RendererBase::GLFWErrorHandler(int error, const char *desc) {
  spdlog::error("glfw error {}: {}", error, desc);
}

