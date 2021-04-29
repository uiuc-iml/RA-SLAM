#include "renderer_base.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <spdlog/spdlog.h>

bool RendererBase::initialized_ = false;

RendererBase::RendererBase(const std::string& name, int width, int height) {
  if (initialized_) {
    spdlog::error("singleton renderer instantiated twice!");
    exit(EXIT_FAILURE);
  }
  initialized_ = true;
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
  glfwSwapInterval(1);  // Enable vsync
  // glew init
  if (glewInit() != GLEW_OK) {
    spdlog::error("cannot initialize glew");
    exit(EXIT_FAILURE);
  }
  spdlog::info("OpenGL {} is used", glGetString(GL_VERSION));
  // attach OpenGL error callbacks
  glDebugMessageCallback(&RendererBase::GLErrorHandler, NULL);
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
  while (running_ && !glfwWindowShouldClose(window_)) {
    // new frame + dispatch events
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    // custom input dispatch / rendering logic to be overloaded
    DispatchInput();
    Render();
    // render imgui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    // double buffer swap
    glfwSwapBuffers(window_);
  }
  RenderExit();
}

void RendererBase::DispatchInput() {}

void RendererBase::RenderExit() {}

void RendererBase::GLFWErrorHandler(int error, const char* desc) {
#ifndef NDEBUG
  spdlog::error("glfw error {}: {}", error, desc);
#endif
}

void RendererBase::GLErrorHandler(GLenum source, GLenum type, GLuint id, GLenum severity,
                                  GLsizei length, const GLchar* msg, const void* args) {
#ifndef NDEBUG
  (void)source;
  (void)type;
  (void)id;
  (void)severity;
  (void)length;
  (void)args;
  const std::string msg_str(msg, msg + length);
  spdlog::error("[GL Error] {}", msg);
#endif
}
