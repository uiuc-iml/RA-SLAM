#include <iostream>

#include "utils/gl/renderer_base.h"
#include "utils/gl/shader.h"

class SimpleRenderer : public RendererBase {
 public:
  SimpleRenderer(const std::string& name)
      : RendererBase(name),
        clear_color_({0.0f, 0.0f, 0.0f, 1.0f}),
        shader_(VERTEX_SHADER_PATH, FRAGMENT_SHADER_PATH) {}  // passed in through CMake

 protected:
  void Render() override {
    // viewport
    int display_w, display_h;
    glfwGetFramebufferSize(window_, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    // shader
    shader_.Bind();
    // bg color
    ImGui::Begin("simple renderer");
    ImGui::ColorEdit3("background color", (float*)&clear_color_);
    ImGui::End();
    glClearColor(clear_color_.x, clear_color_.y, clear_color_.z, clear_color_.w);
    glClear(GL_COLOR_BUFFER_BIT);
  }

 private:
  ImVec4 clear_color_;
  ImVec4 point_color_;
  const Shader shader_;
};

int main() {
  SimpleRenderer renderer("example renderer");
  renderer.Run();
  return 0;
}
