#include "utils/gl/image.h"
#include "utils/gl/renderer_base.h"

#include <string>
#include <GL/glew.h>

const static float vertices[] = {
   1,  1,  0, // top right
   1, -1,  0, // bottom right
  -1, -1,  0, // bottom left
  -1,  1,  0, // top left
};

const static unsigned int indices[] = {
  0, 1, 3, // first triangle
  1, 2, 3, // second triangle
};

const static char *vertex_shader = GLSL_VERSION "\n" R"END(
layout (location = 0) in vec3 pos;
out vec2 tex_coord;

void main() {
  gl_Position = vec4(pos, 1.0);
  tex_coord = vec2((pos.x + 1) / 2, 1 - (pos.y + 1) / 2);
}
)END";

const static char *fragment_shader = GLSL_VERSION "\n" R"END(
out vec4 frag_color;
in vec2 tex_coord;
uniform sampler2D tex;

void main() {
  frag_color = vec4(vec3(texture(tex, tex_coord).r), 1);
}
)END";

GLImage::GLImage(int height, int width, void *data) 
    : height(height), width(width), shader_(vertex_shader, fragment_shader, false) {
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
  ReBindImage(height, width, data);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

GLImage::~GLImage() {
  glDeleteVertexArrays(1, &vao_);
  glDeleteBuffers(1, &vbo_);
  glDeleteBuffers(1, &ebo_);
  glDeleteTextures(1, &texture_);
}

void GLImage::ReBindImage(int height, int width, void *data) {
  glBindTexture(GL_TEXTURE_2D, texture_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_FLOAT, data);
}

void GLImage::Draw() const {
  shader_.Bind();
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glBindVertexArray(vao_);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}
