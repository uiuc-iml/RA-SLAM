#include "shader.h"

#include <spdlog/spdlog.h>

#include <fstream>
#include <sstream>

Shader::Shader(const std::string& vertex_shader, const std::string& fragment_shader, bool load_file)
    : program_id_(glCreateProgram()) {
  const std::string vertex_shader_code = load_file ? ReadFile(vertex_shader) : vertex_shader;
  const std::string frag_shader_code = load_file ? ReadFile(fragment_shader) : fragment_shader;
  const GLuint vertex_id = CompileShader(vertex_shader_code, GL_VERTEX_SHADER);
  const GLuint fragment_id = CompileShader(frag_shader_code, GL_FRAGMENT_SHADER);
  glAttachShader(program_id_, vertex_id);
  glAttachShader(program_id_, fragment_id);
  glLinkProgram(program_id_);
  GLint success;
  glGetProgramiv(program_id_, GL_LINK_STATUS, &success);
  if (!success) {
    spdlog::error("fail to link shaders to program");
    exit(EXIT_FAILURE);
  }
  glDeleteShader(vertex_id);
  glDeleteShader(fragment_id);
}

Shader::~Shader() { glDeleteProgram(program_id_); }

void Shader::Bind() const { glUseProgram(program_id_); }

void Shader::Unbind() const { glUseProgram(0); }

void Shader::SetUniform4f(const std::string& name, float x, float y, float z, float w) const {
  const GLint uniform_idx = GetUniformLocation(name);
  if (uniform_idx < 0) {
    spdlog::error("Uniform name {} does not exist", name);
    return;
  }
  glUniform4f(uniform_idx, x, y, z, w);
}

void Shader::SetUniform3f(const std::string& name, float x, float y, float z) const {
  const GLint uniform_idx = GetUniformLocation(name);
  if (uniform_idx < 0) {
    spdlog::error("Uniform name {} does not exist", name);
    return;
  }
  glUniform3f(uniform_idx, x, y, z);
}

std::string Shader::ReadFile(const std::string& filepath) {
  std::stringstream ss;
  std::ifstream file(filepath);
  ss << file.rdbuf();
  file.close();
  return ss.str();
}

GLuint Shader::CompileShader(const std::string& shader_code, GLenum shader_type) {
  const char* shader_code_c_str = shader_code.c_str();
  // compile shader
  GLint success;
  const GLuint shader_id = glCreateShader(shader_type);
  glShaderSource(shader_id, 1, &shader_code_c_str, NULL);
  glCompileShader(shader_id);
  glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);
  if (!success) {
    spdlog::error("failed to compile shader:\n{}", shader_code);
    exit(EXIT_FAILURE);
  }
  return shader_id;
}

GLint Shader::GetUniformLocation(const std::string& name) const {
  if (uniforms_.find(name) == uniforms_.end()) {
    uniforms_[name] = glGetUniformLocation(program_id_, name.c_str());
  }
  return uniforms_[name];
}
