#include "shader.h"

#include <fstream>
#include <sstream>

#include <spdlog/spdlog.h>

Shader::Shader(const std::string &vertex_shader_filepath, 
               const std::string &fragment_shader_filepath) 
  : program_id_(glCreateProgram()) {
  const GLuint vertex_id = CompileShader(vertex_shader_filepath, GL_VERTEX_SHADER);
  const GLuint fragment_id = CompileShader(fragment_shader_filepath, GL_FRAGMENT_SHADER);
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

Shader::~Shader() {
  glDeleteProgram(program_id_);
}

void Shader::Bind() const {
  glUseProgram(program_id_);
}

void Shader::Unbind() const {
  glUseProgram(0);
}

void Shader::SetUniform4f(const std::string &name, const float4 &value) const {
    const GLint uniform_idx = GetUniformLocation(name);
    if (uniform_idx < 0) {
      spdlog::error("Uniform name {} does not exist", name);
      return;
    }
    glUniform4f(uniform_idx, value.x, value.y, value.z, value.w);
}

void Shader::SetUniform3f(const std::string &name, const float3 &value) const {
    const GLint uniform_idx = GetUniformLocation(name);
    if (uniform_idx < 0) {
      spdlog::error("Uniform name {} does not exist", name);
      return;
    }
    glUniform3f(uniform_idx, value.x, value.y, value.z);
}

GLuint Shader::CompileShader(const std::string &filepath, GLenum shader_type) {
  // read shader file
  std::stringstream ss;
  std::ifstream file(filepath);
  ss << file.rdbuf();
  file.close();
  const std::string shader_code_str = ss.str();
  const char *shader_code = shader_code_str.c_str();
  // compile shader
  GLint success;
  const GLuint shader_id = glCreateShader(shader_type);
  glShaderSource(shader_id, 1, &shader_code, NULL);
  glCompileShader(shader_id);
  glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);
  if (!success) {
    spdlog::error("failed to compile shader from {}", filepath);
    exit(EXIT_FAILURE);
  }
  return shader_id;
}

GLint Shader::GetUniformLocation(const std::string &name) const {
  if (uniforms_.find(name) == uniforms_.end()) {
    uniforms_[name] = glGetUniformLocation(program_id_, name.c_str());
  }
  return uniforms_[name];
}

