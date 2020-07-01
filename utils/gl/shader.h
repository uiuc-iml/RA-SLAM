#pragma once

#include <string>
#include <unordered_map>

#include <GL/glew.h>

class Shader {
 public:
  Shader(const std::string &vertex_shader_filepath, const std::string &fragment_shader_filepath);
  ~Shader();
  
  void Bind() const;
  void Unbind() const;
  
  // TODO(alvin): extend type support when needed
  void SetUniform3f(const std::string &name, float x, float y, float z) const;
  void SetUniform4f(const std::string &name, float x, float y, float z, float w) const;

 private:
  GLuint CompileShader(const std::string &filepath, GLenum shader_type);
  GLint GetUniformLocation(const std::string &name) const;

  const GLuint program_id_;
  mutable std::unordered_map<std::string, GLint> uniforms_;
};
