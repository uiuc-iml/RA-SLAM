#pragma once

#include <string>
#include <unordered_map>

#include <GL/glew.h>

class Shader {
 public:
  /**
   * @brief abstraction of OpenGL shader programs
   *
   * @param vertex_shader   vertex shader
   * @param fragment_shader fragment shader
   * @param load_file       if true
   *                          both shaders are interpreted as path to shader program
   *                        otherwise
   *                          both shaders are interpreted as shader program strings
   */
  Shader(const std::string &vertex_shader,
         const std::string &fragment_shader,
         bool load_file = true);

  /**
   * @brief delete shader program from GL
   */
  ~Shader();

  /**
   * @brief bind GL to this shader
   */
  void Bind() const;

  /**
   * @brief unbind GL from this shader
   */
  void Unbind() const;

  // TODO(alvin): extend type support when needed
  // Set uniform attributes by name
  void SetUniform3f(const std::string &name, float x, float y, float z) const;
  void SetUniform4f(const std::string &name, float x, float y, float z, float w) const;

 private:
  std::string ReadFile(const std::string &filepath);
  GLuint CompileShader(const std::string &shader_code, GLenum shader_type);
  GLint GetUniformLocation(const std::string &name) const;

  const GLuint program_id_;
  mutable std::unordered_map<std::string, GLint> uniforms_;
};
