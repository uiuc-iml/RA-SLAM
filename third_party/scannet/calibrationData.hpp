#pragma once

#include "include.hpp"

namespace ml{
class CalibrationData {
public:
    CalibrationData() {
        setIdentity();
    }

    CalibrationData(const mat4f& intrinsic, const mat4f& extrinsic = mat4f::identity()) {
        m_intrinsic = intrinsic;
        m_extrinsic = extrinsic;
    }

    void setMatrices(const mat4f& intrinsic, const mat4f& extrinsic = mat4f::identity()) {
        m_intrinsic = intrinsic;
        m_extrinsic = extrinsic;
    }

    static mat4f makeIntrinsicMatrix(float fx, float fy, float mx, float my) {
        return mat4f(
            fx, 0.0f, mx, 0.0f,
            0.0f, fy, my, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
            );
    }

    void setIdentity() {
        m_intrinsic.setIdentity();
        m_extrinsic.setIdentity();
    }

    void saveToFile(std::ostream& out) const {
        out.write((const char*)&m_intrinsic, sizeof(mat4f));
        out.write((const char*)&m_extrinsic, sizeof(mat4f));
    }

    void loadFromFile(std::istream& in) {
        in.read((char*)&m_intrinsic, sizeof(mat4f));
        in.read((char*)&m_extrinsic, sizeof(mat4f));
    }

    bool operator==(const CalibrationData& other) const {
        for (unsigned int i = 0; i < 16; i++) {
            if (m_intrinsic.matrix[i] != other.m_intrinsic.matrix[i]) return false;
            if (m_extrinsic.matrix[i] != other.m_extrinsic.matrix[i]) return false;
        }
        return true;
    }

    bool operator!=(const CalibrationData& other) const {
        return !((*this) == other);
    }

    //! Camera-to-Proj matrix
    mat4f m_intrinsic;

    //! it should be the mapping to base frame in a multi-view setup (typically it's depth to color; and the color extrinsic is the identity)
    mat4f m_extrinsic;
};

} // namespace ml