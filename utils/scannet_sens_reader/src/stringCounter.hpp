#pragma once

#include "include.hpp"

namespace ml{
class StringCounter {
public:
    StringCounter(const std::string& base, const std::string fileEnding, unsigned int numCountDigits = 0, unsigned int initValue = 0) {
        m_base = base;
        if (fileEnding[0] != '.') {
            m_fileEnding = ".";
            m_fileEnding.append(fileEnding);
        }
        else {
            m_fileEnding = fileEnding;
        }
        m_numCountDigits = numCountDigits;
        m_initValue = initValue;
        resetCounter();
    }

    ~StringCounter() {
    }

    std::string getNext() {
        std::string curr = getCurrent();
        m_current++;
        return curr;
    }

    std::string getCurrent() {
        std::stringstream ss;
        ss << m_base;
#ifdef WIN32
        for (unsigned int i = std::max(1u, (unsigned int)std::ceilf(std::log10f((float)m_current + 1))); i < m_numCountDigits; i++) ss << "0";
#else
        for (unsigned int i = std::max(1u, (unsigned int)ceilf(log10f((float)m_current + 1))); i < m_numCountDigits; i++) ss << "0";
#endif
        ss << m_current;
        ss << m_fileEnding;
        return ss.str();
    }

    void resetCounter() {
        m_current = m_initValue;
    }
private:
    std::string		m_base;
    std::string		m_fileEnding;
    unsigned int	m_numCountDigits;
    unsigned int	m_current;
    unsigned int	m_initValue;
};

} // namespace ml