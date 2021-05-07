#pragma once

#include "include.hpp"

namespace ml{

class RGBDFrame {
  public:
    RGBDFrame();

    RGBDFrame(const RGBDFrame& other);

    RGBDFrame(RGBDFrame&& other);

    inline unsigned char* getColorCompressed() const {
        return m_colorCompressed;
    }

    inline unsigned char* getDepthCompressed() const {
        return m_depthCompressed;
    }

    inline size_t getColorSizeBytes() const {
        return m_colorSizeBytes;
    }

    inline size_t getDepthSizeBytes() const {
        return m_depthSizeBytes;
    }

    inline const mat4f& getCameraToWorld() const {
        return m_cameraToWorld;
    }

    inline void setCameraToWorld(const mat4f& cameraToWorld) {
        m_cameraToWorld = cameraToWorld;
    }

    //! typically in microseconds
    inline void setTimeStampColor(UINT64 t) {
        m_timeStampColor = t;
    }
    //! typically in microseconds
    inline void setTimeStampDepth(UINT64 t) {
        m_timeStampDepth = t;
    }

    //! returns the color time stamp; typically in microseconds
    inline UINT64 getTimeStampColor() const {
        return m_timeStampColor;
    }

    //! returns the depth time stamp; typically in microseconds
    inline UINT64 getTimeStampDepth() const {
        return m_timeStampDepth;
    }

    inline void free() {
        freeColor();
        freeDepth();
        m_cameraToWorld.setZero(-std::numeric_limits<float>::infinity());
    }

  private:
    friend class SensorData;

    RGBDFrame(
        const vec3uc* color, unsigned int colorWidth, unsigned int colorHeight,
        const unsigned short*  depth, unsigned int depthWidth, unsigned int depthHeight,
        const mat4f& cameraToWorld = mat4f::identity(),
        COMPRESSION_TYPE_COLOR colorType = TYPE_JPEG,
        COMPRESSION_TYPE_DEPTH depthType = TYPE_ZLIB_USHORT,
        UINT64 timeStampColor = 0,
        UINT64 timeStampDepth = 0)
    {
        m_colorCompressed = NULL;
        m_depthCompressed = NULL;
        m_colorSizeBytes = 0;
        m_depthSizeBytes = 0;

        if (color) {
            //Timer t;
            compressColor(color, colorWidth, colorHeight, colorType);
            //std::cout << "compressColor " << t.getElapsedTimeMS() << " [ms] " << std::endl;
        }
        if (depth) {
            //Timer t;
            compressDepth(depth, depthWidth, depthHeight, depthType);
            //std::cout << "compressDepth " << t.getElapsedTimeMS() << " [ms] " << std::endl;
        }

        m_cameraToWorld = cameraToWorld;
        m_timeStampColor = timeStampColor;
        m_timeStampDepth = timeStampDepth;
    }
    
    //! overwrites the depth frame data
    void replaceDepth(const unsigned short* depth, unsigned int depthWidth, unsigned int depthHeight, COMPRESSION_TYPE_DEPTH depthType = TYPE_ZLIB_USHORT) {
        freeDepth();
        compressDepth(depth, depthWidth, depthHeight, depthType);
    }

    //! overwrites the color frame data
    void replaceColor(const vec3uc* color, unsigned int colorWidth, unsigned int colorHeight, COMPRESSION_TYPE_COLOR colorType = TYPE_JPEG) {
        freeColor();
        compressColor(color, colorWidth, colorHeight, colorType);
    }

    void freeColor() {
        if (m_colorCompressed) std::free(m_colorCompressed);
        m_colorCompressed = NULL;
        m_colorSizeBytes = 0;
        m_timeStampColor = 0;
    }
    void freeDepth() {
        if (m_depthCompressed) std::free(m_depthCompressed);
        m_depthCompressed = NULL;
        m_depthSizeBytes = 0;
        m_timeStampDepth = 0;
    }

    //! assignment operator
    bool operator=(const RGBDFrame& other) {
        if (this != &other) {
            free();

            m_colorSizeBytes = other.m_colorSizeBytes;
            m_depthSizeBytes = other.m_depthSizeBytes;
            m_colorCompressed = (unsigned char*)std::malloc(m_colorSizeBytes);
            m_depthCompressed = (unsigned char*)std::malloc(m_depthSizeBytes);

            if (!m_colorCompressed || !m_depthCompressed) throw MLIB_EXCEPTION("out of memory");

            std::memcpy(m_colorCompressed, other.m_colorCompressed, m_colorSizeBytes);
            std::memcpy(m_depthCompressed, other.m_depthCompressed, m_depthSizeBytes);

            m_timeStampColor = other.m_timeStampColor;
            m_timeStampDepth = other.m_timeStampDepth;
            m_cameraToWorld = other.m_cameraToWorld;
        }
        return true;
    }

    //! move operator
    bool operator=(RGBDFrame&& other) {
        if (this != &other) {
            free();

            m_colorSizeBytes = other.m_colorSizeBytes;
            m_depthSizeBytes = other.m_depthSizeBytes;
            m_colorCompressed = other.m_colorCompressed;
            m_depthCompressed = other.m_depthCompressed;

            m_timeStampColor = other.m_timeStampColor;
            m_timeStampDepth = other.m_timeStampDepth;
            m_cameraToWorld = other.m_cameraToWorld;

            other.m_colorCompressed = NULL;
            other.m_depthCompressed = NULL;
        }
        return true;
    }


    void compressColor(const vec3uc* color, unsigned int width, unsigned int height, COMPRESSION_TYPE_COLOR type) {

        if (type == TYPE_RAW) {
            if (m_colorSizeBytes != width*height) {
                freeColor();
                m_colorSizeBytes = width*height*sizeof(vec3uc);
                m_colorCompressed = (unsigned char*)std::malloc(m_colorSizeBytes);
            }
            std::memcpy(m_colorCompressed, color, m_colorSizeBytes);
        }
        else if (type == TYPE_PNG || type == TYPE_JPEG) {
            freeColor();

#ifdef _USE_UPLINK_COMPRESSION
            uplinksimple::graphics_PixelFormat format = uplinksimple::graphics_PixelFormat_RGB;
            uplinksimple::graphics_ImageCodec codec = uplinksimple::graphics_ImageCodec_JPEG;
            if (type == TYPE_PNG) codec = uplinksimple::graphics_ImageCodec_PNG;

            uplinksimple::MemoryBlock block;
            float quality = uplinksimple::defaultQuality;
            //float quality = 1.0f;
            uplinksimple::encode_image(codec, (const uint8_t*)color, width*height, format, width, height, block, quality);
            m_colorCompressed = block.Data;
            m_colorSizeBytes = block.Size;
            block.relinquishOwnership();
#else
            throw MLIB_EXCEPTION("need UPLINK_COMPRESSION");
#endif
        }
        else {
            throw MLIB_EXCEPTION("unknown compression type");
        }
    }

    vec3uc* decompressColorAlloc(COMPRESSION_TYPE_COLOR type) const {
        if (type == TYPE_RAW)	return decompressColorAlloc_raw(type);
#ifdef _USE_UPLINK_COMPRESSION
        else return decompressColorAlloc_occ(type);	//this handles all image formats;
#else
        else return decompressColorAlloc_stb(type);	// this handles all image formats
#endif
    }

    vec3uc* decompressColorAlloc_stb(COMPRESSION_TYPE_COLOR type) const;

    vec3uc* decompressColorAlloc_occ(COMPRESSION_TYPE_COLOR type) const {
#ifdef _USE_UPLINK_COMPRESSION
        if (type != TYPE_JPEG && type != TYPE_PNG) throw MLIB_EXCEPTION("invliad type");
        if (m_colorCompressed == NULL || m_colorSizeBytes == 0) throw MLIB_EXCEPTION("decompression error");
        uplinksimple::graphics_PixelFormat format = uplinksimple::graphics_PixelFormat_RGB;
        uplinksimple::graphics_ImageCodec codec = uplinksimple::graphics_ImageCodec_JPEG;
        if (type == TYPE_PNG) codec = uplinksimple::graphics_ImageCodec_PNG;

        uplinksimple::MemoryBlock block;
        //float quality = uplinksimple::defaultQuality;
        size_t width = 0;
        size_t height = 0;
        uplinksimple::decode_image(codec, (const uint8_t*)m_colorCompressed, m_colorSizeBytes, format, width, height, block);
        vec3uc* res = (vec3uc*)block.Data;
        block.relinquishOwnership();
        return res;
#else
        throw MLIB_EXCEPTION("need UPLINK_COMPRESSION");
        return NULL;
#endif
    }

    vec3uc* decompressColorAlloc_raw(COMPRESSION_TYPE_COLOR type) const {
        if (type != TYPE_RAW) throw MLIB_EXCEPTION("invliad type");
        if (m_colorCompressed == NULL || m_colorSizeBytes == 0) throw MLIB_EXCEPTION("invalid data");
        vec3uc* res = (vec3uc*)std::malloc(m_colorSizeBytes);
        memcpy(res, m_colorCompressed, m_colorSizeBytes);
        return res;
    }

    void compressDepth(const unsigned short* depth, unsigned int width, unsigned int height, COMPRESSION_TYPE_DEPTH type);

    unsigned short* decompressDepthAlloc(unsigned int width, unsigned int height, COMPRESSION_TYPE_DEPTH type) const {
        if (type == TYPE_RAW_USHORT)	return decompressDepthAlloc_raw(type);
        else if (type == TYPE_ZLIB_USHORT) return decompressDepthAlloc_stb(type);
        else if (type == TYPE_OCCI_USHORT) return decompressDepthAlloc_occ(width, height, type);
        else {
            throw MLIB_EXCEPTION("invalid type");
            return NULL;
        }
    }

    unsigned short* decompressDepthAlloc_stb(COMPRESSION_TYPE_DEPTH type) const;

    unsigned int short* decompressDepthAlloc_occ(unsigned int width, unsigned int height, COMPRESSION_TYPE_DEPTH type) const {
#ifdef _USE_UPLINK_COMPRESSION
        if (type != TYPE_OCCI_USHORT) throw MLIB_EXCEPTION("invliad type");
        unsigned short* res = (unsigned short*)std::malloc(width*height * 2);
        uplinksimple::decode(m_depthCompressed, (unsigned int)m_depthSizeBytes, width*height, res);
        uplinksimple::shift2depth(res, width*height);
        return res;
#else
        throw MLIB_EXCEPTION("need UPLINK_COMPRESSION");
        return NULL;
#endif
    }

    unsigned short* decompressDepthAlloc_raw(COMPRESSION_TYPE_DEPTH type) const {
        if (type != TYPE_RAW_USHORT) throw MLIB_EXCEPTION("invliad type");
        if (m_depthCompressed == NULL || m_depthSizeBytes == 0) throw MLIB_EXCEPTION("invalid data");
        unsigned short* res = (unsigned short*)std::malloc(m_depthSizeBytes);
        memcpy(res, m_depthCompressed, m_depthSizeBytes);
        return res;
    }


    void saveToFile(std::ostream& out) const {
        out.write((const char*)&m_cameraToWorld, sizeof(mat4f));
        out.write((const char*)&m_timeStampColor, sizeof(UINT64));
        out.write((const char*)&m_timeStampDepth, sizeof(UINT64));
        out.write((const char*)&m_colorSizeBytes, sizeof(UINT64));
        out.write((const char*)&m_depthSizeBytes, sizeof(UINT64));
        out.write((const char*)m_colorCompressed, m_colorSizeBytes);
        out.write((const char*)m_depthCompressed, m_depthSizeBytes);
    }

    void loadFromFile(std::istream& in) {
        free();
        in.read((char*)&m_cameraToWorld, sizeof(mat4f));
        in.read((char*)&m_timeStampColor, sizeof(UINT64));
        in.read((char*)&m_timeStampDepth, sizeof(UINT64));
        in.read((char*)&m_colorSizeBytes, sizeof(UINT64));
        in.read((char*)&m_depthSizeBytes, sizeof(UINT64));
        m_colorCompressed = (unsigned char*)std::malloc(m_colorSizeBytes);
        in.read((char*)m_colorCompressed, m_colorSizeBytes);
        m_depthCompressed = (unsigned char*)std::malloc(m_depthSizeBytes);
        in.read((char*)m_depthCompressed, m_depthSizeBytes);
    }

    bool operator==(const RGBDFrame& other) const {
        if (m_colorSizeBytes != other.m_colorSizeBytes) return false;
        if (m_depthSizeBytes != other.m_depthSizeBytes) return false;
        if (m_timeStampColor != other.m_timeStampColor) return false;
        if (m_timeStampDepth != other.m_timeStampDepth) return false;
        for (unsigned int i = 0; i < 16; i++) {
            if (m_cameraToWorld.matrix[i] != other.m_cameraToWorld.matrix[i]) return false;
        }
        for (UINT64 i = 0; i < m_colorSizeBytes; i++) {
            if (m_colorCompressed[i] != other.m_colorCompressed[i]) return false;
        }
        for (UINT64 i = 0; i < m_depthSizeBytes; i++) {
            if (m_depthCompressed[i] != other.m_depthCompressed[i]) return false;
        }
        return true;
    }

    bool operator!=(const RGBDFrame& other) const {
        return !((*this) == other);
    }

    UINT64 m_colorSizeBytes;					//compressed byte size
    UINT64 m_depthSizeBytes;					//compressed byte size
    unsigned char* m_colorCompressed;			//compressed color data
    unsigned char* m_depthCompressed;			//compressed depth data
    UINT64 m_timeStampColor;					//time stamp color (convection: in microseconds)
    UINT64 m_timeStampDepth;					//time stamp depth (convention: in microseconds)
    mat4f m_cameraToWorld;						//camera trajectory: from current frame to base frame
};

} // namespace ml