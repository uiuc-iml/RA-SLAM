#include "RGBDFrame.h"

namespace stb {
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"
#undef STB_IMAGE_WRITE_IMPLEMENTATION
}

namespace ml{

RGBDFrame::RGBDFrame() {
    m_colorCompressed = NULL;
    m_depthCompressed = NULL;
    m_colorSizeBytes = 0;
    m_depthSizeBytes = 0;
    m_cameraToWorld.setZero(-std::numeric_limits<float>::infinity());
    m_timeStampColor = 0;
    m_timeStampDepth = 0;
}

RGBDFrame::RGBDFrame(const RGBDFrame& other) {
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

RGBDFrame::RGBDFrame(RGBDFrame&& other) {
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

vec3uc* RGBDFrame::decompressColorAlloc_stb(COMPRESSION_TYPE_COLOR type) const {	//can handle PNG, JPEG etc.
    if (type != TYPE_JPEG && type != TYPE_PNG) throw MLIB_EXCEPTION("invliad type");
    if (m_colorCompressed == NULL || m_colorSizeBytes == 0) throw MLIB_EXCEPTION("decompression error");
    int channels = 3;
    int width, height;
    unsigned char* raw = stb::stbi_load_from_memory(m_colorCompressed, (int)m_colorSizeBytes, &width, &height, NULL, channels);
    return (vec3uc*)raw;
}

void RGBDFrame::compressDepth(const unsigned short* depth, unsigned int width, unsigned int height, COMPRESSION_TYPE_DEPTH type) {
    freeDepth();

    if (type == TYPE_RAW_USHORT) {
        if (m_depthSizeBytes != width*height) {
            freeDepth();
            m_depthSizeBytes = width*height*sizeof(unsigned short);
            m_depthCompressed = (unsigned char*)std::malloc(m_depthSizeBytes);
        }
        std::memcpy(m_depthCompressed, depth, m_depthSizeBytes);
    }
    else if (type == TYPE_ZLIB_USHORT) {
        freeDepth();

        int out_len = 0;
        int quality = 8;
        int n = 2;
        unsigned char* tmpBuff = (unsigned char *)std::malloc((width*n + 1) * height);
        std::memcpy(tmpBuff, depth, width*height*sizeof(unsigned short));
        m_depthCompressed = stb::stbi_zlib_compress(tmpBuff, width*height*sizeof(unsigned short), &out_len, quality);
        std::free(tmpBuff);
        m_depthSizeBytes = out_len;
    }
    else if (type == TYPE_OCCI_USHORT) {
        freeDepth();
#ifdef _USE_UPLINK_COMPRESSION
        //TODO fix the shift here
        int out_len = 0;
        int n = 2;
        unsigned int tmpBuffSize = (width*n + 1) * height;
        unsigned char* tmpBuff = (unsigned char *)std::malloc(tmpBuffSize);
        out_len = uplinksimple::encode(depth, width*height, tmpBuff, tmpBuffSize);
        m_depthSizeBytes = out_len;
        m_depthCompressed = (unsigned char*)std::malloc(out_len);
        std::memcpy(m_depthCompressed, tmpBuff, out_len);
        std::free(tmpBuff);
#else
        throw MLIB_EXCEPTION("need UPLINK_COMPRESSION");
#endif
    }
    else {
        throw MLIB_EXCEPTION("unknown compression type");
    }
}

unsigned short* RGBDFrame::decompressDepthAlloc_stb(COMPRESSION_TYPE_DEPTH type) const {
    if (type != TYPE_ZLIB_USHORT) throw MLIB_EXCEPTION("invliad type");
    unsigned short* res;
    int len;
    res = (unsigned short*)stb::stbi_zlib_decode_malloc((const char*)m_depthCompressed, (int)m_depthSizeBytes, &len);
    return res;
}

}; // namespace ml