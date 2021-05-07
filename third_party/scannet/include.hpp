#pragma once

#include <cstring>
#include <cmath>

#include <vector>
#include <list>
#include <exception>
#include <fstream>
#include <cassert>
#include <iostream>
#include <atomic>
#include <thread> 
#include <mutex>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <iomanip>

//we treat everything that is not WIN32 as linux
#ifndef WIN32
#define LINUX
#endif


#ifdef WIN32
#include "windows.h"
#include <tchar.h>
#endif

#ifdef LINUX
#include <sys/stat.h>
#include <dirent.h>
#endif

#ifdef _USE_UPLINK_COMPRESSION
#if _WIN32
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "Shlwapi.lib")
#include <winsock2.h>
#include <Ws2tcpip.h>
#include "sensorData/uplinksimple.h"
#else 
#undef _USE_UPLINK_COMPRESSION
#endif 
#endif



namespace ml {

#ifndef _HAS_MLIB

#ifndef _MSC_VER
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif

	class MLibException : public std::exception {
	public:
		MLibException(const std::string& what) : std::exception() {
			m_msg = what;
		}
		MLibException(const char* what) : std::exception() {
			m_msg = std::string(what);
		}
		const char* what() const NOEXCEPT{
			return m_msg.c_str();
		}
	private:
		std::string m_msg;
	};



#ifndef MLIB_EXCEPTION
#define MLIB_EXCEPTION(s) ml::MLibException(std::string(__FUNCTION__).append(":").append(std::to_string(__LINE__)).append(": ").append(s).c_str())
#endif

#ifndef SAFE_DELETE
#define SAFE_DELETE(p)       { if (p) { delete (p);     (p)=nullptr; } }
#endif

#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p);   (p)=nullptr; } }
#endif

#ifndef UINT64
#ifdef WIN32
	typedef unsigned __int64 UINT64;
#else
	typedef uint64_t UINT64;
#endif
#endif

	class vec3uc {
		union
		{
			struct
			{
				unsigned char x, y, z; // standard names for components
			};
			unsigned char array[3];     // array access
		};
	};

	class vec4uc {
		union
		{
			struct
			{
				unsigned char x, y, z, w; // standard names for components
			};
			unsigned char array[4];     // array access
		};
	};

	class vec3d {
	public:
		vec3d() {
		}
		vec3d(double d) {
			x = y = z = d;
		}
		bool operator==(const vec3d& other) const {
			if (x != other.x || y != other.y || z != other.z) return false;
			return true;
		}
		bool operator!=(const vec3d& other) const {
			return !(*this == other);
		}
		union
		{
			struct
			{
				double x, y, z; // standard names for components
			};
			double array[3];     // array access
		};
	};


	class vec4d {
	public:
		union
		{
			struct
			{
				double x, y, z, w; // standard names for components
			};
			double array[4];     // array access
		};
	};


	class mat4f {
	public:
		mat4f() { }

		mat4f(
			const float& m00, const float& m01, const float& m02, const float& m03,
			const float& m10, const float& m11, const float& m12, const float& m13,
			const float& m20, const float& m21, const float& m22, const float& m23,
			const float& m30, const float& m31, const float& m32, const float& m33)
		{
			_m00 = m00;	_m01 = m01;	_m02 = m02;	_m03 = m03;
			_m10 = m10;	_m11 = m11;	_m12 = m12;	_m13 = m13;
			_m20 = m20;	_m21 = m21;	_m22 = m22;	_m23 = m23;
			_m30 = m30;	_m31 = m31;	_m32 = m32;	_m33 = m33;
		}

		void setIdentity() {
			matrix[0] = 1.0;	matrix[1] = 0.0f;	matrix[2] = 0.0f; matrix[3] = 0.0f;
			matrix[4] = 0.0f;	matrix[5] = 1.0;	matrix[6] = 0.0f; matrix[7] = 0.0f;
			matrix[8] = 0.0f;	matrix[9] = 0.0f;	matrix[10] = 1.0; matrix[11] = 0.0f;
			matrix[12] = 0.0f;	matrix[13] = 0.0f;	matrix[14] = 0.0f; matrix[15] = 1.0;
		}

		static mat4f identity() {
			mat4f res;	res.setIdentity();
			return res;
		}

		//! sets the matrix zero (or a specified value)
		void setZero(float v = 0.0f) {
			matrix[ 0] = matrix[ 1] = matrix[ 2] = matrix[ 3] = v;
			matrix[ 4] = matrix[ 5] = matrix[ 6] = matrix[ 7] = v;
			matrix[ 8] = matrix[ 9] = matrix[10] = matrix[11] = v;
			matrix[12] = matrix[13] = matrix[14] = matrix[15] = v;
		}
		static mat4f zero(float v = 0.0f) {
			mat4f res;	res.setZero(v);
			return res;
		}

		union {
			//! access matrix using a single array
			float matrix[16];
			//! access matrix using a two-dimensional array
			float matrix2[4][4];
			//! access matrix using single elements
			struct {
				float
					_m00, _m01, _m02, _m03,
					_m10, _m11, _m12, _m13,
					_m20, _m21, _m22, _m23,
					_m30, _m31, _m32, _m33;
			};
		};
	};


	namespace util {
		inline bool directoryExists(const std::string& directory) {
#if defined(WIN32)
			DWORD ftyp = GetFileAttributesA(directory.c_str());
			if (ftyp == INVALID_FILE_ATTRIBUTES)
				return false;  //something is wrong with your path!

			if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
				return true;   // this is a directory!

			return false;    // this is not a directory!
#elif defined(LINUX)
			const char *pzPath = directory.c_str();

			DIR *pDir;
			bool bExists = false;

			pDir = opendir(pzPath);

			if (pDir != NULL)
			{
				bExists = true;
				(void)closedir(pDir);
			}

			return bExists;
#else
#error Unknown OS!
#endif
		}

		inline void makeDirectory(const std::string& directory) {
#if defined(WIN32)
			CreateDirectoryA(directory.c_str(), nullptr);
#elif defined(LINUX)
			mkdir(directory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#else
#error Unknown OS!
#endif

		}
	}

#endif //_NO_MLIB_

enum COMPRESSION_TYPE_COLOR {
    TYPE_COLOR_UNKNOWN = -1,
    TYPE_RAW = 0,
    TYPE_PNG = 1,
    TYPE_JPEG = 2
};
enum COMPRESSION_TYPE_DEPTH {
    TYPE_DEPTH_UNKNOWN = -1,
    TYPE_RAW_USHORT = 0,
    TYPE_ZLIB_USHORT = 1,
    TYPE_OCCI_USHORT = 2
};


}