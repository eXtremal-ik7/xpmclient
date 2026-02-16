#include "cudautil.h"
#include <string.h>
#include <fstream>
#include <iostream>
#include <memory>

// Map CUDA driver version (from cuDriverGetVersion) to max supported PTX ISA version.
// The mapping is NOT a simple formula: CUDA 10.x has an offset, CUDA 12.5/12.6 both
// map to PTX 8.5, CUDA 12.7 was never released, and PTX 8.6 was skipped.
// Sources:
//   https://docs.nvidia.com/cuda/cuda-features-archive/index.html
//   https://milthorpe.org/2022/05/09/matching-cuda-and-nvptx-isa-versions/
//   https://github.com/openxla/xla/issues/16431 (12.5/12.6 both = PTX 8.5)
static bool cudaDriverPtxVersion(int driverVersion, int &ptxMajor, int &ptxMinor)
{
  int major = driverVersion / 1000;
  int minor = (driverVersion % 1000) / 10;

  if (major >= 13) {
    // CUDA 13.0 → PTX 9.0, 13.1 → PTX 9.1
    ptxMajor = 9;
    ptxMinor = minor;
  } else if (major == 12) {
    ptxMajor = 8;
    if (minor <= 4) {
      // CUDA 12.0–12.4 → PTX 8.0–8.4
      ptxMinor = minor;
    } else if (minor <= 6) {
      // CUDA 12.5 and 12.6 both → PTX 8.5 (12.7 was never released, PTX 8.6 skipped)
      ptxMinor = 5;
    } else if (minor == 8) {
      // CUDA 12.8 → PTX 8.7
      ptxMinor = 7;
    } else if (minor >= 9) {
      // CUDA 12.9 → PTX 8.8
      ptxMinor = 8;
    } else {
      return false;
    }
  } else if (major == 11) {
    // CUDA 11.0–11.8 → PTX 7.0–7.8
    ptxMajor = 7;
    ptxMinor = minor;
  } else if (major == 10) {
    // CUDA 10.0 → PTX 6.3, 10.1 → 6.4, 10.2 → 6.5
    ptxMajor = 6;
    ptxMinor = minor + 3;
  } else if (major == 9) {
    // CUDA 9.0 → PTX 6.0, 9.1 → 6.1, 9.2 → 6.2
    ptxMajor = 6;
    ptxMinor = minor;
  } else {
    return false;
  }
  return true;
}

bool cudaCompileKernel(const char *kernelName,
                       const std::vector<const char*> &sources,
                       const char **arguments,
                       int argumentsNum,
                       CUmodule *module,
                       int majorComputeCapability,
                       int,
                       bool needRebuild) 
{
  std::ifstream testfile(kernelName);
  if(needRebuild || !testfile) {
    LOG_F(INFO, "compiling ...");
    
    std::string sourceFile;
    for (auto &i: sources) {
      std::ifstream stream(i);
      std::string str((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
      sourceFile.append(str);
    }
    
    LOG_F(INFO, "source: %u bytes", (unsigned)sourceFile.size());
    if(sourceFile.size() < 1){
      LOG_F(ERROR, "source files not found or empty");
      return false;
    }
    
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(
      nvrtcCreateProgram(&prog,
                         sourceFile.c_str(),
                         "xpm.cu",
                         0,
                         NULL,
                         NULL));

    nvrtcResult compileResult = nvrtcCompileProgram(prog, argumentsNum, arguments);
    
    // Obtain compilation log from the program.
    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    std::unique_ptr<char[]> log(new char[logSize]);
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.get()));
    if (compileResult != NVRTC_SUCCESS) {
      LOG_F(ERROR, "nvrtcCompileProgram error: %s", nvrtcGetErrorString(compileResult));
      LOG_F(ERROR, "%s\n", log.get());
      return false;
    }
    
    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    char *ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
    
    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    
    {
      std::ofstream bin(kernelName, std::ofstream::binary | std::ofstream::trunc);
      bin.write(ptx, ptxSize);
      bin.close();      
    }
    
    delete[] ptx;
  }
  
  std::ifstream bfile(kernelName, std::ifstream::binary);
  if(!bfile) {
    return false;
  }  
  
  bfile.seekg(0, bfile.end);
  size_t binsize = bfile.tellg();
  bfile.seekg(0, bfile.beg);
  if(!binsize){
    LOG_F(ERROR, "%s empty", kernelName);
    return false;
  }
  
  std::unique_ptr<char[]> ptx(new char[binsize+1]);
  bfile.read(ptx.get(), binsize);
  bfile.close();
  
  CUresult result = cuModuleLoadDataEx(module, ptx.get(), 0, 0, 0);
  if (result != CUDA_SUCCESS) {
    if (result == CUDA_ERROR_INVALID_PTX || result == CUDA_ERROR_UNSUPPORTED_PTX_VERSION) {
      int driverVersion = 0;
      cuDriverGetVersion(&driverVersion);
      int ptxMajor, ptxMinor;
      if (!cudaDriverPtxVersion(driverVersion, ptxMajor, ptxMinor)) {
        LOG_F(ERROR, "Unknown CUDA driver version %d, cannot determine PTX version", driverVersion);
        return false;
      }
      LOG_F(WARNING, "PTX version mismatch (toolkit newer than driver), attempting downgrade to PTX %i.%i ...", ptxMajor, ptxMinor);

      // Patch ".version X.Y" in PTX to match driver's supported version
      char *pv = strstr(ptx.get(), ".version ");
      if (pv) {
        char patch[16];
        int patchLen = snprintf(patch, sizeof(patch), ".version %i.%i", ptxMajor, ptxMinor);
        // Find end of existing version line
        char *eol = strchr(pv, '\n');
        if (eol) {
          int oldLen = eol - pv;
          if (patchLen <= oldLen) {
            memcpy(pv, patch, patchLen);
            // Pad remainder with spaces to keep offsets intact
            memset(pv + patchLen, ' ', oldLen - patchLen);
          }
        }
      }

      CUDA_SAFE_CALL(cuModuleLoadDataEx(module, ptx.get(), 0, 0, 0));
    } else {
      const char *msg;
      cuGetErrorName(result, &msg);
      LOG_F(ERROR, "Loading CUDA module failed with error %s", msg);
      return false;
    }
  }

  return true;
}
