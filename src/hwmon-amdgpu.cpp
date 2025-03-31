#include "hwmon-amdgpu.h"
#include <loguru.hpp>
#include <dirent.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

bool HWMonAmdGpu::isAvailable()
{
  int cardIdx = 0;
  for (;;) {
    char fileName[256];

    // Check directory /sys/class/drm/card<number> existent
    {
      struct stat s;
      snprintf(fileName, sizeof(fileName), "/sys/class/drm/card%i", cardIdx);
      if (stat(fileName, &s) != 0)
        break;
    }

    // Check amdgpu specific
    snprintf(fileName, sizeof(fileName), "/sys/class/drm/card%i/device/vendor", cardIdx);
    FILE *hFile = fopen(fileName, "r");
    if (hFile) {
      char vendor[16];
      memset(vendor, 0, sizeof(vendor));
      fread(vendor, 1, 6, hFile);
      fclose(hFile);

      if (strcmp(vendor, "0x1002") == 0) {
        LOG_F(INFO, "amdgpu hardware monitor for linux available");
        return true;
      }
    }

    cardIdx++;
  }

  LOG_F(INFO, "amdgpu/amdgpu-pro not found");
  return false;
}

HWMonAmdGpu::HWMonAmdGpu(int devicesNum)
{
  int cardIdx = 0;
  for (;;) {
    char fileName[256];

    // Check directory /sys/class/drm/card<number> existent
    {
      struct stat s;
      snprintf(fileName, sizeof(fileName), "/sys/class/drm/card%i", cardIdx);
      if (stat(fileName, &s) != 0)
        break;
    }

    // Check amdgpu specific
    snprintf(fileName, sizeof(fileName), "/sys/class/drm/card%i/device/vendor", cardIdx);
    FILE *hFile = fopen(fileName, "r");
    if (hFile) {
      char vendor[16];
      memset(vendor, 0, sizeof(vendor));
      fread(vendor, 1, 6, hFile);
      fclose(hFile);

      std::string hwmonDir;

      if (strcmp(vendor, "0x1002") == 0) {
        // Found AMD GPU
        // Search hwmon directory, try find temperature source
        snprintf(fileName, sizeof(fileName), "/sys/class/drm/card%i/device/hwmon", cardIdx);
        DIR *directory = opendir(fileName);
        if (directory) {
          dirent *ent;
          struct stat s;
          while ( (ent = readdir(directory)) != nullptr ) {
            char tiFileName[256];
            snprintf(tiFileName, sizeof(tiFileName), "/sys/class/drm/card%i/device/hwmon/%s/temp1_input", cardIdx, ent->d_name);
            if (stat(tiFileName, &s) == 0) {
              hwmonDir = fileName;
              hwmonDir.push_back('/');
              hwmonDir.append(ent->d_name);
              break;
            }
          }

          closedir(directory);
        }

        snprintf(fileName, sizeof(fileName), "/sys/kernel/debug/dri/%i/amdgpu_pm_info", cardIdx);
        FILE *pmInfoFile = fopen(fileName, "r");
        if (pmInfoFile) {
          fclose(pmInfoFile);
        } else {
          LOG_F(WARNING, "amdgpu: can't monitor core & memory clock and other info, run sudo chmod 755 /sys/kernel/debug for enable it");
        }

        // Read pwm1_max value
        std::string pwm1MaxPath = hwmonDir + "/pwm1_max";
        std::optional<int> pwm1Max;
        FILE *pwm1MaxFile = fopen(pwm1MaxPath.c_str(), "r");
        if (pwm1MaxFile) {
          int value;
          if (fscanf(pwm1MaxFile, "%d", &value) == 1)
            pwm1Max = value;
          fclose(pwm1MaxFile);
        }

        Device device;
        device.deviceId = cardIdx;
        device.tMon = hwmonDir + "/temp1_input";
        device.pwm1 = hwmonDir + "/pwm1";
        device.pwm1Enable = hwmonDir + "/pwm1_enable";
        device.pwm1Max = pwm1Max;
        device.pmInfo = fileName;
        device.lastUpdateTime = 0;
        device.sclk = std::optional<int>();
        device.mclk = std::optional<int>();
        device.activity = std::optional<int>();

        if (device.tMon.empty()) {
          LOG_F(WARNING, "amdgpu: can't monitor temperature of card %i\n", cardIdx);
        }
        _gpuMap.push_back(device);
      }
    }

    cardIdx++;
  }

  if (_gpuMap.size() != devicesNum) {
    LOG_F(ERROR, "amdgpu hardware monitor found different number of GPUs than OpenCL! Monitoring results may be incorrect");
  }
}

void HWMonAmdGpu::updateGPUInfo(Device &device)
{
  device.sclk = std::optional<int>();
  device.mclk = std::optional<int>();
  device.activity = std::optional<int>();

  FILE *hFile = fopen(device.pmInfo.c_str(), "r");
  if (hFile) {
    char line[256];

    while (fgets(line, sizeof(line), hFile)) {
      constexpr size_t tokensNum = 3;
      std::string token[tokensNum];

      size_t tokenIdx = 0;
      bool insideParenthess = false;
      const char *p = line;
      while (*p && tokenIdx < tokensNum) {
        if (!insideParenthess) {
          if (*p == '(') {
            insideParenthess = true;
          } else if (*p != ' ' && *p != '\t') {
            token[tokenIdx].push_back(*p);
          } else {
            if (!token[tokenIdx].empty())
              tokenIdx++;
          }
        } else {
          if (*p == ')') {
            insideParenthess = false;
            if (!token[tokenIdx].empty())
              tokenIdx++;
          } else {
            token[tokenIdx].push_back(*p);
          }
        }

        p++;
      }

      if (tokenIdx == 3) {
        if (token[2] == "SCLK" && token[1] == "MHz") {
          device.sclk = atoi(token[0].c_str());
        } else if (token[2] == "MCLK" && token[1] == "MHz") {
          device.mclk = atoi(token[0].c_str());
        } else if (token[0] == "GPU" && token[1] == "Load:") {
          device.activity = atoi(token[2].c_str());
        }
      }
    }

    fclose(hFile);
  }
}

bool HWMonAmdGpu::setCoreClock(int gpuIdx, int frequency)
{
  return false;
}

bool HWMonAmdGpu::setMemoryClock(int gpuIdx, int frequency)
{
  return false;
}

bool HWMonAmdGpu::setPowertune(int gpuIdx, int percent)
{
  return false;
}

bool HWMonAmdGpu::setFanSpeed(int gpuIdx, int percent)
{
  static const char *errorMessage = "amdgpu: can't change fan speed, run sudo chmod -R a+w /sys/class/drm/card*/device/hwmon/hwmon*/pwm1* for enable it";

  if (gpuIdx < _gpuMap.size()) {
    Device &device = _gpuMap[gpuIdx];
    if (!device.pwm1.empty()) {
      {
        // Write 1 to /sys/class/drm/card?/device/hwmon/hwmon?/pwm1_enable
        // for manual fan speed controlling
        FILE *hFile = fopen(device.pwm1Enable.c_str(), "w");
        if (!hFile) {
          LOG_F(ERROR, errorMessage);
          return false;
        }

        bool success = fputs("1\n", hFile) > 0;
        fclose(hFile);
        if (!success) {
          LOG_F(ERROR, errorMessage);
          return false;
        }
      }

      {
        // Write fan speed value
        int value = percent / 100.0 * device.pwm1Max.value();
        FILE *hFile = fopen(device.pwm1.c_str(), "w");
        if (!hFile) {
          LOG_F(ERROR, errorMessage);
          return false;
        }

        bool success = fprintf(hFile, "%d\n", value) > 0;
        fclose(hFile);
        if (!success) {
          LOG_F(ERROR, errorMessage);
          return false;
        }

        return true;
      }
    }
  }

  return false;
}

std::optional<int> HWMonAmdGpu::getCoreClock(int gpuIdx)
{
  if (gpuIdx < _gpuMap.size()) {
    Device &device = _gpuMap[gpuIdx];
    if (device.lastUpdateTime != time(nullptr))
      updateGPUInfo(device);
    return device.sclk;
  }
}

std::optional<int> HWMonAmdGpu::getMemoryClock(int gpuIdx)
{
  if (gpuIdx < _gpuMap.size()) {
    Device &device = _gpuMap[gpuIdx];
    if (device.lastUpdateTime != time(nullptr))
      updateGPUInfo(device);
    return device.mclk;
  }
}

std::optional<int> HWMonAmdGpu::getPowertune(int gpuIdx)
{
  return std::optional<int>();
}

std::optional<int> HWMonAmdGpu::getFanSpeed(int gpuIdx)
{
  if (gpuIdx < _gpuMap.size()) {
    Device &device = _gpuMap[gpuIdx];
    if (!device.pwm1.empty()) {
      int pwm1 = 0;
      bool result = false;
      FILE *hFile = fopen(device.pwm1.c_str(), "r");
      if (hFile) {
        result = fscanf(hFile, "%d", &pwm1);
        fclose(hFile);
      }

      return result ? pwm1 * 100 / device.pwm1Max.value() : std::optional<int>();
    }
  }

  return std::optional<int>();
}

std::optional<int> HWMonAmdGpu::getTemperature(int gpuIdx)
{
  if (gpuIdx < _gpuMap.size()) {
    Device &device = _gpuMap[gpuIdx];
    if (!device.tMon.empty()) {
      unsigned temperature = 0;
      bool result = false;
      FILE *hFile = fopen(device.tMon.c_str(), "r");
      if (hFile) {
        result = fscanf(hFile, "%d", &temperature);
        fclose(hFile);
      }

      return result ? temperature / 1000 : std::optional<int>();
    }
  }

  return std::optional<int>();
}

std::optional<int> HWMonAmdGpu::getActivity(int gpuIdx)
{
  if (gpuIdx < _gpuMap.size()) {
    Device &device = _gpuMap[gpuIdx];
    if (device.lastUpdateTime != time(nullptr))
      updateGPUInfo(device);
    return device.activity;
  }
}
