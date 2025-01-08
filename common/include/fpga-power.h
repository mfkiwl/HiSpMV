#ifndef FPGA_POWER_MONITOR_H
#define FPGA_POWER_MONITOR_H

#include <xrt/xrt_device.h>
#include <nlohmann/json.hpp>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <string>

using json = nlohmann::json;

class FpgaPowerMonitor {
public:
    FpgaPowerMonitor();
    ~FpgaPowerMonitor();

    void startMonitoring(const int id, bool debugMode = false);
    void stopMonitoring();
    float getAveragePower(size_t& numSamples) const;
    float getMaxPower() const;

private:
    void monitorPower();
    float extractPowerConsumption(const std::string& jsonStr);

    std::atomic<bool> isMonitoring;
    std::thread monitoringThread;
    xrt::device device;
    mutable std::mutex dataMutex;

    std::vector<float> powerSamples;
    bool debug;
};
#endif // FPGA_POWER_MONITOR_H