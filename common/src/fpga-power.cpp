#include "fpga-power.h"

FpgaPowerMonitor::FpgaPowerMonitor() : isMonitoring(false), debug(false) {}

FpgaPowerMonitor::~FpgaPowerMonitor() {
    if (isMonitoring) stopMonitoring();
}

void FpgaPowerMonitor::startMonitoring(const int id, bool debugMode) {
    if (isMonitoring) return;
    isMonitoring = true;
    device = xrt::device(id);
    debug = debugMode;
    monitoringThread = std::thread(&FpgaPowerMonitor::monitorPower, this);
}

void FpgaPowerMonitor::stopMonitoring() {
    if (!isMonitoring) return;
    isMonitoring = false;
    if (monitoringThread.joinable()) monitoringThread.join();
}

float FpgaPowerMonitor::getAveragePower(size_t& numSamples) const {
    std::lock_guard<std::mutex> lock(dataMutex);
    numSamples = powerSamples.size();
    if (numSamples == 0) return 0.0f;
    float totalPower = std::accumulate(powerSamples.begin(), powerSamples.end(), 0.0f);
    return totalPower / numSamples;
}

float FpgaPowerMonitor::getMaxPower() const {
    std::lock_guard<std::mutex> lock(dataMutex);
    if (powerSamples.empty()) return 0.0f;
    return *std::max_element(powerSamples.begin(), powerSamples.end());
}

void FpgaPowerMonitor::monitorPower() {
    // Polling Frequency is 1Hz
    while (isMonitoring) {
        try {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            std::string output = device.get_info<xrt::info::device::electrical>();
            float power = extractPowerConsumption(output);
            std::lock_guard<std::mutex> lock(dataMutex);
            if (debug)
                std::cout << "sample: " << power << std::endl;
            powerSamples.push_back(power);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        } catch (const std::exception& e) {
            std::cerr << "Error retrieving power info: " << e.what() << std::endl;
        }
    }
}

float FpgaPowerMonitor::extractPowerConsumption(const std::string& jsonStr) {
    try {
        json jsonObj = json::parse(jsonStr);
        return std::stof(jsonObj["power_consumption_watts"].get<std::string>());
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return -1.0f;
    }
}