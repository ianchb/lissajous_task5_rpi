#include "task5/runtime_controller.hpp"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

namespace {

std::atomic_bool stop_requested{false};

void handle_signal(int) { stop_requested.store(true); }

std::optional<task5::RuntimeConfig> parse(int argc, char** argv) {
    task5::RuntimeConfig config;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (index + 1 >= argc) return std::nullopt;
        const std::string value(argv[++index]);
        if (argument == "--camera") config.camera = value;
        else if (argument == "--serial") config.serial = value;
        else if (argument == "--frequency-windows")
            config.frequency_windows = std::stoi(value);
        else if (argument == "--frequency-window-ms")
            config.frequency_window_ms = std::stoi(value);
        else if (argument == "--shape-settle-ms")
            config.shape_settle_ms = std::stoi(value);
        else if (argument == "--grid-exposures") {
            config.grid_exposures.clear();
            std::istringstream stream(value);
            std::string item;
            while (std::getline(stream, item, ',')) {
                if (item.empty()) return std::nullopt;
                config.grid_exposures.push_back(std::stoi(item));
            }
        }
        else if (argument == "--abba-probe-frequency-hz")
            config.abba_probe_frequency_hz = std::stod(value);
        else if (argument == "--abba-center-phase")
            config.abba_center_phase = std::stoi(value);
        else if (argument == "--abba-delta-phase")
            config.abba_delta_phase = std::stoi(value);
        else if (argument == "--abba-settle-ms")
            config.abba_settle_ms = std::stoi(value);
        else if (argument == "--abba-measure-ms")
            config.abba_measure_ms = std::stoi(value);
        else if (argument == "--abba-rounds")
            config.abba_rounds = std::stoi(value);
        else if (argument == "--control-probe-frequency-hz")
            config.control_probe_frequency_hz = std::stod(value);
        else if (argument == "--control-probe-suppress-frequency-trim")
            config.control_probe_suppress_frequency_trim =
                std::stoi(value) != 0;
        else if (argument == "--abba-grid-roi") {
            std::istringstream stream(value);
            cv::Rect roi;
            char comma1 = 0, comma2 = 0, comma3 = 0;
            if (!(stream >> roi.x >> comma1 >> roi.y >> comma2 >>
                  roi.width >> comma3 >> roi.height) ||
                comma1 != ',' || comma2 != ',' || comma3 != ',' ||
                stream.peek() != std::char_traits<char>::eof())
                return std::nullopt;
            config.abba_grid_roi = roi;
        }
        else return std::nullopt;
    }
    if (config.frequency_windows < 3 || config.frequency_window_ms < 300 ||
        config.shape_settle_ms < 100 || config.shape_settle_ms > 2'000 ||
        config.grid_exposures.empty() ||
        std::any_of(config.grid_exposures.begin(),
                    config.grid_exposures.end(),
                    [](int exposure) {
                        return exposure < 1 || exposure > 5'000;
                    }))
        return std::nullopt;
    if (config.abba_probe_frequency_hz < 0.0 ||
        config.abba_probe_frequency_hz > 100'000.0 ||
        config.control_probe_frequency_hz < 0.0 ||
        config.control_probe_frequency_hz > 100'000.0 ||
        config.abba_delta_phase < 1 || config.abba_delta_phase > 48 ||
        config.abba_settle_ms < 100 || config.abba_measure_ms < 100 ||
        config.abba_rounds < 1 || config.abba_rounds > 4)
        return std::nullopt;
    if (config.abba_grid_roi &&
        (std::max(config.abba_probe_frequency_hz,
                  config.control_probe_frequency_hz) <= 0.0 ||
         config.abba_grid_roi->width < 32 ||
         config.abba_grid_roi->height < 32))
        return std::nullopt;
    if (config.control_probe_suppress_frequency_trim &&
        config.control_probe_frequency_hz <= 0.0)
        return std::nullopt;
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    const auto config = parse(argc, argv);
    if (!config) {
        std::cerr << "usage: " << argv[0]
                  << " [--camera /dev/video19] [--serial /dev/ttyUSB0]"
                     " [--frequency-windows 3] [--frequency-window-ms 1000]"
                     " [--shape-settle-ms 600]"
                     " [--grid-exposures 40,60,80]"
                     " [--abba-probe-frequency-hz 50000]"
                     " [--abba-center-phase 0] [--abba-delta-phase 12]"
                     " [--abba-settle-ms 300] [--abba-measure-ms 240]"
                     " [--abba-rounds 1] [--control-probe-frequency-hz 50000]"
                     " [--control-probe-suppress-frequency-trim 1]"
                     " [--abba-grid-roi x,y,w,h]\n";
        return 2;
    }
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    task5::RuntimeController controller(*config);
    return controller.run(&stop_requested);
}
