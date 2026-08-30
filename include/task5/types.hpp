#pragma once

#include <opencv2/core.hpp>

#include <chrono>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace task5 {

using Clock = std::chrono::steady_clock;

enum class AutoMode : uint8_t {
    Line = 3,
    Circle = 4,
    FigureEight = 5,
};

struct Frame {
    cv::Mat bgr;
    Clock::time_point timestamp{};
    uint64_t sequence = 0;
};

struct GridGeometry {
    // Coordinates are in the original camera frame. The normalized output is
    // deliberately fixed so all later modules have stable dimensions.
    cv::Rect roi;
    cv::Mat homography;
    cv::Size normalized_size{640, 480};
    bool valid = false;
};

struct TraceObservation {
    cv::Mat normalized_bgr;
    cv::Mat mask;
    cv::Point2d center{};
    cv::Point2d amplitude{};
    double foreground_fraction = 0.0;
    double quality = 0.0;
    uint64_t frame_sequence = 0;
    Clock::time_point timestamp{};
};

struct ProbeSpec {
    // One monotonic vertical scan maps time to screen Y. The FPGA probe must
    // use the same duration; this value is never estimated from the camera.
    double scan_duration_s = 1.0e-3;
    double probe_frequency_hz = 2.0e3;
    int scan_count = 1;
};

struct PhaseCodeSpec {
    // The 10 ms frame is phase coherent for every legal n * 100 Hz input.
    // Marker 0 is the phase reference; the other starts encode n modulo
    // 7, 11 and 13. Their product 1001 covers bins 10 through 1000.
    double frame_period_s = 10.0e-3;
    double marker_duration_s = 78.125e-6;
    std::array<int, 4> divisors{1, 7, 11, 13};
    std::array<double, 4> band_centers{0.145, 0.395, 0.645, 0.895};
    double band_height = 0.19;
    int samples_per_band = 64;
    int phase_trials = 128;
    double center_x = -1.0;
    double amplitude_x = -1.0;
};

struct MultiRateSpec {
    // The four scans fill one coherent 10 ms frame. Short bands resolve high
    // frequencies spatially; long bands expose enough cycles at the low end.
    std::array<double, 4> durations_s{
        0.15625e-3, 0.625e-3, 2.5e-3, 6.71875e-3};
    std::array<double, 4> band_centers{0.25, 0.625, 0.8125, 0.9375};
    std::array<double, 4> band_heights{0.47, 0.22, 0.10, 0.10};
    std::array<int, 4> samples_per_band{200, 96, 44, 44};
    // Converts normalized camera Y distance to the effective FPGA scan time.
    // It is a fixed installation calibration once the 8x8 grid and scope
    // channel sensitivities have been set.
    double visual_time_scale = 1.0;
    double target_cycles = 10.0;
    double minimum_cycles = 1.3;
    double maximum_cycles = 20.0;
};

struct FrequencyEstimate {
    bool valid = false;
    double frequency_hz = 0.0;
    double runner_up_hz = 0.0;
    double score = 0.0;
    double runner_up_score = 0.0;
    double margin = 0.0;
    double fitted_phase_rad = 0.0;
    int valid_samples = 0;
    int observed_cycles = 0;
    int probe_index = -1;
    int candidate_first_bin = 0;
    std::vector<float> candidate_scores;
    std::string strategy;
    bool visual_geometry_valid = false;
    double visual_scale = 1.0;
    double visual_offset = 0.0;
    double visual_residual_px = 0.0;
    double geometry_alternate_hz = 0.0;
    bool geometry_alternate_scaled = false;
};

struct PhaseSample {
    Clock::time_point timestamp{};
    double phase_rad = 0.0;
    double confidence = 0.0;
};

struct PhaseRateEstimate {
    bool valid = false;
    double delta_frequency_hz = 0.0;
    double phase_rad = 0.0;
    double residual_rad = 0.0;
    int samples = 0;
};

struct DdsCommand {
    uint32_t ftw = 0;
    uint8_t phase = 0;
    uint8_t amplitude = 102;
    uint8_t flags = 0;
};

struct FineDdsCommand {
    // 32 integer FTW bits followed by 8 fractional bits.
    uint64_t ftw_q8 = 0;
    uint8_t phase = 0;
    uint8_t amplitude = 102;
};

struct FpgaResponse {
    uint8_t mode = 0;
    uint8_t flags = 0;
    uint32_t payload = 0;
    uint8_t phase = 0;
    uint8_t amplitude = 0;
    uint8_t command = 0;
    bool checksum_valid = false;

    bool phase_code_mode() const { return (flags & 0x80u) != 0; }
    bool calibration_valid() const { return (flags & 0x80u) != 0; }
    uint16_t calibration_cycle_count() const {
        return static_cast<uint16_t>(phase) |
               (static_cast<uint16_t>(amplitude) << 8);
    }
};

}  // namespace task5
