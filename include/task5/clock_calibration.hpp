#pragma once

#include "task5/types.hpp"

#include <optional>

namespace task5 {

struct ClockCalibration {
    uint32_t sample_sum_q8 = 0;
    uint16_t cycle_count = 0;
    double samples_per_cycle = 0.0;
    double measured_frequency_hz = 0.0;
    double source_grid_frequency_hz = 0.0;
    double residual_hz = 0.0;
    double effective_dds_clock_hz = 0.0;
};

std::optional<ClockCalibration> derive_clock_calibration(
    const FpgaResponse& response,
    double nominal_sample_clock_hz = 20'000'000.0);

}  // namespace task5
