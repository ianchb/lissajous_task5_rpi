#include "task5/clock_calibration.hpp"

#include <cmath>

namespace task5 {

std::optional<ClockCalibration> derive_clock_calibration(
    const FpgaResponse& response, double nominal_sample_clock_hz) {
    if (response.command != 0x03 || !response.checksum_valid ||
        !response.calibration_valid() || response.payload < 100'000'000u ||
        response.calibration_cycle_count() == 0 ||
        !std::isfinite(nominal_sample_clock_hz) ||
        nominal_sample_clock_hz <= 0.0) {
        return std::nullopt;
    }

    ClockCalibration result;
    result.sample_sum_q8 = response.payload;
    result.cycle_count = response.calibration_cycle_count();
    result.samples_per_cycle =
        static_cast<double>(result.sample_sum_q8) /
        (256.0 * result.cycle_count);
    result.measured_frequency_hz =
        nominal_sample_clock_hz / result.samples_per_cycle;
    result.source_grid_frequency_hz =
        std::round(result.measured_frequency_hz / 100.0) * 100.0;
    result.residual_hz = result.measured_frequency_hz -
                         result.source_grid_frequency_hz;
    result.effective_dds_clock_hz =
        result.source_grid_frequency_hz * result.samples_per_cycle;

    if (result.source_grid_frequency_hz < 1'000.0 ||
        result.source_grid_frequency_hz > 100'000.0 ||
        std::abs(result.residual_hz) >= 25.0 ||
        result.effective_dds_clock_hz < 19'900'000.0 ||
        result.effective_dds_clock_hz > 20'100'000.0) {
        return std::nullopt;
    }
    return result;
}

}  // namespace task5
