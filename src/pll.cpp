#include "task5/pll.hpp"

#include <algorithm>
#include <cmath>

namespace task5 {
namespace {
constexpr double kPi = 3.14159265358979323846;

double wrap_phase(double value) {
    while (value > kPi) value -= 2.0 * kPi;
    while (value < -kPi) value += 2.0 * kPi;
    return value;
}
}  // namespace

PhasePll::PhasePll(PllConfig config) : config_(config) {}

void PhasePll::reset() {
    phase_correction_rad_ = 0.0;
    frequency_correction_hz_ = 0.0;
    good_updates_ = 0;
    bad_updates_ = 0;
}

PllOutput PhasePll::update(double measured_phase_error_rad,
                           double confidence,
                           double dt_s) {
    PllOutput result = output();
    if (dt_s <= 0.0 || confidence < 0.45 ||
        !std::isfinite(measured_phase_error_rad)) {
        bad_updates_++;
        good_updates_ = 0;
        result.locked = false;
        return result;
    }
    const double error = wrap_phase(measured_phase_error_rad);
    frequency_correction_hz_ = std::clamp(
        frequency_correction_hz_ +
            config_.integral_gain * error * dt_s /
                (2.0 * 3.14159265358979323846),
        -config_.maximum_frequency_correction_hz,
        config_.maximum_frequency_correction_hz);
    const double step = std::clamp(config_.proportional_gain * error,
                                   -config_.maximum_phase_step_rad,
                                   config_.maximum_phase_step_rad);
    phase_correction_rad_ = wrap_phase(phase_correction_rad_ + step);
    if (std::abs(error) < 0.16 && confidence > 0.7) {
        ++good_updates_;
        bad_updates_ = 0;
    } else {
        good_updates_ = 0;
    }
    if (bad_updates_ > 3) good_updates_ = 0;
    result = output();
    result.locked = good_updates_ >= 8;
    return result;
}

PllOutput PhasePll::output() const {
    return {phase_correction_rad_, frequency_correction_hz_,
            good_updates_ >= 8 && bad_updates_ == 0};
}

}  // namespace task5
