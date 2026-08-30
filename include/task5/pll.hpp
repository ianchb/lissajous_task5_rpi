#pragma once

#include "task5/types.hpp"

namespace task5 {

struct PllConfig {
    double proportional_gain = 0.38;
    double integral_gain = 0.015;
    double maximum_frequency_correction_hz = 20.0;
    double maximum_phase_step_rad = 0.20;
};

struct PllOutput {
    double phase_correction_rad = 0.0;
    double frequency_correction_hz = 0.0;
    bool locked = false;
};

class PhasePll {
public:
    explicit PhasePll(PllConfig config = {});
    void reset();
    PllOutput update(double measured_phase_error_rad,
                     double confidence,
                     double dt_s);
    PllOutput output() const;

private:
    PllConfig config_;
    double phase_correction_rad_ = 0.0;
    double frequency_correction_hz_ = 0.0;
    int good_updates_ = 0;
    int bad_updates_ = 0;
};

}  // namespace task5
