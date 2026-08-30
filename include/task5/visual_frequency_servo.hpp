#pragma once

#include <optional>
#include <vector>

namespace task5 {

struct VisualPhaseSample {
    double time_seconds = 0.0;
    int unwrapped_phase_code = 0;
};

struct VisualFrequencyTrim {
    double phase_rate_codes_per_second = 0.0;
    double output_correction_hz = 0.0;
    double rms_phase_codes = 99.0;
    double direction_consistency = 0.0;
};

struct PhaseTrackSample {
    double time_seconds = 0.0;
    double command_phase_rad = 0.0;
    double correlation = 0.0;
};

struct PhaseTrackFit {
    double frequency_error_hz = 0.0;
    double relative_phase_rad = 0.0;
    double rms = 1.0;
    double amplitude = 0.0;
};

std::optional<PhaseTrackFit> fit_phase_track(
    const std::vector<PhaseTrackSample>& samples,
    double reference_time_seconds,
    double maximum_frequency_error_hz = 2.0,
    double frequency_step_hz = 0.001);

std::optional<VisualFrequencyTrim> estimate_visual_frequency_trim(
    const std::vector<VisualPhaseSample>& samples,
    double minimum_span_seconds = 1.5,
    int minimum_phase_span_codes = 2,
    double maximum_correction_hz = 0.025);

int signed_phase_delta(int next_phase, int previous_phase);

}  // namespace task5
