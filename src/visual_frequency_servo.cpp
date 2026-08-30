#include "task5/visual_frequency_servo.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace task5 {
namespace {
constexpr double kTwoPi = 6.28318530717958647692;
}

std::optional<PhaseTrackFit> fit_phase_track(
    const std::vector<PhaseTrackSample>& samples,
    double reference_time_seconds, double maximum_frequency_error_hz,
    double frequency_step_hz) {
    if (samples.size() < 8 || !std::isfinite(reference_time_seconds) ||
        maximum_frequency_error_hz <= 0.0 || frequency_step_hz <= 0.0)
        return std::nullopt;

    bool distinct_phase = false;
    for (const PhaseTrackSample& sample : samples) {
        if (!std::isfinite(sample.time_seconds) ||
            !std::isfinite(sample.command_phase_rad) ||
            !std::isfinite(sample.correlation))
            return std::nullopt;
        if (std::abs(sample.command_phase_rad -
                     samples.front().command_phase_rad) > 0.1)
            distinct_phase = true;
    }
    if (!distinct_phase) return std::nullopt;

    const double time_origin = samples.front().time_seconds;
    double best_cost = std::numeric_limits<double>::infinity();
    PhaseTrackFit best;
    const int steps = static_cast<int>(std::ceil(
        2.0 * maximum_frequency_error_hz / frequency_step_hz));
    for (int index = 0; index <= steps; ++index) {
        const double rate = -maximum_frequency_error_hz +
            index * (2.0 * maximum_frequency_error_hz / steps);
        double cc = 0.0, ss = 0.0, cs = 0.0;
        double cy = 0.0, sy = 0.0, yy = 0.0;
        for (const PhaseTrackSample& sample : samples) {
            const double argument = kTwoPi * rate *
                (sample.time_seconds - time_origin) +
                sample.command_phase_rad;
            const double c = std::cos(argument);
            const double s = std::sin(argument);
            const double y = std::clamp(sample.correlation, -1.0, 1.0);
            cc += c * c;
            ss += s * s;
            cs += c * s;
            cy += c * y;
            sy += s * y;
            yy += y * y;
        }
        const double determinant = cc * ss - cs * cs;
        if (std::abs(determinant) <= 1.0e-9) continue;
        const double coefficient_a = (cy * ss - sy * cs) / determinant;
        const double coefficient_b = (sy * cc - cy * cs) / determinant;
        const double amplitude = std::hypot(coefficient_a, coefficient_b);
        const double squared_error = std::max(0.0,
            (yy - 2.0 * (coefficient_a * cy + coefficient_b * sy) +
             coefficient_a * coefficient_a * cc +
             2.0 * coefficient_a * coefficient_b * cs +
             coefficient_b * coefficient_b * ss) / samples.size());
        const double cost = squared_error +
            0.03 * (amplitude - 1.0) * (amplitude - 1.0);
        if (cost >= best_cost) continue;
        best_cost = cost;
        best.frequency_error_hz = rate;
        best.rms = std::sqrt(squared_error);
        best.amplitude = amplitude;
        const double phase_at_origin = std::atan2(
            -coefficient_b, coefficient_a);
        const double phase_at_reference = phase_at_origin + kTwoPi * rate *
            (reference_time_seconds - time_origin);
        best.relative_phase_rad = std::atan2(
            std::sin(phase_at_reference), std::cos(phase_at_reference));
    }
    if (!std::isfinite(best_cost)) return std::nullopt;
    return best;
}

std::optional<VisualFrequencyTrim> estimate_visual_frequency_trim(
    const std::vector<VisualPhaseSample>& samples,
    double minimum_span_seconds, int minimum_phase_span_codes,
    double maximum_correction_hz) {
    if (samples.size() < 2 || minimum_span_seconds <= 0.0 ||
        minimum_phase_span_codes <= 0 || maximum_correction_hz <= 0.0)
        return std::nullopt;

    const double span = samples.back().time_seconds -
                        samples.front().time_seconds;
    const int phase_span = samples.back().unwrapped_phase_code -
                           samples.front().unwrapped_phase_code;
    if (span < minimum_span_seconds ||
        std::abs(phase_span) < minimum_phase_span_codes)
        return std::nullopt;

    double mean_time = 0.0;
    double mean_phase = 0.0;
    for (const VisualPhaseSample& sample : samples) {
        if (!std::isfinite(sample.time_seconds)) return std::nullopt;
        mean_time += sample.time_seconds;
        mean_phase += sample.unwrapped_phase_code;
    }
    mean_time /= samples.size();
    mean_phase /= samples.size();

    double covariance = 0.0;
    double time_variance = 0.0;
    for (const VisualPhaseSample& sample : samples) {
        const double dt = sample.time_seconds - mean_time;
        covariance += dt * (sample.unwrapped_phase_code - mean_phase);
        time_variance += dt * dt;
    }
    if (time_variance <= 1.0e-9) return std::nullopt;

    const double phase_rate = covariance / time_variance;
    const double phase_intercept = mean_phase - phase_rate * mean_time;
    double squared_residual = 0.0;
    int direction_matches = 0;
    int direction_trials = 0;
    const int expected_direction = phase_rate > 0.0 ? 1 : -1;
    for (size_t index = 0; index < samples.size(); ++index) {
        const double fitted = phase_intercept +
                              phase_rate * samples[index].time_seconds;
        const double error = samples[index].unwrapped_phase_code - fitted;
        squared_residual += error * error;
        if (index == 0) continue;
        const int delta = samples[index].unwrapped_phase_code -
                          samples[index - 1].unwrapped_phase_code;
        if (delta == 0) continue;
        ++direction_trials;
        if ((delta > 0 ? 1 : -1) == expected_direction)
            ++direction_matches;
    }
    const double rms_phase_codes = std::sqrt(
        squared_residual / static_cast<double>(samples.size()));
    const double direction_consistency = direction_trials == 0 ? 0.0 :
        static_cast<double>(direction_matches) / direction_trials;
    const double residual_limit = std::max(
        1.25, 0.30 * std::abs(static_cast<double>(phase_span)));
    if (rms_phase_codes > residual_limit || direction_consistency < 0.70)
        return std::nullopt;
    // The DDS adds phase_offset to its accumulator. A positive corrective
    // phase slope therefore means that its output is slow by slope/256 Hz.
    const double correction = std::clamp(
        phase_rate / 256.0, -maximum_correction_hz,
        maximum_correction_hz);
    if (!std::isfinite(correction)) return std::nullopt;
    return VisualFrequencyTrim{phase_rate, correction, rms_phase_codes,
                               direction_consistency};
}

int signed_phase_delta(int next_phase, int previous_phase) {
    int delta = ((next_phase & 0xff) - (previous_phase & 0xff)) & 0xff;
    if (delta >= 128) delta -= 256;
    return delta;
}

}  // namespace task5
