#pragma once

#include "task5/types.hpp"

#include <deque>

namespace task5 {

class FrequencyObserver {
public:
    FrequencyObserver(double minimum_hz = 1'000.0,
                      double maximum_hz = 100'000.0,
                      double legal_step_hz = 100.0);

    // Estimate the source from a probe frame in which Y is a monotonic ramp.
    // The ramp duration is known from the FPGA command, so no camera timing
    // assumption enters the estimator.
    FrequencyEstimate observe_ramp(const TraceObservation& observation,
                                   const ProbeSpec& probe) const;

    // Jointly fit the four spatially separated phase-code bands. This is the
    // final 100 Hz-bin discriminator; ramp density is only a range selector.
    FrequencyEstimate observe_phase_code(const TraceObservation& observation,
                                         const PhaseCodeSpec& code) const;

    FrequencyEstimate observe_multirate(
        const TraceObservation& observation,
        const MultiRateSpec& spec) const;

    void reset();
    void update_posterior(const FrequencyEstimate& estimate,
                          Clock::time_point timestamp);
    FrequencyEstimate posterior() const;

private:
    struct Candidate {
        double frequency_hz = 0.0;
        double log_likelihood = 0.0;
    };

    struct ScoredFrame {
        Clock::time_point timestamp{};
        int probe_index = -1;
        int first_bin = 0;
        std::vector<float> scores;
    };

    std::vector<double> sample_ramp(const cv::Mat& mask,
                                     int sample_count,
                                     int* valid_samples) const;
    double fit_candidate(const std::vector<double>& samples,
                         double scan_duration_s,
                         double frequency_hz,
                         double* phase_rad) const;
    std::vector<std::vector<double>> sample_phase_bands(
        const cv::Mat& mask, const PhaseCodeSpec& code) const;
    std::vector<std::vector<double>> sample_multirate_bands(
        const cv::Mat& mask, const MultiRateSpec& spec) const;

    double minimum_hz_;
    double maximum_hz_;
    double legal_step_hz_;
    std::deque<Candidate> posterior_history_;
    std::deque<ScoredFrame> scored_history_;
};

class PhaseSlopeEstimator {
public:
    explicit PhaseSlopeEstimator(size_t maximum_samples = 30);
    void reset();
    void add(PhaseSample sample);
    PhaseRateEstimate estimate() const;

private:
    size_t maximum_samples_;
    std::deque<PhaseSample> samples_;
};

}  // namespace task5
