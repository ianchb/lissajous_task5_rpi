#include "task5/frequency_observer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace task5 {
namespace {
constexpr double kTwoPi = 2.0 * M_PI;

double unwrap_near(double value, double reference) {
    while (value - reference > M_PI) value -= kTwoPi;
    while (value - reference < -M_PI) value += kTwoPi;
    return value;
}

}  // namespace

FrequencyObserver::FrequencyObserver(double minimum_hz,
                                     double maximum_hz,
                                     double legal_step_hz)
    : minimum_hz_(minimum_hz),
      maximum_hz_(maximum_hz),
      legal_step_hz_(legal_step_hz) {}

std::vector<double> FrequencyObserver::sample_ramp(const cv::Mat& mask,
                                                    int sample_count,
                                                    int* valid_samples) const {
    std::vector<double> result(static_cast<size_t>(sample_count),
                               std::numeric_limits<double>::quiet_NaN());
    std::vector<cv::Point> foreground;
    cv::findNonZero(mask, foreground);
    if (foreground.empty()) {
        if (valid_samples) *valid_samples = 0;
        return result;
    }
    const cv::Rect bounds = cv::boundingRect(foreground);
    int valid = 0;
    for (int i = 0; i < sample_count; ++i) {
        const int row = std::clamp(
            bounds.y + static_cast<int>(std::lround(
                           (i + 0.5) * bounds.height / sample_count)),
            0, mask.rows - 1);
        std::vector<int> columns;
        for (int x = 0; x < mask.cols; ++x) {
            if (mask.at<uint8_t>(row, x) != 0) columns.push_back(x);
        }
        if (columns.size() < 2) continue;
        const auto middle = columns.begin() + columns.size() / 2;
        std::nth_element(columns.begin(), middle, columns.end());
        result[static_cast<size_t>(i)] =
            (*middle - 0.5 * mask.cols) / (0.5 * mask.cols);
        ++valid;
    }
    // Small gaps are expected at the ends of a scope trace. Interpolate only
    // between valid rows; do not invent a complete curve from one blob.
    for (int i = 0; i < sample_count; ++i) {
        if (std::isfinite(result[static_cast<size_t>(i)])) continue;
        int left = i - 1;
        while (left >= 0 && !std::isfinite(result[static_cast<size_t>(left)])) --left;
        int right = i + 1;
        while (right < sample_count &&
               !std::isfinite(result[static_cast<size_t>(right)])) {
            ++right;
        }
        if (left >= 0 && right < sample_count && right - left <= sample_count / 5) {
            const double alpha = static_cast<double>(i - left) / (right - left);
            result[static_cast<size_t>(i)] =
                (1.0 - alpha) * result[static_cast<size_t>(left)] +
                alpha * result[static_cast<size_t>(right)];
            ++valid;
        }
    }
    if (valid_samples) *valid_samples = valid;
    return result;
}

double FrequencyObserver::fit_candidate(const std::vector<double>& samples,
                                         double scan_duration_s,
                                         double frequency_hz,
                                         double* phase_rad) const {
    double sum = 0.0;
    int count = 0;
    for (double value : samples) {
        if (std::isfinite(value)) {
            sum += value;
            ++count;
        }
    }
    if (count < 24) return std::numeric_limits<double>::infinity();
    const double mean = sum / count;
    double cc = 0.0;
    double ss = 0.0;
    double yy = 0.0;
    const double denominator = std::max(1, static_cast<int>(samples.size()) - 1);
    for (size_t i = 0; i < samples.size(); ++i) {
        const double value = samples[i];
        if (!std::isfinite(value)) continue;
        const double t = scan_duration_s * static_cast<double>(i) / denominator;
        const double angle = kTwoPi * frequency_hz * t;
        const double centered = value - mean;
        cc += centered * std::cos(angle);
        ss += centered * std::sin(angle);
        yy += centered * centered;
    }
    const double amplitude = std::hypot(cc, ss);
    if (phase_rad) *phase_rad = std::atan2(-ss, cc);
    // Normalized unexplained energy. The legal candidate grid is applied only
    // after this continuous sinusoidal fit, so neighbouring bins can be
    // compared by a common score rather than by an arbitrary threshold.
    const double explained = amplitude * amplitude * 2.0 / count;
    return std::max(0.0, yy - explained) / std::max(1.0, yy);
}

FrequencyEstimate FrequencyObserver::observe_ramp(
    const TraceObservation& observation, const ProbeSpec& probe) const {
    FrequencyEstimate estimate;
    estimate.strategy = "ramp_sinusoid_fit";
    if (observation.mask.empty() || observation.quality < 0.08 ||
        probe.scan_duration_s <= 0.0) {
        return estimate;
    }
    int valid_samples = 0;
    const std::vector<double> samples = sample_ramp(observation.mask, 256,
                                                     &valid_samples);
    estimate.valid_samples = valid_samples;
    if (valid_samples < 48) return estimate;

    const int first_bin = static_cast<int>(std::ceil(
        minimum_hz_ / legal_step_hz_ - 1.0e-9));
    const int last_bin = static_cast<int>(std::floor(
        maximum_hz_ / legal_step_hz_ + 1.0e-9));
    double best_score = std::numeric_limits<double>::infinity();
    double runner_score = std::numeric_limits<double>::infinity();
    int best_bin = first_bin;
    int runner_bin = first_bin;
    double best_phase = 0.0;
    for (int bin = first_bin; bin <= last_bin; ++bin) {
        double phase = 0.0;
        const double score = fit_candidate(samples, probe.scan_duration_s,
                                            bin * legal_step_hz_, &phase);
        if (score < best_score) {
            runner_score = best_score;
            runner_bin = best_bin;
            best_score = score;
            best_bin = bin;
            best_phase = phase;
        } else if (score < runner_score) {
            runner_score = score;
            runner_bin = bin;
        }
    }
    estimate.valid = std::isfinite(best_score);
    estimate.frequency_hz = best_bin * legal_step_hz_;
    estimate.runner_up_hz = runner_bin * legal_step_hz_;
    estimate.score = best_score;
    estimate.runner_up_score = runner_score;
    estimate.margin = runner_score - best_score;
    estimate.fitted_phase_rad = best_phase;
    estimate.observed_cycles = static_cast<int>(std::lround(
        estimate.frequency_hz * probe.scan_duration_s));
    return estimate;
}

std::vector<std::vector<double>> FrequencyObserver::sample_phase_bands(
    const cv::Mat& mask, const PhaseCodeSpec& code) const {
    std::vector<std::vector<double>> bands;
    bands.reserve(code.band_centers.size());
    for (double center : code.band_centers) {
        std::vector<double> samples(
            static_cast<size_t>(code.samples_per_band),
            std::numeric_limits<double>::quiet_NaN());
        const double top = center - 0.5 * code.band_height;
        for (int index = 0; index < code.samples_per_band; ++index) {
            const double fraction = top + code.band_height *
                (index + 0.5) / code.samples_per_band;
            const int row = std::clamp(
                static_cast<int>(std::lround(fraction * (mask.rows - 1))),
                0, mask.rows - 1);
            std::vector<int> columns;
            for (int x = 0; x < mask.cols; ++x) {
                if (mask.at<uint8_t>(row, x) != 0) columns.push_back(x);
            }
            if (columns.size() < 2) continue;
            const auto middle = columns.begin() + columns.size() / 2;
            std::nth_element(columns.begin(), middle, columns.end());
            samples[static_cast<size_t>(index)] = *middle;
        }
        bands.push_back(std::move(samples));
    }
    return bands;
}

std::vector<std::vector<double>> FrequencyObserver::sample_multirate_bands(
    const cv::Mat& mask, const MultiRateSpec& spec) const {
    std::vector<std::vector<double>> bands;
    bands.reserve(spec.band_centers.size());
    for (size_t band_index = 0; band_index < spec.band_centers.size();
         ++band_index) {
        const int sample_count = spec.samples_per_band[band_index];
        std::vector<double> samples(
            static_cast<size_t>(sample_count),
            std::numeric_limits<double>::quiet_NaN());
        const double top = spec.band_centers[band_index] -
                           0.5 * spec.band_heights[band_index];
        for (int index = 0; index < sample_count; ++index) {
            const double fraction = top + spec.band_heights[band_index] *
                (index + 0.5) / sample_count;
            const int row = std::clamp(
                static_cast<int>(std::lround(fraction * (mask.rows - 1))),
                0, mask.rows - 1);
            std::vector<int> columns;
            for (int x = 0; x < mask.cols; ++x) {
                if (mask.at<uint8_t>(row, x) != 0) columns.push_back(x);
            }
            if (columns.size() < 2) continue;
            const auto middle = columns.begin() + columns.size() / 2;
            std::nth_element(columns.begin(), middle, columns.end());
            samples[static_cast<size_t>(index)] =
                (*middle - 0.5 * mask.cols) / (0.5 * mask.cols);
        }
        bands.push_back(std::move(samples));
    }
    return bands;
}

FrequencyEstimate FrequencyObserver::observe_multirate(
    const TraceObservation& observation, const MultiRateSpec& spec) const {
    FrequencyEstimate estimate;
    estimate.strategy = "full_duty_multirate_scan";
    if (observation.mask.empty() || observation.quality < 0.08 ||
        spec.target_cycles <= 0.0 || spec.minimum_cycles <= 0.0 ||
        spec.maximum_cycles <= spec.minimum_cycles) {
        return estimate;
    }
    const auto bands = sample_multirate_bands(observation.mask, spec);
    std::array<int, 4> valid_samples{};
    for (size_t band = 0; band < bands.size(); ++band) {
        valid_samples[band] = static_cast<int>(std::count_if(
            bands[band].begin(), bands[band].end(), [](double value) {
                return std::isfinite(value);
            }));
    }

    int selected_band = -1;
    double selected_distance = std::numeric_limits<double>::infinity();
    for (size_t band = 0; band < bands.size(); ++band) {
        if (valid_samples[band] <
            std::max(24, spec.samples_per_band[band] / 2)) {
            continue;
        }
        double sum = 0.0;
        for (double value : bands[band]) {
            if (std::isfinite(value)) sum += value;
        }
        const double mean = sum / valid_samples[band];
        int previous_sign = 0;
        int transitions = 0;
        constexpr double kCrossingHysteresis = 0.05;
        for (double value : bands[band]) {
            if (!std::isfinite(value)) continue;
            int sign = 0;
            if (value > mean + kCrossingHysteresis) sign = 1;
            if (value < mean - kCrossingHysteresis) sign = -1;
            if (sign == 0) continue;
            if (previous_sign != 0 && sign != previous_sign) ++transitions;
            previous_sign = sign;
        }
        const double observed_cycles = 0.5 * transitions;
        const double spatial_limit = 0.25 * valid_samples[band];

        std::vector<double> row_occupancies;
        row_occupancies.reserve(
            static_cast<size_t>(spec.samples_per_band[band]));
        const double top = spec.band_centers[band] -
                           0.5 * spec.band_heights[band];
        for (int index = 0; index < spec.samples_per_band[band]; ++index) {
            const double fraction = top + spec.band_heights[band] *
                (index + 0.5) / spec.samples_per_band[band];
            const int row = std::clamp(
                static_cast<int>(std::lround(
                    fraction * (observation.mask.rows - 1))),
                0, observation.mask.rows - 1);
            row_occupancies.push_back(
                static_cast<double>(cv::countNonZero(
                    observation.mask.row(row))) /
                observation.mask.cols);
        }
        const auto middle = row_occupancies.begin() +
                            row_occupancies.size() / 2;
        std::nth_element(row_occupancies.begin(), middle,
                         row_occupancies.end());
        const double median_occupancy = *middle;
        if (observed_cycles >= spec.minimum_cycles &&
            observed_cycles <= spec.maximum_cycles &&
            observed_cycles <= spatial_limit && median_occupancy <= 0.55) {
            const double distance = std::abs(
                std::log(observed_cycles / spec.target_cycles));
            if (distance < selected_distance) {
                selected_distance = distance;
                selected_band = static_cast<int>(band);
            }
        }
    }
    if (selected_band < 0) return estimate;

    const int first_bin = static_cast<int>(std::ceil(
        minimum_hz_ / legal_step_hz_ - 1.0e-9));
    const int last_bin = static_cast<int>(std::floor(
        maximum_hz_ / legal_step_hz_ + 1.0e-9));
    estimate.probe_index = selected_band;
    estimate.candidate_first_bin = first_bin;
    estimate.candidate_scores.assign(
        static_cast<size_t>(last_bin - first_bin + 1),
        std::numeric_limits<float>::infinity());
    double best_score = std::numeric_limits<double>::infinity();
    double runner_score = std::numeric_limits<double>::infinity();
    int best_bin = first_bin;
    int runner_bin = first_bin;
    double best_phase = 0.0;
    for (int bin = first_bin; bin <= last_bin; ++bin) {
        const double frequency_hz = bin * legal_step_hz_;
        const double effective_duration =
            spec.durations_s[static_cast<size_t>(selected_band)] *
            spec.visual_time_scale;
        const double cycles = frequency_hz * effective_duration;
        if (cycles < spec.minimum_cycles || cycles > spec.maximum_cycles) {
            continue;
        }
        double candidate_phase = 0.0;
        const double score = fit_candidate(
            bands[static_cast<size_t>(selected_band)],
            effective_duration,
            frequency_hz, &candidate_phase);
        estimate.candidate_scores[static_cast<size_t>(bin - first_bin)] =
            static_cast<float>(score);
        if (score < best_score) {
            runner_score = best_score;
            runner_bin = best_bin;
            best_score = score;
            best_bin = bin;
            best_phase = candidate_phase;
        } else if (score < runner_score) {
            runner_score = score;
            runner_bin = bin;
        }
    }
    estimate.valid = std::isfinite(best_score);
    estimate.frequency_hz = best_bin * legal_step_hz_;
    estimate.runner_up_hz = runner_bin * legal_step_hz_;
    estimate.score = best_score;
    estimate.runner_up_score = runner_score;
    estimate.margin = runner_score - best_score;
    estimate.fitted_phase_rad = best_phase;
    estimate.valid_samples = valid_samples[static_cast<size_t>(selected_band)];
    estimate.observed_cycles = static_cast<int>(std::lround(
        estimate.frequency_hz *
        spec.durations_s[static_cast<size_t>(selected_band)] *
        spec.visual_time_scale));
    return estimate;
}

FrequencyEstimate FrequencyObserver::observe_phase_code(
    const TraceObservation& observation, const PhaseCodeSpec& code) const {
    FrequencyEstimate result;
    result.strategy = "crt_phase_code_7_11_13";
    if (observation.mask.empty() || code.frame_period_s <= 0.0 ||
        code.marker_duration_s <= 0.0 || code.samples_per_band < 16) {
        return result;
    }
    const auto bands = sample_phase_bands(observation.mask, code);
    int available = 0;
    for (const auto& band : bands) {
        available += static_cast<int>(std::count_if(
            band.begin(), band.end(), [](double value) {
                return std::isfinite(value);
            }));
    }
    result.valid_samples = available;
    if (available < code.samples_per_band * 2) return result;

    struct TimedValue {
        double time_s;
        double x;
    };
    std::vector<TimedValue> values;
    values.reserve(static_cast<size_t>(available));
    for (size_t band_index = 0; band_index < bands.size(); ++band_index) {
        const int divisor = code.divisors[band_index];
        const double marker_start = divisor == 1
            ? 0.0
            : code.frame_period_s / divisor;
        const auto& band = bands[band_index];
        auto first = std::find_if(band.begin(), band.end(), [](double value) {
            return std::isfinite(value);
        });
        auto last = std::find_if(band.rbegin(), band.rend(), [](double value) {
            return std::isfinite(value);
        });
        if (first == band.end() || last == band.rend()) continue;
        const size_t first_index = static_cast<size_t>(first - band.begin());
        const size_t last_index = band.size() - 1 -
            static_cast<size_t>(last - band.rbegin());
        const double visible_count = std::max<size_t>(1,
            last_index - first_index + 1);
        for (size_t sample_index = 0; sample_index < band.size();
             ++sample_index) {
            if (!std::isfinite(band[sample_index])) continue;
            const double u = (sample_index - first_index + 0.5) /
                             visible_count;
            values.push_back({marker_start + u * code.marker_duration_s,
                              band[sample_index]});
        }
    }
    if (values.size() < static_cast<size_t>(code.samples_per_band * 2))
        return result;

    double mean_x = 0.0;
    for (const TimedValue& value : values) mean_x += value.x;
    mean_x /= values.size();
    double observed_variance = 0.0;
    for (const TimedValue& value : values) {
        const double centered = value.x - mean_x;
        observed_variance += centered * centered;
    }
    if (observed_variance < values.size()) return result;

    const int first_bin = static_cast<int>(std::ceil(
        minimum_hz_ / legal_step_hz_ - 1.0e-9));
    const int last_bin = static_cast<int>(std::floor(
        maximum_hz_ / legal_step_hz_ + 1.0e-9));
    double best_score = std::numeric_limits<double>::infinity();
    double second_score = std::numeric_limits<double>::infinity();
    int best_bin = first_bin;
    int second_bin = first_bin;
    double best_phase = 0.0;
    for (int bin = first_bin; bin <= last_bin; ++bin) {
        const double frequency_hz = bin * legal_step_hz_;
        double sum_sine = 0.0;
        double sum_cosine = 0.0;
        for (const TimedValue& value : values) {
            const double angle = kTwoPi * frequency_hz * value.time_s;
            sum_sine += std::sin(angle);
            sum_cosine += std::cos(angle);
        }
        const double mean_sine = sum_sine / values.size();
        const double mean_cosine = sum_cosine / values.size();
        double ss = 0.0;
        double cc = 0.0;
        double sc = 0.0;
        double xs = 0.0;
        double xc = 0.0;
        for (const TimedValue& value : values) {
            const double angle = kTwoPi * frequency_hz * value.time_s;
            const double sine = std::sin(angle) - mean_sine;
            const double cosine = std::cos(angle) - mean_cosine;
            const double x = value.x - mean_x;
            ss += sine * sine;
            cc += cosine * cosine;
            sc += sine * cosine;
            xs += x * sine;
            xc += x * cosine;
        }
        const double determinant = ss * cc - sc * sc;
        if (determinant < 1.0e-9) continue;
        const double sine_coefficient = (xs * cc - xc * sc) / determinant;
        const double cosine_coefficient = (xc * ss - xs * sc) / determinant;
        const double explained = sine_coefficient * xs +
                                 cosine_coefficient * xc;
        const double candidate_score = std::max(
            0.0, observed_variance - explained) / observed_variance;
        const double candidate_phase = std::atan2(
            cosine_coefficient, sine_coefficient);
        if (candidate_score < best_score) {
            second_score = best_score;
            second_bin = best_bin;
            best_score = candidate_score;
            best_bin = bin;
            best_phase = candidate_phase;
        } else if (candidate_score < second_score) {
            second_score = candidate_score;
            second_bin = bin;
        }
    }
    result.valid = std::isfinite(best_score);
    result.frequency_hz = best_bin * legal_step_hz_;
    result.runner_up_hz = second_bin * legal_step_hz_;
    result.score = best_score;
    result.runner_up_score = second_score;
    result.margin = second_score - best_score;
    result.fitted_phase_rad = best_phase;
    return result;
}

void FrequencyObserver::reset() {
    posterior_history_.clear();
    scored_history_.clear();
}

void FrequencyObserver::update_posterior(const FrequencyEstimate& estimate,
                                         Clock::time_point timestamp) {
    if (!estimate.valid) return;
    if (!estimate.candidate_scores.empty() && estimate.probe_index >= 0) {
        scored_history_.push_back({timestamp, estimate.probe_index,
                                   estimate.candidate_first_bin,
                                   estimate.candidate_scores});
        const auto oldest = timestamp - std::chrono::seconds(3);
        while (!scored_history_.empty() &&
               (scored_history_.front().timestamp < oldest ||
                scored_history_.size() > 60)) {
            scored_history_.pop_front();
        }
    }
    posterior_history_.push_back({estimate.frequency_hz, estimate.score});
    while (posterior_history_.size() > 9) posterior_history_.pop_front();
}

FrequencyEstimate FrequencyObserver::posterior() const {
    FrequencyEstimate result;
    result.strategy = "joint_candidate_residual";
    if (!scored_history_.empty()) {
        std::array<int, 4> probe_counts{};
        for (const ScoredFrame& frame : scored_history_) {
            if (frame.probe_index >= 0 && frame.probe_index < 4) {
                ++probe_counts[static_cast<size_t>(frame.probe_index)];
            }
        }
        const int selected_probe = static_cast<int>(std::distance(
            probe_counts.begin(),
            std::max_element(probe_counts.begin(), probe_counts.end())));
        int common_first_bin = 0;
        size_t candidate_count = 0;
        for (const ScoredFrame& frame : scored_history_) {
            if (frame.probe_index == selected_probe &&
                !frame.scores.empty()) {
                common_first_bin = frame.first_bin;
                candidate_count = frame.scores.size();
                break;
            }
        }
        double best_score = std::numeric_limits<double>::infinity();
        double runner_score = std::numeric_limits<double>::infinity();
        int best_index = -1;
        int runner_index = -1;
        int best_support = 0;
        for (size_t index = 0; index < candidate_count; ++index) {
            std::vector<float> values;
            for (const ScoredFrame& frame : scored_history_) {
                if (frame.probe_index != selected_probe ||
                    frame.first_bin != common_first_bin ||
                    index >= frame.scores.size() ||
                    !std::isfinite(frame.scores[index])) {
                    continue;
                }
                values.push_back(frame.scores[index]);
            }
            if (values.size() < 5) continue;
            const auto middle = values.begin() + values.size() / 2;
            std::nth_element(values.begin(), middle, values.end());
            const double score = *middle;
            if (score < best_score) {
                runner_score = best_score;
                runner_index = best_index;
                best_score = score;
                best_index = static_cast<int>(index);
                best_support = static_cast<int>(values.size());
            } else if (score < runner_score) {
                runner_score = score;
                runner_index = static_cast<int>(index);
            }
        }
        if (best_index >= 0) {
            result.frequency_hz =
                (common_first_bin + best_index) * legal_step_hz_;
            result.runner_up_hz = runner_index >= 0
                ? (common_first_bin + runner_index) * legal_step_hz_
                : 0.0;
            result.score = best_score;
            result.runner_up_score = runner_score;
            result.margin = runner_score - best_score;
            result.valid_samples = best_support;
            result.probe_index = selected_probe;
            result.valid = best_support >= 8 && best_score < 0.85;
            return result;
        }
    }

    result.strategy = "median_best_bin_fallback";
    if (posterior_history_.empty()) return result;
    std::vector<Candidate> candidates(posterior_history_.begin(),
                                      posterior_history_.end());
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.frequency_hz < b.frequency_hz;
              });
    // A median of the legal-bin observations is deliberately used instead of
    // retuning from a single frame. Frequency changes clear this history.
    const double median = candidates[candidates.size() / 2].frequency_hz;
    int same = 0;
    double score_sum = 0.0;
    for (const Candidate& candidate : candidates) {
        if (std::abs(candidate.frequency_hz - median) < legal_step_hz_ * 0.5) {
            ++same;
            score_sum += candidate.log_likelihood;
        }
    }
    result.frequency_hz = median;
    result.valid = same >= 3;
    result.valid_samples = same;
    result.score = score_sum / std::max(1, same);
    result.margin = same / static_cast<double>(candidates.size());
    return result;
}

PhaseSlopeEstimator::PhaseSlopeEstimator(size_t maximum_samples)
    : maximum_samples_(std::max<size_t>(4, maximum_samples)) {}

void PhaseSlopeEstimator::reset() { samples_.clear(); }

void PhaseSlopeEstimator::add(PhaseSample sample) {
    if (!samples_.empty()) {
        sample.phase_rad = unwrap_near(sample.phase_rad,
                                       samples_.back().phase_rad);
    }
    samples_.push_back(sample);
    while (samples_.size() > maximum_samples_) samples_.pop_front();
}

PhaseRateEstimate PhaseSlopeEstimator::estimate() const {
    PhaseRateEstimate result;
    result.samples = static_cast<int>(samples_.size());
    if (samples_.size() < 4) return result;
    const double t0 = std::chrono::duration<double>(
                          samples_.front().timestamp.time_since_epoch())
                          .count();
    double sum_t = 0.0;
    double sum_p = 0.0;
    double sum_tt = 0.0;
    double sum_tp = 0.0;
    double confidence = 0.0;
    for (const PhaseSample& sample : samples_) {
        const double t = std::chrono::duration<double>(
                              sample.timestamp.time_since_epoch())
                              .count() -
                         t0;
        sum_t += t;
        sum_p += sample.phase_rad;
        sum_tt += t * t;
        sum_tp += t * sample.phase_rad;
        confidence += sample.confidence;
    }
    const double n = static_cast<double>(samples_.size());
    const double denominator = n * sum_tt - sum_t * sum_t;
    if (denominator <= 1.0e-9) return result;
    const double slope = (n * sum_tp - sum_t * sum_p) / denominator;
    double residual = 0.0;
    const double intercept = (sum_p - slope * sum_t) / n;
    for (const PhaseSample& sample : samples_) {
        const double t = std::chrono::duration<double>(
                              sample.timestamp.time_since_epoch())
                              .count() -
                         t0;
        const double error = sample.phase_rad - (intercept + slope * t);
        residual += error * error;
    }
    residual = std::sqrt(residual / n);
    result.valid = confidence / n > 0.45 && residual < 0.45;
    result.delta_frequency_hz = slope / kTwoPi;
    result.phase_rad = intercept;
    result.residual_rad = residual;
    return result;
}

}  // namespace task5
