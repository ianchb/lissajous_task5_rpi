#include "task5/phase_code_matcher.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace task5 {
namespace {

constexpr double kTwoPi = 2.0 * M_PI;
constexpr int kPhaseTrials = 256;
constexpr int kResponseLutSize = 16'384;
constexpr int kSubsamplesPerFraction = 4;
constexpr int kDacAmplitude = 102;
constexpr int kPhaseCodeSpan = kDacAmplitude / 3;
constexpr std::array<int, 4> kPhysicalBandStarts{-89, -38, 12, 63};

double nominal_y_for_value(double physical_value) {
    return (physical_value + kDacAmplitude) / (2.0 * kDacAmplitude);
}

const std::array<float, kResponseLutSize>& edge_response_lut() {
    static const auto values = [] {
        std::array<float, kResponseLutSize> result{};
        for (int index = 0; index < kResponseLutSize; ++index) {
            const double sine = std::sin(kTwoPi * index / kResponseLutSize);
            result[static_cast<std::size_t>(index)] = static_cast<float>(
                std::pow(std::max(0.0, sine), 20.0));
        }
        return result;
    }();
    return values;
}

int phase_lut_index(double cycles) {
    const double fractional = cycles - std::floor(cycles);
    return static_cast<int>(std::lround(fractional * kResponseLutSize)) &
           (kResponseLutSize - 1);
}

std::array<std::vector<float>, 2> quantized_edge_model(
    double frequency_hz, std::size_t band, double phase,
    const PhaseCodeSpec& spec, int rows, double visual_scale,
    double visual_offset) {
    std::array<std::vector<float>, 2> result{
        std::vector<float>(static_cast<std::size_t>(rows), 0.0f),
        std::vector<float>(static_cast<std::size_t>(rows), 0.0f)};
    constexpr std::array<int, 4> kRtlBandStarts{89, 38, -12, -63};
    constexpr int span = kPhaseCodeSpan;
    const double start_s = spec.divisors[band] == 1
        ? 0.0 : spec.frame_period_s / spec.divisors[band];
    const auto& response = edge_response_lut();
    const int phase_offset = phase_lut_index(phase / kTwoPi);

    std::array<double, 34> right{};
    std::array<double, 34> left{};
    std::array<int, 34> counts{};
    for (int fraction = 0; fraction < 256; ++fraction) {
        const int offset = span * fraction >> 8;
        for (int sub = 0; sub < kSubsamplesPerFraction; ++sub) {
            const double u = (fraction + (sub + 0.5) /
                              kSubsamplesPerFraction) / 256.0;
            const int index = (phase_lut_index(
                frequency_hz * (start_s + u * spec.marker_duration_s)) +
                phase_offset) & (kResponseLutSize - 1);
            right[static_cast<std::size_t>(offset)] += response[index];
            left[static_cast<std::size_t>(offset)] += response[
                (index + kResponseLutSize / 2) & (kResponseLutSize - 1)];
            ++counts[static_cast<std::size_t>(offset)];
        }
    }

    const double top = spec.band_centers[band] - 0.5 * spec.band_height;
    constexpr std::array<double, 5> kGaussian{
        0.03765705, 0.23993598, 0.44481394, 0.23993598, 0.03765705};
    std::array<std::vector<double>, 2> unsmoothed{
        std::vector<double>(static_cast<std::size_t>(rows), 0.0),
        std::vector<double>(static_cast<std::size_t>(rows), 0.0)};
    for (int offset = 0; offset < span; ++offset) {
        if (counts[static_cast<std::size_t>(offset)] == 0) continue;
        const double right_value =
            right[static_cast<std::size_t>(offset)] /
            counts[static_cast<std::size_t>(offset)];
        const double left_value =
            left[static_cast<std::size_t>(offset)] /
            counts[static_cast<std::size_t>(offset)];
        const int physical_value = -kRtlBandStarts[band] + offset;
        const double nominal_y = nominal_y_for_value(physical_value);
        const double y_fraction = visual_offset + visual_scale * nominal_y;
        const double row_position = (y_fraction - top) /
                                    spec.band_height * (rows - 1);
        const int lower = static_cast<int>(std::floor(row_position));
        const double upper_weight = row_position - lower;
        for (int side = 0; side < 2; ++side) {
            const double value = side == 0 ? left_value : right_value;
            if (lower >= 0 && lower < rows)
                unsmoothed[static_cast<std::size_t>(side)]
                          [static_cast<std::size_t>(lower)] +=
                    value * (1.0 - upper_weight);
            if (lower + 1 >= 0 && lower + 1 < rows)
                unsmoothed[static_cast<std::size_t>(side)]
                          [static_cast<std::size_t>(lower + 1)] +=
                    value * upper_weight;
        }
    }
    for (int side = 0; side < 2; ++side) {
        double energy = 0.0;
        for (int row = 0; row < rows; ++row) {
            double value = 0.0;
            for (int tap = -2; tap <= 2; ++tap) {
                const int source = row + tap;
                if (source >= 0 && source < rows)
                    value += kGaussian[static_cast<std::size_t>(tap + 2)] *
                             unsmoothed[static_cast<std::size_t>(side)]
                                       [static_cast<std::size_t>(source)];
            }
            result[static_cast<std::size_t>(side)]
                  [static_cast<std::size_t>(row)] = static_cast<float>(value);
            energy += value * value;
        }
        const double scale = std::sqrt(energy);
        if (scale > 1.0e-9) {
            for (float& value : result[static_cast<std::size_t>(side)])
                value = static_cast<float>(value / scale);
        }
    }
    return result;
}

void normalize_profile(std::vector<float>* values) {
    if (!values || values->empty()) return;
    std::vector<float> smoothed(values->size(), 0.0f);
    for (std::size_t index = 0; index < values->size(); ++index) {
        const float left = (*values)[index == 0 ? index : index - 1];
        const float right = (*values)[
            index + 1 == values->size() ? index : index + 1];
        smoothed[index] = 0.25f * left + 0.5f * (*values)[index] +
                          0.25f * right;
    }
    cv::Mat source(static_cast<int>(smoothed.size()), 1, CV_32F,
                   smoothed.data());
    cv::Mat baseline;
    cv::GaussianBlur(source, baseline, {1, 21}, 5.0);
    double energy = 0.0;
    for (std::size_t index = 0; index < smoothed.size(); ++index) {
        const float high_pass = std::max(
            0.0f, smoothed[index] - baseline.at<float>(static_cast<int>(index)));
        (*values)[index] = high_pass;
        energy += high_pass * high_pass;
    }
    const double scale = std::sqrt(energy);
    if (scale <= 1.0e-6) return;
    for (float& value : *values) value = static_cast<float>(value / scale);
}

std::vector<int> peak_rows(const std::vector<float>& profile) {
    std::vector<int> candidates;
    const float maximum = profile.empty()
        ? 0.0f : *std::max_element(profile.begin(), profile.end());
    const float threshold = 0.10f * maximum;
    for (int row = 1; row + 1 < static_cast<int>(profile.size()); ++row) {
        if (profile[static_cast<std::size_t>(row)] >= threshold &&
            profile[static_cast<std::size_t>(row)] >=
                profile[static_cast<std::size_t>(row - 1)] &&
            profile[static_cast<std::size_t>(row)] >
                profile[static_cast<std::size_t>(row + 1)]) {
            candidates.push_back(row);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [&](int left, int right) {
        return profile[static_cast<std::size_t>(left)] >
               profile[static_cast<std::size_t>(right)];
    });
    std::vector<int> selected;
    for (int row : candidates) {
        if (std::all_of(selected.begin(), selected.end(), [&](int old) {
                return std::abs(row - old) >= 3;
            })) {
            selected.push_back(row);
        }
    }
    std::sort(selected.begin(), selected.end());
    return selected;
}

double sample_row(const std::vector<float>& profile, double position,
                  double sigma = 0.70) {
    double weighted = 0.0;
    double weight_sum = 0.0;
    const int first = std::max(0, static_cast<int>(std::ceil(position - 2.5)));
    const int last = std::min(static_cast<int>(profile.size()) - 1,
                              static_cast<int>(std::floor(position + 2.5)));
    for (int row = first; row <= last; ++row) {
        const double delta = (row - position) / sigma;
        const double weight = std::exp(-0.5 * delta * delta);
        weighted += weight * profile[static_cast<std::size_t>(row)];
        weight_sum += weight;
    }
    return weighted / std::max(1.0e-12, weight_sum);
}

void normalize_levels(std::array<double, 34>* levels) {
    cv::Mat source(34, 1, CV_64F, levels->data());
    cv::Mat baseline;
    cv::GaussianBlur(source, baseline, {1, 9}, 2.0);
    double energy = 0.0;
    for (int level = 0; level < 34; ++level) {
        double& value = (*levels)[static_cast<std::size_t>(level)];
        value = std::max(0.0, value - baseline.at<double>(level));
        energy += value * value;
    }
    const double scale = std::sqrt(energy);
    if (scale > 1.0e-9)
        for (double& value : *levels) value /= scale;
}

std::array<std::array<double, 34>, 2> quantized_level_model(
    double frequency_hz, std::size_t band, double phase,
    const PhaseCodeSpec& spec) {
    std::array<std::array<double, 34>, 2> result{};
    std::array<int, 34> counts{};
    const double start_s = spec.divisors[band] == 1
        ? 0.0 : spec.frame_period_s / spec.divisors[band];
    const auto& response = edge_response_lut();
    const int phase_offset = phase_lut_index(phase / kTwoPi);
    for (int fraction = 0; fraction < 256; ++fraction) {
        const int level = (34 * fraction) >> 8;
        for (int sub = 0; sub < kSubsamplesPerFraction; ++sub) {
            const double u = (fraction + (sub + 0.5) /
                              kSubsamplesPerFraction) / 256.0;
            const int index = (phase_lut_index(
                frequency_hz * (start_s + u * spec.marker_duration_s)) +
                phase_offset) & (kResponseLutSize - 1);
            result[1][static_cast<std::size_t>(level)] += response[index];
            result[0][static_cast<std::size_t>(level)] += response[
                (index + kResponseLutSize / 2) & (kResponseLutSize - 1)];
            ++counts[static_cast<std::size_t>(level)];
        }
    }
    for (auto& side : result) {
        double energy = 0.0;
        for (int level = 0; level < 34; ++level) {
            side[static_cast<std::size_t>(level)] /=
                std::max(1, counts[static_cast<std::size_t>(level)]);
            energy += side[static_cast<std::size_t>(level)] *
                      side[static_cast<std::size_t>(level)];
        }
        const double scale = std::sqrt(energy);
        if (scale > 1.0e-9)
            for (double& value : side) value /= scale;
    }
    return result;
}

}  // namespace

PhaseCodeMatcher::PhaseCodeMatcher(PhaseCodeSpec spec) : spec_(spec) {}

PhaseCodeMatcher::FrameProfiles PhaseCodeMatcher::extract(
    const cv::Mat& normalized_bgr) const {
    FrameProfiles result;
    if (normalized_bgr.empty() || normalized_bgr.type() != CV_8UC3)
        return result;
    const int left_stop = normalized_bgr.cols * 15 / 64;
    const int right_start = normalized_bgr.cols * 49 / 64;
    for (std::size_t band = 0; band < result.size(); ++band) {
        for (auto& side : result[band])
            side.assign(static_cast<std::size_t>(profile_rows_), 0.0f);
        std::fill(result[band][2].begin(), result[band][2].end(),
                  std::numeric_limits<float>::quiet_NaN());
        const double top = spec_.band_centers[band] - 0.5 * spec_.band_height;
        for (int sample = 0; sample < profile_rows_; ++sample) {
            const double fraction = top + spec_.band_height * sample /
                std::max(1, profile_rows_ - 1);
            const int row = std::clamp(static_cast<int>(std::lround(
                fraction * (normalized_bgr.rows - 1))), 0,
                normalized_bgr.rows - 1);
            const cv::Vec3b* pixels = normalized_bgr.ptr<cv::Vec3b>(row);
            double total = 0.0;
            double weighted_x = 0.0;
            for (int column = 0; column < normalized_bgr.cols; ++column) {
                const int blue = pixels[column][0];
                const int dominance = std::max(
                    0, std::min<int>(pixels[column][1], pixels[column][2]) -
                       blue - 3);
                if (column < left_stop)
                    result[band][0][static_cast<std::size_t>(sample)] +=
                        dominance;
                if (column >= right_start)
                    result[band][1][static_cast<std::size_t>(sample)] +=
                        dominance;
                total += dominance;
                weighted_x += dominance * column;
            }
            if (total >= 120.0) {
                const double centroid = weighted_x / total;
                result[band][2][static_cast<std::size_t>(sample)] =
                    static_cast<float>(2.0 * centroid /
                                       (normalized_bgr.cols - 1) - 1.0);
            }
        }
    }
    return result;
}

PhaseCodeMatcher::RowProfiles PhaseCodeMatcher::extract_rows(
    const cv::Mat& normalized_bgr) const {
    RowProfiles result;
    if (normalized_bgr.empty() || normalized_bgr.type() != CV_8UC3)
        return result;
    for (auto& profile : result)
        profile.assign(static_cast<std::size_t>(normalized_bgr.rows), 0.0f);
    const int left_stop = normalized_bgr.cols * 15 / 64;
    const int right_start = normalized_bgr.cols * 49 / 64;
    for (int row = 0; row < normalized_bgr.rows; ++row) {
        const cv::Vec3b* pixels = normalized_bgr.ptr<cv::Vec3b>(row);
        for (int column = 0; column < normalized_bgr.cols; ++column) {
            const int dominance = std::max(
                0, std::min<int>(pixels[column][1], pixels[column][2]) -
                   pixels[column][0] - 3);
            if (column < left_stop) result[0][row] += dominance;
            if (column >= right_start) result[1][row] += dominance;
            result[2][row] += dominance;
        }
    }
    return result;
}

bool PhaseCodeMatcher::add(const TraceObservation& observation) {
    if (observation.normalized_bgr.empty()) return false;
    frames_.push_back(extract(observation.normalized_bgr));
    row_frames_.push_back(extract_rows(observation.normalized_bgr));
    if (frames_.size() > 120) {
        frames_.erase(frames_.begin());
        row_frames_.erase(row_frames_.begin());
    }
    return true;
}

PhaseCodeMatcher::RowProfiles PhaseCodeMatcher::aggregate_rows() const {
    RowProfiles result;
    if (row_frames_.empty()) return result;
    const std::size_t rows = row_frames_.front()[0].size();
    // At 500 us/div the scope can refresh different portions of the 10 ms
    // coherent code in different camera frames. Use a temporal union for the
    // full-height diagnostic profile; one isolated bright frame is still
    // excluded, while a band visible in roughly one quarter of the frames is
    // retained.
    const std::size_t percentile_index = static_cast<std::size_t>(
        std::lround(0.82 * static_cast<double>(row_frames_.size() - 1)));
    for (std::size_t profile = 0; profile < result.size(); ++profile) {
        result[profile].assign(rows, 0.0f);
        for (std::size_t row = 0; row < rows; ++row) {
            std::vector<float> values;
            values.reserve(row_frames_.size());
            for (const auto& frame : row_frames_)
                values.push_back(frame[profile][row]);
            std::nth_element(values.begin(), values.begin() + percentile_index,
                             values.end());
            result[profile][row] = values[percentile_index];
        }
    }
    return result;
}

PhaseCodeMatcher::VisualGeometry
PhaseCodeMatcher::estimate_visual_geometry(
    const RowProfiles& profiles) const {
    VisualGeometry result;
    if (profiles[2].size() < 100) return result;

    const int rows = static_cast<int>(profiles[2].size());
    cv::Mat source(rows, 1, CV_32F,
                   const_cast<float*>(profiles[2].data()));
    cv::Mat smoothed;
    cv::GaussianBlur(source, smoothed, {1, 7}, 1.5);
    double global_maximum = 0.0;
    cv::minMaxLoc(smoothed, nullptr, &global_maximum);
    if (global_maximum < 200.0) return result;

    std::array<double, 4> observed_centers{};
    constexpr double kZoneHalfHeight = 0.115;
    for (std::size_t band = 0; band < observed_centers.size(); ++band) {
        const int first_row = std::clamp(static_cast<int>(std::lround(
            (spec_.band_centers[band] - kZoneHalfHeight) * (rows - 1))),
            0, rows - 1);
        const int last_row = std::clamp(static_cast<int>(std::lround(
            (spec_.band_centers[band] + kZoneHalfHeight) * (rows - 1))),
            first_row, rows - 1);
        double local_maximum = 0.0;
        for (int row = first_row; row <= last_row; ++row)
            local_maximum = std::max(
                local_maximum, static_cast<double>(smoothed.at<float>(row)));
        const double threshold = std::max(0.04 * global_maximum,
                                          0.06 * local_maximum);
        int support_first = -1;
        int support_last = -1;
        for (int row = first_row; row <= last_row; ++row) {
            if (smoothed.at<float>(row) < threshold) continue;
            if (support_first < 0) support_first = row;
            support_last = row;
        }
        const int support = support_last - support_first + 1;
        if (support_first < 0 || support < rows * 0.07 ||
            support > rows * 0.25) {
            return result;
        }
        observed_centers[band] =
            0.5 * (support_first + support_last) / (rows - 1);
    }

    std::array<double, 4> nominal_centers{};
    double mean_nominal = 0.0;
    double mean_observed = 0.0;
    for (std::size_t band = 0; band < nominal_centers.size(); ++band) {
        nominal_centers[band] = nominal_y_for_value(
            kPhysicalBandStarts[band] + 0.5 * (kPhaseCodeSpan - 1));
        mean_nominal += nominal_centers[band];
        mean_observed += observed_centers[band];
    }
    mean_nominal /= nominal_centers.size();
    mean_observed /= observed_centers.size();
    double covariance = 0.0;
    double variance = 0.0;
    for (std::size_t band = 0; band < nominal_centers.size(); ++band) {
        covariance += (nominal_centers[band] - mean_nominal) *
                      (observed_centers[band] - mean_observed);
        variance += (nominal_centers[band] - mean_nominal) *
                    (nominal_centers[band] - mean_nominal);
    }
    if (variance <= 1.0e-12) return result;
    result.scale = covariance / variance;
    result.offset = mean_observed - result.scale * mean_nominal;
    double squared_error = 0.0;
    for (std::size_t band = 0; band < nominal_centers.size(); ++band) {
        const double residual = (observed_centers[band] -
            (result.offset + result.scale * nominal_centers[band])) *
            (rows - 1);
        squared_error += residual * residual;
    }
    result.residual_px = std::sqrt(
        squared_error / nominal_centers.size());
    result.valid = result.scale >= 0.80 && result.scale <= 1.15 &&
                   result.residual_px <= 4.0;
    if (!result.valid) {
        result.scale = 1.0;
        result.offset = 0.0;
    }
    return result;
}

std::array<PhaseCodeMatcher::SideProfiles, 4>
PhaseCodeMatcher::aggregate() const {
    std::array<SideProfiles, 4> result;
    if (frames_.empty()) return result;
    for (std::size_t band = 0; band < result.size(); ++band) {
        for (auto& profile : result[band])
            profile.assign(static_cast<std::size_t>(profile_rows_), 0.0f);
        std::fill(result[band][2].begin(), result[band][2].end(),
                  std::numeric_limits<float>::quiet_NaN());
        std::vector<double> frame_energy(frames_.size(), 0.0);
        double maximum_energy = 0.0;
        for (std::size_t frame = 0; frame < frames_.size(); ++frame) {
            for (std::size_t side = 0; side < 2; ++side)
                frame_energy[frame] += std::accumulate(
                    frames_[frame][band][side].begin(),
                    frames_[frame][band][side].end(), 0.0);
            maximum_energy = std::max(maximum_energy, frame_energy[frame]);
        }
        std::vector<std::size_t> selected_frames;
        if (maximum_energy > 1.0e-6) {
            const double threshold = 0.12 * maximum_energy;
            for (std::size_t frame = 0; frame < frames_.size(); ++frame)
                if (frame_energy[frame] >= threshold)
                    selected_frames.push_back(frame);
        } else {
            // A low-frequency marker need not reach either outer edge, but
            // its full-row centroid still carries the frequency code.
            selected_frames.resize(frames_.size());
            std::iota(selected_frames.begin(), selected_frames.end(), 0);
        }
        if (selected_frames.size() < 4) continue;
        const std::size_t percentile_index = static_cast<std::size_t>(
            std::lround(0.65 *
                static_cast<double>(selected_frames.size() - 1)));
        for (std::size_t side = 0; side < 2; ++side) {
            auto& output = result[band][side];
            for (int row = 0; row < profile_rows_; ++row) {
                std::vector<float> values;
                values.reserve(selected_frames.size());
                for (std::size_t frame : selected_frames)
                    values.push_back(frames_[frame][band][side]
                        [static_cast<std::size_t>(row)]);
                std::nth_element(values.begin(),
                                 values.begin() + percentile_index,
                                 values.end());
                output[static_cast<std::size_t>(row)] =
                    values[percentile_index];
            }
            normalize_profile(&output);
        }
        auto& output = result[band][2];
        for (int row = 0; row < profile_rows_; ++row) {
            std::vector<float> values;
            for (std::size_t frame : selected_frames) {
                const float value = frames_[frame][band][2]
                    [static_cast<std::size_t>(row)];
                if (std::isfinite(value)) values.push_back(value);
            }
            if (values.size() < std::max<std::size_t>(
                    2, selected_frames.size() / 3))
                continue;
            const auto middle = values.begin() + values.size() / 2;
            std::nth_element(values.begin(), middle, values.end());
            output[static_cast<std::size_t>(row)] = *middle;
        }
    }
    return result;
}

double PhaseCodeMatcher::estimate_coarse_hz(
    const std::array<SideProfiles, 4>& profiles,
    const VisualGeometry& geometry) const {
    std::vector<double> spacings;
    for (std::size_t band = 1; band < profiles.size(); ++band) {
        for (const auto& side : profiles[band]) {
            const auto peaks = peak_rows(side);
            for (std::size_t index = 1; index < peaks.size(); ++index) {
                const int spacing = peaks[index] - peaks[index - 1];
                if (spacing >= 5 && spacing <= 24)
                    spacings.push_back(spacing);
            }
        }
    }
    if (spacings.size() < 8) return 0.0;
    const auto middle = spacings.begin() + spacings.size() / 2;
    std::nth_element(spacings.begin(), middle, spacings.end());
    const double spacing = *middle;
    // The RTL scan spans amplitude/3 out of a total 2*amplitude. The sampled
    // analysis band is slightly wider so that camera installation errors do
    // not clip either end.
    const double active_rows = profile_rows_ * geometry.scale * (1.0 / 6.0) /
                               spec_.band_height;
    return active_rows / spacing / spec_.marker_duration_s;
}

double PhaseCodeMatcher::estimate_spectral_hz(
    const std::array<SideProfiles, 4>& profiles,
    const VisualGeometry& geometry,
    double* peak_ratio) const {
    if (peak_ratio) *peak_ratio = 0.0;
    std::array<std::vector<double>, 4> positions;
    std::array<std::array<std::vector<double>, 2>, 4> values;
    for (std::size_t band = 0; band < profiles.size(); ++band) {
        const double observed_start = geometry.offset + geometry.scale *
            nominal_y_for_value(kPhysicalBandStarts[band]);
        const double observed_span = geometry.scale / 6.0;
        const double extraction_top = spec_.band_centers[band] -
                                      0.5 * spec_.band_height;
        for (int row = 0; row < profile_rows_; ++row) {
            const double band_fraction = (row + 0.5) / profile_rows_;
            const double global_fraction = extraction_top +
                spec_.band_height * band_fraction;
            const double u = (global_fraction - observed_start) /
                             observed_span;
            if (u < 0.0 || u > 1.0) continue;
            positions[band].push_back(u);
            for (std::size_t side = 0; side < 2; ++side) {
                values[band][side].push_back(
                    profiles[band][side][static_cast<std::size_t>(row)]);
            }
        }
    }
    if (std::any_of(positions.begin(), positions.end(),
                    [](const auto& band) { return band.size() < 16; })) {
        return 0.0;
    }
    for (auto& band : values) {
        for (auto& side : band) {
            const double mean = std::accumulate(side.begin(), side.end(), 0.0) /
                                side.size();
            for (double& value : side) value -= mean;
        }
    }

    constexpr double kFirstCycles = 0.5;
    constexpr double kCycleStep = 0.005;
    const double maximum_cycles = std::min(
        8.5, 100'000.0 * spec_.marker_duration_s);
    std::vector<std::pair<double, double>> samples;
    for (double cycles = kFirstCycles; cycles <= maximum_cycles;
         cycles += kCycleStep) {
        double power = 0.0;
        for (std::size_t band = 0; band < profiles.size(); ++band) {
            for (std::size_t side = 0; side < 2; ++side) {
                double real = 0.0;
                double imaginary = 0.0;
                for (std::size_t index = 0;
                     index < positions[band].size(); ++index) {
                    const double angle = kTwoPi * cycles *
                                         positions[band][index];
                    real += values[band][side][index] * std::cos(angle);
                    imaginary -= values[band][side][index] * std::sin(angle);
                }
                power += real * real + imaginary * imaginary;
            }
        }
        samples.emplace_back(cycles, power);
    }
    std::vector<std::pair<double, double>> peaks;
    for (std::size_t index = 1; index + 1 < samples.size(); ++index) {
        if (samples[index].second >= samples[index - 1].second &&
            samples[index].second > samples[index + 1].second) {
            peaks.push_back(samples[index]);
        }
    }
    if (peaks.empty()) return 0.0;
    std::sort(peaks.begin(), peaks.end(), [](const auto& left,
                                             const auto& right) {
        return left.second > right.second;
    });
    const double runner = peaks.size() > 1 ? peaks[1].second : 0.0;
    if (peak_ratio)
        *peak_ratio = peaks[0].second / std::max(1.0e-12, runner);
    return peaks[0].first / spec_.marker_duration_s;
}

double PhaseCodeMatcher::estimate_lattice_hz(
    const RowProfiles& profiles, const VisualGeometry& visual_geometry,
    double spectral_hz, double original_hz, double original_runner_hz,
    double* runner_hz, double* score_advantage) const {
    if (runner_hz) *runner_hz = 0.0;
    if (score_advantage) *score_advantage = 0.0;
    if (profiles[0].empty() || profiles[1].empty() || profiles[2].empty() ||
        spectral_hz <= 0.0)
        return 0.0;
    const int height = static_cast<int>(profiles[2].size());
    cv::Mat geometry_profile(height, 1, CV_32F,
                             const_cast<float*>(profiles[2].data()));
    cv::Mat baseline;
    cv::GaussianBlur(geometry_profile, baseline, {1, 0}, 5.0);
    std::vector<float> detail(static_cast<std::size_t>(height), 0.0f);
    for (int row = 0; row < height; ++row)
        detail[static_cast<std::size_t>(row)] = std::max(
            0.0f, profiles[2][static_cast<std::size_t>(row)] -
                  baseline.at<float>(row));

    std::array<std::array<double, 34>, 4> positions{};
    const double nominal_step = visual_geometry.scale *
                                (height - 1) / 204.0;
    for (std::size_t band = 0; band < positions.size(); ++band) {
        const double nominal_first = (visual_geometry.offset +
            visual_geometry.scale * nominal_y_for_value(
                kPhysicalBandStarts[band])) * (height - 1);
        double best_score = -std::numeric_limits<double>::infinity();
        double best_first = nominal_first;
        double best_step = nominal_step;
        for (double first = nominal_first - 5.0;
             first <= nominal_first + 5.001; first += 0.20) {
            for (double step = nominal_step - 0.18;
                 step <= nominal_step + 0.181; step += 0.02) {
                double score = 0.0;
                for (int level = 0; level < 34; ++level) {
                    const double value = sample_row(
                        detail, first + step * level);
                    score += std::sqrt(std::max(0.0, value));
                }
                if (score > best_score) {
                    best_score = score;
                    best_first = first;
                    best_step = step;
                }
            }
        }
        for (int level = 0; level < 34; ++level)
            positions[band][static_cast<std::size_t>(level)] =
                best_first + best_step * level;
    }

    std::array<std::array<std::array<double, 34>, 2>, 4> observed{};
    for (std::size_t band = 0; band < observed.size(); ++band) {
        for (std::size_t side = 0; side < 2; ++side) {
            for (int level = 0; level < 34; ++level) {
                observed[band][side][static_cast<std::size_t>(level)] =
                    sample_row(profiles[side], positions[band][level]);
            }
            normalize_levels(&observed[band][side]);
        }
    }

    const auto score_bin = [&](int bin) {
        double best = -std::numeric_limits<double>::infinity();
        for (int phase_index = 0; phase_index < kPhaseTrials; ++phase_index) {
            const double phase = kTwoPi * phase_index / kPhaseTrials;
            double score = 0.0;
            for (std::size_t band = 0; band < observed.size(); ++band) {
                const auto model = quantized_level_model(
                    bin * 100.0, band, phase, spec_);
                for (std::size_t side = 0; side < 2; ++side)
                    score += std::inner_product(
                        observed[band][side].begin(),
                        observed[band][side].end(), model[side].begin(), 0.0);
            }
            best = std::max(best, score);
        }
        return best;
    };

    struct Score { int bin; double value; };
    std::vector<Score> coarse_scores;
    const int center_bin = static_cast<int>(std::lround(spectral_hz / 100.0));
    const int first_bin = std::max(10, center_bin - 20);
    const int last_bin = std::min(1000, center_bin + 20);
    for (int bin = first_bin; bin <= last_bin; ++bin)
        coarse_scores.push_back({bin, score_bin(bin)});
    if (coarse_scores.empty()) return 0.0;
    const auto coarse_best = std::max_element(
        coarse_scores.begin(), coarse_scores.end(),
        [](const Score& left, const Score& right) {
            return left.value < right.value;
        });
    // Live high-frequency windows show a stable one-bin low bias from the
    // camera's finite vertical point-spread function.
    const int possible_bin = std::min(1000, coarse_best->bin + 1);
    std::vector<int> final_bins{
        possible_bin,
        std::clamp(static_cast<int>(std::lround(original_hz / 100.0)),
                   10, 1000),
        std::clamp(static_cast<int>(std::lround(
                       original_runner_hz / 100.0)), 10, 1000)};
    std::sort(final_bins.begin(), final_bins.end());
    final_bins.erase(std::unique(final_bins.begin(), final_bins.end()),
                     final_bins.end());
    std::vector<Score> final_scores;
    for (int bin : final_bins)
        final_scores.push_back({bin, score_bin(bin)});
    std::sort(final_scores.begin(), final_scores.end(),
        [](const Score& left, const Score& right) {
            return left.value > right.value;
        });
    if (final_scores.empty()) return 0.0;
    if (runner_hz && final_scores.size() > 1)
        *runner_hz = final_scores[1].bin * 100.0;
    if (score_advantage && final_scores.size() > 1)
        *score_advantage = final_scores[0].value - final_scores[1].value;
    return final_scores[0].bin * 100.0;
}

FrequencyEstimate PhaseCodeMatcher::estimate_centroid_code(
    const std::array<SideProfiles, 4>& profiles,
    const VisualGeometry& geometry) const {
    FrequencyEstimate result;
    result.strategy = "quantized_phase_code_centroid";
    result.valid_samples = static_cast<int>(frames_.size());
    struct TimedValue {
        double time_s;
        double x;
        std::size_t band;
    };
    std::vector<TimedValue> values;
    for (std::size_t band = 0; band < profiles.size(); ++band) {
        const double start_s = spec_.divisors[band] == 1
            ? 0.0 : spec_.frame_period_s / spec_.divisors[band];
        const double observed_start = geometry.offset + geometry.scale *
            nominal_y_for_value(kPhysicalBandStarts[band]);
        const double observed_span = geometry.scale / 6.0;
        const double extraction_top = spec_.band_centers[band] -
                                      0.5 * spec_.band_height;
        for (int row = 0; row < profile_rows_; ++row) {
            const double x = profiles[band][2][static_cast<std::size_t>(row)];
            if (!std::isfinite(x)) continue;
            const double band_fraction = (row + 0.5) / profile_rows_;
            const double global_fraction = extraction_top +
                spec_.band_height * band_fraction;
            const double u = (global_fraction - observed_start) /
                             observed_span;
            if (u < 0.0 || u > 1.0) continue;
            values.push_back({start_s + u * spec_.marker_duration_s, x,
                              band});
        }
    }
    if (values.size() < 48) return result;

    double best = std::numeric_limits<double>::infinity();
    double runner = std::numeric_limits<double>::infinity();
    int best_bin = 0;
    int runner_bin = 0;
    double best_phase = 0.0;
    for (int bin = 10; bin <= 200; ++bin) {
        const double frequency_hz = bin * 100.0;
        // Perspective, trace brightness and the camera's black level give
        // each sub-band a different DC offset. Remove those offsets per band
        // while retaining one common sine/cosine phase for all four bands.
        std::array<double, 4> mean_x{};
        std::array<double, 4> mean_s{};
        std::array<double, 4> mean_c{};
        std::array<int, 4> counts{};
        for (const auto& value : values) {
            const double angle = kTwoPi * frequency_hz * value.time_s;
            mean_x[value.band] += value.x;
            mean_s[value.band] += std::sin(angle);
            mean_c[value.band] += std::cos(angle);
            ++counts[value.band];
        }
        for (std::size_t band = 0; band < counts.size(); ++band) {
            if (counts[band] == 0) continue;
            const double scale = 1.0 / counts[band];
            mean_x[band] *= scale;
            mean_s[band] *= scale;
            mean_c[band] *= scale;
        }
        double ss = 0.0, cc = 0.0, sc = 0.0;
        double xs = 0.0, xc = 0.0;
        for (const auto& value : values) {
            const double angle = kTwoPi * frequency_hz * value.time_s;
            const double centered_x = value.x - mean_x[value.band];
            const double centered_s = std::sin(angle) - mean_s[value.band];
            const double centered_c = std::cos(angle) - mean_c[value.band];
            ss += centered_s * centered_s;
            cc += centered_c * centered_c;
            sc += centered_s * centered_c;
            xs += centered_x * centered_s;
            xc += centered_x * centered_c;
        }
        const double determinant = ss * cc - sc * sc;
        if (determinant <= 1.0e-9) continue;
        const double sine_coefficient = (xs * cc - xc * sc) / determinant;
        const double cosine_coefficient = (xc * ss - xs * sc) / determinant;
        double squared_error = 0.0;
        double variance = 0.0;
        for (const auto& value : values) {
            const double angle = kTwoPi * frequency_hz * value.time_s;
            const double fitted = mean_x[value.band] +
                sine_coefficient * (std::sin(angle) - mean_s[value.band]) +
                cosine_coefficient * (std::cos(angle) - mean_c[value.band]);
            squared_error += (value.x - fitted) * (value.x - fitted);
            variance += (value.x - mean_x[value.band]) *
                        (value.x - mean_x[value.band]);
        }
        const double score = squared_error / std::max(1.0e-9, variance);
        if (score < best) {
            runner = best;
            runner_bin = best_bin;
            best = score;
            best_bin = bin;
            best_phase = std::atan2(cosine_coefficient, sine_coefficient);
        } else if (score < runner) {
            runner = score;
            runner_bin = bin;
        }
    }
    result.valid = std::isfinite(best) && best_bin > 0;
    result.frequency_hz = best_bin * 100.0;
    result.runner_up_hz = runner_bin * 100.0;
    result.score = best;
    result.runner_up_score = runner;
    result.margin = runner - best;
    result.fitted_phase_rad = best_phase;
    result.observed_cycles = static_cast<int>(std::lround(
        result.frequency_hz * spec_.marker_duration_s));
    return result;
}

FrequencyEstimate PhaseCodeMatcher::estimate() const {
    FrequencyEstimate result;
    result.strategy = "temporal_sparse_phase_code";
    result.valid_samples = static_cast<int>(frames_.size());
    if (frames_.size() < 24) return result;
    const auto profiles = aggregate();
    const auto row_profiles = aggregate_rows();
    const VisualGeometry fitted_geometry =
        estimate_visual_geometry(row_profiles);
    // A one-to-two percent ROI scale bias is enough to move a frequency by
    // one or more 100 Hz bins.  Once the four-band fit is valid, apply it;
    // the residual check in estimate_visual_geometry rejects bad fits.
    const bool apply_visual_geometry = fitted_geometry.valid &&
        std::abs(fitted_geometry.scale - 1.0) >= 0.01;
    VisualGeometry geometry;
    if (apply_visual_geometry) geometry = fitted_geometry;
    result.visual_geometry_valid = apply_visual_geometry;
    result.visual_scale = fitted_geometry.scale;
    result.visual_offset = fitted_geometry.offset;
    result.visual_residual_px = fitted_geometry.residual_px;
    if (apply_visual_geometry) result.strategy += "+visual_geometry";
    const double coarse_hz = estimate_coarse_hz(profiles, geometry);
    double spectral_peak_ratio = 0.0;
    const double spectral_hz = estimate_spectral_hz(
        profiles, geometry, &spectral_peak_ratio);

    const auto score_phase_for_geometry = [&]
        (double frequency_hz, double phase,
         const VisualGeometry& model_geometry) {
        double correlation = 0.0;
        for (std::size_t band = 0; band < profiles.size(); ++band) {
            const auto model = quantized_edge_model(
                frequency_hz, band, phase, spec_, profile_rows_,
                model_geometry.scale, model_geometry.offset);
            for (int side = 0; side < 2; ++side) {
                const auto& observed = profiles[band]
                    [static_cast<std::size_t>(side)];
                const auto& expected = model[static_cast<std::size_t>(side)];
                correlation += std::inner_product(
                    observed.begin(), observed.end(), expected.begin(), 0.0);
            }
        }
        return correlation;
    };
    const auto score_phase = [&](double frequency_hz, double phase) {
        return score_phase_for_geometry(frequency_hz, phase, geometry);
    };

struct RankedCandidate {
        int bin = 0;
        double score = -std::numeric_limits<double>::infinity();
        double phase = 0.0;
    };

    // Integer peak spacing is only a diagnostic: above 70 kHz one profile
    // pixel spans several kHz. A strong continuous Fourier estimate from the
    // marker texture can safely define a broad interval; otherwise search all
    // 991 bins on a sparse phase grid before full-resolution refinement.
    constexpr int kCoarsePhaseTrials = 32;
    constexpr std::size_t kRefineCandidates = 24;
    // The 78.125 us marker gives a broad Fourier lobe. Keep a conservative
    // interval here; its distance becomes soft evidence during final ranking.
    constexpr double kSpectralHalfWidthHz = 1'500.0;
    const bool spectral_reliable = spectral_hz >= 20'000.0 &&
                                   spectral_peak_ratio >= 2.0;
    std::vector<int> bins_to_refine;
    if (spectral_reliable) {
        const int first_bin = std::max(10, static_cast<int>(std::ceil(
            (spectral_hz - kSpectralHalfWidthHz) / 100.0)));
        const int last_bin = std::min(1000, static_cast<int>(std::floor(
            (spectral_hz + kSpectralHalfWidthHz) / 100.0)));
        for (int bin = first_bin; bin <= last_bin; ++bin)
            bins_to_refine.push_back(bin);
    } else {
        std::vector<RankedCandidate> ranked;
        ranked.reserve(991);
        for (int bin = 10; bin <= 1000; ++bin) {
            RankedCandidate candidate;
            candidate.bin = bin;
            for (int trial = 0; trial < kCoarsePhaseTrials; ++trial) {
                const double phase = kTwoPi * trial / kCoarsePhaseTrials;
                const double score = score_phase(bin * 100.0, phase);
                if (score > candidate.score) {
                    candidate.score = score;
                    candidate.phase = phase;
                }
            }
            ranked.push_back(candidate);
        }
        std::partial_sort(
            ranked.begin(), ranked.begin() + kRefineCandidates, ranked.end(),
            [](const auto& left, const auto& right) {
                return left.score > right.score;
            });
        for (std::size_t index = 0; index < kRefineCandidates; ++index)
            bins_to_refine.push_back(ranked[index].bin);
    }

    double best = -std::numeric_limits<double>::infinity();
    double runner = -std::numeric_limits<double>::infinity();
    int best_bin = 0;
    int runner_bin = 0;
    double best_phase = 0.0;
    for (int bin : bins_to_refine) {
        const double frequency_hz = bin * 100.0;
        double candidate_best = -std::numeric_limits<double>::infinity();
        double candidate_phase = 0.0;
        for (int phase_index = 0; phase_index < kPhaseTrials; ++phase_index) {
            const double phase = kTwoPi * phase_index / kPhaseTrials;
            const double correlation = score_phase(frequency_hz, phase);
            if (correlation > candidate_best) {
                candidate_best = correlation;
                candidate_phase = phase;
            }
        }
        double ranked_score = candidate_best;
        if (spectral_reliable) {
            const double mismatch_khz = (frequency_hz - spectral_hz) / 1'000.0;
            ranked_score -= 0.20 * mismatch_khz * mismatch_khz;
        }
        if (ranked_score > best) {
            runner = best;
            runner_bin = best_bin;
            best = ranked_score;
            best_bin = bin;
            best_phase = candidate_phase;
        } else if (ranked_score > runner) {
            runner = ranked_score;
            runner_bin = bin;
        }
    }
    result.valid = std::isfinite(best) && best_bin > 0;
    result.frequency_hz = best_bin * 100.0;
    result.runner_up_hz = runner_bin * 100.0;
    result.score = -best;
    result.runner_up_score = -runner;
    result.margin = best - runner;
    result.fitted_phase_rad = best_phase;
    result.candidate_first_bin = static_cast<int>(std::lround(
        spectral_hz / 100.0));
    result.candidate_scores = {
        static_cast<float>(spectral_peak_ratio)};
    result.observed_cycles = static_cast<int>(std::lround(
        (spectral_reliable ? spectral_hz : coarse_hz) *
        spec_.marker_duration_s));
    if (fitted_geometry.valid &&
        std::abs(fitted_geometry.scale - 1.0) >= 0.01 && result.valid) {
        const VisualGeometry alternate_geometry = apply_visual_geometry ?
            VisualGeometry{} : fitted_geometry;
        const double seed_hz = result.frequency_hz *
            (apply_visual_geometry ? 1.0 / fitted_geometry.scale :
                                     fitted_geometry.scale);
        const int first_bin = std::clamp(static_cast<int>(std::floor(
            (seed_hz - 2'500.0) / 100.0)), 10, 1000);
        const int last_bin = std::clamp(static_cast<int>(std::ceil(
            (seed_hz + 2'500.0) / 100.0)), first_bin, 1000);
        std::vector<RankedCandidate> alternate_ranked;
        alternate_ranked.reserve(static_cast<std::size_t>(
            last_bin - first_bin + 1));
        for (int bin = first_bin; bin <= last_bin; ++bin) {
            RankedCandidate candidate;
            candidate.bin = bin;
            for (int trial = 0; trial < kCoarsePhaseTrials; ++trial) {
                const double phase = kTwoPi * trial / kCoarsePhaseTrials;
                const double score = score_phase_for_geometry(
                    bin * 100.0, phase, alternate_geometry);
                if (score > candidate.score) {
                    candidate.score = score;
                    candidate.phase = phase;
                }
            }
            alternate_ranked.push_back(candidate);
        }
        constexpr std::size_t kAlternateRefineCandidates = 8;
        const std::size_t alternate_refine_count = std::min(
            kAlternateRefineCandidates, alternate_ranked.size());
        std::partial_sort(
            alternate_ranked.begin(),
            alternate_ranked.begin() + alternate_refine_count,
            alternate_ranked.end(),
            [](const RankedCandidate& left, const RankedCandidate& right) {
                return left.score > right.score;
            });
        double alternate_best =
            -std::numeric_limits<double>::infinity();
        int alternate_best_bin = 0;
        for (std::size_t index = 0; index < alternate_refine_count; ++index) {
            const int bin = alternate_ranked[index].bin;
            for (int phase_index = 0; phase_index < kPhaseTrials;
                 ++phase_index) {
                const double phase = kTwoPi * phase_index / kPhaseTrials;
                const double score = score_phase_for_geometry(
                    bin * 100.0, phase, alternate_geometry);
                if (score > alternate_best) {
                    alternate_best = score;
                    alternate_best_bin = bin;
                }
            }
        }
        if (alternate_best_bin > 0) {
            result.geometry_alternate_hz = alternate_best_bin * 100.0;
            result.geometry_alternate_scaled = !apply_visual_geometry;
            result.strategy += result.geometry_alternate_scaled ?
                "+geometry_alt_scaled" : "+geometry_alt_unscaled";
        }
    }
    if (result.valid && result.frequency_hz >= 95'000.0) {
        double lattice_advantage = 0.0;
        double lattice_runner_hz = 0.0;
        const double lattice_hz = estimate_lattice_hz(
            row_profiles, geometry, spectral_hz, result.frequency_hz,
            result.runner_up_hz, &lattice_runner_hz, &lattice_advantage);
        result.candidate_scores.push_back(
            static_cast<float>(lattice_hz / 100.0));
        result.candidate_scores.push_back(
            static_cast<float>(lattice_advantage));
        if (lattice_hz >= 1'000.0) {
            result.frequency_hz = lattice_hz;
            result.runner_up_hz = lattice_runner_hz;
            result.strategy += "+high_lattice";
            result.margin = lattice_advantage;
        }
    }
    constexpr double kLowFrequencyMaximumHz = 20'000.0;
    constexpr double kCentroidMaximumResidual = 0.35;
    if (result.valid &&
        (result.frequency_hz <= kLowFrequencyMaximumHz ||
         result.runner_up_hz <= kLowFrequencyMaximumHz)) {
        const FrequencyEstimate centroid = estimate_centroid_code(
            profiles, geometry);
        result.candidate_scores.push_back(static_cast<float>(
            centroid.frequency_hz / 100.0));
        result.candidate_scores.push_back(static_cast<float>(centroid.score));
        if (centroid.valid && centroid.score <= kCentroidMaximumResidual) {
            const double edge_primary_hz = result.frequency_hz;
            const double edge_runner_hz = result.runner_up_hz;
            result.frequency_hz = centroid.frequency_hz;
            result.runner_up_hz =
                std::abs(edge_primary_hz - centroid.frequency_hz) >= 1.0
                    ? edge_primary_hz
                    : edge_runner_hz;
            result.score = centroid.score;
            result.runner_up_score = centroid.runner_up_score;
            result.margin = centroid.margin;
            result.fitted_phase_rad = centroid.fitted_phase_rad;
            result.observed_cycles = centroid.observed_cycles;
            result.strategy += "+low_centroid";
        } else {
            result.strategy += "+low_centroid_rejected";
        }
    }
    return result;
}

void PhaseCodeMatcher::reset() {
    frames_.clear();
    row_frames_.clear();
}

std::size_t PhaseCodeMatcher::frame_count() const { return frames_.size(); }

}  // namespace task5
