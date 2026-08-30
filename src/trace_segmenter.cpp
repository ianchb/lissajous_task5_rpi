#include "task5/trace_segmenter.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace task5 {
namespace {

cv::Mat remove_idle_line_and_noise(const cv::Mat& input,
                                   const SegmenterConfig& config) {
    cv::Mat cleaned = input.clone();
    const int height = cleaned.rows;
    const int width = cleaned.cols;
    const int start = std::clamp(static_cast<int>(height * 0.35), 0, height);
    const int stop = std::clamp(static_cast<int>(height * 0.65), start + 1,
                                height);
    int center_row = start;
    int center_count = 0;
    for (int row = start; row < stop; ++row) {
        const int count = cv::countNonZero(cleaned.row(row));
        if (count > center_count) {
            center_count = count;
            center_row = row;
        }
    }
    if (center_count >= static_cast<int>(width * 0.55)) {
        const int half_width = std::max(2, static_cast<int>(height * 0.014));
        cv::rectangle(cleaned,
                      cv::Point(0, std::max(0, center_row - half_width)),
                      cv::Point(width - 1,
                                std::min(height - 1, center_row + half_width)),
                      cv::Scalar(0), cv::FILLED);
    }

    cv::Mat labels, stats, centroids;
    const int components = cv::connectedComponentsWithStats(
        cleaned, labels, stats, centroids, 8, CV_32S);
    cv::Mat result = cv::Mat::zeros(cleaned.size(), CV_8U);
    std::vector<double> scores(static_cast<size_t>(components), 0.0);
    for (int label = 1; label < components; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const int component_width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int component_height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        if (area < std::max(1, config.minimum_component_area) ||
            std::max(component_width, component_height) < 8) continue;
        const double span = static_cast<double>(
            std::max(component_width, component_height)) /
            std::max(width, height);
        scores[static_cast<size_t>(label)] = area * (1.0 + 2.0 * span);
    }
    for (int label = 1; label < components; ++label) {
        // The lower, overexposed band can be much larger than the useful
        // short scan above it. Once the 8x8 graticule is the frozen ROI there
        // are no bezel tick marks here; side extrema are real high-frequency
        // samples and must be retained.
        if (scores[static_cast<size_t>(label)] > 0.0) {
            result.setTo(255, labels == label);
        }
    }
    return result;
}

}  // namespace

TraceSegmenter::TraceSegmenter(SegmenterConfig config) : config_(config) {}

TraceObservation TraceSegmenter::process(const cv::Mat& normalized_bgr,
                                         uint64_t sequence,
                                         Clock::time_point timestamp) const {
    TraceObservation observation;
    observation.normalized_bgr = normalized_bgr.clone();
    observation.frame_sequence = sequence;
    observation.timestamp = timestamp;
    if (normalized_bgr.empty()) {
        return observation;
    }

    cv::Mat hsv;
    cv::cvtColor(normalized_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat saturated;
    cv::inRange(hsv,
                cv::Scalar(config_.hue_low, config_.saturation_low,
                           config_.value_low),
                cv::Scalar(config_.hue_high, 255, 255), saturated);

    // Scope traces often lose saturation at the bright core. Keep pale yellow
    // only when red/green dominate blue; this rejects white grid reflections.
    cv::Mat pale_hsv;
    cv::inRange(hsv,
                cv::Scalar(config_.hue_low, 0, config_.pale_value_low),
                cv::Scalar(config_.hue_high, config_.pale_saturation_high, 255),
                pale_hsv);
    cv::Mat bgr_channels[3];
    cv::split(normalized_bgr, bgr_channels);
    cv::Mat rg_dominates = (bgr_channels[1] > bgr_channels[0]) &
                           (bgr_channels[2] > bgr_channels[0]);
    pale_hsv &= rg_dominates;

    cv::Mat mask = saturated | pale_hsv;
    const int radius = std::max(0, config_.morphology_radius);
    if (radius > 0) {
        const int diameter = 2 * radius + 1;
        const cv::Mat element = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(diameter, diameter));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element);
    }
    observation.mask = remove_idle_line_and_noise(mask, config_);

    std::vector<cv::Point> points;
    cv::findNonZero(observation.mask, points);
    if (points.empty()) {
        return observation;
    }
    std::vector<int> xs;
    std::vector<int> ys;
    xs.reserve(points.size());
    ys.reserve(points.size());
    for (const cv::Point& point : points) {
        xs.push_back(point.x);
        ys.push_back(point.y);
    }
    const auto percentile = [](std::vector<int> values, double fraction) {
        const size_t index = static_cast<size_t>(
            std::clamp(fraction * static_cast<double>(values.size() - 1),
                       0.0, static_cast<double>(values.size() - 1)));
        std::nth_element(values.begin(), values.begin() + index, values.end());
        return static_cast<double>(values[index]);
    };
    const double x_low = percentile(xs, 0.002);
    const double x_high = percentile(xs, 0.998);
    const double y_low = percentile(ys, 0.002);
    const double y_high = percentile(ys, 0.998);
    observation.center = {0.5 * (x_low + x_high), 0.5 * (y_low + y_high)};
    observation.amplitude = {std::max(1.0, 0.5 * (x_high - x_low)),
                             std::max(1.0, 0.5 * (y_high - y_low))};
    observation.foreground_fraction =
        static_cast<double>(points.size()) /
        static_cast<double>(observation.mask.total());
    const double extent =
        std::min((x_high - x_low) / normalized_bgr.cols,
                 (y_high - y_low) / normalized_bgr.rows);
    observation.quality = std::clamp(
        std::min(1.0, points.size() / 1200.0) * std::min(1.0, extent / 0.25),
        0.0, 1.0);
    return observation;
}

}  // namespace task5
