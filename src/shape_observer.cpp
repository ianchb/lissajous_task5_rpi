#include "task5/shape_observer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace task5 {
namespace {

double percentile(std::vector<int> values, double fraction) {
    if (values.empty()) return 0.0;
    const size_t index = static_cast<size_t>(std::clamp(
        fraction * static_cast<double>(values.size() - 1), 0.0,
        static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return static_cast<double>(values[index]);
}

}  // namespace

cv::Mat ShapeObserver::isolate_trace(const cv::Mat& mask) {
    if (mask.empty()) return {};
    cv::Mat labels, stats, centroids;
    const int count = cv::connectedComponentsWithStats(
        mask, labels, stats, centroids, 8, CV_32S);
    int best_label = 0;
    double best_score = 0.0;
    for (int label = 1; label < count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const double score = area * (1.0 +
            static_cast<double>(width + height) / (mask.cols + mask.rows));
        if (score > best_score) {
            best_score = score;
            best_label = label;
        }
    }
    cv::Mat result = cv::Mat::zeros(mask.size(), CV_8U);
    if (best_label != 0) result.setTo(255, labels == best_label);
    return result;
}

double ShapeObserver::symmetry_score(const cv::Mat& mask, int flip_code) {
    const int foreground = cv::countNonZero(mask);
    if (foreground == 0) return 0.0;
    cv::Mat flipped;
    cv::flip(mask, flipped, flip_code);
    const int tolerance = std::max(2,
        static_cast<int>(std::lround(std::min(mask.rows, mask.cols) * 0.02)));
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, {2 * tolerance + 1, 2 * tolerance + 1});
    cv::Mat neighborhood, flipped_neighborhood;
    cv::dilate(mask, neighborhood, kernel);
    cv::dilate(flipped, flipped_neighborhood, kernel);
    cv::Mat forward, reverse;
    cv::bitwise_and(mask, flipped_neighborhood, forward);
    cv::bitwise_and(flipped, neighborhood, reverse);
    return 0.5 * (static_cast<double>(cv::countNonZero(forward)) / foreground +
                  static_cast<double>(cv::countNonZero(reverse)) / foreground);
}

ShapeMetrics ShapeObserver::analyze(
    const TraceObservation& observation) const {
    ShapeMetrics result;
    const cv::Mat trace = isolate_trace(observation.mask);
    std::vector<cv::Point> points;
    cv::findNonZero(trace, points);
    if (points.size() < 80 || trace.cols < 16 || trace.rows < 16)
        return result;

    std::vector<int> xs;
    std::vector<int> ys;
    xs.reserve(points.size());
    ys.reserve(points.size());
    cv::Mat samples(static_cast<int>(points.size()), 2, CV_64F);
    for (size_t index = 0; index < points.size(); ++index) {
        xs.push_back(points[index].x);
        ys.push_back(points[index].y);
        samples.at<double>(static_cast<int>(index), 0) = points[index].x;
        samples.at<double>(static_cast<int>(index), 1) = points[index].y;
    }
    const double x0 = percentile(xs, 0.001);
    const double x1 = percentile(xs, 0.999);
    const double y0 = percentile(ys, 0.001);
    const double y1 = percentile(ys, 0.999);
    const double width = std::max(1.0, x1 - x0 + 1.0);
    const double height = std::max(1.0, y1 - y0 + 1.0);

    cv::PCA pca(samples, cv::Mat(), cv::PCA::DATA_AS_ROW);
    const double major = std::max(pca.eigenvalues.at<double>(0), 1.0e-9);
    const double minor = std::max(pca.eigenvalues.at<double>(1), 1.0e-9);
    cv::Mat centered;
    cv::subtract(samples, cv::repeat(pca.mean, samples.rows, 1), centered);
    cv::Mat covariance = centered.t() * centered /
        static_cast<double>(samples.rows);
    const double covariance_xy = covariance.at<double>(0, 1);
    const double variance_x = covariance.at<double>(0, 0);
    const double variance_y = covariance.at<double>(1, 1);

    const double center_x = 0.5 * (x0 + x1);
    const double center_y = 0.5 * (y0 + y1);
    std::vector<double> radii;
    radii.reserve(points.size());
    double moment_sum = 0.0;
    double nx2_sum = 0.0;
    for (const cv::Point& point : points) {
        const double nx = (point.x - center_x) / std::max(width * 0.5, 1.0);
        const double ny = (point.y - center_y) / std::max(height * 0.5, 1.0);
        radii.push_back(std::hypot(nx, ny));
        nx2_sum += nx * nx;
    }
    const double mean_nx2 = nx2_sum / points.size();
    for (const cv::Point& point : points) {
        const double nx = (point.x - center_x) / std::max(width * 0.5, 1.0);
        const double ny = (point.y - center_y) / std::max(height * 0.5, 1.0);
        moment_sum += (nx * nx - mean_nx2) * ny;
    }
    const double radius_mean = std::accumulate(
        radii.begin(), radii.end(), 0.0) / radii.size();
    double radius_variance = 0.0;
    for (double radius : radii)
        radius_variance += (radius - radius_mean) * (radius - radius_mean);
    radius_variance /= radii.size();

    const int local_x0 = std::clamp(static_cast<int>(std::floor(x0)),
                                    0, trace.cols - 1);
    const int local_y0 = std::clamp(static_cast<int>(std::floor(y0)),
                                    0, trace.rows - 1);
    const int local_x1 = std::clamp(static_cast<int>(std::ceil(x1)),
                                    local_x0, trace.cols - 1);
    const int local_y1 = std::clamp(static_cast<int>(std::ceil(y1)),
                                    local_y0, trace.rows - 1);
    const cv::Mat local = trace(cv::Rect(local_x0, local_y0,
        local_x1 - local_x0 + 1, local_y1 - local_y0 + 1));
    const int half_window = std::max(2, std::min(local.cols, local.rows) / 30);
    const int cx = local.cols / 2;
    const int cy = local.rows / 2;
    const cv::Rect crossing_rect(
        std::max(0, cx - half_window), std::max(0, cy - half_window),
        std::min(local.cols, cx + half_window + 1) -
            std::max(0, cx - half_window),
        std::min(local.rows, cy + half_window + 1) -
            std::max(0, cy - half_window));
    const int crossing_half_width = std::max(
        3, static_cast<int>(std::lround(width / 20.0)));
    const int crossing_x0 = std::clamp(
        static_cast<int>(std::lround(center_x)) - crossing_half_width,
        0, trace.cols - 1);
    const int crossing_x1 = std::clamp(
        static_cast<int>(std::lround(center_x)) + crossing_half_width,
        crossing_x0, trace.cols - 1);
    double crossing_y_sum = 0.0;
    int crossing_points = 0;
    for (const cv::Point& point : points) {
        if (point.x < crossing_x0 || point.x > crossing_x1) continue;
        crossing_y_sum += point.y;
        ++crossing_points;
    }

    result.valid = true;
    result.pixels = static_cast<int>(points.size());
    result.bbox = {x0, y0, width, height};
    result.center_div = {center_x * 8.0 / trace.cols,
                         center_y * 8.0 / trace.rows};
    result.span_div = {width * 8.0 / trace.cols,
                       height * 8.0 / trace.rows};
    result.coverage = points.size() / (width * height);
    result.correlation = covariance_xy /
        std::sqrt(std::max(variance_x * variance_y, 1.0e-9));
    result.thinness = std::sqrt(minor / major);
    const cv::Vec2d minor_axis(pca.eigenvectors.at<double>(1, 0),
                               pca.eigenvectors.at<double>(1, 1));
    const double div_per_pixel = std::hypot(
        minor_axis[0] * 8.0 / trace.cols,
        minor_axis[1] * 8.0 / trace.rows);
    result.minor_rms_div = std::sqrt(minor) * div_per_pixel;
    result.radial_cv = std::sqrt(radius_variance) /
        std::max(radius_mean, 1.0e-6);
    result.symmetry_x = symmetry_score(local, 0);
    result.symmetry_y = symmetry_score(local, 1);
    result.crossing_fill = static_cast<double>(
        cv::countNonZero(local(crossing_rect))) / crossing_rect.area();
    if (crossing_points >= 10) {
        const double crossing_y = crossing_y_sum / crossing_points;
        result.crossing_offset_y_div =
            crossing_y * 8.0 / trace.rows - 4.0;
    }
    result.phase_feature = moment_sum / points.size();
    return result;
}

double ShapeObserver::search_score(AutoMode mode,
                                   const ShapeMetrics& metrics) const {
    if (!metrics.valid || metrics.coverage >= 0.35) return -1000.0;
    if (mode == AutoMode::Line) {
        return -5.0 * metrics.thinness -
               1.5 * std::abs(metrics.correlation + 1.0);
    }
    if (mode == AutoMode::Circle) {
        return -3.0 * metrics.radial_cv -
               2.0 * std::abs(metrics.correlation) -
               1.5 * std::abs(metrics.span_div.x - metrics.span_div.y);
    }
    return 1.5 * (metrics.symmetry_x + metrics.symmetry_y) +
           metrics.crossing_fill -
           1.5 * std::abs(metrics.span_div.x - metrics.span_div.y) -
           6.0 * std::abs(metrics.phase_feature) -
           2.0 * std::abs(metrics.crossing_offset_y_div);
}

bool ShapeObserver::shape_ok(AutoMode mode,
                             const ShapeMetrics& metrics) const {
    if (!metrics.valid || metrics.coverage >= 0.35) return false;
    if (mode == AutoMode::Line) {
        // A locked line need not fill the calibrated grid: input amplitude and
        // oscilloscope volts/div both change its size. Require enough visible
        // extent to reject short trace fragments, then judge the line by its
        // scale-independent geometry.
        const bool visible_extent = metrics.span_div.x >= 4.5 &&
                                    metrics.span_div.y >= 3.8;
        const bool compact_trace = metrics.thinness < 0.28 &&
                                   metrics.correlation < -0.82 &&
                                   metrics.minor_rms_div < 0.30;
        // Preserve the camera-thick full-grid line regression while the
        // phase acquisition sequence itself follows the initial controller.
        const bool full_grid_crisp_trace = metrics.thinness < 0.17 &&
                                           metrics.correlation < -0.94 &&
                                           metrics.minor_rms_div < 0.50;
        return visible_extent && (compact_trace || full_grid_crisp_trace);
    }
    const bool vertical_amplitude_ok =
        std::abs(metrics.span_div.y - 8.0) <= 0.25;
    if (mode == AutoMode::Circle) {
        // Camera perspective and scope persistence bias the measured
        // correlation by roughly 0.2 even for a visually round trace.
        return vertical_amplitude_ok && std::abs(metrics.correlation) < 0.28 &&
               metrics.radial_cv < 0.23 &&
               std::abs(metrics.span_div.x - metrics.span_div.y) < 0.35 &&
               std::abs(metrics.symmetry_x - metrics.symmetry_y) < 0.50 &&
               std::abs(metrics.phase_feature) < 0.08;
    }
    return vertical_amplitude_ok && metrics.thinness > 0.55 &&
           metrics.symmetry_x > 0.52 && metrics.symmetry_y > 0.75 &&
           metrics.crossing_fill > 0.08 &&
           std::abs(metrics.phase_feature) < 0.015 &&
           std::abs(metrics.crossing_offset_y_div) < 0.25;
}

bool ShapeObserver::phase_servo_needed(
    AutoMode mode, const ShapeMetrics& metrics) const {
    if (!metrics.valid || metrics.coverage >= 0.35) return false;
    // Start tracking before the permissive scoring threshold declares a
    // visual unlock. This leaves enough time for a four-code probe to catch
    // the moving optimum without visibly pulling a good target away.
    if (mode == AutoMode::Line) {
        return metrics.correlation > -0.965 || metrics.thinness > 0.12 ||
               metrics.minor_rms_div > 0.42;
    }
    if (mode == AutoMode::Circle) {
        return std::abs(metrics.correlation) > 0.24 ||
               std::abs(metrics.span_div.x - metrics.span_div.y) > 0.22;
    }
    return std::abs(metrics.phase_feature) > 0.008 ||
           std::abs(metrics.crossing_offset_y_div) > 0.12 ||
           metrics.symmetry_x < 0.72;
}

bool ShapeObserver::frequency_shape_coherent(
    AutoMode, const ShapeMetrics& metrics) const {
    // A bounded trace at several independent phase commands proves that the
    // output and input frequencies agree even when the target phase has not
    // been reached yet. This applies to the 2:1 figure-eight output as well;
    // without it, a near-target figure eight could fall back to the sparse
    // frequency code after one unsuccessful phase prediction.
    return metrics.valid && metrics.coverage < 0.35 &&
           metrics.span_div.x >= 4.5 && metrics.span_div.y >= 3.8 &&
           metrics.span_div.x <= 8.25 && metrics.span_div.y <= 8.25;
}

bool ShapeObserver::frequency_mismatch_fill(
    const ShapeMetrics& metrics) const {
    if (!metrics.valid || metrics.coverage < 0.35 ||
        metrics.span_div.x < 7.0 || metrics.span_div.y < 4.0 ||
        metrics.crossing_fill < 0.80)
        return false;
    // A slightly drifting frequency does not always fill the two axes
    // symmetrically: the camera and scope persistence can leave one axis at
    // 0.65--0.85 even though the centre and both outer edges are saturated.
    // Require evidence from both axes, but do not wait for an ideal rectangle.
    const double symmetry_mean = 0.5 *
        (metrics.symmetry_x + metrics.symmetry_y);
    return std::min(metrics.symmetry_x, metrics.symmetry_y) >= 0.52 &&
           std::max(metrics.symmetry_x, metrics.symmetry_y) >= 0.78 &&
           symmetry_mean >= 0.68;
}

bool ShapeObserver::frequency_mismatch_harmonic(
    AutoMode mode, const ShapeMetrics& metrics) const {
    return mode == AutoMode::Circle && metrics.valid &&
           metrics.coverage < 0.35 && metrics.thinness > 0.55 &&
           std::abs(metrics.symmetry_x - metrics.symmetry_y) > 0.55 &&
           std::abs(metrics.phase_feature) > 0.05 &&
           metrics.crossing_fill < 0.15;
}

}  // namespace task5
