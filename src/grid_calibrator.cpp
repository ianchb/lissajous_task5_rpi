#include "task5/grid_calibrator.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace task5 {
namespace {

bool grid_debug_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("TASK5_GRID_DEBUG");
        return value && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

struct ScreenCandidate {
    double score = 0.0;
    cv::Rect rect;
    int dark_cutoff = 150;
};

struct LatticeSequence {
    double score = 0.0;
    int origin = 0;
    int spacing = 0;
};

struct FittedLine {
    double slope = 0.0;
    double intercept = 0.0;
    double rms = std::numeric_limits<double>::infinity();
    bool valid = false;
};

struct RefinedGrid {
    std::array<cv::Point2f, 4> corners{};
    double reprojection_rms = std::numeric_limits<double>::infinity();
    bool nominal = false;
    bool ransac = false;
};

struct MagentaMarker {
    cv::Point2d center{};
    int area = 0;
    cv::Rect bounds;
};

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

FittedLine robust_weighted_line(const std::vector<double>& independent,
                                const std::vector<double>& dependent,
                                const std::vector<double>& weights) {
    FittedLine result;
    if (independent.size() != dependent.size() ||
        independent.size() != weights.size() || independent.size() < 12) {
        return result;
    }
    std::vector<uint8_t> keep(independent.size(), 1);
    for (int iteration = 0; iteration < 4; ++iteration) {
        double sw = 0.0, sx = 0.0, sy = 0.0;
        double sxx = 0.0, sxy = 0.0;
        int count = 0;
        for (std::size_t index = 0; index < independent.size(); ++index) {
            if (!keep[index]) continue;
            const double weight = weights[index];
            const double x = independent[index];
            const double y = dependent[index];
            sw += weight;
            sx += weight * x;
            sy += weight * y;
            sxx += weight * x * x;
            sxy += weight * x * y;
            ++count;
        }
        const double determinant = sw * sxx - sx * sx;
        if (count < 12 || sw <= 0.0 || std::abs(determinant) < 1.0e-9)
            return result;
        result.slope = (sw * sxy - sx * sy) / determinant;
        result.intercept = (sy - result.slope * sx) / sw;

        std::vector<double> residuals;
        residuals.reserve(static_cast<std::size_t>(count));
        for (std::size_t index = 0; index < independent.size(); ++index) {
            if (!keep[index]) continue;
            residuals.push_back(dependent[index] -
                (result.slope * independent[index] + result.intercept));
        }
        const double center = median(residuals);
        std::vector<double> deviations;
        deviations.reserve(residuals.size());
        for (double residual : residuals)
            deviations.push_back(std::abs(residual - center));
        const double threshold = std::max(1.2, 3.5 *
            (median(deviations) + 0.15));
        for (std::size_t index = 0; index < independent.size(); ++index) {
            const double residual = dependent[index] -
                (result.slope * independent[index] + result.intercept);
            keep[index] = std::abs(residual - center) <= threshold;
        }
    }

    double weighted_error = 0.0;
    double total_weight = 0.0;
    int count = 0;
    for (std::size_t index = 0; index < independent.size(); ++index) {
        if (!keep[index]) continue;
        const double residual = dependent[index] -
            (result.slope * independent[index] + result.intercept);
        weighted_error += weights[index] * residual * residual;
        total_weight += weights[index];
        ++count;
    }
    result.rms = total_weight > 0.0
        ? std::sqrt(weighted_error / total_weight)
        : std::numeric_limits<double>::infinity();
    result.valid = count >= 12 && std::abs(result.slope) < 0.10 &&
                   result.rms < 1.2;
    return result;
}

FittedLine fit_vertical_line(const cv::Mat& gray, const cv::Mat& excluded,
                             double nominal_x, int y0, int y1) {
    std::vector<double> independent, dependent, weights;
    const int center = static_cast<int>(std::lround(nominal_x));
    constexpr int kRadius = 7;
    for (int y = y0; y <= y1; ++y) {
        const int begin = std::max(0, center - kRadius);
        const int end = std::min(gray.cols, center + kRadius + 1);
        if (!excluded.empty() && cv::countNonZero(
                excluded(cv::Rect(begin, y, end - begin, 1))) > 0) {
            continue;
        }
        std::vector<int> values;
        values.reserve(static_cast<std::size_t>(end - begin));
        for (int x = begin; x < end; ++x)
            values.push_back(gray.at<uint8_t>(y, x));
        if (values.empty()) continue;
        const auto percentile = values.begin() +
            static_cast<std::ptrdiff_t>(values.size() * 35 / 100);
        std::nth_element(values.begin(), percentile, values.end());
        const double baseline = *percentile;
        double strength = 0.0, position_sum = 0.0;
        for (int x = begin; x < end; ++x) {
            const double weight = std::max(
                0.0, gray.at<uint8_t>(y, x) - baseline - 3.0);
            strength += weight;
            position_sum += weight * x;
        }
        if (strength < 18.0) continue;
        independent.push_back(y);
        dependent.push_back(position_sum / strength);
        weights.push_back(strength);
    }
    return robust_weighted_line(independent, dependent, weights);
}

FittedLine fit_horizontal_line(const cv::Mat& gray, const cv::Mat& excluded,
                               double nominal_y, int x0, int x1) {
    std::vector<double> independent, dependent, weights;
    const int center = static_cast<int>(std::lround(nominal_y));
    constexpr int kRadius = 7;
    for (int x = x0; x <= x1; ++x) {
        const int begin = std::max(0, center - kRadius);
        const int end = std::min(gray.rows, center + kRadius + 1);
        if (!excluded.empty() && cv::countNonZero(
                excluded(cv::Rect(x, begin, 1, end - begin))) > 0) {
            continue;
        }
        std::vector<int> values;
        values.reserve(static_cast<std::size_t>(end - begin));
        for (int y = begin; y < end; ++y)
            values.push_back(gray.at<uint8_t>(y, x));
        if (values.empty()) continue;
        const auto percentile = values.begin() +
            static_cast<std::ptrdiff_t>(values.size() * 35 / 100);
        std::nth_element(values.begin(), percentile, values.end());
        const double baseline = *percentile;
        double strength = 0.0, position_sum = 0.0;
        for (int y = begin; y < end; ++y) {
            const double weight = std::max(
                0.0, gray.at<uint8_t>(y, x) - baseline - 3.0);
            strength += weight;
            position_sum += weight * y;
        }
        if (strength < 18.0) continue;
        independent.push_back(x);
        dependent.push_back(position_sum / strength);
        weights.push_back(strength);
    }
    return robust_weighted_line(independent, dependent, weights);
}

cv::Point2f intersection(const FittedLine& vertical,
                         const FittedLine& horizontal) {
    const double denominator = 1.0 - horizontal.slope * vertical.slope;
    const double y = (horizontal.slope * vertical.intercept +
                      horizontal.intercept) / denominator;
    const double x = vertical.slope * y + vertical.intercept;
    return {static_cast<float>(x), static_cast<float>(y)};
}

std::optional<RefinedGrid> refine_projective_grid(
    const cv::Mat& crop, const cv::Mat& excluded, const LatticeSequence& xs,
    const LatticeSequence& ys) {
    static int debug_calls = 0;
    const bool print_debug = grid_debug_enabled() && debug_calls++ < 160;
    std::array<FittedLine, 7> verticals{};
    std::array<FittedLine, 7> horizontals{};
    const int x0 = xs.origin - xs.spacing;
    const int x1 = xs.origin + 7 * xs.spacing;
    const int y0 = ys.origin - ys.spacing;
    const int y1 = ys.origin + 7 * ys.spacing;
    int valid_vertical_count = 0;
    int valid_horizontal_count = 0;
    for (int index = 0; index < 7; ++index) {
        verticals[static_cast<std::size_t>(index)] = fit_vertical_line(
            crop, excluded, xs.origin + index * xs.spacing, y0, y1);
        horizontals[static_cast<std::size_t>(index)] = fit_horizontal_line(
            crop, excluded, ys.origin + index * ys.spacing, x0, x1);
        valid_vertical_count += verticals[static_cast<std::size_t>(index)].valid;
        valid_horizontal_count += horizontals[static_cast<std::size_t>(index)].valid;
    }
    // A trace or a reflection can hide one or two graticule lines. The strict
    // all-point fit below still needs five measurements per axis. The RANSAC
    // fallback keeps that same measurement floor and additionally requires
    // 34/49 geometric inliers later on.
    const bool ransac_support = valid_vertical_count >= 5 &&
                                valid_horizontal_count >= 5;
    if (!ransac_support) {
        if (print_debug) {
            std::cerr << "GRID_DIAG_REFINE_FAIL x_origin=" << xs.origin
                      << " x_spacing=" << xs.spacing
                      << " y_origin=" << ys.origin
                      << " y_spacing=" << ys.spacing
                      << " valid_v=" << valid_vertical_count
                      << " valid_h=" << valid_horizontal_count << "\n";
        }
        return std::nullopt;
    }
    for (int index = 0; index < 7; ++index) {
        auto& vertical = verticals[static_cast<std::size_t>(index)];
        if (!vertical.valid) {
            vertical.slope = 0.0;
            vertical.intercept = xs.origin + index * xs.spacing;
            vertical.rms = 0.0;
            vertical.valid = true;
        }
        auto& horizontal = horizontals[static_cast<std::size_t>(index)];
        if (!horizontal.valid) {
            horizontal.slope = 0.0;
            horizontal.intercept = ys.origin + index * ys.spacing;
            horizontal.rms = 0.0;
            horizontal.valid = true;
        }
    }

    std::vector<cv::Point2f> source, target;
    source.reserve(49);
    target.reserve(49);
    for (int row = 0; row < 7; ++row) {
        for (int column = 0; column < 7; ++column) {
            source.push_back(intersection(
                verticals[static_cast<std::size_t>(column)],
                horizontals[static_cast<std::size_t>(row)]));
            target.emplace_back((column + 1) * 80.0f,
                                (row + 1) * 60.0f);
        }
    }
    // Each point comes from a line that already passed robust fitting. Prefer
    // the all-point solution, then use a bounded RANSAC recovery only when
    // glare has displaced one or two nominal rows.
    cv::Mat homography = cv::findHomography(source, target, 0);
    if (homography.empty())
        return std::nullopt;

    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(source, projected, homography);
    double squared_error = 0.0;
    double maximum_error = 0.0;
    for (std::size_t index = 0; index < target.size(); ++index) {
        const double error = cv::norm(projected[index] - target[index]);
        squared_error += error * error;
        maximum_error = std::max(maximum_error, error);
    }
    double reprojection_rms = std::sqrt(squared_error / target.size());
    bool ransac_recovery = false;
    // Camera perspective and the 640x480 resampling make a perfectly fitted
    // grid land a few normalized pixels away from an exact homography. Keep
    // the check strict enough to reject unrelated text, but do not reject a
    // real graticule merely because one corner is 4-5 pixels off.
    if (reprojection_rms > 2.25 || maximum_error > 5.0) {
        cv::Mat inlier_mask;
        const cv::Mat ransac_homography = cv::findHomography(
            source, target, cv::RANSAC, 4.0, inlier_mask);
        int inlier_count = 0;
        double inlier_squared_error = 0.0;
        double inlier_maximum_error = 0.0;
        if (!ransac_homography.empty() && !inlier_mask.empty()) {
            std::vector<cv::Point2f> ransac_projected;
            cv::perspectiveTransform(source, ransac_projected,
                                     ransac_homography);
            for (std::size_t index = 0; index < target.size(); ++index) {
                if (inlier_mask.at<uint8_t>(static_cast<int>(index), 0) == 0)
                    continue;
                const double error = cv::norm(
                    ransac_projected[index] - target[index]);
                inlier_squared_error += error * error;
                inlier_maximum_error = std::max(inlier_maximum_error, error);
                ++inlier_count;
            }
        }
        const double inlier_rms = inlier_count > 0 ?
            std::sqrt(inlier_squared_error / inlier_count) :
            std::numeric_limits<double>::infinity();
        if (ransac_support && inlier_count >= 34 && inlier_rms <= 1.75 &&
            inlier_maximum_error <= 4.0) {
            homography = ransac_homography;
            reprojection_rms = inlier_rms;
            maximum_error = inlier_maximum_error;
            ransac_recovery = true;
        } else {
            if (print_debug) {
                std::cerr << "GRID_DIAG_REFINE_ERROR x_origin=" << xs.origin
                          << " x_spacing=" << xs.spacing
                          << " y_origin=" << ys.origin
                          << " y_spacing=" << ys.spacing
                          << " rms=" << reprojection_rms
                          << " max=" << maximum_error
                          << " ransac_inliers=" << inlier_count
                          << " ransac_rms=" << inlier_rms << "\n";
            }
            return std::nullopt;
        }
    }

    const std::vector<cv::Point2f> ideal_corners{
        {0.0f, 0.0f}, {640.0f, 0.0f},
        {640.0f, 480.0f}, {0.0f, 480.0f}};
    std::vector<cv::Point2f> corners;
    cv::perspectiveTransform(ideal_corners, corners, homography.inv());
    if (corners.size() != 4) {
        if (print_debug) {
            std::cerr << "GRID_DIAG_REFINE_CORNER_FAIL x_origin=" << xs.origin
                      << " x_spacing=" << xs.spacing
                      << " y_origin=" << ys.origin
                      << " y_spacing=" << ys.spacing
                      << " corners=" << corners.size() << "\n";
        }
        return std::nullopt;
    }
    if (print_debug) {
        std::cerr << "GRID_DIAG_REFINE_OK x_origin=" << xs.origin
                  << " x_spacing=" << xs.spacing
                  << " y_origin=" << ys.origin
                  << " y_spacing=" << ys.spacing
                  << " rms=" << reprojection_rms
                  << " max=" << maximum_error
                  << " ransac=" << ransac_recovery << "\n";
    }
    return RefinedGrid{{corners[0], corners[1], corners[2], corners[3]},
                       reprojection_rms, false, ransac_recovery};
}

// The trace is yellow while the graticule is gray.  Ambient light can wash
// the trace core almost white, so use the same pale-yellow colour dominance
// rule as TraceSegmenter instead of relying on saturation alone.
cv::Mat trace_exclusion_mask(const cv::Mat& gray, const cv::Mat& bgr) {
    if (gray.empty() || bgr.empty() || gray.size() != bgr.size())
        return {};

    cv::Mat hsv, saturated, pale;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(12, 35, 55), cv::Scalar(48, 255, 255),
                saturated);
    cv::inRange(hsv, cv::Scalar(12, 0, 105), cv::Scalar(48, 145, 255),
                pale);
    cv::Mat channels[3];
    cv::split(bgr, channels);
    pale &= (channels[1] > channels[0]) & (channels[2] > channels[0]);
    cv::Mat yellow = saturated | pale;
    if (cv::countNonZero(yellow) == 0)
        return yellow;

    // A one-pixel scope cursor can cross and join the actual trace. Remove
    // that hairline before expanding the mask around the much thicker trace.
    cv::morphologyEx(yellow, yellow, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    cv::dilate(yellow, yellow,
               cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7}));
    return yellow;
}

cv::Mat primary_grid_trace(const cv::Mat& excluded,
                           const cv::Rect& screen) {
    if (excluded.empty()) return {};
    cv::Mat labels, stats, centroids;
    const int count = cv::connectedComponentsWithStats(
        excluded, labels, stats, centroids, 8, CV_32S);
    int best_label = -1;
    int best_area = 0;
    for (int label = 1; label < count; ++label) {
        const cv::Rect bounds(
            stats.at<int>(label, cv::CC_STAT_LEFT),
            stats.at<int>(label, cv::CC_STAT_TOP),
            stats.at<int>(label, cv::CC_STAT_WIDTH),
            stats.at<int>(label, cv::CC_STAT_HEIGHT));
        const cv::Rect overlap = bounds & screen;
        if (overlap.area() < 0.80 * bounds.area()) continue;
        if (bounds.width < 0.25 * screen.width) continue;
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area > best_area) {
            best_area = area;
            best_label = label;
        }
    }
    if (best_label < 0) return {};
    return labels == best_label;
}

std::vector<MagentaMarker> magenta_markers(const cv::Mat& bgr) {
    if (bgr.empty()) return {};
    cv::Mat hsv, saturated, pale;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(120, 45, 45), cv::Scalar(179, 255, 255),
                saturated);
    cv::inRange(hsv, cv::Scalar(120, 12, 80), cv::Scalar(179, 150, 255),
                pale);
    cv::Mat channels[3];
    cv::split(bgr, channels);
    cv::Mat red_dominant, blue_dominant, dominance;
    cv::compare(channels[2], channels[1] + 12, red_dominant,
                cv::CMP_GE);
    cv::compare(channels[0], channels[1] + 4, blue_dominant,
                cv::CMP_GE);
    dominance = red_dominant & blue_dominant;
    cv::Mat mask = (saturated | pale) & dominance;
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));

    cv::Mat labels, stats, centroids;
    const int count = cv::connectedComponentsWithStats(
        mask, labels, stats, centroids, 8, CV_32S);
    std::vector<MagentaMarker> result;
    for (int label = 1; label < count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const cv::Rect bounds(
            stats.at<int>(label, cv::CC_STAT_LEFT),
            stats.at<int>(label, cv::CC_STAT_TOP),
            stats.at<int>(label, cv::CC_STAT_WIDTH),
            stats.at<int>(label, cv::CC_STAT_HEIGHT));
        if (area < 10 || area > 250 || bounds.width < 3 ||
            bounds.height < 3 || bounds.width > 20 || bounds.height > 20) {
            continue;
        }
        result.push_back({
            {centroids.at<double>(label, 0),
             centroids.at<double>(label, 1)},
            area, bounds});
    }
    return result;
}

std::vector<ScreenCandidate> screen_candidates(const cv::Mat& gray) {
    const bool debug = grid_debug_enabled();
    static int debug_calls = 0;
    const bool print_debug = debug && debug_calls++ < 3;
    cv::Mat flattened = gray.reshape(1, 1).clone();
    cv::sort(flattened, flattened, cv::SORT_ASCENDING);
    const int percentile_index = std::clamp(
        static_cast<int>(0.32 * flattened.cols), 0, flattened.cols - 1);
    // The camera's automatic exposure can move the whole scope image above
    // gray 100. A fixed upper bound of 75 then sees only bezel artifacts and
    // produces no screen candidate at all. Keep the percentile-relative
    // threshold bounded so ordinary dark frames retain the old behavior.
    const int threshold = std::clamp(
        static_cast<int>(flattened.at<uint8_t>(0, percentile_index)), 35, 180);
    cv::Mat dark;
    cv::threshold(gray, dark, threshold, 255, cv::THRESH_BINARY_INV);
    cv::morphologyEx(dark, dark, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, {7, 7}));
    cv::morphologyEx(dark, dark, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dark, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    if (print_debug) {
        std::cerr << "GRID_DIAG_DARK threshold=" << threshold
                  << " dark_pixels=" << cv::countNonZero(dark)
                  << " contours=" << contours.size() << "\n";
    }
    std::vector<ScreenCandidate> result;
    for (const auto& contour : contours) {
        const cv::Rect rect = cv::boundingRect(contour);
        const double area_fraction = static_cast<double>(rect.area()) /
                                     std::max(1, gray.rows * gray.cols);
        const double aspect = static_cast<double>(rect.width) /
                              std::max(1, rect.height);
        if (print_debug) {
            std::cerr << "GRID_DIAG_CONTOUR x=" << rect.x
                      << " y=" << rect.y << " w=" << rect.width
                      << " h=" << rect.height
                      << " area_fraction=" << area_fraction
                      << " aspect=" << aspect << "\n";
        }
        // A phone camera can compress the scope's nominally wide display into
        // a near-square candidate when the optical axis is oblique. The
        // lattice and projective-error checks below still reject unrelated
        // dark regions, so do not discard that valid geometry here.
        if (area_fraction < 0.12 || aspect < 0.80 || aspect > 2.4)
            continue;
        const cv::Mat crop = gray(rect);
        const int candidate_dark_cutoff = std::min(threshold + 45, 220);
        const double dark_fraction = static_cast<double>(cv::countNonZero(
            crop < candidate_dark_cutoff)) / std::max(1, rect.area());
        if (print_debug)
            std::cerr << "GRID_DIAG_CONTOUR_DARK value=" << dark_fraction
                      << " accepted=" << (dark_fraction >= 0.58) << "\n";
        if (dark_fraction < 0.58) continue;
        const double fill = cv::contourArea(contour) /
                            std::max(1, rect.area());
        cv::Rect candidate_rect = rect;
        // Reflections can split the upper screen from its dark contour. A
        // very wide candidate is a clipped lower screen; restore its height
        // from the display geometry while retaining the reliable lower edge.
        if (aspect > 1.90) {
            constexpr double kDisplayAspect = 1.65;
            const int target_height = static_cast<int>(std::lround(
                rect.width / kDisplayAspect));
            const int bottom = std::min(gray.rows, rect.y + rect.height);
            const int top = std::max(0, bottom - target_height);
            candidate_rect.y = top;
            candidate_rect.height = bottom - top;
        }
        result.push_back({2.0 * area_fraction + dark_fraction + 0.3 * fill,
                          candidate_rect, candidate_dark_cutoff});
    }
    std::sort(result.begin(), result.end(),
              [](const auto& left, const auto& right) {
                  return left.score > right.score;
              });
    return result;
}

std::vector<LatticeSequence> lattice_sequences(const cv::Mat& projection,
                                                int count = 7) {
    CV_Assert(projection.rows == 1 && projection.type() == CV_32F);
    const int length = projection.cols;
    cv::Mat sorted;
    cv::sort(projection, sorted, cv::SORT_ASCENDING);
    const double baseline = sorted.at<float>(
        0, std::clamp(length / 2, 0, length - 1)) + 1.0;
    cv::Mat local_peak;
    cv::dilate(projection, local_peak,
               cv::getStructuringElement(cv::MORPH_RECT, {5, 1}));
    std::vector<LatticeSequence> candidates;
    const int minimum_spacing = std::max(7, static_cast<int>(length * 0.045));
    const int maximum_spacing = std::max(
        minimum_spacing + 1, static_cast<int>(length * 0.12));
    for (int spacing = minimum_spacing; spacing <= maximum_spacing; ++spacing) {
        for (int origin = 3;
             origin + (count - 1) * spacing < length - 3; ++origin) {
            std::vector<double> values(static_cast<std::size_t>(count));
            double sum = 0.0;
            for (int index = 0; index < count; ++index) {
                values[static_cast<size_t>(index)] =
                    local_peak.at<float>(0, origin + index * spacing) /
                    baseline;
                sum += values[static_cast<size_t>(index)];
            }
            const double mean = sum / count;
            double variance = 0.0;
            for (double value : values) variance += (value - mean) *
                                                     (value - mean);
            std::sort(values.begin(), values.end());
            const double score = mean - 0.5 * std::sqrt(variance / count) +
                                 0.3 * values[1];
            candidates.push_back({score, origin, spacing});
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& left, const auto& right) {
                  return left.score > right.score;
              });
    std::vector<LatticeSequence> distinct;
    for (const auto& candidate : candidates) {
        const bool duplicate = std::any_of(
            distinct.begin(), distinct.end(), [&](const auto& old) {
                return std::abs(candidate.origin - old.origin) <= 4 &&
                       std::abs(candidate.spacing - old.spacing) <= 1;
            });
        if (!duplicate) distinct.push_back(candidate);
        if (distinct.size() == 16) break;
    }
    return distinct;
}

double local_projection_peak(const cv::Mat& projection, int position) {
    CV_Assert(projection.rows == 1 && projection.type() == CV_32F);
    const int begin = std::max(0, position - 2);
    const int end = std::min(projection.cols - 1, position + 2);
    double peak = 0.0;
    for (int index = begin; index <= end; ++index)
        peak = std::max(peak, static_cast<double>(projection.at<float>(0, index)));
    return peak;
}

double boundary_support(const cv::Mat& projection,
                        const LatticeSequence& sequence) {
    double interior_sum = 0.0;
    for (int index = 0; index < 7; ++index) {
        interior_sum += local_projection_peak(
            projection, sequence.origin + index * sequence.spacing);
    }
    const double interior_mean = interior_sum / 7.0;
    if (interior_mean <= 1.0e-9) return 0.0;
    const double first = local_projection_peak(
        projection, sequence.origin - sequence.spacing);
    const double last = local_projection_peak(
        projection, sequence.origin + 7 * sequence.spacing);
    // Both extrapolated edges must have line evidence.  A one-cell shift into
    // a blank menu/status area leaves one edge unsupported even though the
    // seven internal lines remain periodic.
    return std::min(first, last) / interior_mean;
}

double center_axis_support(const cv::Mat& projection,
                           const LatticeSequence& sequence) {
    double other_sum = 0.0;
    for (int index = 0; index < 7; ++index) {
        if (index == 3) continue;
        other_sum += local_projection_peak(
            projection, sequence.origin + index * sequence.spacing);
    }
    const double other_mean = other_sum / 6.0;
    if (other_mean <= 1.0e-9) return 0.0;
    return local_projection_peak(
        projection, sequence.origin + 3 * sequence.spacing) / other_mean;
}

double vertical_band_strength(const cv::Mat& gradient, int x,
                              int y0, int y1) {
    const int left = std::max(0, x - 2);
    const int right = std::min(gradient.cols - 1, x + 2);
    const int top = std::clamp(y0, 0, gradient.rows - 1);
    const int bottom = std::clamp(y1, top, gradient.rows - 1);
    return cv::sum(gradient(cv::Rect(
        left, top, right - left + 1, bottom - top + 1)))[0];
}

double horizontal_band_strength(const cv::Mat& gradient, int y,
                                int x0, int x1) {
    const int top = std::max(0, y - 2);
    const int bottom = std::min(gradient.rows - 1, y + 2);
    const int left = std::clamp(x0, 0, gradient.cols - 1);
    const int right = std::clamp(x1, left, gradient.cols - 1);
    return cv::sum(gradient(cv::Rect(
        left, top, right - left + 1, bottom - top + 1)))[0];
}

double vertical_boundary_support(const cv::Mat& gradient,
                                 const LatticeSequence& xs,
                                 int y0, int y1) {
    double interior_sum = 0.0;
    for (int index = 0; index < 7; ++index) {
        interior_sum += vertical_band_strength(
            gradient, xs.origin + index * xs.spacing, y0, y1);
    }
    const double interior_mean = interior_sum / 7.0;
    if (interior_mean <= 1.0e-9) return 0.0;
    return std::min(
        vertical_band_strength(gradient, xs.origin - xs.spacing, y0, y1),
        vertical_band_strength(gradient, xs.origin + 7 * xs.spacing,
                               y0, y1)) / interior_mean;
}

double horizontal_boundary_support(const cv::Mat& gradient,
                                   const LatticeSequence& ys,
                                   int x0, int x1) {
    double interior_sum = 0.0;
    for (int index = 0; index < 7; ++index) {
        interior_sum += horizontal_band_strength(
            gradient, ys.origin + index * ys.spacing, x0, x1);
    }
    const double interior_mean = interior_sum / 7.0;
    if (interior_mean <= 1.0e-9) return 0.0;
    return std::min(
        horizontal_band_strength(gradient, ys.origin - ys.spacing, x0, x1),
        horizontal_band_strength(gradient, ys.origin + 7 * ys.spacing,
                                 x0, x1)) / interior_mean;
}

std::optional<std::pair<std::array<cv::Point2f, 4>, double>> locate_grid(
    const cv::Mat& gray, const cv::Mat& bgr) {
    const auto screens = screen_candidates(gray);
    const bool debug = grid_debug_enabled();
    if (debug) {
        std::cerr << "GRID_DIAG_FRAME width=" << gray.cols
                  << " height=" << gray.rows
                  << " screen_candidates=" << screens.size() << "\n";
        for (size_t index = 0; index < screens.size(); ++index) {
            const auto& candidate = screens[index];
            std::cerr << "GRID_DIAG_SCREEN index=" << index
                      << " x=" << candidate.rect.x
                      << " y=" << candidate.rect.y
                      << " w=" << candidate.rect.width
                      << " h=" << candidate.rect.height
                      << " score=" << candidate.score << "\n";
        }
    }
    const cv::Mat excluded = trace_exclusion_mask(gray, bgr);
    double best_score = -1.0;
    double best_confidence = 0.0;
    std::array<cv::Point2f, 4> best{};
    int lattice_pairs = 0;
    int cell_rejects = 0;
    int support_rejects = 0;
    int dark_rejects = 0;
    int confidence_rejects = 0;
    int refine_rejects = 0;
    for (size_t screen_index = 0;
        screen_index < std::min<size_t>(3, screens.size()); ++screen_index) {
        const auto& screen_candidate = screens[screen_index];
        const cv::Rect detected_screen = screen_candidate.rect;
        // Strong ambient light can make the upper part of the LCD fail the
        // dark-screen threshold even though the graticule is still visible.
        // Recover that area before lattice search. Keep the horizontal bounds
        // unchanged because they carry the strongest evidence against a
        // one-cell left/right shift.
        const int top_padding = std::min(
            detected_screen.y,
            std::max(12, static_cast<int>(std::lround(
                detected_screen.height * 0.10))));
        const cv::Rect screen(
            detected_screen.x, detected_screen.y - top_padding,
            detected_screen.width, detected_screen.height + top_padding);
        if (debug && top_padding > 0) {
            std::cerr << "GRID_DIAG_SEARCH screen=" << screen_index
                      << " x=" << screen.x << " y=" << screen.y
                      << " w=" << screen.width << " h=" << screen.height
                      << " top_padding=" << top_padding << "\n";
        }
        const cv::Mat crop = gray(screen);
        const cv::Mat color_crop = bgr(screen);
        const auto markers = magenta_markers(color_crop);
        const bool marker_available = std::any_of(
            markers.begin(), markers.end(), [&](const auto& marker) {
                return marker.area >= 15 &&
                    marker.center.x <= 0.58 * screen.width &&
                    marker.center.y >= 0.10 * screen.height &&
                    marker.center.y <= 0.80 * screen.height;
            });
        if (debug) {
            std::cerr << "GRID_DIAG_MARKERS screen=" << screen_index
                      << " available=" << marker_available
                      << " count=" << markers.size();
            for (const auto& marker : markers) {
                std::cerr << " marker=" << marker.center.x << ','
                          << marker.center.y << '/' << marker.area;
            }
            std::cerr << "\n";
        }
        const cv::Mat excluded_crop = excluded.empty()
            ? cv::Mat{} : excluded(screen);
        const cv::Mat primary_trace = primary_grid_trace(excluded, screen);
        const cv::Mat primary_trace_crop = primary_trace.empty()
            ? cv::Mat{} : primary_trace(screen);
        const int primary_trace_area = primary_trace_crop.empty()
            ? 0 : cv::countNonZero(primary_trace_crop);
        const cv::Rect primary_trace_bounds = primary_trace_area > 0
            ? cv::boundingRect(primary_trace_crop) : cv::Rect{};
        // Candidate-independent classification is essential: a shifted grid
        // must not evade the horizontal anchor merely because the same trace
        // extends slightly beyond that wrong candidate rectangle.  Requiring
        // one dominant component prevents any one of the four phase-code bands
        // from being mistaken for the blanking trace.
        const bool horizontal_primary_trace =
            primary_trace_area >= 200 &&
            primary_trace_bounds.width >= 0.50 * screen.width &&
            primary_trace_bounds.width >= 4 * primary_trace_bounds.height;
        cv::Mat gradient_x, gradient_y;
        cv::Sobel(crop, gradient_x, CV_32F, 1, 0, 3);
        cv::Sobel(crop, gradient_y, CV_32F, 0, 1, 3);
        gradient_x = cv::abs(gradient_x);
        gradient_y = cv::abs(gradient_y);
        if (!excluded_crop.empty()) {
            gradient_x.setTo(0, excluded_crop);
            gradient_y.setTo(0, excluded_crop);
        }
        cv::Mat vertical, horizontal;
        cv::reduce(gradient_x, vertical, 0, cv::REDUCE_SUM, CV_32F);
        cv::reduce(gradient_y, horizontal, 1, cv::REDUCE_SUM, CV_32F);
        horizontal = horizontal.t();
        const auto x_sequences = lattice_sequences(vertical);
        const auto y_sequences = lattice_sequences(horizontal);
        if (debug) {
            std::cerr << "GRID_DIAG_LATTICE screen=" << screen_index
                      << " x_candidates=" << x_sequences.size()
                      << " y_candidates=" << y_sequences.size() << "\n";
            for (size_t index = 0; index < x_sequences.size(); ++index) {
                const auto& sequence = x_sequences[index];
                std::cerr << "GRID_DIAG_X index=" << index
                          << " origin=" << sequence.origin
                          << " spacing=" << sequence.spacing
                          << " score=" << sequence.score
                          << " boundary="
                          << boundary_support(vertical, sequence) << "\n";
            }
            for (size_t index = 0; index < y_sequences.size(); ++index) {
                const auto& sequence = y_sequences[index];
                std::cerr << "GRID_DIAG_Y index=" << index
                          << " origin=" << sequence.origin
                          << " spacing=" << sequence.spacing
                          << " score=" << sequence.score
                          << " boundary="
                          << boundary_support(horizontal, sequence) << "\n";
            }
        }
        for (const auto& xs : x_sequences) {
            for (const auto& ys : y_sequences) {
                ++lattice_pairs;
                const double cell_ratio = static_cast<double>(xs.spacing) /
                                          std::max(1, ys.spacing);
                if (cell_ratio < 0.72 || cell_ratio > 1.38) {
                    ++cell_rejects;
                    continue;
                }
                const int x0 = xs.origin - xs.spacing;
                const int y0 = ys.origin - ys.spacing;
                const int x1 = xs.origin + 7 * xs.spacing;
                const int y1 = ys.origin + 7 * ys.spacing;
                // The upper graticule can be clipped by a raised camera.  Let
                // the homography/line-consistency checks decide whether a
                // zero-margin candidate is real rather than rejecting it up
                // front.
                constexpr int kGridMargin = 0;
                if (x0 < kGridMargin || y0 < kGridMargin ||
                    x1 >= screen.width || y1 >= screen.height) continue;
                const double grid_center_x = 0.5 * (x0 + x1) / screen.width;
                const double grid_center_y = 0.5 * (y0 + y1) / screen.height;
                const double layout_dx = (grid_center_x - 0.44) / 0.11;
                const double layout_dy = (grid_center_y - 0.45) / 0.11;
                // If glare joins the display to the black status/menu area,
                // the detected screen rectangle becomes much too tall. Its
                // vertical center then actively favors a grid shifted down by
                // one cell. Horizontal placement is still useful; apply the
                // vertical prior only when the screen shape is credible.
                const double screen_aspect = static_cast<double>(screen.width) /
                                             screen.height;
                const double vertical_layout_factor = screen_aspect >= 1.35
                    ? std::exp(-0.5 * layout_dy * layout_dy)
                    : 1.0;
                const double layout_factor =
                    std::exp(-0.5 * layout_dx * layout_dx) *
                    vertical_layout_factor;
                if (layout_factor < 0.15) continue;
                const cv::Rect grid_rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
                const double marker_axis_y = y0 + 4.0 * ys.spacing;
                double marker_dx = std::numeric_limits<double>::infinity();
                double marker_dy = std::numeric_limits<double>::infinity();
                double marker_error = std::numeric_limits<double>::infinity();
                bool marker_matches = false;
                for (const auto& marker : markers) {
                    if (marker.area < 15 ||
                        marker.center.x > 0.58 * screen.width ||
                        marker.center.y < 0.10 * screen.height ||
                        marker.center.y > 0.80 * screen.height) {
                        continue;
                    }
                    const double dx = (marker.center.x - x0) /
                                      std::max(1, xs.spacing);
                    const double dy = (marker.center.y - marker_axis_y) /
                                      std::max(1, ys.spacing);
                    const double error = std::pow((dx - 0.20) / 0.30, 2.0) +
                                         std::pow(dy / 0.32, 2.0);
                    if (error < marker_error) {
                        marker_error = error;
                        marker_dx = dx;
                        marker_dy = dy;
                    }
                }
                marker_matches = marker_error <= 1.0;
                // The channel-position triangle is an absolute reference for
                // both the left graticule edge and its horizontal centre. It
                // resolves the integer-cell ambiguity left by seven periodic
                // internal lines. Keep the old path when glare or cropping
                // removes the marker entirely.
                if (marker_available && !marker_matches) {
                    ++support_rejects;
                    continue;
                }
                double trace_support_x = 1.0;
                double trace_support_y = 1.0;
                double trace_center_dx = 0.0;
                double trace_center_dy = 0.0;
                double trace_span_x = 0.0;
                double trace_span_y = 0.0;
                double trace_top_gap = 0.0;
                double trace_bottom_gap = 0.0;
                bool trace_anchor_valid = false;
                bool horizontal_trace_anchor_valid = false;
                if (primary_trace_area > 0) {
                    const int padding = std::max(6, xs.spacing / 2);
                    const int support_x0 = std::max(0, x0 - padding);
                    const int support_x1 = std::min(
                        screen.width - 1, x1 + padding);
                    const cv::Rect support_rect(
                        support_x0, 0,
                        support_x1 - support_x0 + 1,
                        screen.height);
                    trace_support_x = static_cast<double>(cv::countNonZero(
                        primary_trace_crop(support_rect))) /
                        primary_trace_area;
                    const int support_y0 = std::max(0, y0 - padding);
                    const int support_y1 = std::min(
                        screen.height - 1, y1 + padding);
                    const cv::Rect vertical_support_rect(
                        0, support_y0, screen.width,
                        support_y1 - support_y0 + 1);
                    trace_support_y = static_cast<double>(cv::countNonZero(
                        primary_trace_crop(vertical_support_rect))) /
                        primary_trace_area;
                    const cv::Rect& trace_bounds = primary_trace_bounds;
                    const double trace_center_x =
                        trace_bounds.x + 0.5 * trace_bounds.width;
                    const double trace_center_y =
                        trace_bounds.y + 0.5 * trace_bounds.height;
                    trace_center_dx = std::abs(
                        trace_center_x - 0.5 * (x0 + x1)) / xs.spacing;
                    trace_center_dy = std::abs(
                        trace_center_y - 0.5 * (y0 + y1)) / ys.spacing;
                    trace_span_x = static_cast<double>(trace_bounds.width) /
                        std::max(1, x1 - x0 + 1);
                    trace_span_y = static_cast<double>(trace_bounds.height) /
                        std::max(1, y1 - y0 + 1);
                    trace_top_gap = std::max(0.0,
                        static_cast<double>(trace_bounds.y - y0) /
                        std::max(1, y1 - y0 + 1));
                    trace_bottom_gap = std::max(0.0,
                        static_cast<double>(y1 -
                            (trace_bounds.y + trace_bounds.height - 1)) /
                        std::max(1, y1 - y0 + 1));
                    // Circle and figure-eight calibration traces occupy most
                    // of the graticule in both axes. Their center resolves the
                    // otherwise ambiguous integer-cell phase of seven evenly
                    // spaced internal lines. If auto-exposure turns one end
                    // of a yellow trace white, the saturated component is
                    // visibly clipped inside the candidate and its centroid
                    // must not be used as an absolute row anchor.
                    trace_anchor_valid = primary_trace_area >= 200 &&
                        trace_span_x >= 0.65 && trace_span_x <= 1.35 &&
                        trace_span_y >= 0.60 && trace_span_y <= 1.35 &&
                        trace_top_gap <= 0.25 && trace_bottom_gap <= 0.25;
                    // With Y blanked during automatic-mode grid acquisition,
                    // the scope shows a pale horizontal X trace whose endpoints
                    // coincide with the graticule sides.  It is an absolute
                    // cell-phase reference and removes the otherwise ambiguous
                    // one/two-cell lattice shift.  Keep this separate from the
                    // two-dimensional circle/figure-eight anchor above.
                    horizontal_trace_anchor_valid = horizontal_primary_trace;
                    // The trace can legitimately extend beyond the graticule
                    // (for example a direct-mode line or a scope cursor).
                    // Use its overlap only as a ranking hint; requiring the
                    // whole yellow component to fit inside the grid rejects
                    // the real grid in that situation.
                }
                if (trace_anchor_valid &&
                    (trace_center_dx > 0.65 || trace_center_dy > 0.65)) {
                    ++support_rejects;
                    continue;
                }
                if (horizontal_trace_anchor_valid &&
                    (trace_center_dx > 0.75 || trace_center_dy > 0.70)) {
                    ++support_rejects;
                    continue;
                }
                const double dark_fraction = static_cast<double>(
                    cv::countNonZero(crop(grid_rect) <
                                     screen_candidate.dark_cutoff)) /
                    std::max(1, grid_rect.area());
                // A thick circle/figure-eight trace can occupy a large part
                // of the cells, and glare can lift the remaining background
                // above the adaptive cutoff.  Geometry and line-fit checks
                // below provide the stronger anti-false-positive gate, so
                // keep a modest dark-pixel floor here.
                constexpr double kMinimumGridDarkFraction = 0.52;
                if (dark_fraction < kMinimumGridDarkFraction) {
                    ++dark_rejects;
                    continue;
                }
                const double lattice_score = std::sqrt(
                    std::max(0.0, xs.score) * std::max(0.0, ys.score));
                const double edge_support_x = std::clamp(
                    vertical_boundary_support(gradient_x, xs, y0, y1),
                    0.0, 1.0);
                const double edge_support_y = std::clamp(
                    horizontal_boundary_support(gradient_y, ys, x0, x1),
                    0.0, 1.0);
                // The extrapolated outer graticule lines are the only
                // reliable absolute reference when a bright trace hides the
                // interior. Keep this as a soft ranking term; the hard gate
                // below is deliberately stricter than the score.
                const double boundary_x = std::clamp(
                    boundary_support(vertical, xs), 0.0, 2.0);
                const double boundary_y = std::clamp(
                    boundary_support(horizontal, ys), 0.0, 2.0);
                const double boundary_factor =
                    (0.72 + 0.28 * std::clamp(boundary_x / 0.80, 0.0, 1.0)) *
                    (0.72 + 0.28 * std::clamp(boundary_y / 0.20, 0.0, 1.0));
                const double center_support_x =
                    center_axis_support(vertical, xs);
                const double center_support_y =
                    center_axis_support(horizontal, ys);
                double trace_anchor_factor = 1.0;
                if (trace_anchor_valid) {
                    trace_anchor_factor = std::exp(-0.5 * (
                        std::pow(trace_center_dx / 0.40, 2.0) +
                        std::pow(trace_center_dy / 0.40, 2.0)));
                } else if (horizontal_trace_anchor_valid) {
                    trace_anchor_factor = std::exp(-0.5 * (
                        std::pow(trace_center_dx / 0.50, 2.0) +
                        std::pow(trace_center_dy / 0.45, 2.0)));
                }
                const double center_axis_factor =
                    (0.75 + 0.25 * std::clamp(
                        center_support_x / 1.30, 0.0, 1.0)) *
                    (0.75 + 0.25 * std::clamp(
                        center_support_y / 1.30, 0.0, 1.0));
                const double edge_factor =
                    (0.60 + 0.40 * edge_support_x) *
                    (0.60 + 0.40 * edge_support_y);
                const double trace_support = std::sqrt(
                    std::max(0.0, trace_support_x * trace_support_y));
                const double selection_score = lattice_score *
                    (0.9 + 0.1 * dark_fraction) *
                    (0.55 + 0.45 * trace_support) * edge_factor *
                    boundary_factor * layout_factor * trace_anchor_factor *
                    center_axis_factor *
                    (marker_matches ? 1.15 : 1.0);
                if (debug && edge_support_x >= 0.40) {
                        std::cerr << "GRID_DIAG_PAIR"
                              << " x=" << xs.origin << '/' << xs.spacing
                              << " y=" << ys.origin << '/' << ys.spacing
                              << " edge=" << edge_support_x << ','
                              << edge_support_y
                              << " boundary=" << boundary_x << ','
                              << boundary_y
                              << " trace=" << trace_support_x << ','
                              << trace_support_y
                              << " trace_center=" << trace_center_dx << ','
                              << trace_center_dy
                              << " trace_span=" << trace_span_x << ','
                              << trace_span_y
                              << " trace_gap=" << trace_top_gap << ','
                              << trace_bottom_gap
                              << " trace_anchor=" << trace_anchor_valid << ','
                              << trace_anchor_factor
                              << " horizontal_trace_anchor="
                              << horizontal_trace_anchor_valid
                              << " center_axis=" << center_support_x << ','
                              << center_support_y
                              << " marker=" << marker_matches << ','
                              << marker_dx << ',' << marker_dy
                              << " layout=" << layout_factor
                              << " dark=" << dark_fraction
                              << " score=" << selection_score << "\n";
                }
                const double confidence = std::min(
                    1.0, std::min(xs.score, ys.score) / 5.0) *
                    std::min(1.0, std::max(0.0,
                        (dark_fraction - kMinimumGridDarkFraction) /
                        (0.72 - kMinimumGridDarkFraction)));
                // When ambient glare washes out the upper graticule, the
                // correct rectangle can be only just dark enough while a
                // one-row-lower rectangle includes the black menu area and
                // appears artificially more confident. Let a centered,
                // high-period candidate reach the geometric fit in that
                // narrow case; the RANSAC and final center gate below still
                // decide whether it is usable.
                const bool centered_glare_candidate =
                    screen_aspect >= 1.35 && vertical_layout_factor >= 0.85 &&
                    std::min(xs.score, ys.score) >= 2.30 &&
                    dark_fraction >= 0.535;
                if ((confidence >= 0.30 || centered_glare_candidate) &&
                    selection_score > best_score) {
                    auto refined = refine_projective_grid(
                        crop, excluded_crop, xs, ys);
                    // A dense trace can corrupt the projective fit even when
                    // the lattice period and both outer edges are clear. In
                    // that case use the nominal rectangle, but only when the
                    // edge pair agrees on both axes. This prevents a status
                    // bar from fabricating a seventh row and still rejects
                    // the one-cell-shifted trace candidate above.
                    const bool strong_outer_edges =
                        edge_support_x >= 0.70 && edge_support_y >= 0.12 &&
                        boundary_x >= 0.80 && boundary_y >= 0.15;
                    if (!refined && strong_outer_edges &&
                        layout_factor >= 0.45 &&
                        std::min(xs.score, ys.score) >= 2.0) {
                        refined = RefinedGrid{{
                            cv::Point2f(static_cast<float>(x0),
                                        static_cast<float>(y0)),
                            cv::Point2f(static_cast<float>(x1),
                                        static_cast<float>(y0)),
                            cv::Point2f(static_cast<float>(x1),
                                        static_cast<float>(y1)),
                            cv::Point2f(static_cast<float>(x0),
                                        static_cast<float>(y1))},
                            std::numeric_limits<double>::infinity(), true};
                        if (debug) {
                            std::cerr << "GRID_DIAG_BOUNDARY_NOMINAL"
                                      << " x_origin=" << xs.origin
                                      << " x_spacing=" << xs.spacing
                                      << " y_origin=" << ys.origin
                                      << " y_spacing=" << ys.spacing
                                  << " edge=" << edge_support_x << ','
                                      << edge_support_y
                                      << " boundary=" << boundary_x << ','
                                      << boundary_y
                                      << " trace=" << trace_support_x << ','
                                      << trace_support_y << "\n";
                        }
                    }
                    if (!refined) {
                        ++refine_rejects;
                        continue;
                    }
                    bool inside_frame = true;
                    constexpr float kImageMargin = 3.0f;
                    for (const auto& point : refined->corners) {
                        const float image_x =
                            point.x + static_cast<float>(screen.x);
                        const float image_y =
                            point.y + static_cast<float>(screen.y);
                        inside_frame = inside_frame &&
                            std::isfinite(image_x) && std::isfinite(image_y) &&
                            image_x >= kImageMargin &&
                            image_y >= kImageMargin &&
                            image_x <= gray.cols - 1 - kImageMargin &&
                            image_y <= gray.rows - 1 - kImageMargin;
                    }
                    if (!inside_frame) {
                        ++support_rejects;
                        continue;
                    }
                    // A RANSAC recovery intentionally discards a few glare
                    // outliers. Keep its absolute vertical placement tight so
                    // it cannot replace the real top row with a dark menu
                    // strip one division below the graticule.
                    if (refined->ransac && screen_aspect >= 1.35 &&
                        vertical_layout_factor < 0.75) {
                        ++support_rejects;
                        continue;
                    }
                    // A one-cell-shifted sequence can contain seven periodic
                    // lines by borrowing a status/menu edge. Its homography is
                    // measurably less self-consistent than the real graticule,
                    // so include that evidence in candidate ranking.
                    const double refinement_factor = refined->nominal
                        ? 0.50
                        : std::exp(-0.5 * std::pow(
                              refined->reprojection_rms / 1.50, 2.0));
                    const double refined_score =
                        selection_score * refinement_factor;
                    if (debug) {
                        std::cerr << "GRID_DIAG_RANK"
                                  << " x=" << xs.origin << '/'
                                  << xs.spacing
                                  << " y=" << ys.origin << '/'
                                  << ys.spacing
                                  << " raw=" << selection_score
                                  << " refine=" << refinement_factor
                                  << " final=" << refined_score
                                  << " rms=" << refined->reprojection_rms
                                  << " trace_center=" << trace_center_dx
                                  << ',' << trace_center_dy
                              << " trace_span=" << trace_span_x << ','
                              << trace_span_y
                              << " trace_gap=" << trace_top_gap << ','
                              << trace_bottom_gap
                              << " boundary=" << boundary_x << ','
                              << boundary_y
                                  << " center_axis=" << center_support_x
                                  << ',' << center_support_y
                                  << " marker=" << marker_matches << ','
                                  << marker_dx << ',' << marker_dy
                                  << " nominal=" << refined->nominal
                                  << " ransac=" << refined->ransac
                                  << " glare_candidate="
                                  << centered_glare_candidate
                                  << "\n";
                    }
                    if (refined_score <= best_score) continue;
                    best_score = refined_score;
                    best_confidence = confidence;
                    best = refined->corners;
                    for (auto& point : best) {
                        point.x += static_cast<float>(screen.x);
                        point.y += static_cast<float>(screen.y);
                    }
                } else {
                    ++confidence_rejects;
                }
            }
        }
    }
    if (debug) {
        std::cerr << "GRID_DIAG_RESULT best_score=" << best_score
                  << " best_confidence=" << best_confidence
                  << " lattice_pairs=" << lattice_pairs
                  << " cell_rejects=" << cell_rejects
                  << " support_rejects=" << support_rejects
                  << " dark_rejects=" << dark_rejects
                  << " confidence_rejects=" << confidence_rejects
                  << " refine_rejects=" << refine_rejects << "\n";
    }
    if (best_score < 0.0) return std::nullopt;
    return std::make_pair(best, best_confidence);
}

cv::Rect bounds_for(const std::array<cv::Point2f, 4>& quad,
                    const cv::Size& size) {
    float left = quad[0].x, right = quad[0].x;
    float top = quad[0].y, bottom = quad[0].y;
    for (const auto& point : quad) {
        left = std::min(left, point.x);
        right = std::max(right, point.x);
        top = std::min(top, point.y);
        bottom = std::max(bottom, point.y);
    }
    const int x0 = std::clamp(static_cast<int>(std::floor(left)), 0,
                              size.width - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(top)), 0,
                              size.height - 1);
    const int x1 = std::clamp(static_cast<int>(std::ceil(right)), x0,
                              size.width - 1);
    const int y1 = std::clamp(static_cast<int>(std::ceil(bottom)), y0,
                              size.height - 1);
    return {x0, y0, x1 - x0 + 1, y1 - y0 + 1};
}

}  // namespace

GridCalibrator::GridCalibrator(cv::Size normalized_size)
    : normalized_size_(normalized_size) {}

void GridCalibrator::set_roi(const cv::Rect& roi) {
    if (roi.width < 32 || roi.height < 32) {
        throw std::invalid_argument("camera ROI is too small");
    }
    const std::array<cv::Point2f, 4> source{{
        {static_cast<float>(roi.x), static_cast<float>(roi.y)},
        {static_cast<float>(roi.x + roi.width - 1), static_cast<float>(roi.y)},
        {static_cast<float>(roi.x + roi.width - 1),
         static_cast<float>(roi.y + roi.height - 1)},
        {static_cast<float>(roi.x), static_cast<float>(roi.y + roi.height - 1)}}};
    const std::array<cv::Point2f, 4> target{{
        {0.0f, 0.0f}, {static_cast<float>(normalized_size_.width - 1), 0.0f},
        {static_cast<float>(normalized_size_.width - 1),
         static_cast<float>(normalized_size_.height - 1)},
        {0.0f, static_cast<float>(normalized_size_.height - 1)}}};
    GridGeometry geometry;
    geometry.roi = roi;
    geometry.normalized_size = normalized_size_;
    geometry.homography = cv::getPerspectiveTransform(source.data(),
                                                       target.data());
    geometry.valid = true;
    geometry_ = geometry;
    history_.clear();
}

bool GridCalibrator::auto_locate(const cv::Mat& frame) {
    if (geometry_) return true;
    if (frame.empty()) return false;
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    const auto detection = locate_grid(gray, frame);
    if (!detection) return false;
    history_.push_back(detection->first);
    if (history_.size() > 7) history_.erase(history_.begin());
    if (history_.size() < 5) return false;

    std::array<cv::Point2f, 4> median{};
    for (size_t corner = 0; corner < 4; ++corner) {
        std::vector<float> xs, ys;
        for (const auto& quad : history_) {
            xs.push_back(quad[corner].x);
            ys.push_back(quad[corner].y);
        }
        std::sort(xs.begin(), xs.end());
        std::sort(ys.begin(), ys.end());
        median[corner] = {xs[xs.size() / 2], ys[ys.size() / 2]};
    }
    int inliers = 0;
    for (const auto& quad : history_) {
        double maximum_error = 0.0;
        for (size_t corner = 0; corner < 4; ++corner) {
            maximum_error = std::max(maximum_error,
                static_cast<double>(cv::norm(quad[corner] - median[corner])));
        }
        if (maximum_error <= 4.0) ++inliers;
    }
    if (inliers < 5) return false;

    const std::array<cv::Point2f, 4> target{{
        {0.0f, 0.0f}, {static_cast<float>(normalized_size_.width - 1), 0.0f},
        {static_cast<float>(normalized_size_.width - 1),
         static_cast<float>(normalized_size_.height - 1)},
        {0.0f, static_cast<float>(normalized_size_.height - 1)}}};
    GridGeometry geometry;
    geometry.roi = bounds_for(median, frame.size());
    geometry.normalized_size = normalized_size_;
    geometry.homography = cv::getPerspectiveTransform(median.data(),
                                                       target.data());
    geometry.valid = true;
    geometry_ = geometry;
    return true;
}

bool GridCalibrator::locked() const { return geometry_.has_value(); }

void GridCalibrator::reset() {
    geometry_.reset();
    history_.clear();
}

std::optional<GridGeometry> GridCalibrator::geometry() const {
    return geometry_;
}

cv::Mat GridCalibrator::normalize(const cv::Mat& frame) const {
    if (!geometry_) return frame.clone();
    cv::Mat result;
    cv::warpPerspective(frame, result, geometry_->homography,
                        normalized_size_, cv::INTER_AREA,
                        cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return result;
}

}  // namespace task5
