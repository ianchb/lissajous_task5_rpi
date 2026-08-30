#include "task5/sine_matcher.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <vector>

namespace task5 {
namespace {

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 1.0;
    const size_t index = static_cast<size_t>(std::clamp(
        fraction * static_cast<double>(values.size() - 1), 0.0,
        static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

}  // namespace

SineMatchScore SineMatcher::score(
    const TraceObservation& observation) const {
    SineMatchScore result;
    if (observation.mask.empty()) return result;

    const int edge = std::max(2, observation.mask.cols / 40);
    const cv::Rect interior(edge, 0, observation.mask.cols - 2 * edge,
                            observation.mask.rows);
    if (interior.width < 32) return result;
    const cv::Mat mask = observation.mask(interior);
    const int foreground = cv::countNonZero(mask);
    if (foreground < 300) return result;

    std::vector<double> row_occupancies;
    row_occupancies.reserve(static_cast<size_t>(mask.rows));
    int occupied_rows = 0;
    for (int row = 0; row < mask.rows; ++row) {
        const int count = cv::countNonZero(mask.row(row));
        if (count == 0) continue;
        ++occupied_rows;
        row_occupancies.push_back(
            static_cast<double>(count) / mask.cols);
    }
    if (occupied_rows < mask.rows / 8) return result;

    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);
    const cv::Rect bounds = cv::boundingRect(points);
    result.foreground_fraction =
        static_cast<double>(foreground) / mask.total();
    result.median_row_occupancy = percentile(row_occupancies, 0.50);
    result.p90_row_occupancy = percentile(row_occupancies, 0.90);
    result.bounding_fill = static_cast<double>(foreground) /
        std::max(1, bounds.area());
    result.score = result.foreground_fraction +
        0.20 * result.median_row_occupancy +
        0.10 * result.p90_row_occupancy +
        0.05 * result.bounding_fill;
    result.valid = result.foreground_fraction < 0.75;
    return result;
}

}  // namespace task5
