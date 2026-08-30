#include "task5/candidate_bank_matcher.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <limits>
#include <vector>

namespace task5 {
namespace {

double median(std::vector<double> values) {
    if (values.empty()) return 1.0;
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

CandidateBandScore score_band(const cv::Mat& mask, int band) {
    CandidateBandScore result;
    // Ignore two pixels at each inter-band transition. FPGA emits the new
    // center at a slot boundary, and scope interpolation can brighten it.
    const int y0 = static_cast<int>(
        std::lround((band + 0.08) * mask.rows / 16.0));
    const int y1 = static_cast<int>(
        std::lround((band + 0.92) * mask.rows / 16.0));
    const int edge = std::max(2, mask.cols / 50);
    const cv::Rect area(edge, std::clamp(y0, 0, mask.rows - 1),
                        mask.cols - 2 * edge,
                        std::max(1, std::min(mask.rows, y1) -
                                    std::clamp(y0, 0, mask.rows - 1)));
    const cv::Mat crop = mask(area);
    const int foreground = cv::countNonZero(crop);
    if (foreground < 20) return result;
    std::vector<double> row_occupancies;
    for (int row = 0; row < crop.rows; ++row) {
        const int count = cv::countNonZero(crop.row(row));
        if (count > 0) row_occupancies.push_back(
            static_cast<double>(count) / crop.cols);
    }
    if (row_occupancies.size() < 3) return result;
    std::vector<cv::Point> points;
    cv::findNonZero(crop, points);
    const cv::Rect bounds = cv::boundingRect(points);
    result.foreground_fraction = static_cast<double>(foreground) /
                                 crop.total();
    result.median_row_occupancy = median(row_occupancies);
    result.bounding_fill = static_cast<double>(foreground) /
                           std::max(1, bounds.area());
    result.score = result.foreground_fraction +
                   0.20 * result.median_row_occupancy +
                   0.05 * result.bounding_fill;
    result.valid = result.foreground_fraction < 0.90;
    return result;
}

}  // namespace

CandidateBankResult CandidateBankMatcher::score(
    const TraceObservation& observation) const {
    CandidateBankResult result;
    if (observation.mask.empty() || observation.mask.rows < 160 ||
        observation.mask.cols < 160) return result;
    double best_score = std::numeric_limits<double>::infinity();
    double runner_score = std::numeric_limits<double>::infinity();
    for (int band = 0; band < 16; ++band) {
        result.bands[static_cast<size_t>(band)] =
            score_band(observation.mask, band);
        const auto& score = result.bands[static_cast<size_t>(band)];
        if (!score.valid) continue;
        if (score.score < best_score) {
            runner_score = best_score;
            result.runner_up_band = result.best_band;
            best_score = score.score;
            result.best_band = band;
        } else if (score.score < runner_score) {
            runner_score = score.score;
            result.runner_up_band = band;
        }
    }
    result.valid = result.best_band >= 0 && result.runner_up_band >= 0;
    result.margin = result.valid ? runner_score - best_score : 0.0;
    return result;
}

}  // namespace task5
