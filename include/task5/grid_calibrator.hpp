#pragma once

#include "task5/types.hpp"

#include <optional>
#include <vector>

namespace task5 {

class GridCalibrator {
public:
    explicit GridCalibrator(cv::Size normalized_size = {640, 480});

    void set_roi(const cv::Rect& roi);
    bool auto_locate(const cv::Mat& frame);
    bool locked() const;
    void reset();
    std::optional<GridGeometry> geometry() const;
    cv::Mat normalize(const cv::Mat& frame) const;

private:
    cv::Size normalized_size_;
    std::optional<GridGeometry> geometry_;
    std::vector<std::array<cv::Point2f, 4>> history_;
};

}  // namespace task5
