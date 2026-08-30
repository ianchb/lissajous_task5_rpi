#pragma once

#include "task5/types.hpp"

namespace task5 {

struct SegmenterConfig {
    int hue_low = 12;
    int hue_high = 48;
    int saturation_low = 35;
    int value_low = 55;
    int pale_saturation_high = 145;
    int pale_value_low = 105;
    int morphology_radius = 1;
    int minimum_component_area = 10;
};

class TraceSegmenter {
public:
    explicit TraceSegmenter(SegmenterConfig config = {});
    TraceObservation process(const cv::Mat& normalized_bgr,
                             uint64_t sequence,
                             Clock::time_point timestamp) const;

private:
    SegmenterConfig config_;
};

}  // namespace task5
