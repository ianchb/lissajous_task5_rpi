#pragma once

#include "task5/types.hpp"

namespace task5 {

struct ShapeMetrics {
    bool valid = false;
    int pixels = 0;
    cv::Rect2d bbox;
    cv::Point2d center_div{};
    cv::Point2d span_div{};
    double coverage = 1.0;
    double correlation = 0.0;
    double thinness = 1.0;
    double minor_rms_div = 99.0;
    double radial_cv = 1.0;
    double symmetry_x = 0.0;
    double symmetry_y = 0.0;
    double crossing_fill = 0.0;
    // Vertical offset of the figure-eight crossing from the calibrated grid
    // centre, in divisions. Positive values are below the centre.
    double crossing_offset_y_div = 99.0;
    double phase_feature = 0.0;
};

class ShapeObserver {
public:
    ShapeMetrics analyze(const TraceObservation& observation) const;
    double search_score(AutoMode mode, const ShapeMetrics& metrics) const;
    bool shape_ok(AutoMode mode, const ShapeMetrics& metrics) const;
    bool phase_servo_needed(AutoMode mode,
                            const ShapeMetrics& metrics) const;
    bool frequency_shape_coherent(AutoMode mode,
                                  const ShapeMetrics& metrics) const;
    bool frequency_mismatch_fill(const ShapeMetrics& metrics) const;
    bool frequency_mismatch_harmonic(AutoMode mode,
                                     const ShapeMetrics& metrics) const;

private:
    static cv::Mat isolate_trace(const cv::Mat& mask);
    static double symmetry_score(const cv::Mat& mask, int flip_code);
};

}  // namespace task5
