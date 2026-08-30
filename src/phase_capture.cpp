#include "task5/phase_capture.hpp"

#include <cmath>

namespace task5 {
namespace {
constexpr double kTwoPi = 6.28318530717958647692;

int radians_to_phase(double radians) {
    return static_cast<int>(std::lround(radians * 256.0 / kTwoPi)) & 0xff;
}
}  // namespace

std::optional<QuadraturePhaseSolution> solve_quadrature_phase(
    AutoMode mode, double response_at_phase_0,
    double response_at_phase_64) {
    if (!std::isfinite(response_at_phase_0) ||
        !std::isfinite(response_at_phase_64))
        return std::nullopt;
    const double amplitude = std::hypot(
        response_at_phase_0, response_at_phase_64);
    if (amplitude < 0.02) return std::nullopt;

    QuadraturePhaseSolution result;
    result.response_amplitude = amplitude;
    if (mode == AutoMode::Line || mode == AutoMode::Circle) {
        // r(q)=cos(delta+q), so r(0)=cos(delta) and
        // r(pi/2)=-sin(delta).
        result.relative_phase_rad = std::atan2(
            -response_at_phase_64, response_at_phase_0);
        const double target = mode == AutoMode::Line ?
            3.14159265358979323846 : 1.57079632679489661923;
        result.phase = radians_to_phase(target - result.relative_phase_rad);
    } else {
        // feature(q)=a*cos(q)+b*sin(q). Its two zeros differ by pi and
        // produce vertically inverted but equivalent horizontal figures.
        result.relative_phase_rad = std::atan2(
            response_at_phase_64, response_at_phase_0);
        result.phase = radians_to_phase(std::atan2(
            -response_at_phase_0, response_at_phase_64));
    }
    return result;
}

}  // namespace task5
