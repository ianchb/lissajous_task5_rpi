#pragma once

#include "task5/types.hpp"

#include <optional>

namespace task5 {

struct QuadraturePhaseSolution {
    int phase = 0;
    double relative_phase_rad = 0.0;
    double response_amplitude = 0.0;
};

std::optional<QuadraturePhaseSolution> solve_quadrature_phase(
    AutoMode mode, double response_at_phase_0,
    double response_at_phase_64);

}  // namespace task5
