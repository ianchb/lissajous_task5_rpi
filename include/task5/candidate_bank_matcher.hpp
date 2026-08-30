#pragma once

#include "task5/types.hpp"

#include <array>

namespace task5 {

struct CandidateBandScore {
    bool valid = false;
    double score = 1.0;
    double foreground_fraction = 1.0;
    double median_row_occupancy = 1.0;
    double bounding_fill = 1.0;
};

struct CandidateBankResult {
    bool valid = false;
    int best_band = -1;
    int runner_up_band = -1;
    double margin = 0.0;
    std::array<CandidateBandScore, 16> bands{};
};

// Scores all sixteen narrow horizontal candidate bands in one frozen-grid
// frame. Wrong frequencies fill their band as relative phase sweeps; the
// matching frequency remains a thin line or ellipse.
class CandidateBankMatcher {
public:
    CandidateBankResult score(const TraceObservation& observation) const;
};

}  // namespace task5
