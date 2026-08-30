#pragma once

#include "task5/types.hpp"

namespace task5 {

struct SineMatchScore {
    bool valid = false;
    double score = 1.0;
    double foreground_fraction = 1.0;
    double median_row_occupancy = 1.0;
    double p90_row_occupancy = 1.0;
    double bounding_fill = 1.0;
};

// Scores a continuous-sine XY trace. A frequency match is a thin line or
// ellipse; a 100 Hz mismatch sweeps phase during the scope acquisition and
// occupies a much larger area. Lower scores are better.
class SineMatcher {
public:
    SineMatchScore score(const TraceObservation& observation) const;
};

}  // namespace task5
