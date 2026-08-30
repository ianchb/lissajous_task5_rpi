#pragma once

#include "task5/types.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace task5 {

// Accumulates sparse phase-code frames. High-frequency traces are brightest
// at their left/right turning points, so the matcher uses temporal edge
// profiles instead of requiring every dim middle section to be segmented.
class PhaseCodeMatcher {
public:
    explicit PhaseCodeMatcher(PhaseCodeSpec spec = {});

    bool add(const TraceObservation& observation);
    FrequencyEstimate estimate() const;
    void reset();
    std::size_t frame_count() const;

private:
    // Profiles 0/1 are left/right edge energy. Profile 2 is the normalized
    // horizontal centroid of the complete yellow trace at each sampled row.
    using SideProfiles = std::array<std::vector<float>, 3>;
    using FrameProfiles = std::array<SideProfiles, 4>;
    // Full-height left/right/whole-row energy is retained only for the
    // high-frequency lattice disambiguator and visual scan calibration.
    using RowProfiles = std::array<std::vector<float>, 3>;

    struct VisualGeometry {
        bool valid = false;
        double scale = 1.0;
        double offset = 0.0;
        double residual_px = 0.0;
    };

    FrameProfiles extract(const cv::Mat& normalized_bgr) const;
    RowProfiles extract_rows(const cv::Mat& normalized_bgr) const;
    std::array<SideProfiles, 4> aggregate() const;
    RowProfiles aggregate_rows() const;
    VisualGeometry estimate_visual_geometry(
        const RowProfiles& profiles) const;
    double estimate_coarse_hz(
        const std::array<SideProfiles, 4>& profiles,
        const VisualGeometry& geometry) const;
    double estimate_spectral_hz(
        const std::array<SideProfiles, 4>& profiles,
        const VisualGeometry& geometry,
        double* peak_ratio) const;
    double estimate_lattice_hz(const RowProfiles& profiles,
                               const VisualGeometry& geometry,
                               double spectral_hz,
                               double original_hz,
                               double original_runner_hz,
                               double* runner_hz,
                               double* score_advantage) const;
    FrequencyEstimate estimate_centroid_code(
        const std::array<SideProfiles, 4>& profiles,
        const VisualGeometry& geometry) const;

    PhaseCodeSpec spec_;
    int profile_rows_ = 80;
    std::vector<FrameProfiles> frames_;
    std::vector<RowProfiles> row_frames_;
};

}  // namespace task5
