#include "task5/camera.hpp"
#include "task5/candidate_bank_matcher.hpp"
#include "task5/clock_calibration.hpp"
#include "task5/frequency_observer.hpp"
#include "task5/grid_calibrator.hpp"
#include "task5/phase_code_matcher.hpp"
#include "task5/sine_matcher.hpp"
#include "task5/shape_observer.hpp"
#include "task5/trace_segmenter.hpp"
#include "task5/uart.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::string camera = "/dev/video19";
    std::string serial;
    std::string image;
    std::string snapshot_dir;
    int duration_s = 10;
    int roi_x = 0;
    int roi_y = 0;
    int roi_w = 0;
    int roi_h = 0;
    double scan_ms = 1.0;
    double multirate_scale = 0.8760;
    double dds_clock_hz = 19'999'858.0;
    double fixed_sine_hz = 0.0;
    std::vector<double> sine_candidates;
    int sine_gate_profile = -1;
    double candidate_bank_base_hz = 0.0;
    int candidate_settle_ms = 700;
    int candidate_measure_ms = 500;
    int phase_windows = 1;
    double lock_frequency_hz = 0.0;
    int lock_mode = 0;
    int lock_direct_phase = -1;
    bool read_calibration = false;
    bool drive = false;
    bool phase_code = false;
    bool grid_only = false;
};

std::optional<Arguments> parse(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string value(argv[i]);
        const auto next = [&]() -> std::optional<std::string> {
            if (i + 1 >= argc) return std::nullopt;
            return std::string(argv[++i]);
        };
        if (value == "--camera") {
            auto item = next(); if (!item) return std::nullopt; args.camera = *item;
        } else if (value == "--serial") {
            auto item = next(); if (!item) return std::nullopt; args.serial = *item;
        } else if (value == "--image") {
            auto item = next(); if (!item) return std::nullopt; args.image = *item;
        } else if (value == "--snapshot-dir") {
            auto item = next(); if (!item) return std::nullopt; args.snapshot_dir = *item;
        } else if (value == "--duration") {
            auto item = next(); if (!item) return std::nullopt; args.duration_s = std::stoi(*item);
        } else if (value == "--scan-ms") {
            auto item = next(); if (!item) return std::nullopt; args.scan_ms = std::stod(*item);
        } else if (value == "--multirate-scale") {
            auto item = next(); if (!item) return std::nullopt;
            args.multirate_scale = std::stod(*item);
        } else if (value == "--dds-clock-hz") {
            auto item = next(); if (!item) return std::nullopt;
            args.dds_clock_hz = std::stod(*item);
        } else if (value == "--read-calibration") {
            args.read_calibration = true;
        } else if (value == "--fixed-sine-hz") {
            auto item = next(); if (!item) return std::nullopt;
            args.fixed_sine_hz = std::stod(*item);
        } else if (value == "--sine-candidates") {
            auto item = next(); if (!item) return std::nullopt;
            std::replace(item->begin(), item->end(), ',', ' ');
            std::stringstream stream(*item);
            double frequency_hz = 0.0;
            while (stream >> frequency_hz) {
                args.sine_candidates.push_back(frequency_hz);
            }
            if (args.sine_candidates.empty()) return std::nullopt;
        } else if (value == "--candidate-settle-ms") {
            auto item = next(); if (!item) return std::nullopt;
            args.candidate_settle_ms = std::stoi(*item);
        } else if (value == "--candidate-measure-ms") {
            auto item = next(); if (!item) return std::nullopt;
            args.candidate_measure_ms = std::stoi(*item);
        } else if (value == "--phase-windows") {
            auto item = next(); if (!item) return std::nullopt;
            args.phase_windows = std::stoi(*item);
        } else if (value == "--lock-frequency-hz") {
            auto item = next(); if (!item) return std::nullopt;
            args.lock_frequency_hz = std::stod(*item);
        } else if (value == "--lock-mode") {
            auto item = next(); if (!item) return std::nullopt;
            args.lock_mode = std::stoi(*item);
        } else if (value == "--lock-direct-phase") {
            auto item = next(); if (!item) return std::nullopt;
            args.lock_direct_phase = std::stoi(*item);
        } else if (value == "--sine-gate-profile") {
            auto item = next(); if (!item) return std::nullopt;
            args.sine_gate_profile = std::stoi(*item);
        } else if (value == "--candidate-bank-base") {
            auto item = next(); if (!item) return std::nullopt;
            args.candidate_bank_base_hz = std::stod(*item);
        } else if (value == "--roi") {
            auto item = next(); if (!item) return std::nullopt;
            std::replace(item->begin(), item->end(), ',', ' ');
            std::stringstream stream(*item);
            if (!(stream >> args.roi_x >> args.roi_y >> args.roi_w >> args.roi_h)) return std::nullopt;
        } else if (value == "--drive") {
            args.drive = true;
        } else if (value == "--phase-code") {
            args.phase_code = true;
        } else if (value == "--grid-only") {
            args.grid_only = true;
        } else if (value == "--help" || value == "-h") {
            return std::nullopt;
        } else {
            std::cerr << "unknown argument: " << value << "\n";
            return std::nullopt;
        }
    }
    return args;
}

void print_usage(const char* name) {
    std::cout << "usage: " << name
              << " [--camera /dev/video19] [--serial /dev/ttyUSB0]"
                 " [--image file] [--roi x,y,w,h] [--scan-ms 1.0]"
                 " [--multirate-scale 0.8760]"
                 " [--dds-clock-hz 19999858] [--read-calibration]"
                 " [--duration 10] [--snapshot-dir path]"
                 " [--phase-code | --fixed-sine-hz 10000 |"
                 " --sine-candidates 62200,62300,62400]"
                 " [--sine-gate-profile 0]"
                 " [--candidate-bank-base 98500]"
                 " [--candidate-settle-ms 700]"
                 " [--candidate-measure-ms 500] [--phase-windows 1]"
                 " [--lock-frequency-hz 44400 --lock-mode 3]"
                 " [--lock-direct-phase 64]"
                 " [--drive]\n";
}

uint32_t ftw_for_frequency(double frequency_hz, double clock_hz) {
    return static_cast<uint32_t>(std::llround(
        frequency_hz * 4294967296.0 / clock_hz));
}

}  // namespace

int main(int argc, char** argv) {
    const auto parsed = parse(argc, argv);
    if (!parsed) {
        print_usage(argv[0]);
        return 2;
    }
    const Arguments args = *parsed;
    const int output_modes = static_cast<int>(args.phase_code) +
        static_cast<int>(args.fixed_sine_hz > 0.0) +
        static_cast<int>(!args.sine_candidates.empty()) +
        static_cast<int>(args.candidate_bank_base_hz > 0.0) +
        static_cast<int>(args.lock_frequency_hz > 0.0);
    if (output_modes > 1) {
        std::cerr << "output diagnostic modes are mutually exclusive\n";
        return 2;
    }
    if (args.grid_only && (!args.drive || output_modes != 0)) {
        std::cerr << "--grid-only requires --drive and no output mode\n";
        return 2;
    }
    if (args.candidate_bank_base_hz > 0.0 &&
        (!args.drive || args.candidate_bank_base_hz < 1'000.0 ||
         args.candidate_bank_base_hz + 1'500.0 > 100'000.0)) {
        std::cerr << "candidate bank requires --drive and 1000..98500 Hz base\n";
        return 2;
    }
    if (args.fixed_sine_hz < 0.0 || args.fixed_sine_hz > 100'000.0) {
        std::cerr << "--fixed-sine-hz must be in (0, 100000]\n";
        return 2;
    }
    for (double frequency_hz : args.sine_candidates) {
        if (frequency_hz <= 0.0 || frequency_hz > 100'000.0) {
            std::cerr << "sine candidates must be in (0, 100000]\n";
            return 2;
        }
    }
    if ((!args.sine_candidates.empty() && !args.drive) ||
        args.candidate_settle_ms < 0 || args.candidate_measure_ms < 100) {
        std::cerr << "sine candidate sweep requires --drive and valid timing\n";
        return 2;
    }
    if (args.sine_gate_profile < -1 || args.sine_gate_profile > 3 ||
        (args.sine_gate_profile >= 0 && args.sine_candidates.empty())) {
        std::cerr << "--sine-gate-profile requires a candidate sweep and 0..3\n";
        return 2;
    }
    if (args.phase_windows < 1 || (!args.phase_code && args.phase_windows != 1)) {
        std::cerr << "--phase-windows requires --phase-code and a positive count\n";
        return 2;
    }
    if ((args.lock_frequency_hz > 0.0 &&
         (args.lock_frequency_hz < 1'000.0 ||
          args.lock_frequency_hz > 100'000.0 || !args.drive ||
          args.lock_mode < 3 || args.lock_mode > 5)) ||
        (args.lock_frequency_hz <= 0.0 && args.lock_mode != 0) ||
        args.lock_direct_phase < -1 || args.lock_direct_phase > 255) {
        std::cerr << "lock experiment requires --drive, frequency 1k..100k, "
                     "and --lock-mode 3..5\n";
        return 2;
    }
    task5::GridCalibrator calibrator;
    if (args.roi_w > 0 && args.roi_h > 0) {
        calibrator.set_roi({args.roi_x, args.roi_y, args.roi_w, args.roi_h});
    }
    task5::TraceSegmenter segmenter;
    task5::FrequencyObserver observer;
    task5::ProbeSpec probe;
    probe.scan_duration_s = args.scan_ms * 1.0e-3;
    task5::MultiRateSpec multirate;
    multirate.visual_time_scale = args.multirate_scale;
    task5::PhaseCodeSpec phase_code;
    task5::UartTransport uart;
    if (!args.serial.empty() && !uart.open(args.serial)) {
        std::cerr << "warning: cannot open serial " << args.serial << "\n";
    }
    if (args.drive && !uart.is_open()) {
        std::cerr << "--drive requires a working serial connection\n";
        return 3;
    }
    double dds_clock_hz = args.dds_clock_hz;
    const bool require_live_calibration = args.read_calibration ||
        (args.drive && (args.phase_code || args.lock_frequency_hz > 0.0));
    if (require_live_calibration) {
        if (!uart.is_open()) {
            std::cerr << "live clock calibration requires serial\n";
            return 22;
        }
        task5::FpgaResponse calibration;
        if (!uart.request(0x03, &calibration, 500)) {
            std::cerr << "cannot read live FPGA clock calibration\n";
            return 22;
        }
        const auto decoded = task5::derive_clock_calibration(calibration);
        if (!decoded) {
            std::cerr << "live FPGA clock calibration is unavailable or invalid\n";
            return 23;
        }
        dds_clock_hz = decoded->effective_dds_clock_hz;
        std::cout << "calibration_source=fpga_live sample_sum_q8="
                  << decoded->sample_sum_q8 << " cycles="
                  << decoded->cycle_count << " measured_hz=" << std::fixed
                  << std::setprecision(6) << decoded->measured_frequency_hz
                  << " grid_hz=" << std::setprecision(1)
                  << decoded->source_grid_frequency_hz << " residual_hz="
                  << std::setprecision(6) << decoded->residual_hz
                  << " dds_clock_hz=" << std::setprecision(3)
                  << dds_clock_hz << "\n";
    } else if (uart.is_open()) {
        std::cout << "calibration_source=diagnostic_argument dds_clock_hz="
                  << std::fixed << std::setprecision(3)
                  << dds_clock_hz << "\n";
    }
    bool phase_code_command_sent = false;
    const auto send_phase_code = [&]() {
        task5::DdsCommand command;
        command.ftw = ftw_for_frequency(100.0, dds_clock_hz);
        command.amplitude = 102;
        command.flags = 0x80;
        if (!uart.send_dds(command)) {
            std::cerr << "failed to send phase-code command\n";
            return false;
        }
        task5::FpgaResponse acknowledgement;
        uart.read_response(&acknowledgement, 500, 0x01);
        task5::FpgaResponse status;
        if (!uart.request(0x02, &status, 500) || !status.phase_code_mode()) {
            std::cerr << "FPGA did not acknowledge phase-code mode\n";
            return false;
        }
        std::cout << "phase_code_ack=1 mode=" << static_cast<int>(status.mode)
                  << " ftw=" << status.payload << "\n";
        phase_code_command_sent = true;
        return true;
    };
    if (args.drive && args.phase_code && calibrator.locked()) {
        if (!send_phase_code()) return 7;
    } else if (args.drive && args.phase_code) {
        task5::DdsCommand command;
        command.ftw = ftw_for_frequency(100'000.0, dds_clock_hz);
        command.amplitude = 0;
        command.flags = 0x00;
        if (!uart.send_dds(command)) {
            std::cerr << "failed to blank trace for grid calibration\n";
            return 12;
        }
        task5::FpgaResponse acknowledgement;
        uart.read_response(&acknowledgement, 300, 0x01);
        std::cout << "phase_code_grid_blank=1\n";
    }
    if (args.grid_only) {
        task5::DdsCommand command;
        command.ftw = ftw_for_frequency(100'000.0, dds_clock_hz);
        command.amplitude = 0;
        command.flags = 0x00;
        if (!uart.send_dds(command)) {
            std::cerr << "failed to send grid calibration command\n";
            return 12;
        }
        task5::FpgaResponse acknowledgement;
        uart.read_response(&acknowledgement, 300, 0x01);
    }
    if (args.candidate_bank_base_hz > 0.0) {
        task5::DdsCommand command;
        command.ftw = ftw_for_frequency(args.candidate_bank_base_hz,
                                         dds_clock_hz);
        command.phase = 1;  // one legal 100 Hz bin per band
        command.amplitude = 102;
        command.flags = 0xc0;  // phase-code + sine-gate = candidate bank
        if (!uart.send_dds(command)) {
            std::cerr << "failed to send candidate-bank command\n";
            return 14;
        }
        task5::FpgaResponse acknowledgement;
        uart.read_response(&acknowledgement, 300, 0x01);
    }
    if (args.drive && args.fixed_sine_hz > 0.0) {
        task5::DdsCommand command;
        command.ftw = ftw_for_frequency(args.fixed_sine_hz, dds_clock_hz);
        command.amplitude = 102;
        command.flags = 0x00;
        if (!uart.send_dds(command)) {
            std::cerr << "failed to send fixed-sine command\n";
            return 8;
        }
        task5::FpgaResponse acknowledgement;
        uart.read_response(&acknowledgement, 500, 0x01);
        task5::FpgaResponse status;
        if (!uart.request(0x02, &status, 500) || status.phase_code_mode()) {
            std::cerr << "FPGA did not acknowledge fixed-sine mode\n";
            return 9;
        }
        std::cout << "fixed_sine_ack=1 mode=" << static_cast<int>(status.mode)
                  << " requested_hz=" << std::fixed << std::setprecision(1)
                  << args.fixed_sine_hz << " ftw=" << status.payload << "\n";
    }
    if (!args.snapshot_dir.empty()) {
        std::filesystem::create_directories(args.snapshot_dir);
    }

    auto process = [&](const task5::Frame& frame) {
        if (!calibrator.geometry()) calibrator.auto_locate(frame.bgr);
        const cv::Mat normalized = calibrator.normalize(frame.bgr);
        const task5::TraceObservation observation =
            segmenter.process(normalized, frame.sequence, frame.timestamp);
        const task5::FrequencyEstimate estimate = args.phase_code
            ? observer.observe_phase_code(observation, phase_code)
            : observer.observe_ramp(observation, probe);
        observer.update_posterior(estimate, frame.timestamp);
        const task5::FrequencyEstimate posterior = observer.posterior();
        std::cout << "frame=" << frame.sequence << " quality="
                  << std::fixed << std::setprecision(3) << observation.quality
                  << " pixels=" << cv::countNonZero(observation.mask)
                  << " estimate=" << std::setprecision(1)
                  << estimate.frequency_hz << "Hz score="
                  << std::setprecision(4) << estimate.score
                  << " margin=" << estimate.margin
                  << " probe=" << estimate.probe_index
                  << " posterior=" << posterior.frequency_hz
                  << " posterior_probe=" << posterior.probe_index
                  << " posterior_valid=" << posterior.valid
                  << " grid=" << calibrator.geometry().has_value() << "\n";
        if (!args.snapshot_dir.empty()) {
            const std::filesystem::path directory(args.snapshot_dir);
            const std::string stem = "frame_" + std::to_string(frame.sequence);
            cv::imwrite((directory / (stem + ".jpg")).string(), frame.bgr);
            if (const auto geometry = calibrator.geometry()) {
                cv::Mat annotated = frame.bgr.clone();
                const std::vector<cv::Point2f> normalized_corners{
                    {0.0f, 0.0f},
                    {static_cast<float>(geometry->normalized_size.width - 1),
                     0.0f},
                    {static_cast<float>(geometry->normalized_size.width - 1),
                     static_cast<float>(geometry->normalized_size.height - 1)},
                    {0.0f,
                     static_cast<float>(geometry->normalized_size.height - 1)}};
                std::vector<cv::Point2f> source_corners;
                cv::perspectiveTransform(normalized_corners, source_corners,
                                         geometry->homography.inv());
                std::vector<cv::Point> polygon;
                polygon.reserve(source_corners.size());
                for (const auto& point : source_corners) {
                    polygon.emplace_back(cvRound(point.x), cvRound(point.y));
                }
                cv::polylines(annotated, polygon, true,
                              cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                cv::rectangle(annotated, geometry->roi,
                              cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
                cv::imwrite((directory / (stem + "_grid.jpg")).string(),
                            annotated);
            }
            if (!observation.mask.empty()) {
                cv::imwrite((directory / (stem + "_mask.png")).string(),
                            observation.mask);
            }
        }
        if (args.drive && !args.phase_code && args.fixed_sine_hz <= 0.0 &&
            estimate.valid &&
            estimate.margin > 0.015) {
            task5::DdsCommand command;
            command.ftw = ftw_for_frequency(estimate.frequency_hz, dds_clock_hz);
            command.flags = 0x02 | (1u << 2);  // probe/ramp profile 1
            uart.send_dds(command);
        }
    };

    if (!args.image.empty()) {
        cv::Mat image = cv::imread(args.image, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "cannot read image " << args.image << "\n";
            return 4;
        }
        // Auto-location deliberately requires five agreeing detections. Feed
        // a still image through the same history path so offline regression
        // can produce the exact grid annotation used by live capture.
        for (uint64_t sequence = 0; sequence < 5; ++sequence) {
            task5::Frame frame{image, task5::Clock::now(), sequence};
            process(frame);
        }
        return 0;
    }

    task5::CameraSource camera;
    if (!camera.open(args.camera)) {
        std::cerr << "cannot open camera " << args.camera << "\n";
        return 5;
    }
    if (args.lock_frequency_hz > 0.0) {
        const auto mode = static_cast<task5::AutoMode>(args.lock_mode);
        task5::FpgaResponse status;
        if (!uart.request(0x02, &status, 500)) {
            std::cerr << "cannot read FPGA mode before lock search\n";
            return 18;
        }
        if (status.mode != static_cast<uint8_t>(mode)) {
            std::cerr << "lock mode mismatch requested=" << args.lock_mode
                      << " fpga=" << static_cast<int>(status.mode) << "\n";
            return 19;
        }

        task5::DdsCommand command;
        const double output_frequency_hz = args.lock_frequency_hz *
            (mode == task5::AutoMode::FigureEight ? 2.0 : 1.0);
        command.ftw = ftw_for_frequency(output_frequency_hz, dds_clock_hz);
        command.amplitude = 0;
        command.flags = 0x00;
        if (!uart.send_dds(command) ||
            !uart.read_response(&status, 500, 0x01)) {
            std::cerr << "cannot blank trace before lock search\n";
            return 20;
        }

        const auto grid_deadline = task5::Clock::now() +
            std::chrono::seconds(3);
        while (!calibrator.locked() && task5::Clock::now() < grid_deadline) {
            task5::Frame frame;
            if (!camera.read(&frame)) continue;
            calibrator.auto_locate(frame.bgr);
        }
        if (!calibrator.locked()) {
            std::cerr << "lock search could not calibrate grid\n";
            return 21;
        }
        command.amplitude = 102;
        if (!uart.send_dds(command) ||
            !uart.read_response(&status, 500, 0x01)) {
            std::cerr << "cannot start lock search\n";
            return 20;
        }

        task5::ShapeObserver shape_observer;
        const std::vector<int> coarse_phases{0, 64, 128, 192};
        std::vector<std::pair<int, task5::ShapeMetrics>> results;
        int saved_phase_frames = 0;
        const auto measure_phase = [&](int phase, int settle_ms, int measure_ms) {
            command.phase = static_cast<uint8_t>(phase & 0xff);
            uart.send_dds(command);
            uart.read_response(&status, 500, 0x01);
            const auto settle_deadline = task5::Clock::now() +
                std::chrono::milliseconds(settle_ms);
            while (task5::Clock::now() < settle_deadline) {
                task5::Frame discarded;
                camera.read(&discarded);
            }
            std::vector<task5::ShapeMetrics> samples;
            const auto measure_deadline = task5::Clock::now() +
                std::chrono::milliseconds(measure_ms);
            while (task5::Clock::now() < measure_deadline) {
                task5::Frame frame;
                if (!camera.read(&frame)) continue;
                const cv::Mat normalized = calibrator.normalize(frame.bgr);
                const auto observation = segmenter.process(
                    normalized, frame.sequence, frame.timestamp);
                const auto metrics = shape_observer.analyze(observation);
                if (metrics.valid) samples.push_back(metrics);
                if (!args.snapshot_dir.empty() && metrics.valid) {
                    const std::filesystem::path directory(args.snapshot_dir);
                    const std::string stem = "lock_phase_" +
                        std::to_string(phase & 0xff) + "_frame_" +
                        std::to_string(saved_phase_frames++);
                    cv::imwrite((directory / (stem + ".jpg")).string(),
                                normalized);
                    cv::imwrite((directory / (stem + "_mask.png")).string(),
                                observation.mask);
                }
            }
            task5::ShapeMetrics median;
            if (!samples.empty()) {
                auto select = [&](auto accessor) {
                    std::vector<double> values;
                    values.reserve(samples.size());
                    for (const auto& sample : samples)
                        values.push_back(accessor(sample));
                    auto middle = values.begin() + values.size() / 2;
                    std::nth_element(values.begin(), middle, values.end());
                    return *middle;
                };
                median = samples[samples.size() / 2];
                median.valid = true;
                median.pixels = static_cast<int>(std::lround(select(
                    [](const auto& value) { return value.pixels; })));
                median.span_div.x = select(
                    [](const auto& value) { return value.span_div.x; });
                median.span_div.y = select(
                    [](const auto& value) { return value.span_div.y; });
                median.coverage = select(
                    [](const auto& value) { return value.coverage; });
                median.correlation = select(
                    [](const auto& value) { return value.correlation; });
                median.thinness = select(
                    [](const auto& value) { return value.thinness; });
                median.minor_rms_div = select(
                    [](const auto& value) { return value.minor_rms_div; });
                median.radial_cv = select(
                    [](const auto& value) { return value.radial_cv; });
                median.symmetry_x = select(
                    [](const auto& value) { return value.symmetry_x; });
                median.symmetry_y = select(
                    [](const auto& value) { return value.symmetry_y; });
                median.crossing_fill = select(
                    [](const auto& value) { return value.crossing_fill; });
                median.phase_feature = select(
                    [](const auto& value) { return value.phase_feature; });
            }
            const double score = shape_observer.search_score(mode, median);
            std::cout << "lock_phase=" << (phase & 0xff)
                      << " frames=" << samples.size()
                      << " valid=" << median.valid
                      << " score=" << std::fixed << std::setprecision(4)
                      << score << " corr=" << median.correlation
                      << " thin=" << median.thinness
                      << " minor_div=" << median.minor_rms_div
                      << " span_div=" << median.span_div.x << ','
                      << median.span_div.y << " radial=" << median.radial_cv
                      << " sym=" << median.symmetry_x << ','
                      << median.symmetry_y << " cross=" << median.crossing_fill
                      << " feature=" << median.phase_feature << "\n";
            return median;
        };

        if (args.lock_direct_phase >= 0) {
            const int phase = args.lock_direct_phase;
            const auto initial_metrics = measure_phase(phase, 2'000, 600);
            int stable_windows = 0;
            for (int window = 0; window < 20; ++window) {
                const auto metrics = measure_phase(phase, 0, 500);
                const bool good = shape_observer.shape_ok(mode, metrics);
                if (good) ++stable_windows;
                std::cout << "lock_direct_hold_window=" << window + 1
                          << " good=" << good << "\n";
            }
            const bool stable_5s = shape_observer.shape_ok(mode, initial_metrics) &&
                stable_windows >= 10;
            command.phase = static_cast<uint8_t>(phase);
            command.flags = stable_5s ? 0x01 : 0x00;
            uart.send_dds(command);
            uart.read_response(&status, 500, 0x01);
            std::cout << "lock_direct_stable=" << stable_5s
                      << " good_windows=" << stable_windows << "/20\n";
            return 0;
        }

        for (int phase : coarse_phases) {
            const auto metrics = measure_phase(phase,
                mode == task5::AutoMode::FigureEight ? 900 : 650,
                mode == task5::AutoMode::FigureEight ? 400 : 250);
            results.emplace_back(phase, metrics);
            if (mode == task5::AutoMode::FigureEight &&
                shape_observer.shape_ok(mode, metrics)) {
                int stable_windows = 0;
                for (int window = 0; window < 10; ++window) {
                    const auto held = measure_phase(phase, 0, 500);
                    const bool good = shape_observer.shape_ok(mode, held);
                    if (good) ++stable_windows;
                    std::cout << "lock_direct_hit_window=" << window + 1
                              << " good=" << good << "\n";
                }
                const bool stable_5s = stable_windows == 10;
                command.phase = static_cast<uint8_t>(phase);
                command.flags = stable_5s ? 0x01 : 0x00;
                uart.send_dds(command);
                uart.read_response(&status, 500, 0x01);
                std::cout << "lock_direct_hit_phase=" << phase
                          << " stable_5s=" << stable_5s
                          << " good_windows=" << stable_windows << "/10\n";
                return 0;
            }
        }
        int best_phase = results.front().first;
        if (mode == task5::AutoMode::Line) {
            cv::Mat design(static_cast<int>(results.size()), 3, CV_64F);
            cv::Mat correlations(static_cast<int>(results.size()), 1, CV_64F);
            for (size_t index = 0; index < results.size(); ++index) {
                const double radians = 2.0 * 3.14159265358979323846 *
                    results[index].first / 256.0;
                design.at<double>(static_cast<int>(index), 0) = 1.0;
                design.at<double>(static_cast<int>(index), 1) = std::cos(radians);
                design.at<double>(static_cast<int>(index), 2) = std::sin(radians);
                correlations.at<double>(static_cast<int>(index)) =
                    results[index].second.valid ?
                    results[index].second.correlation : 0.0;
            }
            cv::Mat coefficients;
            cv::solve(design, correlations, coefficients, cv::DECOMP_SVD);
            double minimum = std::numeric_limits<double>::infinity();
            for (int phase = 0; phase < 256; ++phase) {
                const double radians = 2.0 * 3.14159265358979323846 * phase / 256.0;
                const double prediction = coefficients.at<double>(0) +
                    coefficients.at<double>(1) * std::cos(radians) +
                    coefficients.at<double>(2) * std::sin(radians);
                if (prediction < minimum) {
                    minimum = prediction;
                    best_phase = phase;
                }
            }
        } else {
            best_phase = std::max_element(results.begin(), results.end(),
                [&](const auto& left, const auto& right) {
                    return shape_observer.search_score(mode, left.second) <
                           shape_observer.search_score(mode, right.second);
                })->first;
        }
        std::cout << "lock_coarse_best_phase=" << best_phase << "\n";
        std::vector<int> fine_phases;
        for (int offset : {-8, -4, 0, 4, 8})
            fine_phases.push_back((best_phase + offset) & 0xff);
        std::vector<std::pair<int, task5::ShapeMetrics>> fine_results;
        for (int phase : fine_phases)
            fine_results.emplace_back(phase, measure_phase(phase, 500, 250));
        best_phase = std::max_element(fine_results.begin(), fine_results.end(),
            [&](const auto& left, const auto& right) {
                return shape_observer.search_score(mode, left.second) <
                       shape_observer.search_score(mode, right.second);
            })->first;
        std::cout << "lock_fine_best_phase=" << best_phase << "\n";
        const auto final_metrics = measure_phase(best_phase, 650, 600);
        command.phase = static_cast<uint8_t>(best_phase);
        command.flags = shape_observer.shape_ok(mode, final_metrics) ? 0x01 : 0x00;
        uart.send_dds(command);
        uart.read_response(&status, 500, 0x01);
        const bool final_ok = shape_observer.shape_ok(mode, final_metrics);
        std::cout << "lock_final_phase=" << best_phase
                  << " shape_ok=" << final_ok
                  << " output_hz=" << output_frequency_hz << "\n";
        int stable_windows = 0;
        for (int window = 0; window < 10; ++window) {
            const auto metrics = measure_phase(best_phase, 0, 500);
            const bool good = shape_observer.shape_ok(mode, metrics);
            if (good) ++stable_windows;
            std::cout << "lock_hold_window=" << window + 1
                      << " good=" << good << "\n";
        }
        const bool stable_5s = final_ok && stable_windows == 10;
        command.flags = stable_5s ? 0x01 : 0x00;
        uart.send_dds(command);
        uart.read_response(&status, 500, 0x01);
        std::cout << "lock_stable_5s=" << stable_5s
                  << " good_windows=" << stable_windows << "/10\n";
        return 0;
    }
    if (args.grid_only) {
        const auto deadline = task5::Clock::now() +
            std::chrono::seconds(std::max(1, args.duration_s));
        while (task5::Clock::now() < deadline && !calibrator.locked()) {
            task5::Frame frame;
            if (!camera.read(&frame)) continue;
            calibrator.auto_locate(frame.bgr);
        }
        const auto geometry = calibrator.geometry();
        if (!geometry) {
            std::cout << "grid_locked=0\n";
            return 13;
        }
        std::cout << "grid_locked=1 roi=" << geometry->roi.x << ','
                  << geometry->roi.y << ',' << geometry->roi.width << ','
                  << geometry->roi.height << "\n";
        return 0;
    }
    if (args.candidate_bank_base_hz > 0.0) {
        task5::CandidateBankMatcher matcher;
        std::array<std::vector<double>, 16> band_scores;
        const auto settle_deadline = task5::Clock::now() +
            std::chrono::milliseconds(args.candidate_settle_ms);
        while (task5::Clock::now() < settle_deadline) {
            task5::Frame discarded;
            camera.read(&discarded);
        }
        const auto measure_deadline = task5::Clock::now() +
            std::chrono::milliseconds(args.candidate_measure_ms);
        while (task5::Clock::now() < measure_deadline) {
            task5::Frame frame;
            if (!camera.read(&frame)) continue;
            if (!calibrator.geometry()) calibrator.auto_locate(frame.bgr);
            const cv::Mat normalized = calibrator.normalize(frame.bgr);
            const task5::TraceObservation observation = segmenter.process(
                normalized, frame.sequence, frame.timestamp);
            const task5::CandidateBankResult result = matcher.score(observation);
            for (int band = 0; band < 16; ++band) {
                const auto& score = result.bands[static_cast<size_t>(band)];
                if (score.valid) band_scores[static_cast<size_t>(band)].push_back(
                    score.score);
            }
        }
        int best_band = -1;
        double best_score = std::numeric_limits<double>::infinity();
        double runner_score = std::numeric_limits<double>::infinity();
        for (int band = 0; band < 16; ++band) {
            auto& values = band_scores[static_cast<size_t>(band)];
            if (values.empty()) {
                std::cout << "bank_band=" << band << " valid_frames=0\n";
                continue;
            }
            const auto middle = values.begin() + values.size() / 2;
            std::nth_element(values.begin(), middle, values.end());
            const double score = *middle;
            std::cout << "bank_band=" << band << " frequency_hz="
                      << std::fixed << std::setprecision(1)
                      << args.candidate_bank_base_hz + 100.0 * band
                      << " valid_frames=" << values.size() << " score="
                      << std::setprecision(5) << score << "\n";
            if (score < best_score) {
                runner_score = best_score;
                best_score = score;
                best_band = band;
            } else if (score < runner_score) {
                runner_score = score;
            }
        }
        const double best_frequency = best_band >= 0
            ? args.candidate_bank_base_hz + 100.0 * best_band : 0.0;
        if (best_band >= 0) {
            task5::DdsCommand command;
            command.ftw = ftw_for_frequency(best_frequency, dds_clock_hz);
            command.amplitude = 102;
            command.flags = 0x00;
            uart.send_dds(command);
            task5::FpgaResponse acknowledgement;
            uart.read_response(&acknowledgement, 300, 0x01);
        }
        std::cout << "candidate_bank_best_hz=" << std::fixed
                  << std::setprecision(1) << best_frequency << " score="
                  << std::setprecision(5) << best_score << " margin="
                  << runner_score - best_score << "\n";
        return best_band >= 0 ? 0 : 15;
    }
    if (args.phase_code) {
        int calibration_frames = 0;
        const auto calibration_deadline = task5::Clock::now() +
            std::chrono::seconds(3);
        while (!calibrator.locked() &&
               task5::Clock::now() < calibration_deadline) {
            task5::Frame frame;
            if (!camera.read(&frame)) continue;
            ++calibration_frames;
            if (!args.snapshot_dir.empty()) {
                const std::filesystem::path directory(args.snapshot_dir);
                const std::string stem = "calibration_" +
                    std::to_string(frame.sequence);
                cv::imwrite((directory / (stem + ".jpg")).string(), frame.bgr);
            }
            calibrator.auto_locate(frame.bgr);
        }
        const auto geometry = calibrator.geometry();
        if (!geometry) {
            std::cout << "phase_code_grid_locked=0 calibration_frames="
                      << calibration_frames << "\n";
            return 16;
        }
        std::cout << "phase_code_grid_locked=1 calibration_frames="
                  << calibration_frames << " roi=" << geometry->roi.x << ','
                  << geometry->roi.y << ',' << geometry->roi.width << ','
                  << geometry->roi.height << "\n";
        if (args.drive && !phase_code_command_sent) {
            if (!send_phase_code()) return 7;
            const auto settle_deadline = task5::Clock::now() +
                std::chrono::milliseconds(400);
            while (task5::Clock::now() < settle_deadline) {
                task5::Frame discarded;
                camera.read(&discarded);
            }
        }

        std::vector<double> valid_frequencies;
        for (int window = 0; window < args.phase_windows; ++window) {
            task5::PhaseCodeMatcher matcher(phase_code);
            const auto deadline = task5::Clock::now() +
                std::chrono::seconds(std::max(1, args.duration_s));
            while (task5::Clock::now() < deadline) {
                task5::Frame frame;
                if (!camera.read(&frame)) continue;
                const cv::Mat normalized = calibrator.normalize(frame.bgr);
                const task5::TraceObservation observation = segmenter.process(
                    normalized, frame.sequence, frame.timestamp);
                matcher.add(observation);
                if (!args.snapshot_dir.empty()) {
                    const std::filesystem::path directory(args.snapshot_dir);
                    const std::string stem = "window_" +
                        std::to_string(window + 1) + "_frame_" +
                        std::to_string(frame.sequence);
                    cv::imwrite((directory / (stem + ".jpg")).string(),
                                frame.bgr);
                    cv::imwrite((directory / (stem + "_normalized.jpg")).string(),
                                normalized);
                }
            }
            const auto estimate = matcher.estimate();
            std::cout << "phase_code_window=" << window + 1
                      << " frames=" << matcher.frame_count()
                      << " frequency_hz=" << std::fixed << std::setprecision(1)
                      << estimate.frequency_hz << " runner_hz="
                      << estimate.runner_up_hz << " score="
                      << std::setprecision(5) << estimate.score << " margin="
                      << estimate.margin << " coarse_cycles="
                      << estimate.observed_cycles << " valid="
                      << estimate.valid << "\n";
            if (estimate.valid)
                valid_frequencies.push_back(estimate.frequency_hz);
        }
        if (valid_frequencies.empty()) {
            std::cout << "phase_code_consensus_hz=0.0 valid_windows=0\n";
            return 16;
        }
        std::sort(valid_frequencies.begin(), valid_frequencies.end());
        double consensus_hz = 0.0;
        int consensus_support = 0;
        for (double candidate : valid_frequencies) {
            std::vector<double> cluster;
            for (double value : valid_frequencies) {
                if (std::abs(value - candidate) <= 100.0)
                    cluster.push_back(value);
            }
            if (static_cast<int>(cluster.size()) > consensus_support) {
                consensus_support = static_cast<int>(cluster.size());
                const auto middle = cluster.begin() + cluster.size() / 2;
                std::nth_element(cluster.begin(), middle, cluster.end());
                consensus_hz = *middle;
            }
        }
        const int required_support = std::max(
            1, static_cast<int>(valid_frequencies.size() / 2 + 1));
        if (consensus_support < required_support) {
            std::cout << "phase_code_consensus_hz=0.0 valid_windows="
                      << valid_frequencies.size() << " support="
                      << consensus_support << " required="
                      << required_support << "\n";
            return 17;
        }
        std::cout << "phase_code_consensus_hz=" << std::fixed
                  << std::setprecision(1) << consensus_hz
                  << " valid_windows=" << valid_frequencies.size()
                  << " support=" << consensus_support << "\n";
        return 0;
    }
    if (!args.sine_candidates.empty()) {
        task5::SineMatcher matcher;
        double best_frequency_hz = 0.0;
        double best_score = std::numeric_limits<double>::infinity();
        for (double frequency_hz : args.sine_candidates) {
            const int frequency_bin = static_cast<int>(std::lround(
                frequency_hz / 100.0));
            task5::DdsCommand command;
            command.ftw = ftw_for_frequency(frequency_hz, dds_clock_hz);
            command.amplitude = 102;
            if (args.sine_gate_profile >= 0) {
                command.phase = static_cast<uint8_t>(frequency_bin & 0xff);
                command.flags = static_cast<uint8_t>(
                    0x40 | ((args.sine_gate_profile & 0x03) << 4) |
                    ((frequency_bin >> 8) & 0x03));
            } else {
                command.flags = 0x00;
            }
            if (!uart.send_dds(command)) {
                std::cerr << "failed to send sine candidate\n";
                return 10;
            }
            task5::FpgaResponse acknowledgement;
            uart.read_response(&acknowledgement, 300, 0x01);

            const auto settle_deadline = task5::Clock::now() +
                std::chrono::milliseconds(args.candidate_settle_ms);
            while (task5::Clock::now() < settle_deadline) {
                task5::Frame discarded;
                camera.read(&discarded);
            }

            std::vector<task5::SineMatchScore> scores;
            int measured_frames = 0;
            const auto measure_deadline = task5::Clock::now() +
                std::chrono::milliseconds(args.candidate_measure_ms);
            while (task5::Clock::now() < measure_deadline) {
                task5::Frame frame;
                if (!camera.read(&frame)) continue;
                ++measured_frames;
                if (!calibrator.geometry()) calibrator.auto_locate(frame.bgr);
                const cv::Mat normalized = calibrator.normalize(frame.bgr);
                const task5::TraceObservation observation =
                    segmenter.process(normalized, frame.sequence,
                                      frame.timestamp);
                const task5::SineMatchScore score = matcher.score(observation);
                if (score.valid) scores.push_back(score);
            }
            if (scores.empty()) {
                std::cout << "candidate_hz=" << frequency_hz
                          << " valid_frames=0 total_frames="
                          << measured_frames << "\n";
                continue;
            }
            std::sort(scores.begin(), scores.end(),
                      [](const task5::SineMatchScore& left,
                         const task5::SineMatchScore& right) {
                          return left.score < right.score;
                      });
            const task5::SineMatchScore& median = scores[scores.size() / 2];
            const double valid_fraction = static_cast<double>(scores.size()) /
                std::max(1, measured_frames);
            const double ranked_score = median.score + 0.60 *
                (1.0 - valid_fraction);
            std::cout << "candidate_hz=" << std::fixed
                      << std::setprecision(1) << frequency_hz
                      << " valid_frames=" << scores.size()
                      << " total_frames=" << measured_frames
                      << " valid_fraction=" << std::setprecision(3)
                      << valid_fraction
                      << " score=" << std::setprecision(5) << median.score
                      << " ranked=" << ranked_score
                      << " foreground=" << median.foreground_fraction
                      << " row50=" << median.median_row_occupancy
                      << " row90=" << median.p90_row_occupancy
                      << " fill=" << median.bounding_fill << "\n";
            if (ranked_score < best_score) {
                best_score = ranked_score;
                best_frequency_hz = frequency_hz;
            }
        }
        if (best_frequency_hz > 0.0) {
            const int frequency_bin = static_cast<int>(std::lround(
                best_frequency_hz / 100.0));
            task5::DdsCommand command;
            command.ftw = ftw_for_frequency(best_frequency_hz, dds_clock_hz);
            command.amplitude = 102;
            if (args.sine_gate_profile >= 0) {
                command.phase = static_cast<uint8_t>(frequency_bin & 0xff);
                command.flags = static_cast<uint8_t>(
                    0x40 | ((args.sine_gate_profile & 0x03) << 4) |
                    ((frequency_bin >> 8) & 0x03));
            } else {
                command.flags = 0x00;
            }
            uart.send_dds(command);
            task5::FpgaResponse acknowledgement;
            uart.read_response(&acknowledgement, 300, 0x01);
        }
        std::cout << "sine_match_best_hz=" << std::fixed
                  << std::setprecision(1) << best_frequency_hz
                  << " score=" << std::setprecision(5) << best_score << "\n";
        return best_frequency_hz > 0.0 ? 0 : 11;
    }
    const auto deadline = task5::Clock::now() +
                          std::chrono::seconds(std::max(1, args.duration_s));
    while (task5::Clock::now() < deadline) {
        task5::Frame frame;
        if (!camera.read(&frame)) {
            std::cerr << "camera read failed\n";
            continue;
        }
        process(frame);
    }
    return 0;
}
