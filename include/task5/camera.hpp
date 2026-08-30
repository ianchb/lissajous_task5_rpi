#pragma once

#include "task5/types.hpp"

#include <opencv2/videoio.hpp>

#include <string>

namespace task5 {

class CameraSource {
public:
    CameraSource() = default;
    ~CameraSource();
    bool open(const std::string& device, int width = 640, int height = 480,
              int fps = 30);
    bool read(Frame* frame);
    bool set_manual_exposure(int exposure_absolute);
    void close();
    bool is_open() const;

private:
    cv::VideoCapture capture_;
    std::string device_;
    uint64_t sequence_ = 0;
};

}  // namespace task5
