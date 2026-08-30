#include "task5/camera.hpp"

#include <opencv2/videoio.hpp>

#include <cmath>

#ifdef __linux__
#include <cerrno>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace task5 {

CameraSource::~CameraSource() { close(); }

bool CameraSource::open(const std::string& device, int width, int height,
                        int fps) {
    close();
    if (!capture_.open(device, cv::CAP_V4L2)) {
        return false;
    }
    device_ = device;
    capture_.set(cv::CAP_PROP_FOURCC,
                 static_cast<double>(cv::VideoWriter::fourcc('M', 'J', 'P', 'G')));
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    capture_.set(cv::CAP_PROP_FPS, fps);
    const int actual_width = static_cast<int>(std::lround(
        capture_.get(cv::CAP_PROP_FRAME_WIDTH)));
    const int actual_height = static_cast<int>(std::lround(
        capture_.get(cv::CAP_PROP_FRAME_HEIGHT)));
    const double actual_fps = capture_.get(cv::CAP_PROP_FPS);
    if (!capture_.isOpened() || actual_width != width ||
        actual_height != height || actual_fps + 0.5 < fps) {
        close();
        return false;
    }
    sequence_ = 0;
    return true;
}

bool CameraSource::read(Frame* frame) {
    if (!frame || !capture_.isOpened()) return false;
    cv::Mat image;
    if (!capture_.read(image) || image.empty()) return false;
    frame->bgr = std::move(image);
    frame->timestamp = Clock::now();
    frame->sequence = ++sequence_;
    return true;
}

bool CameraSource::set_manual_exposure(int exposure_absolute) {
    if (!capture_.isOpened() || device_.empty() || exposure_absolute <= 0)
        return false;
#ifdef __linux__
    const int descriptor = ::open(device_.c_str(), O_RDWR | O_NONBLOCK);
    if (descriptor < 0) return false;
    const auto set_control = [&](uint32_t id, int value) {
        v4l2_control control{};
        control.id = id;
        control.value = value;
        int result = 0;
        do {
            result = ::ioctl(descriptor, VIDIOC_S_CTRL, &control);
        } while (result < 0 && errno == EINTR);
        return result == 0;
    };
    const bool manual = set_control(V4L2_CID_EXPOSURE_AUTO,
                                    V4L2_EXPOSURE_MANUAL);
    const bool exposure = manual && set_control(
        V4L2_CID_EXPOSURE_ABSOLUTE, exposure_absolute);
    ::close(descriptor);
    return exposure;
#else
    return false;
#endif
}

void CameraSource::close() {
    if (capture_.isOpened()) capture_.release();
    device_.clear();
}

bool CameraSource::is_open() const { return capture_.isOpened(); }

}  // namespace task5
