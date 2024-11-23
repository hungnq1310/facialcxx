#ifndef FACEDETECTIONANDSPOOFING_H
#define FACEDETECTIONANDSPOOFING_H

#include "pipeline.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>

class FaceDetectionAndSpoofing : public Pipeline {
public:
    FaceDetectionAndSpoofing(const std::string& face_model_path, const std::string& spoof_model_path);
    std::vector<float> preprocess(const std::vector<float>& input) override;
    std::vector<float> postprocess(const std::vector<float>& input) override;
    std::vector<cv::Rect> detectFaces(const cv::Mat& image);
    float checkSpoofing(const cv::Mat& face);

private:
    Ort::Session face_session;
    Ort::Session spoof_session;
    std::string face_input_name;
    std::string face_output_name;
    std::string spoof_input_name;
    std::string spoof_output_name;
    cv::Size face_model_input_size;
    cv::Size spoof_model_input_size;

    Ort::Session loadModel(const std::string& path);
};

#endif // FACEDETECTIONANDSPOOFING_H