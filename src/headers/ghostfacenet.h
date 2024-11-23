#ifndef GHOSTFACENET_H
#define GHOSTFACENET_H

#include "pipeline.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>
#include <string>

class GhostFaceNet : public Pipeline {
public:
    GhostFaceNet(Model model);
    Ort::Value preprocess(Ort::Value input) override;
    Ort::Value postprocess(Ort::Value input) override;
    Ort::Value inference(Ort::Value input) override;

private:
    Ort::Value createOrtValueFromMat(const cv::Mat& mat);
    cv::Mat createMatFromOrtValue(const Ort::Value& ort_value);
};

#endif // GHOSTFACENET_H