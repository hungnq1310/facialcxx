#ifndef YOLOV7_H
#define YOLOV7_H

#include "pipeline.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>
#include <string>

class YoloV7 : public Pipeline {
public:
    YoloV7(Model model);
    Ort::Value preprocess(Ort::Value input) override;
    Ort::Value postprocess(Ort::Value input) override;
    Ort::Value inference(Ort::Value input) override;

private:
    Ort::Value createOrtValueFromMat(const cv::Mat& mat);
    cv::Mat createMatFromOrtValue(const Ort::Value& ort_value);
    std::vector<float> postprocessDetections(const cv::Mat& detections);
    Ort::Value createOrtValueFromVector(const std::vector<float>& vec);
};

#endif // YOLOV7_H