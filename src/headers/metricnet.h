#ifndef METRICNET_H
#define METRICNET_H

#include "pipeline.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>
#include <string>

struct PredictResultMetricNet {
    float results[1];
};

const int BATCH_SIZE = 1;
const int EMBEDDING_SIZE = 512; // Assuming EMBEDDING_SIZE is 512

class MetricNet : public Pipeline {
public:
    MetricNet(Model model);
    Ort::Value preprocess(Ort::Value input) override;
    Ort::Value postprocess(Ort::Value input) override;
    Ort::Value inference(Ort::Value input) override;
    void predict(float feat1[EMBEDDING_SIZE], float feat2[BATCH_SIZE * EMBEDDING_SIZE], PredictResultMetricNet *res);

private:
    Ort::Value createOrtValueFromMat(const cv::Mat& mat);
    cv::Mat createMatFromOrtValue(const Ort::Value& ort_value);
};

#endif // METRICNET_H