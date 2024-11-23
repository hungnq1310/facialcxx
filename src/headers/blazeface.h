// #include <opencv2/dnn.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"
// #include "tensorflow/lite/optional_debug_tools.h"

// using namespace cv;
// using namespace std;

// struct FaceResult {
// 	float score = 0.0;
// 	Rect bbox;
//     float keypoints[10] = {0.0};
// };

// class BlazeFace {
// public:
// 	BlazeFace();
//     ~BlazeFace();
//     int detect(Mat src, FaceResult *res);
//     // Methods
//     void init(const char *model, long modelSize);
// 	void init(const char *model);
// 	void init(const char *model, int numThreads);
//     static const int MAX_OUTPUT = 2304;
// 	float conf_threshold = 0.5;
// 	float nms_threshold = 0.3;
//     const int OUTPUT_WEIGHT = 16;
// private:
// 	// members
// 	const int INPUT_SIZE = 192;
// 	const int INPUT_CHANNELS = 3;
// 	float anchors[MAX_OUTPUT][2];
// 	char *m_modelBytes = nullptr;
// 	std::unique_ptr<tflite::FlatBufferModel> m_model;
// 	std::unique_ptr<tflite::Interpreter> m_interpreter;
// 	void preprocess(Mat input, Mat &input_data, float padding[]);
	
// };

#ifndef BLAZEFACE_H
#define BLAZEFACE_H

#include "pipeline.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>
#include <string>

class BlazeFace : public Pipeline {
public:
    BlazeFace(Model model);
    ~BlazeFace();
    Ort::Value preprocess(Ort::Value input) override;
    Ort::Value postprocess(Ort::Value input) override;
    Ort::Value inference(Ort::Value input) override;

private:
    Ort::Value createOrtValueFromMat(const cv::Mat& mat);
    cv::Mat createMatFromOrtValue(const Ort::Value& ort_value);
    std::vector<float> postprocessDetections(const cv::Mat& detections);
    Ort::Value createOrtValueFromVector(const std::vector<float>& vec);
    void generateAnchors(float anchors[][2]);

    // Additional member variables if needed
    char* m_modelBytes = nullptr;
};

#endif // BLAZEFACE_H