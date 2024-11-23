#include "yolov7.h"
#include <iostream>
#include <model.h>
#include "pipeline.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

YoloV7::YoloV7(Model model)
    : Pipeline(model) {
    // Additional initialization if needed
}

Ort::Value YoloV7::preprocess(Ort::Value input) {
    // Implement preprocessing logic here
    std::cout << "YoloV7 Preprocessing..." << std::endl;
    cv::Mat image = createMatFromOrtValue(input);
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(640, 640));
    // Convert the image to float and normalize to [0, 1]
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255);
    // Return to Ort::Value
    Ort::Value preprocessed_image = createOrtValueFromMat(resized_image);
    return preprocessed_image;
}

Ort::Value YoloV7::postprocess(Ort::Value input) {
    // Implement postprocessing logic here
    std::cout << "YoloV7 Postprocessing..." << std::endl;
    cv::Mat detections = createMatFromOrtValue(input);
    std::vector<float> output = postprocessDetections(detections);
    // Convert std::vector<float> to Ort::Value
    Ort::Value postprocessed_output = createOrtValueFromVector(output);
    return postprocessed_output;
}

Ort::Value YoloV7::inference(Ort::Value input) {
    // Run the model with the preprocessed input
    Ort::Value preprocessed_input = preprocess(input);
    Ort::Value output = model->run(preprocessed_input);
    // Postprocess the output
    Ort::Value postprocessed_output = postprocess(output);
    return postprocessed_output;
}

Ort::Value YoloV7::createOrtValueFromMat(const cv::Mat& mat) {
    // Ensure the input mat is of type CV_32F (float)
    cv::Mat mat_float;
    if (mat.type() != CV_32F) {
        mat.convertTo(mat_float, CV_32F);
    } else {
        mat_float = mat;
    }

    // Define the dimensions of the tensor
    std::vector<int64_t> dims = {1, mat_float.rows, mat_float.cols, mat_float.channels()};

    // Calculate the size of the tensor
    size_t tensor_size = mat_float.total() * mat_float.elemSize();

    // Create the tensor from the image data
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info, mat_float.ptr<float>(), tensor_size, dims.data(), dims.size());

    return tensor;
}

cv::Mat YoloV7::createMatFromOrtValue(const Ort::Value& ort_value) {
    // Ensure the Ort::Value is a tensor
    if (!ort_value.IsTensor()) {
        throw std::invalid_argument("Ort::Value is not a tensor");
    }

    // Get the tensor information
    Ort::TensorTypeAndShapeInfo tensor_info = ort_value.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = tensor_info.GetShape();
    size_t total_elements = tensor_info.GetElementCount();

    // Ensure the tensor has 4 dimensions (batch, height, width, channels)
    if (dims.size() != 4) {
        throw std::invalid_argument("Tensor does not have 4 dimensions");
    }

    // Extract dimensions
    int batch_size = dims[0];
    int height = dims[1];
    int width = dims[2];
    int channels = dims[3];

    // Ensure batch size is 1
    if (batch_size != 1) {
        throw std::invalid_argument("Batch size is not 1");
    }

    // Get the data pointer
    float* tensor_data = ort_value.GetTensorMutableData<float>();

    // Create a cv::Mat from the tensor data
    cv::Mat mat(height, width, CV_32FC(channels), tensor_data);

    return mat;
}

std::vector<float> YoloV7::postprocessDetections(const cv::Mat& detections) {
    // Implement postprocessing logic here
    std::vector<float> output;
    for (int i = 0; i < detections.rows; ++i) {
        const float* detection = detections.ptr<float>(i);
        // Example: extract bounding box coordinates and confidence
        float confidence = detection[4];
        if (confidence > 0.5) { // Threshold for detection
            float x = detection[0];
            float y = detection[1];
            float w = detection[2];
            float h = detection[3];
            // Add detection to output
            output.insert(output.end(), {x, y, w, h, confidence});
        }
    }
    return output;
}

Ort::Value YoloV7::createOrtValueFromVector(const std::vector<float>& vec) {
    // Define the dimensions of the tensor
    std::vector<int64_t> dims = {1, static_cast<int64_t>(vec.size())};

    // Create the tensor from the vector data
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(vec.data()), vec.size() * sizeof(float), dims.data(), dims.size());

    return tensor;
}