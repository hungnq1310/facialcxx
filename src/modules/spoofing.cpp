#include "spoofing.h"
#include <iostream>
#include <model.h>
#include "pipeline.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

Spoofing::Spoofing(Model model)
    : Pipeline(model) {
    // Additional initialization if needed
}

Ort::Value Spoofing::preprocess(Ort::Value input) {
    // Implement preprocessing logic here
    std::cout << "Spoofing Preprocessing..." << std::endl;
    cv::Mat image = createMatFromOrtValue(input);
    cv::Mat resized_image;
    cv::resize(input, resized_image, cv::Size(256, 256));
    // Return to Ort::Value
    Ort::Value preprocessed_image = createOrtValueFromMat(resized_image);
    return preprocessed_image;
}

Ort::Value Spoofing::postprocess(Ort::Value input) {
    // Implement postprocessing logic here
    std::cout << "Spoofing Postprocessing..." << std::endl;
    // Example: return the input as is
    return input;
}

Ort::Value Spoofing::inference(Ort::Value input) {
    // Run the model with the preprocessed input
    Ort::Value preprocessed_input = preprocess(input);
    Ort::Value output = model->run(preprocessed_input);
    // Postprocess the output
    Ort::Value postprocessed_output = postprocess(output);
    return postprocessed_output;
}

Ort::Value Spoofing::createOrtValueFromMat(const cv::Mat& mat) {
    // Implement conversion from cv::Mat to Ort::Value
    // Example: create a tensor from the image data
    std::vector<int64_t> dims = {1, mat.rows, mat.cols, mat.channels()};
    size_t tensor_size = mat.total() * mat.elemSize();
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info, mat.ptr<float>(), tensor_size, dims.data(), dims.size());
    return tensor;
}

cv::Mat Spoofing::createMatFromOrtValue(const Ort::Value& ort_value) {
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