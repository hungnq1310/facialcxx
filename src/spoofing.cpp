#include "spoofing.h"
#include <iostream>
#include <ortcxx/model.h>
#include <ortcxx/pipeline.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace ortcxx::pipeline;
using namespace ortcxx::model;

Spoofing::Spoofing(shared_ptr<Model> extractor, shared_ptr<Model> classifier)
{
    // Additional initialization if needed
    this->extractor = extractor;
    this->classifier = classifier;
}

void Spoofing::preprocess(cv::Mat input, cv::Mat &output) {
    // Implement preprocessing logic here
    std::cout << "Spoofing Preprocessing..." << std::endl;

    // Convert BGR to RGB
    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);

    // Resize the image to (256, 256)
    cv::resize(input, input, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);

    // Convert the image to float and scale to [0, 1]
    input.convertTo(input, CV_32FC1, 1.0 / 255.0);

    // Normalize the image with mean [0.5, 0.5, 0.5] and std [0.5, 0.5, 0.5]
    input = (input - 0.5) / 0.5;

    // Assign the processed image to output
    output = input;
}

std::vector<float> Spoofing::postprocess(std::vector<float> input) {
    // Implement postprocessing logic here
    std::cout << "Spoofing Postprocessing..." << std::endl;

    // Apply softmax
    std::vector<float> softmax_output(input.size());
    for (size_t i = 0; i < input.size(); i+=2) {
        softmax_output[i] = std::exp(input[i]);
        softmax_output[i+1] = std::exp(input[i+1]);
        // call softmax
        float sum = softmax_output[i] + softmax_output[i+1];
        softmax_output[i] /= sum;
        softmax_output[i+1] /= sum;        
    }
    return softmax_output;
}

std::vector<float> Spoofing::inference(cv::Mat input) {
    // Preprocess the input
    cv::Mat preprocessed_input;
    preprocess(input, preprocessed_input);

    // Convert preprocessed input to Ort::Value
    Ort::Value input_tensor = this->createOrtValueFromMat(preprocessed_input);
    auto dims_input = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
    // print dims2
    for (int i = 0; i < dims_input.size(); i++) {
        std::cout << "dims_input: " << dims_input[i] << std::endl;
    }

    // Run the extractor model
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input_tensor));
    
    
    auto extract_outputs = this->extractor->run(
        inputs, 
        shared_ptr<const char*>(),
        Ort::RunOptions()
    );

    // Get the second output from the extractor
    auto second_output = std::move(extract_outputs->at(0));

    // Run the classifier model with the second output
    std::vector<Ort::Value> classifier_inputs;
    classifier_inputs.push_back(std::move(second_output));
    auto classifier_outputs = this->classifier->run(
        classifier_inputs, 
        std::shared_ptr<const char*>(),
        Ort::RunOptions()
    );

    // Convert the final output to a vector of floats
    auto final_output = classifier_outputs->at(0).GetTensorMutableData<float>();
    std::vector<float> output_vector;
    for (int i = 0; i < 2; i++) {
        output_vector.push_back(final_output[i]);
    }; //MAKE SURE THIS IS CORRECT

    // Postprocess the output
    return postprocess(output_vector);
}

Ort::Value Spoofing::createOrtValueFromMat(const cv::Mat& mat) {
    // Ensure the input mat is of type CV_32F
    cv::Mat input;
    if (mat.type() != CV_32F) {
        mat.convertTo(input, CV_32F);
    } else {
        input = mat;
    }

    // Get the dimensions of the input mat
    std::vector<int64_t> dims = {1, input.channels(), input.rows, input.cols};

    // Create an Ort::MemoryInfo object
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create an Ort::Value tensor from the input mat data
    return Ort::Value::CreateTensor<float>(
        memory_info,
        input.ptr<float>(),
        input.total() * input.channels(),
        dims.data(),
        dims.size()
    );
}


void cropFace(cv::Mat src, cv::Mat& dst, cv::Rect rectPoint, int out_height, int out_width) {
    // warp
    if (rectPoint.x < 0) {
        rectPoint.x = 0;
    }
    if (rectPoint.y < 0) {
        rectPoint.y = 0;
    }
    if (rectPoint.width > src.cols) {
        rectPoint.width = src.cols;
    }
    if (rectPoint.height > src.rows) {
        rectPoint.height = src.rows;
    }
    // crop
    dst = src(
        rectPoint
    );
}
