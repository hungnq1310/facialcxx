// #include "metricnet.h"

// TFLiteMetricNet::TFLiteMetricNet() {
// }


// TFLiteMetricNet::~TFLiteMetricNet() {
//     if (m_modelBytes != nullptr) {
//         free(m_modelBytes);
//         m_modelBytes = nullptr;
//     }
// }


// void TFLiteMetricNet::initModel(const char *tfliteModel, long modelSize) {

//     // Copy to model bytes as the caller might release this memory while we need it (EXC_BAD_ACCESS error on ios)
//     m_modelBytes = (char *) malloc(sizeof(char) * modelSize);
//     memcpy(m_modelBytes, tfliteModel, sizeof(char) * modelSize);
//     m_model = tflite::FlatBufferModel::BuildFromBuffer(m_modelBytes, modelSize);
//     assert(m_model != nullptr);

//     // Build the interpreter
//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     tflite::InterpreterBuilder builder(*m_model, resolver);
//     builder(&m_interpreter);
//     assert(m_interpreter != nullptr);

//     // Allocate tensor buffers.
//     assert(m_interpreter->AllocateTensors() == kTfLiteOk);
//     assert(m_interpreter->Invoke() == kTfLiteOk);
// }

// void TFLiteMetricNet::initModel(const char *tfliteModel) {

//     m_model = tflite::FlatBufferModel::BuildFromFile(tfliteModel);
//     assert(m_model != nullptr);

//     // Build the interpreter
//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     tflite::InterpreterBuilder builder(*m_model, resolver);
//     builder(&m_interpreter);
//     assert(m_interpreter != nullptr);

//     // Allocate tensor buffers.
//     assert(m_interpreter->AllocateTensors() == kTfLiteOk);
//     assert(m_interpreter->Invoke() == kTfLiteOk);
// }

// void TFLiteMetricNet::initModel(const char *tfliteModel, int numThreads) {

//     m_model = tflite::FlatBufferModel::BuildFromFile(tfliteModel);
//     assert(m_model != nullptr);

//     // Build the interpreter
//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     tflite::InterpreterBuilder builder(*m_model, resolver);
//     builder(&m_interpreter);
//     assert(m_interpreter != nullptr);

//     // Allocate tensor buffers.
//     assert(m_interpreter->AllocateTensors() == kTfLiteOk);
//     assert(m_interpreter->Invoke() == kTfLiteOk);
//     // interpreter->SetAllowFp16PrecisionForFp32(true);
//     m_interpreter->SetNumThreads(numThreads);
// }

// void TFLiteMetricNet::predict(float feat1[], float feat2[], PredictResultMetricNet *res) {
//     // get input & output layer of tflite model
//     float *inputLayer1 = m_interpreter->typed_input_tensor<float>(0);
// //    float *inputLayer2 = m_interpreter->typed_input_tensor<float>(1);
//     float *outputLayer = m_interpreter->typed_output_tensor<float>(0);

//     // merge input
//     float feat[BATCH_SIZE * EMBEDDING_SIZE + EMBEDDING_SIZE];
//     for (int i = 0; i < EMBEDDING_SIZE; i++){
//         feat[i] = feat1[i];
//     }
//     for (int i = 0; i < BATCH_SIZE * EMBEDDING_SIZE; i++) {
//         feat[i + EMBEDDING_SIZE] = feat2[i];
//     }

//     // copy the input image to input layer
//     memcpy(inputLayer1, feat, (BATCH_SIZE * EMBEDDING_SIZE + EMBEDDING_SIZE) * sizeof(float));
// //    memcpy(inputLayer1, feat1, EMBEDDING_SIZE * sizeof(float));
// //    memcpy(inputLayer2, feat2, BATCH_SIZE * EMBEDDING_SIZE * sizeof(float));

//     // compute model instance
//     if (m_interpreter->Invoke() != kTfLiteOk) {
//         printf("Error invoking detection model");
//     } else{
//         for (int i = 0; i < BATCH_SIZE; i++){
//             res[i].results[0] = outputLayer[i];
//         }

//     }

// }

#include "metricnet.h"
#include <iostream>
#include <model.h>
#include "pipeline.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

MetricNet::MetricNet(Model model)
    : Pipeline(model) {
    // Additional initialization if needed
}

Ort::Value MetricNet::preprocess(Ort::Value input) {
    // Implement preprocessing logic here
    std::cout << "MetricNet Preprocessing..." << std::endl;
    cv::Mat image = createMatFromOrtValue(input);
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(256, 256));
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255);
    Ort::Value preprocessed_image = createOrtValueFromMat(resized_image);
    return preprocessed_image;
}

Ort::Value MetricNet::postprocess(Ort::Value input) {
    // Implement postprocessing logic here
    std::cout << "MetricNet Postprocessing..." << std::endl;
    // Example: return the input as is
    return input;
}

Ort::Value MetricNet::inference(Ort::Value input) {
    // Run the model with the preprocessed input
    Ort::Value preprocessed_input = preprocess(input);
    Ort::Value output = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &preprocessed_input, 1, output_names.data(), 1);
    Ort::Value postprocessed_output = postprocess(output);
    return postprocessed_output;
}

Ort::Value MetricNet::createOrtValueFromMat(const cv::Mat& mat) {
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

cv::Mat MetricNet::createMatFromOrtValue(const Ort::Value& ort_value) {
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

void MetricNet::predict(float feat1[EMBEDDING_SIZE], float feat2[BATCH_SIZE * EMBEDDING_SIZE], PredictResultMetricNet *res) {
    float* input1 = session.GetInputTensorMutableData<float>(0);
    float* input2 = session.GetInputTensorMutableData<float>(1);

    std::copy(feat1, feat1 + EMBEDDING_SIZE, input1);
    std::copy(feat2, feat2 + BATCH_SIZE * EMBEDDING_SIZE, input2);

    if (session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input1, 1, output_names.data(), 1) != kTfLiteOk) {
        std::cerr << "Failed to invoke ONNX Runtime session" << std::endl;
        return;
    }

    float* output = session.GetOutputTensorMutableData<float>(0);
    res->results[0] = output[0];
}