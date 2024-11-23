// Continue to update

#include "facepipeline.h"
#include <iostream>

FaceDetectionAndSpoofing::FaceDetectionAndSpoofing(const std::string& face_model_path, const std::string& spoof_model_path)
    : Pipeline(face_model_path),
      face_session(loadModel(face_model_path)),
      spoof_session(loadModel(spoof_model_path)) {
    face_input_name = face_session.GetInputName(0, env);
    face_output_name = face_session.GetOutputName(0, env);
    auto face_input_shape = face_session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    face_model_input_size = cv::Size(face_input_shape[2], face_input_shape[1]);

    spoof_input_name = spoof_session.GetInputName(0, env);
    spoof_output_name = spoof_session.GetOutputName(0, env);
    auto spoof_input_shape = spoof_session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    spoof_model_input_size = cv::Size(spoof_input_shape[2], spoof_input_shape[1]);
}

Ort::Session FaceDetectionAndSpoofing::loadModel(const std::string& path) {
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return Ort::Session(env, path.c_str(), session_options);
}

std::vector<float> FaceDetectionAndSpoofing::preprocess(const std::vector<float>& input) {
    // Default implementation: return the input as is
    std::cout << "FaceDetectionAndSpoofing Preprocessing..." << std::endl;
    return input;
}

std::vector<float> FaceDetectionAndSpoofing::postprocess(const std::vector<float>& input) {
    // Default implementation: return the input as is
    std::cout << "FaceDetectionAndSpoofing Postprocessing..." << std::endl;
    return input;
}

std::vector<cv::Rect> FaceDetectionAndSpoofing::detectFaces(const cv::Mat& image) {
    cv::Mat resized_image;
    cv::resize(image, resized_image, face_model_input_size);
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255);

    std::vector<float> input_tensor_values(resized_image.begin<float>(), resized_image.end<float>());
    std::vector<int64_t> input_shape = {1, face_model_input_size.height, face_model_input_size.width, 3};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(env, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    auto output_tensors = face_session.Run(Ort::RunOptions{nullptr}, &face_input_name, &input_tensor, 1, &face_output_name, 1);
    std::vector<float> result = output_tensors.front().GetTensorMutableData<float>();

    std::vector<cv::Rect> faces;
    for (size_t i = 0; i < result.size(); i += 6) {
        float confidence = result[i + 4];
        if (confidence > 0.5) { // Threshold for detection
            int x = static_cast<int>(result[i] * image.cols);
            int y = static_cast<int>(result[i + 1] * image.rows);
            int w = static_cast<int>(result[i + 2] * image.cols - x);
            int h = static_cast<int>(result[i + 3] * image.rows - y);
            faces.emplace_back(x, y, w, h);
        }
    }
    return faces;
}

float FaceDetectionAndSpoofing::checkSpoofing(const cv::Mat& face) {
    cv::Mat resized_face;
    cv::resize(face, resized_face, spoof_model_input_size);
    resized_face.convertTo(resized_face, CV_32F, 1.0 / 255);

    std::vector<float> input_tensor_values(resized_face.begin<float>(), resized_face.end<float>());
    std::vector<int64_t> input_shape = {1, spoof_model_input_size.height, spoof_model_input_size.width, 3};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(env, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    auto output_tensors = spoof_session.Run(Ort::RunOptions{nullptr}, &spoof_input_name, &input_tensor, 1, &spoof_output_name, 1);
    std::vector<float> result = output_tensors.front().GetTensorMutableData<float>();

    return result[0];
}