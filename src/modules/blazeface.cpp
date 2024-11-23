// #include "blazeface.h"
// #include <iostream>
// void _ssd_generate_anchors(float anchors[][2]) {
//     // This function is build base on python version.
//     // Only use for face_detection_full_range_sparse.tflite
//     int layer_id = 0;
//     int num_layers = 1;
//     int strides[1] = {4};
//     int input_height = 192;
//     int input_width = 192;
//     int anchor_index = 0;
//     float anchor_offset_x = 0.5;
//     float anchor_offset_y = 0.5;
//     float interpolated_scale_aspect_ratio = 0.0;

//     int last_same_stride_layer, repeats;
//     while (layer_id < num_layers) {
//         last_same_stride_layer = layer_id;
//         repeats = 0;
//         while (last_same_stride_layer < num_layers &&
//                strides[last_same_stride_layer] == strides[layer_id]) {
//             last_same_stride_layer += 1;
//             // aspect_ratios are added twice per iteration
//             if (interpolated_scale_aspect_ratio == 1.0) {
//                 repeats += 2;
//             } else {
//                 repeats += 1;
//             }
//         }
//         int stride = strides[layer_id];
//         int feature_map_height = input_height / stride;
//         int feature_map_width = input_width / stride;
//         for (int y = 0; y < feature_map_height; y++) {
//             float y_center =
//                 ((float)y + anchor_offset_y) / (float)feature_map_height;
//             for (int x = 0; x < feature_map_width; x++) {
//                 float x_center =
//                     ((float)x + anchor_offset_x) / (float)feature_map_width;
//                 for (int i = 0; i < repeats; i++) {
//                     anchors[anchor_index][0] = x_center;
//                     anchors[anchor_index][1] = y_center;
//                     anchor_index+=1;
//                 }
//             }
//         }
//         layer_id = last_same_stride_layer;
//     }
// }

// void find_equations_of_line(float x1, float y1, float x2, float y2,
//                             float ret[]) {
//     float m = (y2 - y1) / (x2 - x1);
//     float c = y1 - m * x1;
//     ret[0] = m;
//     ret[1] = c;
// }

// void find_parallel_eqautions_line(float left_eye[2], float right_eye[2],
//                                   float mouth_center[2], float ret[]) {
//     float m, c, b;
//     find_equations_of_line(left_eye[0], left_eye[1], right_eye[0], right_eye[1],
//                            ret);
//     m = ret[0];
//     c = ret[1];
//     b = mouth_center[1] - m * mouth_center[0];
//     ret[0] = m;
//     ret[1] = b;
// }

// BlazeFace::BlazeFace() { _ssd_generate_anchors(anchors); }

// BlazeFace::~BlazeFace() {
//     if (m_modelBytes != nullptr) {
//         free(m_modelBytes);
//         m_modelBytes = nullptr;
//     }
// }

// void BlazeFace::init(const char *tfliteModel, long modelSize) {
//      // Copy to model bytes as the caller might release this memory while we need
//      // it (EXC_BAD_ACCESS error on ios)
//      m_modelBytes = (char *)malloc(sizeof(char) * modelSize);
//      memcpy(m_modelBytes, tfliteModel, sizeof(char) * modelSize);
//      m_model = tflite::FlatBufferModel::BuildFromBuffer(m_modelBytes,
//      modelSize); assert(m_model != nullptr);

//      // Build the interpreter
//      tflite::ops::builtin::BuiltinOpResolver resolver;
//      tflite::InterpreterBuilder builder(*m_model, resolver);
//      builder(&m_interpreter);
//      assert(m_interpreter != nullptr);

//      // Allocate tensor buffers.
//      assert(m_interpreter->AllocateTensors() == kTfLiteOk);
//      assert(m_interpreter->Invoke() == kTfLiteOk);

// //    // for PC
// //    m_model = tflite::FlatBufferModel::BuildFromFile(tfliteModel);
// //    tflite::ops::builtin::BuiltinOpResolver resolver;
// //    tflite::InterpreterBuilder(*m_model, resolver)(&m_interpreter);
// //    // Allocate tensor buffers.
// //    assert(m_interpreter != nullptr);
// //    assert(m_interpreter->AllocateTensors() == kTfLiteOk);
// //    assert(m_interpreter->Invoke() == kTfLiteOk);
// }

// void BlazeFace::init(const char *tfliteModel) {
//      // Copy to model bytes as the caller might release this memory while we need
//      // it (EXC_BAD_ACCESS error on ios)
//      m_model = tflite::FlatBufferModel::BuildFromFile(tfliteModel);
//      assert(m_model != nullptr);

//      // Build the interpreter
//      tflite::ops::builtin::BuiltinOpResolver resolver;
//      tflite::InterpreterBuilder builder(*m_model, resolver);
//      builder(&m_interpreter);
//      assert(m_interpreter != nullptr);

//      // Allocate tensor buffers.
//      assert(m_interpreter->AllocateTensors() == kTfLiteOk);
//      assert(m_interpreter->Invoke() == kTfLiteOk);
// }

// void BlazeFace::init(const char *tfliteModel, int numThreads) {
//      // Copy to model bytes as the caller might release this memory while we need
//      // it (EXC_BAD_ACCESS error on ios)
//      m_model = tflite::FlatBufferModel::BuildFromFile(tfliteModel);
//      assert(m_model != nullptr);

//      // Build the interpreter
//      tflite::ops::builtin::BuiltinOpResolver resolver;
//      tflite::InterpreterBuilder builder(*m_model, resolver);
//      builder(&m_interpreter);
//      assert(m_interpreter != nullptr);

//      // Allocate tensor buffers.
//      assert(m_interpreter->AllocateTensors() == kTfLiteOk);
//      assert(m_interpreter->Invoke() == kTfLiteOk);
//     //  interpreter->SetAllowFp16PrecisionForFp32(true);
//      m_interpreter->SetNumThreads(numThreads);

// }

// void BlazeFace::preprocess(Mat input, Mat &input_data, float padding[]) {
//     input_data = input.clone();
//     // Convert to RGB
//     cvtColor(input_data, input_data, COLOR_BGR2RGB);
//     // Padding the input image to square
//     int input_data_height = input_data.size().height;
//     int input_data_width = input_data.size().width;
//     int padding_left = 0;
//     int padding_right = 0;
//     int padding_top = 0;
//     int padding_bottom = 0;
//     // If height > width, padding left and right
//     if (input_data_height > input_data_width) {
//         padding_left = (input_data_height - input_data_width) / 2;
//         padding_right = (input_data_height - input_data_width) / 2;
//         // else padding top and bottom
//     } else {
//         padding_top = (input_data_width - input_data_height) / 2;
//         padding_bottom = (input_data_width - input_data_height) / 2;
//     }
//     copyMakeBorder(input_data, input_data, padding_top, padding_bottom,
//                    padding_left, padding_right, BORDER_CONSTANT,
//                    Scalar(0, 0, 0));
//     // convert to float 32, 3 channels
//     input_data.convertTo(input_data, CV_32FC3);
//     // normalize the data to -1, 1
//     input_data = (input_data - 127.5) / 127.5;
//     // resize to the input size
//     resize(input_data, input_data, Size(INPUT_SIZE, INPUT_SIZE));
//     // calculate the padding ratio for the original image
//     padding[0] = padding_left / (float)max(input_data_height, input_data_width);
//     padding[1] = padding_top / (float)max(input_data_height, input_data_width);
//     padding[2] =
//         padding_right / (float)max(input_data_height, input_data_width);
//     padding[3] =
//         padding_bottom / (float)max(input_data_height, input_data_width);
// }

// int BlazeFace::detect(Mat input, FaceResult *res) {
//     // Input image is expected to be BGR, channels last, 0-255, uint8 type
//     Mat input_data;
//     float padding[4];
//     preprocess(input, input_data, padding);
//     // get input & output layer of tflite model
//     float *inputLayer = m_interpreter->typed_input_tensor<float>(0);
//     float *outputData = m_interpreter->typed_output_tensor<float>(0);
//     float *raw_scores = m_interpreter->typed_output_tensor<float>(1);
//     // copy the input image to input layer
//     memcpy(inputLayer, input_data.data,
//            input_data.total() * input_data.elemSize());
//     // compute model instance
//     if (m_interpreter->Invoke() != kTfLiteOk) {
//         printf("Error invoking detection model");
//         return -1;
//     }
//     // reshape the output layer from 1D [MAX_OUTPUT*16] -> 3D [MAX_OUTPUT, 8, 2]
//     float raw_boxes[MAX_OUTPUT][8][2];
    
//     // filter out the boxes with low confidence score
//     float boxes[MAX_OUTPUT][8][2];
//     int boxes_index = 0;
//     vector<float> scores;
//     vector<Rect> rects;
//     for (unsigned i = 0; i < MAX_OUTPUT; i++) {
//         for (unsigned j = 0; j < 8; j++) {
//             raw_boxes[i][j][0] = outputData[i * 16 + j * 2];
//             raw_boxes[i][j][1] = outputData[i * 16 + j * 2 + 1];
//         }
//         // signmoid
//         raw_scores[i] = 1 / (1 + exp(-raw_scores[i]));
//         // filter out low confidence detections
//         if (raw_scores[i] > conf_threshold) {
//             for (unsigned j = 0; j < 8; j++) {
//                 // scale all values (applies to positions, width, and height
//                 // alike)
//                 boxes[boxes_index][j][0] = raw_boxes[i][j][0] / (float)INPUT_SIZE;
//                 boxes[boxes_index][j][1] = raw_boxes[i][j][1] / (float)INPUT_SIZE;
//             }
//             // adjust center coordinates and key points to anchor positions
//             boxes[boxes_index][0][0] += anchors[i][0];
//             boxes[boxes_index][0][1] += anchors[i][1];
//             for (unsigned k = 2; k < 8; k++) {
//                 boxes[boxes_index][k][0] += anchors[i][0];
//                 boxes[boxes_index][k][1] += anchors[i][1];
//             }
            
//             // convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
//             float center[2] = {boxes[boxes_index][0][0], boxes[boxes_index][0][1]};
//             float half_size[2] = {boxes[boxes_index][1][0] / 2,
//                                   boxes[boxes_index][1][1] / 2};
//             boxes[boxes_index][0][0] = center[0] - half_size[0];
//             boxes[boxes_index][0][1] = center[1] - half_size[1];
//             boxes[boxes_index][1][0] = center[0] + half_size[0];
//             boxes[boxes_index][1][1] = center[1] + half_size[1];
            
//             // remove padding effect
//             float left = padding[0], top = padding[1], right = padding[2], bottom = padding[3];
//             float h_scale = 1.0 - (left + right);
//             float v_scale = 1.0 - (top + bottom);
//             for (unsigned j = 0; j < 8; j++) {
//                 boxes[boxes_index][j][0] = (boxes[boxes_index][j][0] - left) / h_scale * input.size().width;
//                 boxes[boxes_index][j][1] = (boxes[boxes_index][j][1] - top) / v_scale * input.size().height;
//             }
//             // push back the Rect and scores for using nms OpenCV
//             rects.push_back(Rect(boxes[boxes_index][0][0], boxes[boxes_index][0][1], 
//                                  boxes[boxes_index][1][0] - boxes[boxes_index][0][0],
//                                  boxes[boxes_index][1][1] - boxes[boxes_index][0][1]));
//             scores.push_back(raw_scores[i]);
//             boxes_index += 1;
//         }
//     }

//     // perform non-maxima suppression
//     vector<int> selected_indices;
//     // std::cout<<scores.size()<<std::endl;
//     cv::dnn::NMSBoxes(rects, scores, conf_threshold, nms_threshold,
//                       selected_indices);
//     for (unsigned i = 0; i < selected_indices.size(); i++) {
//         res[i].bbox = Rect(boxes[selected_indices[i]][0][0],
//                            boxes[selected_indices[i]][0][1],
//                            boxes[selected_indices[i]][1][0] - boxes[selected_indices[i]][0][0],
//                            boxes[selected_indices[i]][1][1] - boxes[selected_indices[i]][0][1]);
//         res[i].score = scores[selected_indices[i]];
//         // Convert from 6 points landmark to 5 points landmark
//         // Get the position of left_eye, right_eye, nose, mouth_center
//         float left_eye[2] = {boxes[selected_indices[i]][2][0],
//                              boxes[selected_indices[i]][2][1]};
//         float right_eye[2] = {boxes[selected_indices[i]][3][0],
//                               boxes[selected_indices[i]][3][1]};
//         float nose[2] = {boxes[selected_indices[i]][4][0],
//                          boxes[selected_indices[i]][4][1]};
//         float mouth_center[2] = {boxes[selected_indices[i]][5][0],
//                                  boxes[selected_indices[i]][5][1]};
//         float ab[2];
//         // Find the line go through left_eye and right_eye. Then find the
//         // parallel line with this line and go through mouth_center
//         find_parallel_eqautions_line(left_eye, right_eye, mouth_center, ab);
//         // Rough estimation of the position of left_mouth and right_mouth
//         float left_mouth_x = mouth_center[0] + (left_eye[0] - right_eye[0]) / 3;
//         float left_mouth[2] = {left_mouth_x, ab[0] * left_mouth_x + ab[1]};
//         float right_mouth_x =
//             mouth_center[0] - (left_eye[0] - right_eye[0]) / 3;
//         float right_mouth[2] = {right_mouth_x, ab[0] * right_mouth_x + ab[1]};
        
//         // Assign the keypoints to output.
//         for (unsigned j = 0; j < 2; j++) {
//             res[i].keypoints[j] = left_eye[j];
//             res[i].keypoints[j + 2] = right_eye[j];
//             res[i].keypoints[j + 4] = nose[j];
//             res[i].keypoints[j + 6] = left_mouth[j];
//             res[i].keypoints[j + 8] = right_mouth[j];
//         }
//     }
//     return selected_indices.size();
// }

#include "blazeface.h"
#include <iostream>
#include <model.h>
#include "pipeline.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

BlazeFace::BlazeFace(Model model)
    : Pipeline(model) {
    // Additional initialization if needed
}

BlazeFace::~BlazeFace() {
    if (m_modelBytes != nullptr) {
        free(m_modelBytes);
        m_modelBytes = nullptr;
    }
}

Ort::Value BlazeFace::preprocess(Ort::Value input) {
    // Implement preprocessing logic here
    std::cout << "BlazeFace Preprocessing..." << std::endl;
    cv::Mat image = createMatFromOrtValue(input);
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(192, 192));
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255);
    Ort::Value preprocessed_image = createOrtValueFromMat(resized_image);
    return preprocessed_image;
}

Ort::Value BlazeFace::postprocess(Ort::Value input) {
    // Implement postprocessing logic here
    std::cout << "BlazeFace Postprocessing..." << std::endl;
    cv::Mat detections = createMatFromOrtValue(input);
    std::vector<float> output = postprocessDetections(detections);
    Ort::Value postprocessed_output = createOrtValueFromVector(output);
    return postprocessed_output;
}

Ort::Value BlazeFace::inference(Ort::Value input) {
    // Run the model with the preprocessed input
    Ort::Value preprocessed_input = preprocess(input);
    Ort::Value output = model->run(preprocessed_input);
    Ort::Value postprocessed_output = postprocess(output);
    return postprocessed_output;
}

Ort::Value BlazeFace::createOrtValueFromMat(const cv::Mat& mat) {
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

cv::Mat BlazeFace::createMatFromOrtValue(const Ort::Value& ort_value) {
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

std::vector<float> BlazeFace::postprocessDetections(const cv::Mat& detections) {
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

Ort::Value BlazeFace::createOrtValueFromVector(const std::vector<float>& vec) {
    // Define the dimensions of the tensor
    std::vector<int64_t> dims = {1, static_cast<int64_t>(vec.size())};

    // Create the tensor from the vector data
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(vec.data()), vec.size() * sizeof(float), dims.data(), dims.size());

    return tensor;
}

void BlazeFace::generateAnchors(float anchors[][2]) {
    // This function is built based on the Python version.
    // Only use for face_detection_full_range_sparse.tflite
    int layer_id = 0;
    int num_layers = 1;
    int strides[1] = {4};
    int input_height = 192;
    int input_width = 192;
    int anchor_index = 0;
    float anchor_offset_x = 0.5;
    float anchor_offset_y = 0.5;
    float interpolated_scale_aspect_ratio = 0.0;

    int last_same_stride_layer, repeats;
    while (layer_id < num_layers) {
        last_same_stride_layer = layer_id;
        repeats = 0;
        while (last_same_stride_layer < num_layers &&
               strides[last_same_stride_layer] == strides[layer_id]) {
            last_same_stride_layer += 1;
            // aspect_ratios are added twice per iteration
            if (interpolated_scale_aspect_ratio == 1.0) {
                repeats += 2;
            } else {
                repeats += 1;
            }
        }
        layer_id = last_same_stride_layer;
    }
}