// #include "featurenet.h"

// TFLiteFeatureNet::TFLiteFeatureNet() {
// }


// TFLiteFeatureNet::~TFLiteFeatureNet() {
//     if (m_modelBytes != nullptr) {
//         free(m_modelBytes);
//         m_modelBytes = nullptr;
//     }
// }


// void TFLiteFeatureNet::initModel(const char *tfliteModel, long modelSize) {

// 	// Copy to model bytes as the caller might release this memory while we need it (EXC_BAD_ACCESS error on ios)
// 	m_modelBytes = (char *) malloc(sizeof(char) * modelSize);
// 	memcpy(m_modelBytes, tfliteModel, sizeof(char) * modelSize);
// 	m_model = tflite::FlatBufferModel::BuildFromBuffer(m_modelBytes, modelSize);
// 	assert(m_model != nullptr);

// 	// Build the interpreter
// 	tflite::ops::builtin::BuiltinOpResolver resolver;
// 	tflite::InterpreterBuilder builder(*m_model, resolver);
// 	builder(&m_interpreter);
// 	assert(m_interpreter != nullptr);

// 	// Allocate tensor buffers.
// 	assert(m_interpreter->AllocateTensors() == kTfLiteOk);
// 	assert(m_interpreter->Invoke() == kTfLiteOk);
// }

// void TFLiteFeatureNet::initModel(const char *tfliteModel) {
// 	m_model = tflite::FlatBufferModel::BuildFromFile(tfliteModel);
// 	assert(m_model != nullptr);

// 	// Build the interpreter
// 	tflite::ops::builtin::BuiltinOpResolver resolver;
// 	tflite::InterpreterBuilder builder(*m_model, resolver);
// 	builder(&m_interpreter);
// 	assert(m_interpreter != nullptr);

// 	// Allocate tensor buffers.
// 	assert(m_interpreter->AllocateTensors() == kTfLiteOk);
// 	assert(m_interpreter->Invoke() == kTfLiteOk);
// }

// void TFLiteFeatureNet::initModel(const char *tfliteModel, int numThreads) {
// 	m_model = tflite::FlatBufferModel::BuildFromFile(tfliteModel);
// 	assert(m_model != nullptr);

// 	// Build the interpreter
// 	tflite::ops::builtin::BuiltinOpResolver resolver;
// 	tflite::InterpreterBuilder builder(*m_model, resolver);
// 	builder(&m_interpreter);
// 	assert(m_interpreter != nullptr);

// 	// Allocate tensor buffers.
// 	assert(m_interpreter->AllocateTensors() == kTfLiteOk);
// 	assert(m_interpreter->Invoke() == kTfLiteOk);
//     // interpreter->SetAllowFp16PrecisionForFp32(true);
//     m_interpreter->SetNumThreads(numThreads);
// }


// void TFLiteFeatureNet::predict(Mat input, PredictResultFeatureNet &res) {

// 	// get input & output layer of tflite model
// 	float *inputLayer = m_interpreter->typed_input_tensor<float>(0);
// 	float *outputLayer = m_interpreter->typed_output_tensor<float>(0);

//     // copy the input image to input layer
//     memcpy(inputLayer, input.data, input.total() * input.elemSize());

// 	// compute model instance
// 	if (m_interpreter->Invoke() != kTfLiteOk) {
// 		printf("Error invoking detection model");
// 	} else{
// 		for (int i = 0; i < EMBEDDING_SIZE; ++i) {
// 			res.embedding[i] = outputLayer[i];
// 		}
// 	}
    
// }

// // Square each element in 2D Mat
// static Mat mat2DSquare(Mat mat){
//     Mat result = mat.clone();
//     for(int y=0; y<mat.size().height; y++){
//         for(int x=0; x<mat.size().width; x++){
//             result.at<float>(y,x) = mat.at<float>(y,x) * mat.at<float>(y,x);
//         }
//     }
//     return result;
// }

// // Elements wise subtration
// static Mat mat2DSubtract(Mat m1, Mat m2){
//     Mat result = m1.clone();
//     for(int y=0; y<m1.size().height; y++){
//         for(int x=0; x<m1.size().width; x++){
//             result.at<float>(y,x) = m1.at<float>(y,x) - m2.at<float>(y,x);
//         }
//     }
//     return result;
// }

// // Sum elements of 2D Mat with axis =1
// static Mat matSumAxis1(Mat mat){
//     Mat result(mat.size().height, 1 , CV_64F);
//     for(int y=0; y<mat.size().height; y++){
//         result.at<float>(y,0) = mat.at<float>(y,0) + mat.at<float>(y,1);
//     }
//     return result;

// }

// // Square root all elements in 1D Mat
// static Mat matSqrt1D(Mat mat){
//     Mat result = mat.clone();
//     for(int y=0; y<mat.size().height; y++){
//         result.at<float>(y,0) = sqrt(mat.at<float>(y,0));
//     }
//     return result;
// }

// // Sum all elements in 1D Mat
// static float matReduceSum1D(Mat mat){
//     float sum = 0.f;
//     for(int y=0; y<mat.size().height; y++){
//         sum += mat.at<float>(y,0);
//     }
//     return sum;
// }

// void cropFace(Mat src, Mat& dst, float landmarks[5][2], int out_height, int out_width){
//     // Init the result of crop face
//     Mat min_M(2, 3, CV_64F);
//     float min_error = numeric_limits<float>::infinity();

//     // the points detected by model
//     Mat src_points(1, 5, CV_32FC2);
//     Mat src_points_tran(5, 3, CV_32FC1);
//     // expand the src_points from 5,1<points> to 5,3<float>
//     for(int i = 0; i < 5; i ++){
//         src_points.at<Point2f>(i) = Point2f(landmarks[i][0], landmarks[i][1]);
//         src_points_tran.at<float>(i,0) = landmarks[i][0];
//         src_points_tran.at<float>(i,1) = landmarks[i][1];
//         src_points_tran.at<float>(i,2) = 1.0;

//     }

//     // the points target for align face
//     Mat dst_points(1, 5, CV_32FC2);
//     // loop through all possible points
//     for (int src_index = 0; src_index< 5; src_index++){
//         // Mat (5,2) for calculating the error of align face.
//         Mat dst_mat(5,2,CV_32FC1);
//         for(int i = 0; i < 5; i ++){
//             dst_points.at<Point2f>(i) = Point2f(SRC_MAP[src_index][i][0], SRC_MAP[src_index][i][1]);
//             dst_mat.at<float>(i,0) = SRC_MAP[src_index][i][0];
//             dst_mat.at<float>(i,1) = SRC_MAP[src_index][i][1];
//         }

//         // Calculate the affine matrix
//         vector<uchar> inliers;
//         Mat aff_est = estimateAffinePartial2D(src_points, dst_points, inliers);

//         // Calculate the error and choose the align with lowest error.
//         Mat result_mat = aff_est.clone();
//         result_mat.convertTo(result_mat,CV_32FC1);     // numpy.astype()
//         result_mat = result_mat * src_points_tran.t();          // numpy.dot()
//         result_mat = result_mat.t();                            // numpy.tranpose()
//         result_mat = mat2DSubtract(result_mat,dst_mat); // numpy.subtract()
//         result_mat = mat2DSquare(result_mat);               // numpy.square()
//         result_mat = matSumAxis1(result_mat);               // numpy.sum(mat, axis=1)
//         result_mat = matSqrt1D(result_mat);                 // numpy.sqrt()
//         float error = matReduceSum1D(result_mat);           // numpy.sum()

//         if (error < min_error){
//             min_error = error;
//             min_M = aff_est.clone();
//         }
// //        __android_log_print(ANDROID_LOG_INFO, "NgB_JNI", "error %f %f" , error, min_error);
//     }
//     // perform align and crop the face based on affine transformation matrix, the output size is 112x112
//     warpAffine(src,dst,min_M,Size(out_width,out_height), INTER_LINEAR,BORDER_CONSTANT, 0.0);
// }

// float calCosimilarity(float* com, float* ref){

//     // caculate cosine similarity
//     float dot = 0.0f, denom_a = 0.0f, denom_b = 0.0f, output = 0.0f;
//     for(unsigned int i = 0u; i < EMBEDDING_SIZE; ++i) {
//         dot += com[i] * ref[i] ;
//         denom_a += com[i] * com[i] ;
//         denom_b += ref[i] * ref[i] ;
//     }
//     output =  dot / (sqrt(denom_a) * sqrt(denom_b));
//     // convert from -1 1 to 0,1
//     output = (output + 1.0f)/2.0f;

//     return output;
// }

#include "featurenet.h"
#include <iostream>
#include <model.h>
#include "pipeline.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

FeatureNet::FeatureNet(Model model)
    : Pipeline(model) {
    // Additional initialization if needed
}

Ort::Value FeatureNet::preprocess(Ort::Value input) {
    // Implement preprocessing logic here
    std::cout << "FeatureNet Preprocessing..." << std::endl;
    cv::Mat image = createMatFromOrtValue(input);
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(256, 256));
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255);
    resized_image = (resized_image - 0.5) * 2.0;
    Ort::Value preprocessed_image = createOrtValueFromMat(resized_image);
    return preprocessed_image;
}

Ort::Value FeatureNet::postprocess(Ort::Value input) {
    // Implement postprocessing logic here
    std::cout << "FeatureNet Postprocessing..." << std::endl;
    // Example: return the input as is
    return input;
}

Ort::Value FeatureNet::inference(Ort::Value input) {
    // Run the model with the preprocessed input
    Ort::Value preprocessed_input = preprocess(input);
    Ort::Value output = model->run(preprocessed_input);
    // Postprocess the output
    Ort::Value postprocessed_output = postprocess(output);
    return postprocessed_output;
}

Ort::Value FeatureNet::createOrtValueFromMat(const cv::Mat& mat) {
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

cv::Mat FeatureNet::createMatFromOrtValue(const Ort::Value& ort_value) {
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