#ifndef _FACE_TFLITE_H_
#define _FACE_TFLITE_H_

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/types_c.h>

#include "blazeface.h"
#include "featurenet.h"
#include "metricnet.h"

using namespace cv;
using namespace std;

const float THRESHOLD_RECOGNITION = 10.0f;

class FacialTFlite {
public:
    FacialTFlite();
    ~FacialTFlite();
    int init(const char* blaze_face_path, long blaze_face_size,
             const char* feature_net_path, long feature_net_size,
             const char* metric_net_path, long metric_net_size);
    int init(const char* blaze_face_path,
             const char* feature_net_path,
             const char* metric_net_path);
    int init(const char* blaze_face_path, int num_threads_blaze,
             const char* feature_net_path, int num_threads_feature,
             const char* metric_net_path, int num_threads_metric);
    int detectFace(Mat src, FaceResult *res);
    void predictID(float face_embsC[][EMBEDDING_SIZE], float database_embsC[][EMBEDDING_SIZE],float output[], int num_faces, int num_database);
    void predictEmb(const Mat& img_src, float landmarks[], float output[]);

private:
    bool initialized_;
    // Initialize the interpreter for headface, landmark and embedder
    BlazeFace* blazeFace = new BlazeFace();
    TFLiteFeatureNet* featureNet = new TFLiteFeatureNet();
    TFLiteMetricNet* metricNet = new TFLiteMetricNet();
};


#endif  // !_FACE_TFLITE_H_
