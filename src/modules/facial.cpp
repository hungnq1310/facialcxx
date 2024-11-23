#include "facial.h"
#include <opencv2/opencv.hpp>
// #include <android/log.h>

FacialTFlite::FacialTFlite() {
    initialized_ = false;
}

FacialTFlite::~FacialTFlite() {
    blazeFace -> ~BlazeFace();
    featureNet -> ~TFLiteFeatureNet();
    metricNet -> ~TFLiteMetricNet();
}

int FacialTFlite::detectFace(Mat src, FaceResult *res) {
    if (!initialized_) {
        return -1;
    }
    if (src.empty()) {
        return -1;
    }
    int num_detected = blazeFace -> detect(src, res);
    return num_detected;
}

void FacialTFlite::predictEmb(const Mat & img_src, float landmarks[], float output[]) {
    // Crop the face base on the landmarks position
    float lmk_src[5][2];
    for (int j = 0, row = 0; j < 10; j+=2, row+=1){
        lmk_src[row][0] = landmarks[j];
        lmk_src[row][1] = landmarks[j + 1];
    }
    Mat img_crop_lmk_src;
    cropFace(img_src,img_crop_lmk_src, lmk_src, TFLiteFeatureNet::INPUT_SIZE, TFLiteFeatureNet::INPUT_SIZE);
    cvtColor(img_crop_lmk_src, img_crop_lmk_src, COLOR_BGR2RGB);
    PredictResultFeatureNet emb;
    // predict
    img_crop_lmk_src.convertTo(img_crop_lmk_src, CV_32FC3);
    featureNet-> predict(img_crop_lmk_src, emb);
    // get the embedding
    for (int i = 0; i < EMBEDDING_SIZE; i++){
        output[i] = emb.embedding[i];
    }
}

void FacialTFlite::predictID(float face_embsC[][EMBEDDING_SIZE], float database_embsC[][EMBEDDING_SIZE],float output[], int num_faces, int num_database){
    float listPredictScore[num_database];

    for (int face_index = 0; face_index < num_faces; face_index++){
        // find the lowest score
        // Metric net will output the same pairs with the value close to 0 and the different pairs with the value close to 40
        // Best score means the value is closer to 0 than 40
        float best_score = 100.0f;
        int best_index = 0;

        // compare with all database
        for (int batch_index = 0; batch_index < num_database; batch_index+=BATCH_SIZE){
            // loop through 0, 1 * BATCH_SIZE, 2 * BATCH_SIZE, ... indexs
            PredictResultMetricNet batchMetricNetOutput[BATCH_SIZE];
            float database_embsC_flatten[BATCH_SIZE * EMBEDDING_SIZE];

            // Index for assign the value to database_embsC_flatten
            int database_index = 0;
            for (int sample_index = 0; sample_index < BATCH_SIZE; sample_index++){
                // Sample index run from 0 to BATCH_SIZE
                // Batch index run from 0 to num_database with step is BATCH_SIZE
                // So the sample_index + batch_index is the start index of the database_embsC in batch n.
                int current_sample_index = batch_index + sample_index;
                if (current_sample_index < num_database){
                    for (int i = 0; i < EMBEDDING_SIZE; i++){
                        database_embsC_flatten[database_index * EMBEDDING_SIZE + i] = database_embsC[current_sample_index][i];
                    }
                    database_index += 1;
                }
                else{
                    for (int i = 0; i < EMBEDDING_SIZE; i++){
                        database_embsC_flatten[database_index * EMBEDDING_SIZE + i] = 1.0f;
                    }
                    database_index += 1;
                }
            }
            metricNet ->predict(face_embsC[face_index], database_embsC_flatten, batchMetricNetOutput);
            for (int i = 0 ; i < BATCH_SIZE; i++){
                if ((batch_index + i) < num_database){
                    listPredictScore[batch_index + i] = batchMetricNetOutput[i].results[0];
                }
            }
        }
        for (int j =0; j < num_database; j ++){
            if (listPredictScore[j] < best_score){
                best_score = listPredictScore[j];
                best_index = j;
            }
            
        }
        // ignore the score lower than threshold
        if (best_score > THRESHOLD_RECOGNITION) best_index = -1;

        output[face_index*2] = (float)best_index;
        output[face_index*2 + 1] = best_score;
        // __android_log_print(ANDROID_LOG_ERROR, "TRACKERS", "%f, %f", (float)best_index, best_score);
    }
}

int FacialTFlite::init(const char* blaze_face_path, long blaze_face_size,
                     const char* feature_net_path, long feature_net_size,
                     const char* metric_net_path, long metric_net_size) {
    // BlazeFace model
    blazeFace ->init(blaze_face_path,blaze_face_size);
    // Feature net model
    featureNet -> initModel(feature_net_path, feature_net_size);
    // Metric net model
    metricNet -> initModel(metric_net_path, metric_net_size);
    initialized_ = true;
    return 0;
}

int FacialTFlite::init(const char* blaze_face_path,
                     const char* feature_net_path,
                     const char* metric_net_path) {
    // BlazeFace model
    blazeFace ->init(blaze_face_path);
    // Feature net model
    featureNet -> initModel(feature_net_path);
    // Metric net model
    metricNet -> initModel(metric_net_path);
    initialized_ = true;
    return 0;
}

int FacialTFlite::init(const char* blaze_face_path, int num_threads_blaze,
                     const char* feature_net_path, int num_threads_feature,
                     const char* metric_net_path, int num_threads_metric) {
    // BlazeFace model
    blazeFace ->init(blaze_face_path, num_threads_blaze);
    // Feature net model
    featureNet -> initModel(feature_net_path, num_threads_feature);
    // Metric net model
    metricNet -> initModel(metric_net_path, num_threads_metric);
    initialized_ = true;
    return 0;
}
