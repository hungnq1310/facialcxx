#include <jni.h>
#include <string>
#include <math.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <ortcxx/model.h>
#include <opencv2/opencv.hpp>

// #include <blazeface.h>
#include <spoofing.h>
#include <yolov7.h>

using namespace cv;
using namespace std;
using namespace Ort;
using namespace ortcxx;


////////////////////////
#define TENSOR(METHOD_NAME) \
  Java_ai_spoofing_TensorUtils_##METHOD_NAME
////////////////////////
#ifdef __cplusplus
extern "C" {
#endif


// BlazeFace* blazefacePipeline;
YoloV7* yoloPipeline;
Spoofing* SpoofingPipeline;



// Convert the bitmap to OpenCV Mat
// Reference https://github.com/opencv/opencv/blob/17234f82d025e3bbfbf611089637e5aa2038e7b8/modules/java/generator/src/cpp/utils.cpp
void bitmapToMat(JNIEnv *env, jobject bitmap, Mat &dst, jboolean needUnPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = 0;

    try {
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                  info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        dst.create(info.height, info.width, CV_8UC4);
        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (needUnPremultiplyAlpha) cvtColor(tmp, dst, COLOR_mRGBA2RGBA);
            else tmp.copyTo(dst);
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            cvtColor(tmp, dst, COLOR_BGR5652RGBA);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch (const cv::Exception &e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nBitmapToMat}");
        return;
    }
}

// Convert a mat to Bitmap
// Reference https://github.com/opencv/opencv/blob/17234f82d025e3bbfbf611089637e5aa2038e7b8/modules/java/generator/src/cpp/utils.cpp
void matToBitmap(JNIEnv *env, Mat src, jobject bitmap, jboolean needPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = 0;

    try {
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                  info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        CV_Assert(src.dims == 2);
        CV_Assert(info.height == (uint32_t) src.rows);
        CV_Assert(info.width == (uint32_t) src.cols);
        CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);
        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (src.type() == CV_8UC1) {
                cvtColor(src, tmp, COLOR_GRAY2RGBA);
            } else if (src.type() == CV_8UC3) {
                cvtColor(src, tmp, COLOR_RGB2RGBA);
            } else if (src.type() == CV_8UC4) {
                if (needPremultiplyAlpha) cvtColor(src, tmp, COLOR_RGBA2mRGBA);
                else src.copyTo(tmp);
            }
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            Mat tmp(info.height, info.width, CV_8UC2, pixels);
            if (src.type() == CV_8UC1) {
                cvtColor(src, tmp, COLOR_GRAY2BGR565);
            } else if (src.type() == CV_8UC3) {
                cvtColor(src, tmp, COLOR_RGB2BGR565);
            } else if (src.type() == CV_8UC4) {
                cvtColor(src, tmp, COLOR_RGBA2BGR565);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return;
    } catch (const cv::Exception &e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
        return;
    }
}


extern "C"
JNIEXPORT void JNICALL
Java_ai_spoofing_TensorUtils_initModel(
    JNIEnv *env, jclass clazz, jobject assetManager,
    // jstring blazeFacePath,
    jstring yoloPath,
    jstring extractorPath,
    jstring embedderPath
) {

    // YoloPath
    const char *yolo = env->GetStringUTFChars(yoloPath, 0);
    // convert to string
    std::string yoloString(yolo);

    // Extractor Spoofing
    const char *extractor = env->GetStringUTFChars(extractorPath, 0);
    // convert to string
    std::string extractorString(extractor);

    // Embedder Spoofing
    const char *embed = env->GetStringUTFChars(embedderPath, 0);
    // convert to string
    std::string embedString(embed);

    // Load the model
    auto providers = Ort::GetAvailableProviders();
    for (std::string p : providers) {
        std::cout << "- " << p << std::endl;
    };
    map<string, any> c;
    c["parallel"] = false;
    c["inter_ops_threads"] = 1;
    c["intra_ops_threads"] = 1;
    c["graph_optimization_level"] = 1;

    // Load the model with share env and allocator
    std::shared_ptr<Ort::Env> ortEnv = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");

    std::shared_ptr<Model> yoloModel = Model::create(yoloString, ortEnv, c, providers, false);
    std::shared_ptr<Model> extractorModel = Model::create(extractorString, ortEnv, c, providers, false);
    std::shared_ptr<Model> embedderModel = Model::create(embedString, ortEnv, c, providers, false);

    // Assign pointers
    // blazefacePipeline = new BlazeFace(blazefaceModel);
    yoloPipeline = new YoloV7(yoloModel);
    SpoofingPipeline = new Spoofing(extractorModel, embedderModel);

    //DONE
}

extern "C"
JNIEXPORT jobject JNICALL
Java_ai_spoofing_TensorUtils_checkspoof(JNIEnv *env, jclass clazz, jobject bitmap) {

    // From mobile to OpenCV Mat
    Mat input_mat;
    bitmapToMat(env, bitmap, input_mat, false);
    cvtColor(input_mat, input_mat, COLOR_RGBA2BGR);

    cv::Mat original = input_mat.clone();
    cv::Mat processed_img = yoloPipeline->preprocess(input_mat);
    std::vector<PredictResultHeadFace> res_yolo;
    yoloPipeline->detect(processed_img, &res_yolo);

    if (res_yolo.size() > 0) {

        // Create a list to store DetectionResult objects
        jclass detectionResultClass = env->FindClass("ai/spoofing/DetectionResult");
        jmethodID detectionResultConstructor = env->GetMethodID(detectionResultClass, "<init>",
                                                                "(Landroid/graphics/Bitmap;[F)V");
        jobjectArray results = env->NewObjectArray(res_yolo.size(), detectionResultClass, nullptr);

        for (int i = 0; i < res_yolo.size(); i++) {

            PredictResultHeadFace res = res_yolo[i];
//            float score = (res.score_face == 0.0) ? res.score_head : res.score_face;

            cv::Rect bbox = cv::Rect(
                static_cast<int>(std::round(res.xmin)),
                static_cast<int>(std::round(res.ymin)),
                static_cast<int>(std::round(res.xmax - res.xmin)),
                static_cast<int>(std::round(res.ymax - res.ymin))
            );

            cv::Mat cropped_face;
            cropFace(original, cropped_face, bbox, 256, 256);
            cvtColor(cropped_face, cropped_face, COLOR_BGR2RGB);

            // Preprocess the cropped face
            cv::Mat preprocessed_face;
            SpoofingPipeline->preprocess(cropped_face, preprocessed_face);

            // Inference
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<float> probs = SpoofingPipeline->inference(preprocessed_face);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            // Log inference time to Android logcat
            __android_log_print(ANDROID_LOG_DEBUG, "SpoofingInference", "Inference time: %lld ms", duration.count());

            // create jobject
            // Create a new Bitmap with the correct size and format
            jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
            jmethodID createBitmapMethod = env->GetStaticMethodID(bitmapClass, "createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
            jobject config = env->GetStaticObjectField(env->FindClass("android/graphics/Bitmap$Config"), env->GetStaticFieldID(env->FindClass("android/graphics/Bitmap$Config"), "ARGB_8888", "Landroid/graphics/Bitmap$Config;"));

            //convert to RGB before create bitmap
            cvtColor(cropped_face, cropped_face, COLOR_BGR2RGB);
            jobject faceBitmap = env->CallStaticObjectMethod(bitmapClass, createBitmapMethod, cropped_face.cols, cropped_face.rows, config);
            matToBitmap(env, cropped_face, faceBitmap, false); // Function to create Bitmap from Mat

            //create crops
            jfloatArray probsArray = env->NewFloatArray(probs.size());
            env->SetFloatArrayRegion(probsArray, 0, probs.size(), probs.data());

            jobject detectionResult = env->NewObject(
                    detectionResultClass,
                    detectionResultConstructor,
                    faceBitmap,
                    probsArray
            );
            env->SetObjectArrayElement(results, i, detectionResult);
        }
        return results;
    } else {
        // Return null if no faces are detected
        return nullptr;
    }
}


///////////////////////////
#ifdef __cplusplus
}
#endif
////////////////////////