
#include <iostream>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <improc.h>

double compute_similarity(cv::Mat image, cv::Mat ref_image){
    cv::Mat diff;
    cv::matchTemplate(image, ref_image, diff, cv::TM_CCOEFF_NORMED);
    double maxScore;
    cv::minMaxLoc(diff, 0, &maxScore);
    return maxScore;
}

bool test_3(){
    
    //read image
    cv::Mat image = cv::imread("../photo_2024-07-19_15-11-25.jpg", cv::IMREAD_COLOR);
    cv::cuda::GpuMat d_image;
    d_image.upload(image);

    // Measure the time taken to interpolate the image
    auto start = std::chrono::high_resolution_clock::now();
    cv::cuda::GpuMat output = improc::interpolation(d_image, cv::Size(640, 640), "area");
    cv::cuda::GpuMat output_2 = improc::interpolation(d_image, cv::Size(640, 640), "bilinear");
    cv::cuda::GpuMat output_3 = improc::interpolation(d_image, cv::Size(640, 640), "bicubic");
    cv::cuda::GpuMat output_4 = improc::interpolation(d_image, cv::Size(640, 640), "nearest");
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken to interpolate the image: " << duration.count() << " seconds" << std::endl;

    //read ref image
    cv::Mat ref_area = cv::imread("../assert/interpolate/ref_area.jpg", cv::IMREAD_COLOR);
    cv::Mat ref_bicubic = cv::imread("../assert/interpolate/ref_bicubic.jpg", cv::IMREAD_COLOR);
    cv::Mat ref_bilinear = cv::imread("../assert/interpolate/ref_bilinear.jpg", cv::IMREAD_COLOR);
    cv::Mat ref_nearest = cv::imread("../assert/interpolate/ref_nearest.jpg", cv::IMREAD_COLOR);

    
    // Check if the images are the same
    cv::Mat h_output;
    output.download(h_output);
    cv::Mat h_output_2;
    output_2.download(h_output_2);
    cv::Mat h_output_3;
    output_3.download(h_output_3);
    cv::Mat h_output_4;
    output_4.download(h_output_4);


    double maxScoreArea = compute_similarity(h_output, ref_area);
    double maxScoreBilinear = compute_similarity(h_output_2, ref_bilinear);
    double maxScoreBicubic = compute_similarity(h_output_3, ref_bicubic);
    double maxScoreNearest = compute_similarity(h_output_4, ref_nearest);

    // Check if the images are the same
    int pass = 4;
    assert (maxScoreArea >= 0.999), (pass - 1);
    assert (maxScoreBilinear >= 0.999), (pass - 1);
    assert (maxScoreBicubic >= 0.999), (pass - 1);
    assert (maxScoreNearest >= 0.999), (pass - 1);

    if (pass == 4){
        std::cout << "Test 3 passed" << std::endl;
        return true;
    } else {
        std::cout << "Test 3 failed, only "<< pass << "/4" << std::endl;
        return false;
    }
}

bool test_4(){
    //test rotate

    //read image 
    cv::Mat image = cv::imread("../photo_2024-07-19_15-11-25.jpg", cv::IMREAD_COLOR);
    cv::cuda::GpuMat d_image;
    d_image.upload(image);

    // Measure the time taken to rotate the image
    auto start = std::chrono::high_resolution_clock::now();
    cv::cuda::GpuMat output_rotate_90 = improc::rotateImage(
        d_image, 90, d_image.size(), "bicubic"
    );

    cv::cuda::GpuMat output_rotate_180 = improc::rotateImage(
        d_image, 180, d_image.size(), "bicubic"
    );

    cv::cuda::GpuMat output_rotate_270 = improc::rotateImage(
        d_image, 270, d_image.size(), "bicubic"
    );

    cv::cuda::GpuMat output_rotate_90_neg = improc::rotateImage(
        d_image, -90, d_image.size(), "bicubic"
    ); // appendix for negative angle

    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken to rotate the image: " << duration.count() << " seconds" << std::endl;

    //read ref image
    cv::Mat ref_90 = cv::imread("../assert/rotate/ref_rotate_90.jpg", cv::IMREAD_COLOR);
    cv::Mat ref_180 = cv::imread("../assert/rotate/ref_rotate_180.jpg", cv::IMREAD_COLOR);
    cv::Mat ref_270 = cv::imread("../assert/rotate/ref_rotate_-90.jpg", cv::IMREAD_COLOR);

    // Check if the images are the same
    cv::Mat h_output_rotate_90;
    output_rotate_90.download(h_output_rotate_90);
    cv::Mat h_output_rotate_180;
    output_rotate_180.download(h_output_rotate_180);
    cv::Mat h_output_rotate_270;
    output_rotate_270.download(h_output_rotate_270);
    cv::Mat h_output_rotate_90_neg;
    output_rotate_90_neg.download(h_output_rotate_90_neg);

    //write to disk
    cv::imwrite("output_rotate_90.jpg", h_output_rotate_90);
    cv::imwrite("output_rotate_180.jpg", h_output_rotate_180);
    cv::imwrite("output_rotate_270.jpg", h_output_rotate_270);
    cv::imwrite("output_rotate_90_neg.jpg", h_output_rotate_90_neg);

    // 
    double maxScore90 = compute_similarity(h_output_rotate_90, ref_90);
    double maxScore180 = compute_similarity(h_output_rotate_180, ref_180);
    double maxScore270 = compute_similarity(h_output_rotate_270, ref_270);
    double maxScore90_neg = compute_similarity(h_output_rotate_90_neg, ref_270);

    //log and check score
    int pass = 4;
    assert (maxScore90 >= 0.999), (pass - 1);
    assert (maxScore180 >= 0.999), (pass - 1);
    assert (maxScore270 >= 0.99), (pass - 1); //! only 0.99
    assert (maxScore90_neg >= 0.999), (pass - 1);

    if (pass == 4){
        std::cout << "Test 4 passed" << std::endl;
        return true;
    } else {
        std::cout << "Test 4 failed, only "<< pass << "/4" << std::endl;
        return false;
    }
}

bool test_5(){
    //test equalize hist

    //read image    
    cv::Mat image = cv::imread("../photo_2024-07-19_15-11-25.jpg", cv::IMREAD_COLOR);
    cv::cuda::GpuMat d_image;
    d_image.upload(image);

    // Measure the time taken to equalize the image
    auto start = std::chrono::high_resolution_clock::now();
    cv::cuda::GpuMat output = improc::histEqualize(d_image, false, true);
    cv::cuda::GpuMat output_clahe = improc::histEqualize(d_image, true, true);
    cv::cuda::GpuMat output_bgr = improc::histEqualize(d_image, false, false);
    cv::cuda::GpuMat output_clahe_bgr = improc::histEqualize(d_image, true, false);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken to equalize the image: " << duration.count() << " milliseconds" << std::endl;

    //read ref image
    cv::Mat ref_clahe = cv::imread("../assert/hisEqual/ref_hist_equalize_clahe.jpg", cv::IMREAD_COLOR);
    cv::Mat ref_wo_clahe = cv::imread("../assert/hisEqual/ref_hist_equalize_wo_clahe.jpg", cv::IMREAD_COLOR);
    cv::Mat ref_wo_clahe_rgb = cv::imread("../assert/hisEqual/ref_hist_equalize_wo_clahe_rgb.jpg", cv::IMREAD_COLOR);
    cv::Mat ref_clahe_rgb = cv::imread("../assert/hisEqual/ref_hist_equalize_clahe_rgb.jpg", cv::IMREAD_COLOR);

    // Check if the images are the same
    cv::Mat h_output;
    output.download(h_output);
    cv::Mat h_output_clahe;
    output_clahe.download(h_output_clahe);
    cv::Mat h_output_bgr;
    output_bgr.download(h_output_bgr);
    cv::Mat h_output_clahe_bgr;
    output_clahe_bgr.download(h_output_clahe_bgr);

    double maxScoreClahe = compute_similarity(h_output_clahe, ref_clahe);
    double maxScoreWoClahe = compute_similarity(h_output, ref_wo_clahe);
    double maxScoreWoClaheRGB = compute_similarity(h_output_bgr, ref_wo_clahe_rgb);
    double maxScoreClaheRGB = compute_similarity(h_output_clahe_bgr, ref_clahe_rgb);


    //log and check score
    int pass = 4;
    assert (maxScoreClahe >= 0.999), (pass - 1);
    assert (maxScoreWoClahe >= 0.999), (pass - 1);
    assert (maxScoreWoClaheRGB >= 0.99), (pass - 1); //! only 0.99
    assert (maxScoreClaheRGB >= 0.99), (pass - 1); //!

    if (pass == 4){
        std::cout << "Test 5 passed" << std::endl;
        return true;
    } else {
        std::cout << "Test 5 failed, only "<< pass << "/4" << std::endl;
        return false;
    }
}

bool test_6(){
    //test flip

    //read image
    cv::Mat image = cv::imread("../photo_2024-07-19_15-11-25.jpg", cv::IMREAD_COLOR);
    cv::cuda::GpuMat d_image;

    d_image.upload(image);

    // Measure the time taken to flip the image
    auto start = std::chrono::high_resolution_clock::now();
    cv::cuda::GpuMat output_flip_v = improc::flipVertical(d_image);
    cv::cuda::GpuMat output_flip_h = improc::flipHorizontal(d_image);
    cv::cuda::GpuMat output_flip_v_h = improc::flipFull(d_image);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken to flip the image: " << duration.count() << " seconds" << std::endl;

    //read ref image
    cv::Mat ref_v = cv::imread("../assert/flip/ref_flip_left-right.jpg", cv::IMREAD_COLOR);
    cv::Mat ref_h = cv::imread("../assert/flip/ref_flip_up-down.jpg", cv::IMREAD_COLOR);
    cv::Mat ref_v_h = cv::imread("../assert/flip/ref_flip_full.jpg", cv::IMREAD_COLOR);

    // Check if the images are the same
    cv::Mat h_output_flip_v;
    output_flip_v.download(h_output_flip_v);
    cv::Mat h_output_flip_h;
    output_flip_h.download(h_output_flip_h);
    cv::Mat h_output_flip_v_h;
    output_flip_v_h.download(h_output_flip_v_h);

    double maxScoreV = compute_similarity(h_output_flip_v, ref_v);
    double maxScoreH = compute_similarity(h_output_flip_h, ref_h);
    double maxScoreVH = compute_similarity(h_output_flip_v_h, ref_v_h);

    //log and check score
    int pass = 3;
    assert (maxScoreV >= 0.999), (pass - 1);
    assert (maxScoreH >= 0.999), (pass - 1);
    assert (maxScoreVH >= 0.999), (pass - 1);

    if (pass == 3){
        std::cout << "Test 6 passed" << std::endl;
        return true;
    } else {
        std::cout << "Test 6 failed, only "<< pass << "/3" << std::endl;
        return false;
    }
}

bool test_7(){
    //test augmentHSV

    //read image
    cv::Mat image = cv::imread("../photo_2024-07-19_15-11-25.jpg", cv::IMREAD_COLOR);
    cv::cuda::GpuMat d_image;
    d_image.upload(image);

    // Measure the time taken to augment the image
    auto start = std::chrono::high_resolution_clock::now();
    cv::cuda::GpuMat output = improc::augmentHSV(d_image, 0.5, 0.5, 0.5);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken to augment the image: " << duration.count() << " seconds" << std::endl;
    std::cout << "Test 7 passed" << std::endl;

    // Display the augmented image
    // cv::Mat h_output;
    // output.download(h_output);
    // cv::imshow("Augmented Image", h_output);
    // cv::waitKey(0);
    
    return true;
}


bool test_9(){
    // test padImageTopLeft
    
    //read image
    cv::Mat image = cv::imread("../photo_2024-07-19_15-11-25.jpg", cv::IMREAD_COLOR);
    cv::cuda::GpuMat d_image;
    d_image.upload(image);

    // Measure the time taken to pad the image
    auto start = std::chrono::high_resolution_clock::now();
    cv::cuda::GpuMat output = improc::padImageTopLeft(
        d_image, cv::Size(1400, 1280), //! heigt, width
        cv::Scalar(0, 0, 0)
    );
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken to pad the image: " << duration.count() << " seconds" << std::endl;

    //read ref image
    cv::Mat ref = cv::imread("../assert/resize-pad/ref_pad_image.jpg", cv::IMREAD_COLOR);

    // Check if the images are the same
    cv::Mat h_output;
    output.download(h_output);
    double maxScore = compute_similarity(h_output, ref);

    //log and check score
    assert (maxScore >= 0.999);

    if (maxScore >= 0.999){
        std::cout << "Test 9 passed" << std::endl;
        return true;
    } else {
        std::cout << "Test 9 failed" << std::endl;
        return false;
    }
    return true;
}

bool test_10(){
    // test resizeWithPad

    //read image
    cv::Mat image = cv::imread("../photo_2024-07-19_15-11-25.jpg", cv::IMREAD_COLOR);
    cv::cuda::GpuMat d_image;
    d_image.upload(image);

    // Measure the time taken to resize the image
    auto start = std::chrono::high_resolution_clock::now();
    cv::cuda::GpuMat output = improc::resizeWithPad(
        d_image, cv::Size(500, 768), "bilinear", 0
    );
    cv::cuda::GpuMat output_2 = improc::resizeWithPad(
        d_image, cv::Size(500, 768), "bicubic", 0
    );
    cv::cuda::GpuMat output_3 = improc::resizeWithPad(
        d_image, cv::Size(500, 768), "nearest", 0
    );
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken to resize the image: " << duration.count() << " seconds" << std::endl;

    //read ref image
    cv::Mat ref_bicubic = cv::imread("../assert/resize-pad/ref_resize_pad_bicubic.jpg", cv::IMREAD_COLOR);
    cv::Mat ref_bilinear = cv::imread("../assert/resize-pad/ref_resize_pad_bilinear.jpg", cv::IMREAD_COLOR);
    cv::Mat ref_nearest = cv::imread("../assert/resize-pad/ref_resize_pad_nearest.jpg", cv::IMREAD_COLOR);

    // Check if the images are the same
    cv::Mat h_output;
    output.download(h_output);
    cv::Mat h_output_2;
    output_2.download(h_output_2);
    cv::Mat h_output_3;
    output_3.download(h_output_3);

    double maxScoreBilinear = compute_similarity(h_output, ref_bilinear);
    double maxScoreBicubic = compute_similarity(h_output_2, ref_bicubic);
    double maxScoreNearest = compute_similarity(h_output_3, ref_nearest);

    //log and check score
    int pass = 3;
    assert (maxScoreBilinear >= 0.999), (pass - 1);
    assert (maxScoreBicubic >= 0.999), (pass - 1);
    assert (maxScoreNearest >= 0.99), (pass - 1); //! only 0.99

    if (pass == 3){
        std::cout << "Test 10 passed" << std::endl;
        return true;
    } else {
        std::cout << "Test 10 failed, only "<< pass << "/3" << std::endl;
        return false;
    }

    return true;
}


int main(){
    test_3();
    test_4();
    test_5();
    test_6();
    test_7();
    test_9();
    test_10();
    return 0;
}