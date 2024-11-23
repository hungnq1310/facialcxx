#pragma once
#include <opencv2/core/cuda.hpp>

namespace improc {
    int make_divisible(int x, int divisor);
    int check_img_size(int imgSize, int stride);
    cv::Size getNewResWithoutDistortion(
        const cv::Size& imageSize, 
        const cv::Size& targetSize
    );
    cv::Size getNewResWithDistortion(
        const cv::Size& imageSize, 
        const cv::Size& targetSize
    );
    cv::Size getNewRes(
        const cv::Size& imageSize, 
        const cv::Size& targetSize, 
        bool keepAspectRatio
    );
    void clipCoords(
        cv::cuda::GpuMat& boxes, 
        const cv::Size& imgShape, 
        int step
    );
    cv::cuda::GpuMat interpolation(
        const cv::cuda::GpuMat& image, 
        const cv::Size& size, 
        const std::string& mode
    );
    cv::cuda::GpuMat padImageTopLeft(
        const cv::cuda::GpuMat& image, 
        const cv::Size& targetSize,
        const cv::Scalar& padValue
    );
    cv::cuda::GpuMat resizeWithPad(
        const cv::cuda::GpuMat& image, 
        const cv::Size& targetSize, 
        const std::string& mode, 
        int maxUpscalingSize
    );
    cv::cuda::GpuMat augmentHSV(
        const cv::cuda::GpuMat& inputImage, 
        float hgain, 
        float sgain, 
        float vagin
    );
    cv::cuda::GpuMat histEqualize(
        const cv::cuda::GpuMat& inputImage, 
        bool clahe, 
        bool bgr
    );
    cv::cuda::GpuMat flipVertical(
        const cv::cuda::GpuMat& inputImage
    );
    cv::cuda::GpuMat flipHorizontal(
        const cv::cuda::GpuMat& inputImage
    );
    cv::cuda::GpuMat flipFull(
        const cv::cuda::GpuMat& inputImage
    );
    cv::cuda::GpuMat scaleCoords(
        cv::cuda::GpuMat& coords, 
        const cv::Size& img1Shape, 
        const cv::Size& img0Shape, 
        std::pair<double, cv::Point2d> ratioPad, 
        int step
    );
    cv::cuda::GpuMat rotateImage(
        const cv::cuda::GpuMat& inputImage, 
        double angle, 
        const cv::Size& size, 
        std::string mode
    );
    cv::cuda::GpuMat xyxy2xywh(const cv::cuda::GpuMat& x);
    cv::cuda::GpuMat xywh2xyxy(const cv::cuda::GpuMat& x);
}
