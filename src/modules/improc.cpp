#include <iostream>
#include <stdio.h> 
#include <random>

#include <opencv2/cudaarithm.hpp>
#include "opencv2/cudawarping.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>

#include <improc.h>

namespace improc {

// ******************************
// *       SUPPORTED FUNCTIONS
// ******************************

/**
 * @brief Returns a value that is evenly divisible by the given divisor.
 *
 * This function takes an integer `x` and a `divisor`, and returns the smallest integer
 * that is greater than or equal to `x` and is evenly divisible by the `divisor`.
 *
 * @param x The integer to be made divisible.
 * @param divisor The divisor by which the result should be evenly divisible.
 *
 * @return An integer that is evenly divisible by the divisor.
 */
int make_divisible(int x, int divisor) {
    // Returns x evenly divisible by divisor
    return std::ceil(static_cast<float>(x) / divisor) * divisor;
}

/**
 * @brief Verifies that the image size is a multiple of the given stride.
 *
 * This function checks if the provided `img_size` is a multiple of the stride `s`.
 * If it is not, it adjusts the `img_size` to the nearest multiple of `s` and prints a warning message.
 *
 * @param img_size The original image size to be checked.
 * @param s The stride value to which the image size should be a multiple. Default is 32.
 *
 * @return The adjusted image size that is a multiple of the stride `s`.
 */
int check_img_size(int imgSize, int stride = 32) {
    // Verify img_size is a multiple of stride
    int newSize = make_divisible(imgSize, stride);  // ceil gs-multiple
    if (newSize != imgSize) {
        std::cout << "WARNING: Image size " << imgSize << " must be multiple of max stride " << stride << ", updating to " << newSize << std::endl;
    }
    return newSize;
}


/**
 * @brief Calculates the new resolution for an image to fit within a target size without distortion.
 *
 * This function computes the possible dimensions for an image to fit within a given target size
 * while maintaining the original aspect ratio, thus avoiding any distortion.
 *
 * @param imageSize The original size of the image. It should be a `cv::Size` object representing the width and height.
 * @param targetSize The target size within which the image should fit. It should be a `cv::Size` object representing the width and height.
 *
 * @return A `cv::Size` object representing the new dimensions of the image that fit within the target size without distortion.
 */
cv::Size getNewResWithoutDistortion(const cv::Size& imageSize, const cv::Size& targetSize) {
    double aspectRatio = static_cast<double>(imageSize.width) / imageSize.height;
    int newWidth = targetSize.width;
    int newHeight = static_cast<int>(newWidth / aspectRatio);

    if (newHeight > targetSize.height) {
        newHeight = targetSize.height;
        newWidth = static_cast<int>(newHeight * aspectRatio);
    }

    return cv::Size(newWidth, newHeight);
}


/**
 * @brief Clips the coordinates of bounding boxes to fit within the image dimensions.
 *
 * This function ensures that the coordinates of bounding boxes do not exceed the boundaries of the image.
 * It clips the x and y coordinates of the bounding boxes to be within the range [0, imgShape.width - 1] and [0, imgShape.height - 1], respectively.
 *
 * @param boxes A `cv::cuda::GpuMat` object containing the bounding box coordinates. The coordinates are expected to be in the format [x1, y1, x2, y2, ...].
 * @param imgShape A `cv::Size` object representing the dimensions of the image (width and height).
 * @param step The step size for iterating through the coordinates. Default is 2, assuming the format [x1, y1, x2, y2, ...].
 */
void clipCoords(cv::cuda::GpuMat& boxes, const cv::Size& imgShape, int step = 2) {
    // Create a GpuMat for the minimum and maximum values
    cv::cuda::GpuMat minVal(boxes.size(), boxes.type(), cv::Scalar(0));
    cv::cuda::GpuMat maxValX(boxes.size(), boxes.type(), cv::Scalar(imgShape.width - 1));
    cv::cuda::GpuMat maxValY(boxes.size(), boxes.type(), cv::Scalar(imgShape.height - 1));

    // Clip x coordinates
    for (int i = 0; i < boxes.cols; i += step) {
        cv::cuda::max(boxes.col(i), minVal.col(i), boxes.col(i));
        cv::cuda::min(boxes.col(i), maxValX.col(i), boxes.col(i));
    }

    // Clip y coordinates
    for (int i = 1; i < boxes.cols; i += step) {
        cv::cuda::max(boxes.col(i), minVal.col(i), boxes.col(i));
        cv::cuda::min(boxes.col(i), maxValX.col(i), boxes.col(i));
    }
}


// ******************************
// *       MAIN FUNCTION
// ******************************

/**
 * @brief Resizes an image using the specified interpolation mode.
 *
 * This function resizes the input image to the given size using the specified interpolation mode.
 * It supports several interpolation methods including nearest neighbor, bilinear, bicubic, and Lanczos.
 *
 * @param image The input image to be resized. It should be a `cv::cuda::GpuMat` object.
 * @param size The target size for the output image. It should be a `cv::Size` object.
 * @param mode The interpolation mode to be used. It should be one of the following strings:
 *             - "nearest": Nearest neighbor interpolation.
 *             - "bilinear": Bilinear interpolation.
 *             - "bicubic": Bicubic interpolation.
 *             - "lanczos": Lanczos interpolation.
 *
 * @return A `cv::cuda::GpuMat` object containing the resized image.
 *
 * @throws `std::invalid_argument` If an invalid interpolation mode is provided.
 */
cv::cuda::GpuMat interpolation(
    const cv::cuda::GpuMat& image, 
    const cv::Size& size, 
    const std::string& mode = "bilinear"
){
    int interpolationFlag;
    if (mode == "nearest"){
        interpolationFlag = cv::INTER_NEAREST;
    }
    else if (mode == "bilinear"){
        interpolationFlag = cv::INTER_LINEAR;
    }
    else if (mode == "bicubic"){
        interpolationFlag = cv::INTER_CUBIC;
    }
    else if (mode == "area"){
        interpolationFlag = cv::INTER_AREA;
    }
    else{
        throw std::invalid_argument("Invalid interpolation mode. Supported modes are 'nearest', 'bilinear', 'bicubic', and 'area'.");
    }
    
    cv::cuda::GpuMat output_image;
    cv::cuda::resize(image, output_image, size, 0, 0, interpolationFlag);
    return output_image;
}


/**
 * @brief Pads an image to the target size by adding borders to the right and bottom.
 *
 * This function pads the input image to the specified target size by adding borders to the right and bottom.
 * The padding is filled with a constant value (black by default).
 *
 * @param image The input image to be padded. It should be a `cv::cuda::GpuMat` object.
 * @param targetSize The target size for the padded image. It should be a `cv::Size` object representing the width and height.
 * @param padValue The value to be used for padding. Default is black (0, 0, 0).
 * 
 * @return A `cv::cuda::GpuMat` object containing the padded image.
 */
cv::cuda::GpuMat padImageTopLeft(
    const cv::cuda::GpuMat& image, 
    const cv::Size& targetSize,
    const cv::Scalar& padValue = cv::Scalar(0, 0, 0)
){
    cv::Size imageSize = image.size();
    int padX = targetSize.width - imageSize.width;
    int padY = targetSize.height - imageSize.height;

    cv::cuda::GpuMat paddedImage;
    cv::cuda::copyMakeBorder(image, paddedImage, 0, padY, 0, padX, cv::BORDER_CONSTANT, padValue);
    return paddedImage;
}


/**
 * @brief Resizes an image to fit within a target size while preserving the aspect ratio and pads the image to the target size.
 *
 * This function resizes the input image to fit within the specified target size while maintaining the original aspect ratio.
 * If the resized image does not match the target size, it is padded with a constant value (black by default) to reach the target size.
 * Optionally, a maximum upscaling size can be specified to limit the upscaling of the image.
 *
 * @param image The input image to be resized and padded. It should be a `cv::cuda::GpuMat` object.
 * @param targetSize The target size for the output image. It should be a `cv::Size` object representing the width and height.
 * @param mode The interpolation mode to be used for resizing. Currently, only "bilinear" is supported.
 * @param maxUpscalingSize The maximum size to which the image can be upscaled. Default is -1, which means no limit on upscaling.
 *
 * @return A `cv::cuda::GpuMat` object containing the resized and padded image.
 */
cv::cuda::GpuMat resizeWithPad(
    const cv::cuda::GpuMat& image, 
    const cv::Size& targetSize, 
    const std::string& mode, 
    int maxUpscalingSize = -1
){
    cv::Size imageSize = image.size();
    cv::Size targetSizeResize = targetSize;

    if (maxUpscalingSize > 0) {
        int newTargetHeight = std::min(std::max(imageSize.height, maxUpscalingSize), targetSize.height);
        int newTargetWidth = std::min(std::max(imageSize.width, maxUpscalingSize), targetSize.width);
        targetSizeResize = cv::Size(newTargetWidth, newTargetHeight);
    }

    cv::Size newSizePreservingAspectRatio = getNewResWithoutDistortion(imageSize, targetSizeResize);

    cv::cuda::GpuMat resizedImage;
    interpolation(image, newSizePreservingAspectRatio, mode).copyTo(resizedImage);

    return padImageTopLeft(resizedImage, targetSize, cv::Scalar(0, 0, 0));
}



/**
 * @brief Augments the HSV values of an input image by applying random gains to the hue, saturation, and value channels.
 *
 * This function converts the input image from BGR to HSV color space, applies random gains to the hue, saturation, and value channels, and then converts the image back to BGR color space. The random gains are controlled by the `hgain`, `sgain`, and `vagin` parameters.
 *
 * @param inputImage The input image to be augmented. It should be a `cv::cuda::GpuMat` object.
 * @param hgain The gain factor for the hue channel. Default is 0.5.
 * @param sgain The gain factor for the saturation channel. Default is 0.5.
 * @param vagin The gain factor for the value channel. Default is 0.5.
 *
 * @return A `cv::cuda::GpuMat` object containing the augmented image.
 */
cv::cuda::GpuMat augmentHSV(
    const cv::cuda::GpuMat& inputImage, 
    float hgain = 0.5, 
    float sgain = 0.5, 
    float vagin = 0.5
){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

    float randomeHSV[3] = {
        static_cast<float>(dis(gen) * hgain + 1),
        static_cast<float>(dis(gen) * sgain + 1),
        static_cast<float>(dis(gen) * vagin + 1),
    };

    cv::cuda::GpuMat hsvImage;
    cv::cuda::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV); 

    //Get list of matrix (hsv channels)
    std::vector<cv::cuda::GpuMat> hsvChannels;
    cv::cuda::split(hsvImage, hsvChannels);

    //Multiply the hsv channels with the random values
    cv::Mat lut_hue(1, 256, CV_8U);
    cv::Mat lut_sat(1, 256, CV_8U);
    cv::Mat lut_val(1, 256, CV_8U);

    for (int i = 0; i < 256; i++){
        lut_hue.at<uchar>(i) = static_cast<uchar>(static_cast<int>((float) i * randomeHSV[0]) % 180);
        lut_sat.at<uchar>(i) = static_cast<uchar>(std::min(std::max(i * randomeHSV[1], 0.0f), 255.0f));
        lut_val.at<uchar>(i) = static_cast<uchar>(std::min(std::max(i * randomeHSV[2], 0.0f), 255.0f));
    }

    // Create pointer LUT
    cv::Ptr ptr_h = cv::cuda::createLookUpTable(lut_hue);
    cv::Ptr pth_s = cv::cuda::createLookUpTable(lut_sat);
    cv::Ptr pth_v = cv::cuda::createLookUpTable(lut_val);

    //Transform the hsv channels
    ptr_h->transform(hsvChannels[0], hsvChannels[0]);
    pth_s->transform(hsvChannels[1], hsvChannels[1]);
    pth_v->transform(hsvChannels[2], hsvChannels[2]);

    //Merge the hsv channels
    cv::cuda::GpuMat outputImage;
    cv::cuda::merge(hsvChannels, hsvImage);
    cv::cuda::cvtColor(hsvImage, outputImage, cv::COLOR_HSV2BGR);
    return outputImage;
}


/**
 * @brief Performs histogram equalization on the input image using either CLAHE or standard histogram equalization.
 *
 * This function converts the input image from BGR/RGB to YUV color space, applies histogram equalization to the Y channel, and then converts the image back to BGR/RGB color space. The type of histogram equalization (CLAHE or standard) can be specified.
 *
 * @param inputImage The input image to be equalized. It should be a `cv::cuda::GpuMat` object.
 * @param clahe A boolean flag indicating whether to use CLAHE (Contrast Limited Adaptive Histogram Equalization). Default is true.
 *              If false, standard histogram equalization is used.
 * @param bgr A boolean flag indicating whether the input image is in BGR color space. Default is false (assumes RGB color space).
 *
 * @return A `cv::cuda::GpuMat` object containing the histogram equalized image.
 */
cv::cuda::GpuMat histEqualize(
    const cv::cuda::GpuMat& inputImage, 
    bool clahe = true, 
    bool bgr = false
){
    cv::cuda::GpuMat yuv;
    cv::cuda::cvtColor(inputImage, yuv, bgr ? cv::COLOR_BGR2YUV : cv::COLOR_RGB2YUV);

    std::vector<cv::cuda::GpuMat> yuvChannels;
    cv::cuda::split(yuv, yuvChannels);

    if (clahe){
        cv::Ptr<cv::cuda::CLAHE> clahe = cv::cuda::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(yuvChannels[0], yuvChannels[0]);
    }
    else{
        cv::cuda::equalizeHist(yuvChannels[0], yuvChannels[0]);
    }

    cv::cuda::merge(yuvChannels, yuv);
    cv::cuda::GpuMat outputImage = inputImage.clone();
    cv::cuda::cvtColor(yuv, outputImage, bgr ? cv::COLOR_YUV2BGR : cv::COLOR_YUV2RGB);
    return outputImage;
}


/**
 * @brief Flips the input image vertically.
 *
 * This function flips the input image around the x-axis, resulting in a vertical flip.
 *
 * @param inputImage The input image to be flipped. It should be a `cv::cuda::GpuMat` object.
 *
 * @return A `cv::cuda::GpuMat` object containing the vertically flipped image.
 */
cv::cuda::GpuMat flipVertical(
    const cv::cuda::GpuMat& inputImage
){
    cv::cuda::GpuMat flippedImage;
    cv::cuda::flip(inputImage, flippedImage, 1); // 1 means flipping around the x-axis (vertical flip)
    return flippedImage;
}


/**
 * @brief Flips the input image horizontally.
 *
 * This function flips the input image around the y-axis, resulting in a horizontal flip.
 *
 * @param inputImage The input image to be flipped. It should be a `cv::cuda::GpuMat` object.
 *
 * @return A `cv::cuda::GpuMat` object containing the horizontally flipped image.
 */
cv::cuda::GpuMat flipHorizontal(
    const cv::cuda::GpuMat& inputImage
){
    cv::cuda::GpuMat flippedImage;
    cv::cuda::flip(inputImage, flippedImage, 0); // 0 means flipping around the y-axis (horizontal flip)
    return flippedImage;
}

/**
 * @brief Flips the input image both horizontally and vertically.
 * 
 * This function flips the input image around both the x-axis and y-axis, resulting in a full flip.
 * 
 * @param inputImage The input image to be flipped. It should be a `cv::cuda::GpuMat` object.
 * 
 * @return A `cv::cuda::GpuMat` object containing the fully flipped image.
 */
cv::cuda::GpuMat flipFull(
    const cv::cuda::GpuMat& inputImage
){
    cv::cuda::GpuMat flippedImage;
    cv::cuda::flip(inputImage, flippedImage, -1); // 1 means flipping around the y-axis (horizontal flip)
    return flippedImage;
}

/**
 * @brief Scales coordinates from one image shape to another, optionally adjusting for padding and keypoint labels.
 *
 * This function scales the coordinates in the `coords` matrix from the shape of `img0Shape` to the shape of `img1Shape`.
 * It can also handle optional padding and keypoint labels. The scaling is performed using CUDA operations for efficiency.
 *
 * @param coords The coordinates to be scaled. It should be a `cv::cuda::GpuMat` object.
 * @param img1Shape The target image shape to which the coordinates will be scaled.
 * @param img0Shape The original image shape from which the coordinates are scaled.
 * @param ratioPad An optional pair consisting of a scaling factor and padding. If not provided, it is calculated based on the image shapes.
 * @param kptLabel A boolean flag indicating whether the coordinates are keypoint labels. Default is false.
 * @param step The step size for processing coordinates when `kptLabel` is true. Default is 2.
 *
 * @return A `cv::cuda::GpuMat` object containing the scaled coordinates.
 */
cv::cuda::GpuMat scaleCoords(
    cv::cuda::GpuMat& coords, 
    const cv::Size& img1Shape, 
    const cv::Size& img0Shape, 
    std::pair<double, cv::Point2d> ratioPad = {}, 
    int step = 2
){
    double gain;
    cv::Point2d pad;

    if (ratioPad == std::pair<double, cv::Point2d>()) {
        gain = std::min(
            static_cast<double>(img1Shape.height) / img0Shape.height, 
            static_cast<double>(img1Shape.width) / img0Shape.width
        );
        pad = cv::Point2d(
            (img1Shape.height - img0Shape.height * gain) / 2,
            (img1Shape.width - img0Shape.width * gain) / 2
        );
    } else {
        gain = ratioPad.first;
        pad = ratioPad.second;
    }

    // Scale the coordinates
    for (int i = 0; i < coords.cols; i += step) {
        cv::cuda::subtract(coords.col(i), cv::Scalar(pad.x), coords.col(i));
        cv::cuda::subtract(coords.col(i + 1), cv::Scalar(pad.y), coords.col(i + 1));
        cv::cuda::divide(coords.col(i), cv::Scalar(gain), coords.col(i));
        cv::cuda::divide(coords.col(i + 1), cv::Scalar(gain), coords.col(i + 1));
    }

    // Clip the coordinates
    clipCoords(coords, img0Shape, step);
    return coords;
}


/**
 * @brief Rotates an image by a specified angle.
 *
 * This function rotates an input image by a given angle and returns the rotated image.
 * The rotation is performed using the specified size and interpolation mode.
 *
 * @param inputImage A `cv::cuda::GpuMat` containing the input image to be rotated.
 * @param angle The angle by which to rotate the image, in degrees.
 * @param size The size of the output image.
 * @param mode The interpolation mode to be used for rotation. Currently, this parameter is not used in the function.
 *
 * @return A `cv::cuda::GpuMat` containing the rotated image.
 */
cv::cuda::GpuMat rotateImage(
    const cv::cuda::GpuMat& inputImage, 
    double angle, 
    const cv::Size& size, 
    std::string mode
){
    int interpolationFlag;
    if (mode == "nearest"){
        interpolationFlag = cv::INTER_NEAREST;
    }
    else if (mode == "bilinear"){
        interpolationFlag = cv::INTER_LINEAR;
    }
    else if (mode == "bicubic"){
        interpolationFlag = cv::INTER_CUBIC;
    }
    else{
        throw std::invalid_argument("Invalid interpolation mode. Supported modes are 'nearest', 'bilinear', 'bicubic'.");
    }

    // get rotation matrix
    cv::Point2f center_coord(
        static_cast<float>(inputImage.cols-1) / 2.0f,
        static_cast<float>(inputImage.rows-1) / 2.0f
    );
    cv::Mat rotation_matix = getRotationMatrix2D(
        center_coord,
        static_cast<double>(-angle),
        1.0
    );
    // determine bounding rectangle, center not relevant
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), size, angle).boundingRect2f();
    // adjust transformation matrix
    rotation_matix.at<double>(0,2) += bbox.width/2.0 - inputImage.cols/2.0;
    rotation_matix.at<double>(1,2) += bbox.height/2.0 - inputImage.rows/2.0;

    cv::cuda::GpuMat rotatedImage;
    cv::cuda::warpAffine(inputImage, rotatedImage, rotation_matix, bbox.size());
    return rotatedImage;
}


/**
 * @brief Converts bounding box coordinates from (x1, y1, x2, y2) format to (x, y, w, h) format.
 *
 * This function takes a `cv::cuda::GpuMat` containing bounding box coordinates in the format (x1, y1, x2, y2)
 * and converts them to the format (x, y, w, h), where (x, y) is the center of the bounding box, and (w, h) are the width and height.
 *
 * @param x A `cv::cuda::GpuMat` containing bounding box coordinates in (x1, y1, x2, y2) format.
 *
 * @return A `cv::cuda::GpuMat` containing bounding box coordinates in (x, y, w, h) format.
 */
cv::cuda::GpuMat xyxy2xywh(const cv::cuda::GpuMat& x) {
    cv::cuda::GpuMat y(x.size(), x.type());
    cv::cuda::addWeighted(x.col(0), 0.5, x.col(2), 0.5, 0, y.col(0));
    cv::cuda::addWeighted(x.col(1), 0.5, x.col(3), 0.5, 0, y.col(1));
    cv::cuda::subtract(x.col(2), x.col(0), y.col(2));
    cv::cuda::subtract(x.col(3), x.col(1), y.col(3));
    //TODO: implement for kpt_label
    return y;
}


/**
 * @brief Converts bounding box coordinates from (x, y, w, h) format to (x1, y1, x2, y2) format.
 *
 * This function takes a `cv::cuda::GpuMat` containing bounding box coordinates in the format (x, y, w, h)
 * and converts them to the format (x1, y1, x2, y2), where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
 *
 * @param x A `cv::cuda::GpuMat` containing bounding box coordinates in (x, y, w, h) format.
 *
 * @return A `cv::cuda::GpuMat` containing bounding box coordinates in (x1, y1, x2, y2) format.
 */
cv::cuda::GpuMat xywh2xyxy(const cv::cuda::GpuMat& x) {
    cv::cuda::GpuMat y(x.size(), x.type());
    cv::cuda::divide(x.col(2), 2, x.col(2));
    cv::cuda::divide(x.col(3), 2, x.col(3));
    
    cv::cuda::subtract(x.col(0), x.col(2), y.col(0));
    cv::cuda::subtract(x.col(1), x.col(3), y.col(1));
    cv::cuda::add(x.col(0), x.col(2), y.col(2));
    cv::cuda::add(x.col(1), x.col(3), y.col(3));
    //TODO: implement for kpt_label
    return y;
}


/**
 * @brief Convert normalized segments into pixel segments, shape (n,2)
 *
 * This function takes a `cv::cuda::GpuMat` containing normalized segments in the format (x, y)
 * and converts them to absolute coordinates based on the provided image dimensions and padding.
 *
 * @param x A `cv::cuda::GpuMat` containing normalized coordinates in (x, y) format.
 * @param w The width of the image. Default is 640.
 * @param h The height of the image. Default is 640.
 * @param padw The padding width. Default is 0.
 * @param padh The padding height. Default is 0.
 *
 * @return A `cv::cuda::GpuMat` containing absolute coordinates.
 */
//! segmentation
// cv::cuda::GpuMat xyn2xy(
//     const cv::cuda::GpuMat& x, 
//     int w = 640, 
//     int h = 640, 
//     int padw = 0, 
//     int padh = 0
// ){
//     cv::cuda::GpuMat y;
//     x.copyTo(y);
//     cv::cuda::GpuMat w_mat(x.size(), CV_32F, cv::Scalar(w));
//     cv::cuda::GpuMat h_mat(x.size(), CV_32F, cv::Scalar(h));
//     cv::cuda::GpuMat padw_mat(x.size(), CV_32F, cv::Scalar(padw));
//     cv::cuda::GpuMat padh_mat(x.size(), CV_32F, cv::Scalar(padh));
//     cv::cuda::addWeighted(x.colRange(0, 1), w, padw_mat, 1, 0, y.colRange(0, 1));
//     cv::cuda::addWeighted(x.colRange(1, 2), h, padh_mat, 1, 0, y.colRange(1, 2));
//     return y;
// }


/**
 * @brief Converts a segmented region to a bounding box.
 *
 * This function takes a segmented region represented by a `cv::cuda::GpuMat` and converts it to a bounding box.
 * The bounding box is represented by a `cv::cuda::GpuMat` containing the coordinates (x_min, y_min, x_max, y_max).
 *
 * @param segment A `cv::cuda::GpuMat` containing the segmented region.
 * @param width The width of the image. Default is 640.
 * @param height The height of the image. Default is 640.
 *
 * @return A `cv::cuda::GpuMat` containing the bounding box coordinates (x_min, y_min, x_max, y_max).
 */
//! segment is matrix, each element is 1 vector [x, y] 
// cv::cuda::GpuMat segment2box(const cv::cuda::GpuMat& segment, int width = 640, int height = 640) {
//     cv::cuda::GpuMat x, y;
//     cv::cuda::split(segment, std::vector<cv::cuda::GpuMat>{x, y});

//     cv::cuda::GpuMat inside;
//     cv::cuda::compare(x, cv::Scalar(0), inside, cv::CMP_GE);
//     cv::cuda::compare(y, cv::Scalar(0), inside, cv::CMP_GE, inside);
//     cv::cuda::compare(x, cv::Scalar(width), inside, cv::CMP_LE, inside);
//     cv::cuda::compare(y, cv::Scalar(height), inside, cv::CMP_LE, inside);

//     cv::cuda::GpuMat x_inside, y_inside;
//     x.copyTo(x_inside, inside);
//     y.copyTo(y_inside, inside);

//     double x_min, x_max, y_min, y_max;
//     cv::cuda::minMax(x_inside, &x_min, &x_max);
//     cv::cuda::minMax(y_inside, &y_min, &y_max);

//     cv::cuda::GpuMat box(1, 4, CV_32F);
//     if (x_inside.empty()) {
//         box.setTo(cv::Scalar(0));
//     } else {
//         box.at<float>(0) = x_min;
//         box.at<float>(1) = y_min;
//         box.at<float>(2) = x_max;
//         box.at<float>(3) = y_max;
//     }

//     return box;
// }


/**
 * @brief Converts multiple segmented regions to bounding boxes.
 *
 * This function takes a vector of segmented regions represented by `cv::cuda::GpuMat` and converts each to a bounding box.
 * The bounding boxes are concatenated into a single `cv::cuda::GpuMat`.
 *
 * @param segments A vector of `cv::cuda::GpuMat` containing the segmented regions.
 *
 * @return A `cv::cuda::GpuMat` containing the bounding box coordinates for each segmented region.
 */
//! Fix segment2box first
// cv::cuda::GpuMat segments2boxes(const std::vector<cv::cuda::GpuMat>& segments) {
//     std::vector<cv::cuda::GpuMat> boxes;
//     for (const auto& s : segments) {
//         boxes.push_back(segment2box(s));
//     }

//     cv::cuda::GpuMat boxes_mat;
//     cv::cuda::vconcat(boxes, boxes_mat);

//     return boxes_mat;
// }


/**
 * @brief Resamples segmented regions to a fixed number of points.
 *
 * This function takes a vector of segmented regions represented by `cv::cuda::GpuMat` and resamples each to a fixed number of points.
 * The resampled segments are returned as a vector of `cv::cuda::GpuMat`.
 *
 * @param segments A vector of `cv::cuda::GpuMat` containing the segmented regions.
 * @param n The number of points to resample each segment to. Default is 1000.
 *
 * @return A vector of `cv::cuda::GpuMat` containing the resampled segmented regions.
 */
//! Fix segment2box first
// std::vector<cv::cuda::GpuMat> resample_segments(const std::vector<cv::cuda::GpuMat>& segments, int n = 1000) {
//     std::vector<cv::cuda::GpuMat> resampled_segments;
//     for (const auto& s : segments) {
//         cv::cuda::GpuMat x, y;
//         cv::cuda::split(s, std::vector<cv::cuda::GpuMat>{x, y});

//         std::vector<float> xp(x.cols);
//         std::iota(xp.begin(), xp.end(), 0);

//         std::vector<float> x_resampled(n), y_resampled(n);
//         cv::cuda::GpuMat x_resampled_gpu, y_resampled_gpu;
//         cv::cuda::GpuMat xp_gpu(xp);

//         cv::cuda::GpuMat x_gpu, y_gpu;
//         cv::cuda::resize(x, x_resampled_gpu, cv::Size(n, 1), 0, 0, cv::INTER_LINEAR);
//         cv::cuda::resize(y, y_resampled_gpu, cv::Size(n, 1), 0, 0, cv::INTER_LINEAR);

//         cv::cuda::GpuMat resampled_segment;
//         cv::cuda::merge(std::vector<cv::cuda::GpuMat>{x_resampled_gpu, y_resampled_gpu}, resampled_segment);

//         resampled_segments.push_back(resampled_segment);
//     }

//     return resampled_segments;
// }

}