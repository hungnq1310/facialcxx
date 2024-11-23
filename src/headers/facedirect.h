#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

class FaceDirect {
public:
    FaceDirect();
    ~FaceDirect();
    void vis(Mat& img, const Point eye_left, const Point eye_right, const Point nose);
private:

};